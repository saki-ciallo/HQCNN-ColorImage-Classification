import torch
from torch import nn
import pennylane as qml
import numpy as np

# from tqdm import tqdm
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


from scripts.qcnn.component import q_emb_layer
from scripts.qcnn.component import q_conv_layer
from scripts.qcnn.component import q_pool_layer

import importlib
importlib.reload(q_emb_layer)
importlib.reload(q_pool_layer)
importlib.reload(q_conv_layer)


class QuantumLayer(nn.Module):
    def __init__(self, stride=1, device="default.qubit", num_qubit=6, out_channels=1, option_layer=None, emb_layer=None,
                 qc2_Ising=None):
        super(QuantumLayer, self).__init__()
        # init device
        self.num_qubit = num_qubit
        self.dev = qml.device(device, wires=self.num_qubit)
        self.out_channels = out_channels
        self.stride = stride
        self.emb_layer = emb_layer
        self.option_layer = option_layer
        self.qc2_Ising = qc2_Ising

        @qml.qnode(device=self.dev, interface="torch")
        def circuit(inputs, weights):
            # Embedding Layer
            if self.emb_layer == 'type1':
                # YZ
                q_emb_layer.emb_type1(inputs, self.num_qubit)  # example: [[a00,a01],[a10,a11]] --> q0: Y_a00, Z_a10; q1: Y_a01, Z_a11
            elif self.emb_layer == 'type2':
                # XZ
                q_emb_layer.emb_type2(inputs, self.num_qubit)
            elif self.emb_layer == 'type3':
                # XY
                q_emb_layer.emb_type3(inputs, self.num_qubit)
            else:
                raise NotImplementedError
            # -------------------------------------------------------------------------------------------------------------#

            # Quantum Convolutional Layer 1
            conv1_params = 6
            q_conv_layer.conv_layer1_type1(weights[0:conv1_params], self.num_qubit) # 6
            # -------------------------------------------------------------------------------------------------------------#

            # Optional Addition Layer 1
            #
            option_params = 0
            if self.option_layer == 'type1':
                # CRX
                option_params = conv1_params + 3
                q_conv_layer.option_layer_type1(weights[conv1_params:option_params], self.num_qubit) # 3
            elif self.option_layer == 'type2':
                # CRY
                option_params = conv1_params + 3
                q_conv_layer.option_layer_type2(weights[conv1_params:option_params], self.num_qubit)  # 3
            elif self.option_layer == 'type3':
                # CRZ
                option_params = conv1_params + 3
                q_conv_layer.option_layer_type3(weights[conv1_params:option_params], self.num_qubit)  # 3
            else:
                option_params = conv1_params
            # -------------------------------------------------------------------------------------------------------------#

            # Quantum Pooling Layer 1
            pool1_params = option_params + 3
            q_pool_layer.pool_layer1_type1(weights[option_params:pool1_params], self.num_qubit) # 3
            # -------------------------------------------------------------------------------------------------------------#

            # Quantum Convolutional Layer 2
            conv2_params = pool1_params + 13
            q_conv_layer.conv_layer2_type1(weights[pool1_params:conv2_params], self.num_qubit) # 13
            # -------------------------------------------------------------------------------------------------------------#

            # Quantum Pooling Layer 2
            pool2_params = conv2_params + 2
            q_pool_layer.pool_layer2_type1(weights[conv2_params:pool2_params], self.num_qubit) # 2
            # -------------------------------------------------------------------------------------------------------------#

            # qml.Barrier(wires=list(range(self.num_qubit)), only_visual=True)
            # qml.ArbitraryUnitary(weights_Arb, wires=[0,4]) # 4**n - 1, n=2, 15 params
            # -------------------------------------------------------------------------------------------------------------#

            return qml.expval(qml.PauliZ(0))

        if self.option_layer in ['type1', 'type2', 'type3']:
            weight_shapes = {"weights": 27}
        else:
            weight_shapes = {"weights": 24}  # cp2

            # weight_shapes = {"weights": 48 if self.qc2_Ising == 'IsingXY' else 50, "weights_Arb": 15}
        self.circuit = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)

    # def _quanv(self, image):
    #     # 10*10
    #     h, w, _ = image.size()
    #     kernel_size = 2
    #     h_out = (h - kernel_size) // self.stride + 1
    #     w_out = (w - kernel_size) // self.stride + 1
    #
    #     # 使用 Unfold 提取所有 2×2 窗口
    #     unfold = torch.nn.Unfold(kernel_size=2, stride=self.stride)
    #     patches = unfold(image.permute(2, 0, 1).unsqueeze(0))  # (1, 12, h_out * w_out)
    #     img_list = patches.squeeze(0).T  # (h_out * w_out, 12)
    #     expavals = self.circuit(img_list)  # Parameter Broadcasting in QNodes
    #
    #     temp_array = torch.zeros((1, h_out, w_out, self.out_channels))
    #     temp_array[0, :, :, 0] = expavals.detach().view(h_out, w_out)  # new version: fewer words, more powerful
    #
    #     return temp_array

    def _quanv_optimized(self, image):
        # print(image.size())
        batch_size, h, w, _ = image.size()  # image: (batch_size, h, w, 3)
        kernel_size = 2  # 假设卷积核大小为 2×2
        h_out = (h - kernel_size) // self.stride + 1
        w_out = (w - kernel_size) // self.stride + 1

        # 使用 Unfold 提取所有 2×2 窗口
        unfold = torch.nn.Unfold(kernel_size=2, stride=self.stride)
        patches = unfold(image.permute(0, 3, 1, 2))  # (batch_size, 3, h, w) -> (batch_size, 12, h_out * w_out)
        # 调整形状以适应量子电路
        img_list = patches.permute(0, 2, 1).reshape(-1, 12)  # (batch_size * h_out * w_out, 12)
        # 应用量子电路（假设 self.circuit 处理 (N, 12) 的输入）
        expavals = self.circuit(img_list)  # (batch_size * h_out * w_out)
        output = expavals.view(batch_size, h_out * w_out) # 重新按顺序整理成有batch的结果
        return output

    # def _evaluate_qnode(self, x):
        # batch_round = x.shape[0]
        # res_list = self._quanv(x[0])
        # if batch_round > 1:
        #     for i in range(1, batch_round):
        #         temp = self._quanv(x[i])
        #         res_list = torch.cat([res_list, temp], 0)
        #         # torch.Size([32, 9, 9, 1])
        # return res_list
        # return self._quanv1(x)

    def forward(self, inputs):
        # results = self._evaluate_qnode(inputs)
        results = self._quanv_optimized(inputs)

        return results





def train(model, train_dataloaders, valid_dataloaders, test_dataloaders, epochs, lr):
    print("Starting Training for {} epochs".format(epochs))
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-5)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # base on train set loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-5) # base on valid set loss
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    train_epochs_loss = []
    valid_epochs_loss = []
    test_epochs_loss = []

    train_report_full = []
    valid_report_full = []
    test_report_full = []

    for epoch in range(epochs):
        # MODEL TRAIN START
        model.train()
        train_iteration_loss = [] # 用于平均训练集的每个epoch的loss
        train_targets_all = []
        train_predictions_all = []
        # loop = tqdm(enumerate(train_dataloaders), total=len(train_dataloaders))
        # for batch_idx, (data, targets) in loop:
        for batch_idx, (data, targets) in enumerate(train_dataloaders):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            # print(torch.nn.utils.clip_grad_norm_(model.parameters(), 1)) # 打印梯度
            # print(outputs)
            # print(targets)

            optimizer.step()

            train_iteration_loss.append(loss.item())
            predictions = outputs.argmax(-1).numpy()
            train_targets_all.extend(targets)
            train_predictions_all.extend(predictions)

            # loop.set_description(f"Epoch [{epoch + 1}/{epochs}] ")
            # loop.set_postfix(loss=loss.item(), acc=acc)

        train_perf_report = classification_report(train_targets_all, train_predictions_all, output_dict=True) # dict_keys(['0', '1', 'accuracy', 'macro avg', 'weighted avg'])
        train_epochs_loss.append(np.average(train_iteration_loss))
        train_report_full.append(train_perf_report)  # 训练集的report保存

        # scheduler.step() # StepLR

        # MODEL VALID START
        model.eval()
        valid_epoch_loss = []
        valid_targets_all = []
        valid_predictions_all = []
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(valid_dataloaders):
                outputs = model(data)
                loss = criterion(outputs, targets)

                valid_epoch_loss.append(loss.item())
                predictions = outputs.argmax(-1).numpy()
                valid_targets_all.extend(targets)
                valid_predictions_all.extend(predictions)


        valid_perf_report = classification_report(valid_targets_all, valid_predictions_all, output_dict=True)
        valid_report_full.append(valid_perf_report)  # 验证集的report保存

        avg_valid_loss = np.average(valid_epoch_loss)
        valid_epochs_loss.append(avg_valid_loss)

        scheduler.step(avg_valid_loss)  # ReduceLROnPlateau
        # scheduler.step()  # CosineAnnealingLR

        # valid set loss, acc and learning rate
        # current_lr = optimizer.param_groups[0]['lr']
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"[Validation-Epoch: {epoch + 1}] Loss: {avg_valid_loss:.4f}, Acc: {valid_perf_report['accuracy']:.4f}, precision: {valid_perf_report['macro avg']['precision']:.4f}, "
            f"recall: {valid_perf_report['macro avg']['recall']:.4f}, f1-score: {valid_perf_report['macro avg']['f1-score']:.4f}, Lr: {current_lr:.5f}"
        )

        if test_dataloaders is not None:
            # MODEL TEST START
            model.eval()
            test_epoch_loss = []
            test_targets_all = []
            test_predictions_all = []
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(test_dataloaders):
                    outputs = model(data)
                    loss = criterion(outputs, targets)

                    test_epoch_loss.append(loss.item())
                    predictions = outputs.argmax(-1).numpy()
                    test_targets_all.extend(targets)
                    test_predictions_all.extend(predictions)

            test_perf_report = classification_report(test_targets_all, test_predictions_all, output_dict=True)
            test_report_full.append(test_perf_report)  # 验证集的report保存

            avg_test_loss = np.average(test_epoch_loss)
            test_epochs_loss.append(avg_test_loss)

            # test set loss, acc and learning rate
            print(
                f"[Test-Epoch: {epoch + 1}] Loss: {avg_test_loss:.4f}, Acc: {test_perf_report['accuracy']:.4f}, precision: {test_perf_report['macro avg']['precision']:.4f}, "
                f"recall: {test_perf_report['macro avg']['recall']:.4f}, f1-score: {test_perf_report['macro avg']['f1-score']:.4f}, Lr: {current_lr:.5f}"
            )

    if test_dataloaders is not None:
        best_entry = max(enumerate(test_report_full), key=lambda x: x[1]['accuracy'])
        # enumerate -> (index, report)
        # x: (index, report), x[1]: report, 取正确率。max之后，形如 (index, {'accuracy'}: 0.95)
        max_epoch, best_report = best_entry
        max_accuracy = best_report['accuracy']
        print(f"Training completed, test-set max accuracy: {max_accuracy}, epoch: {max_epoch + 1}, Lr: {current_lr:.5f}")

        return train_epochs_loss, valid_epochs_loss, test_epochs_loss, train_report_full, valid_report_full, test_report_full
    else:
        best_entry = max(enumerate(valid_report_full), key=lambda x: x[1]['accuracy'])
        # enumerate -> (index, report)
        # x: (index, report), x[1]: report, 取正确率。max之后，形如 (index, {'accuracy'}: 0.95)
        max_epoch, best_report = best_entry
        max_accuracy = best_report['accuracy']
        print(f"Training completed, valid-set max accuracy: {max_accuracy}, epoch: {max_epoch + 1}, Lr: {current_lr:.5f}")

        return train_epochs_loss, valid_epochs_loss, train_report_full, valid_report_full


class simpleCNN(nn.Module):
    def __init__(self, num_classes, linear_size):
        super(simpleCNN, self).__init__()
        self.linear_size = linear_size
        # 卷积块1：输入 10x10x3 -> 输出 8x8x16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出 4x4x16

        # 卷积块2：输入 4x4x16 -> 输出 2x2x32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(24)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出 1x1x32

        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(out_features=self.linear_size)  # 自动推断输入维度
        self.bn_fc = nn.LazyBatchNorm1d()
        # self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(self.linear_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # 卷积块1
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # 卷积块2
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.flatten(x)  # 展平
        x = self.bn_fc(x)
        x = self.fc1(x)
        x = self.activation(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

class QuantumNet(nn.Module):
    def __init__(self, num_classes, linear_size, option_layer='type1', emb_layer='type1'):
        super(QuantumNet, self).__init__()
        self.linear_size = linear_size
        self.quantum_layer = QuantumLayer(option_layer=option_layer, emb_layer=emb_layer)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(out_features=linear_size)  # 自动推断输入维度
        self.bn_fc = nn.LazyBatchNorm1d()
        # self.bn_fc = nn.BatchNorm1d(32)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        # self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(self.linear_size, num_classes)
        # self.fc2 = nn.LazyLinear(out_features=num_classes)

    def forward(self, x):
        # 带batch进行，目前的瓶颈是quantum_layer
        x = self.quantum_layer(x)
        x = self.flatten(x) # [batch, flattened size]
        x = self.bn_fc(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

