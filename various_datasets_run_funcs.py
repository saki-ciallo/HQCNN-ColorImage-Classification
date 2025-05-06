import json
import os
import importlib
import pandas as pd

from datetime import datetime

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import  DataLoader

from scripts.dataset import dataset_process as data_funcs
from scripts.dataset import color_preprocess as color_funcs
import hqccnn as hqc

importlib.reload(hqc)
importlib.reload(color_funcs)
importlib.reload(data_funcs)


def run_model(type_info):
    # 使用全局变量选择所需训练集
    train_dataset = type_info['train_dataset']
    test_dataset = type_info['test_dataset']

    if type_info['dataset_pick'] == 'CIFAR100':
        super_class, sub_class_list = data_funcs.cifar100_pick_classes_from_superclass(data_funcs.cifar100_dict,
                                                                                       type_info['superclass_name'],
                                                                                       type_info['subclass_range'])

        print(f"Processing Superclass: {super_class}")
        print(f"Selected Subclasses: {sub_class_list}")
        # 下标和对应标签
        # train_indices, train_labels = data_funcs.cifar100_pick_dataset_mix(
        #                                 type_info['train_map'], super_class, sub_class_list, train_size=400, val_size=100)
        # todo cifar100_pick_dataset 函数出现问题, cifar100_pick_dataset_mix正常
        train_indices, train_labels = data_funcs.cifar100_pick_dataset(type_info['train_map'], super_class, sub_class_list)
        # CIFAR-100测试集用作验证集
        valid_indices, valid_labels = data_funcs.cifar100_pick_dataset(type_info['test_map'], super_class, sub_class_list)
    elif type_info['dataset_pick'] in ['CIFAR10', 'SAT4', 'MNIST']:
        # train_indices, train_labels = data_funcs.general_pick_dataset(train_dataset, type_info['sliding_window'], pick_size=500)
        train_indices, train_labels, valid_indices, valid_labels = data_funcs.general_split_train_valid_dataset(
                                            train_dataset, type_info['sliding_window'], train_size=500, val_size=100, seed=None)
        test_indices, test_labels = data_funcs.general_pick_dataset(test_dataset, type_info['sliding_window'], pick_size=100)

    elif type_info['dataset_pick'] == 'EuroSAT':
        train_indices, train_labels, valid_indices, valid_labels, test_indices, test_labels = data_funcs.special_pick_dataset(
                                            train_dataset, type_info['sliding_window'], train_size=500, valid_size=100, test_size=100)
    # elif type_info['dataset_pick'] == 'SVHN':
    #     train_indices, train_labels = data_funcs.general_pick_dataset(train_dataset, type_info['sliding_window'], pick_size=500)
    #     test_indices, test_labels = data_funcs.general_pick_dataset(test_dataset, type_info['sliding_window'], pick_size=100)

    else:
        raise ValueError(f"Dataset pick: {type_info['dataset_pick']}")

    # CIFAR-100 测试集作为验证集/EuroSAT 加载图片，手动划分/CIFAR-10、SAT4、MNIST有独立测试集
    train_dataset_selected = color_funcs.preprocessed_Dataset(
        dataset=train_dataset,
        indices=train_indices,
        labels=train_labels,
        size=type_info['resize'],
        color_space=type_info['color_space'],
    )
    valid_dataset_selected = color_funcs.preprocessed_Dataset(
        dataset=train_dataset,
        indices=valid_indices,
        labels=valid_labels,
        size=type_info['resize'],
        color_space=type_info['color_space'],
    )

    train_dataloaders = DataLoader(train_dataset_selected, batch_size=type_info['batch_size'], shuffle=True, num_workers=0)
    valid_dataloaders = DataLoader(valid_dataset_selected, batch_size=type_info['batch_size'], shuffle=True, num_workers=0)
    # start training
    if type_info['CNN'] is True:
        print("Building CNN")
        model = hqc.simpleCNN(num_classes=type_info['classes'], linear_size=type_info['linear_size'])
    else:
        model = hqc.QuantumNet(num_classes=type_info['classes'], linear_size=type_info['linear_size'],
                               option_layer=type_info['option_layer'], emb_layer=type_info['emb_layer'])
    if type_info['dataset_pick'] == 'CIFAR100':
        train_epochs_loss, valid_epochs_loss, train_report_full, valid_report_full = hqc.train(model, train_dataloaders=train_dataloaders,
                                                                                               valid_dataloaders=valid_dataloaders,
                                                                                               test_dataloaders=None,
                                                                                               epochs=type_info['epochs'],
                                                                                               lr=type_info['lr'])
        return train_epochs_loss, valid_epochs_loss, train_report_full, valid_report_full
    else:
        # 放在这少套一层if
        test_dataset_selected = color_funcs.preprocessed_Dataset(
            dataset=test_dataset,
            indices=test_indices,
            labels=test_labels,
            size=type_info['resize'],
            color_space=type_info['color_space'],
        )
        test_dataloaders = DataLoader(test_dataset_selected, batch_size=type_info['batch_size'], shuffle=True, num_workers=0)

        train_epochs_loss, valid_epochs_loss, test_epochs_loss, train_report_full, valid_report_full, test_report_full = hqc.train(model,
                                                                                                train_dataloaders=train_dataloaders,
                                                                                                valid_dataloaders=valid_dataloaders,
                                                                                                test_dataloaders=test_dataloaders,
                                                                                                epochs=type_info['epochs'],
                                                                                                lr=type_info['lr'])
        return train_epochs_loss, valid_epochs_loss, test_epochs_loss, train_report_full, valid_report_full, test_report_full


def general_go(info):
    if info['dataset_pick'] == 'CIFAR10':
        train_dataset = CIFAR10(root='./datasets/CIFAR10', train=True, transform=None, download=True)
        test_dataset = CIFAR10(root='./datasets/CIFAR10', train=False, transform=None, download=True)
        tc = train_dataset.classes

    elif info['dataset_pick'] == 'SAT4':
        train_dataset = data_funcs.SAT4(root='./datasets/SAT4', train=True, transform=None)
        test_dataset = data_funcs.SAT4(root='./datasets/SAT4', train=False, transform=None)
        tc = train_dataset.classes

    elif info['dataset_pick'] == 'EuroSAT':
        train_dataset = data_funcs.EuroSAT2Data(root='./datasets/EuroSAT', transform=None, download=True)
        test_dataset = train_dataset
        tc = train_dataset.classes

    # elif info['dataset_pick'] == 'SVHN':
    #     train_dataset = data_funcs.FormatDataShape(dataset_name=info['dataset_pick'], dataset=SVHN(root='./datasets/SVHN', split='train', download=True))
    #     test_dataset = data_funcs.FormatDataShape(dataset_name=info['dataset_pick'], dataset=SVHN(root='./datasets/SVHN', split='test', download=True))
    #     print(train_dataset)
    #     tc = [i for i in range(10)]
    elif info['dataset_pick'] == 'MNIST':
        train_dataset = data_funcs.MNIST2RGB(root='./datasets/MNIST', train=True, transform=None, download=True)
        test_dataset = data_funcs.MNIST2RGB(root='./datasets/MNIST', train=False, transform=None, download=True)
        tc = train_dataset.classes

    # -------------------------------------------------------------------------
    # 1. Image
    dataset_pick = info['dataset_pick']
    color_spaces = info['color_spaces']
    resize = info['resize']
    # 2. Classic
    epochs = info['epochs']
    linear_size = info['linear_size']
    batch_size = info['batch_size']
    learning_rate = info['learning_rate']
    num_classes = info['num_classes']
    # 3. Quantum
    emb_layer = info['emb_layer']
    option_layer = info['option_layer']
    # 4. Other
    test_mode = info['test_mode']
    save_output = info['save_output']
    repeat = info['repeat']
    # -------------------------------------------------------------------------
    if test_mode == 'True':
        sliding_windows = info['test_sliding_window']
        color_spaces = info['test_color_spaces']
    else:
        # 二分类
        if dataset_pick in ['EuroSAT', 'CIFAR10', 'MNIST']:
            sliding_windows = [[i, j] for i in range(10) for j in range(10) if i != j and i < j]
        elif dataset_pick == 'SAT4':
            sliding_windows = [[i, j] for i in range(4) for j in range(4) if i != j and i < j]
        else:
            sliding_windows = [[i, i + 1] for i in range(len(train_dataset.classes) - 1)] # 滑动窗口


    for idx, sliding_window in enumerate(sliding_windows):
        for color_space in color_spaces:
            type_info = {
                'dataset_pick': dataset_pick,
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'sliding_window': sliding_window,
                'classes': num_classes,
                'resize': resize,
                'linear_size': linear_size,  # static
                'batch_size': batch_size,  # static
                'lr': learning_rate,  # static
                'epochs': epochs,  # static
                'color_space': color_space,
                'emb_layer': emb_layer,
                'option_layer': option_layer,
                'CNN': info['CNN']
            }
            slid_category1 = tc[sliding_window[0]]
            slid_category2 = tc[sliding_window[1]]
            slid_category = [tc[slid] for slid in sliding_window]
            print(f"Processing Dataset: {dataset_pick}")
            # print(f"Selected Subclasses: {slid_category1}, {slid_category2}")
            print(f"Selected Classes: {slid_category[0]}, {slid_category[-1]}, Total Classes: {len(slid_category)}")
            # use various environments to run the model
            start_time = datetime.now()
            train_epochs_loss, valid_epochs_loss, test_epochs_loss, train_report_full, valid_report_full, test_report_full = run_model(type_info)
            end_time = datetime.now()
            run_time = (end_time - start_time).total_seconds()
            # -------------------------------------------------------------------------

            if save_output == 'True':
                if test_mode == 'True':
                    # 除了二分类，都在测试中控制参数
                    if info['CNN'] is True:
                        if num_classes == 10:
                            path_pick = f'./results/test_only/{dataset_pick}-CNN-10/repeat{repeat}/{color_space}/emb-{emb_layer}-option-{option_layer}/{slid_category[0]}-{slid_category[-1]}'
                        else:
                            path_pick = f'./results/test_only/{dataset_pick}-CNN/repeat{repeat}/{color_space}/emb-{emb_layer}-option-{option_layer}/{slid_category1}-{slid_category2}-{idx}'
                    # not cnn
                    elif num_classes == 10:
                        path_pick = f'./results/test_only/{dataset_pick}/10classes/repeat{repeat}/{color_space}/emb-{emb_layer}-option-{option_layer}/{slid_category[0]}-{slid_category[-1]}'

                    else: path_pick = f'./results/test_only/{dataset_pick}/repeat{repeat}/{color_space}/emb-{emb_layer}-option-{option_layer}/{slid_category1}-{slid_category2}-{idx}'

                elif info['CNN'] is False:
                    path_pick = f'./results/{dataset_pick}/repeat{repeat}/{color_space}/emb-{emb_layer}-option-{option_layer}/{slid_category1}-{slid_category2}-{idx}'
                else:
                    path_pick = f'./results/{dataset_pick}-CNN/repeat{repeat}/{color_space}/emb-{emb_layer}-option-{option_layer}/{slid_category1}-{slid_category2}-{idx}'


                os.makedirs(path_pick, exist_ok=True)
                # 损失值、正确率保存为csv，便于panda绘图
                df = pd.DataFrame({
                    'epoch': list(range(1, len(train_epochs_loss) + 1)),
                    'train_loss': train_epochs_loss,
                    'valid_loss': valid_epochs_loss,
                    'test_loss': test_epochs_loss,
                    'train_accuracy': [report['accuracy'] for report in train_report_full],
                    'valid_accuracy': [report['accuracy'] for report in valid_report_full],
                    'test_accuracy': [report['accuracy'] for report in test_report_full]
                })
                df.to_csv(f'{path_pick}/training_metrics.csv', index=False)

                with open(f'{path_pick}/train_reports.json', 'w') as f:
                    json.dump(train_report_full, f, indent=4)
                with open(f'{path_pick}/valid_reports.json', 'w') as f:
                    json.dump(valid_report_full, f, indent=4)
                with open(f'{path_pick}/test_reports.json', 'w') as f:
                    json.dump(test_report_full, f, indent=4)

                metadata = {'run_time': run_time}
                with open(f'{path_pick}/metadata.json', 'w') as f:
                    json.dump(metadata, f)
            # ————————————————————————————————————————————————————————————
            print(f"Completed {num_classes}-class task for {info['dataset_pick']} with {color_space}, "
                  f"{slid_category1}-{slid_category2}, "
                  f"run time: {run_time}")
            print('--------')


# MyGO!!!!!
# def cifar100_go(info):
#     # 固定筛选出的数据集，测试不同色彩空间的表现
#     cifar100_train_dataset = CIFAR100(root='./datasets/cifar100', train=True, transform=None, download=True)
#     cifar100_test_dataset = CIFAR100(root='./datasets/cifar100', train=False, transform=None, download=False)
#     cifar100_train_map = data_funcs.cifar100_classes_map_create(cifar100_train_dataset)
#     cifar100_test_map = data_funcs.cifar100_classes_map_create(cifar100_test_dataset)
#     # todo 正常
#
#     # 1. Image
#     dataset_pick = info['dataset_pick']
#     color_spaces = info['color_spaces']
#     train_dataset = cifar100_train_dataset
#     test_dataset = cifar100_test_dataset
#     train_map = cifar100_train_map
#     test_map = cifar100_test_map
#     resize = info['resize']
#     # 2. Classic
#     epochs = info['epochs']
#     linear_size = info['linear_size']
#     batch_size = info['batch_size']
#     learning_rate = info['learning_rate']
#     num_classes = info['num_classes']
#     # 3. Quantum
#     emb_layer = info['emb_layer']
#     option_layer = info['option_layer']
#     # 4. Other
#     test_mode = info['test_mode']
#     save_output = info['save_output']
#     # ————————————————————————————————————————————————————————————
#     if test_mode == 'True':
#         superclass_mapping_sub = {
#             'fruit and vegetables': ['mushroom', 'orange']
#         }
#     else:
#         superclass_mapping_sub = data_funcs.cifar100_dict
#
#     for superclass_name, subclasses in superclass_mapping_sub.items():
#         sliding_windows = [[i, i + 1] for i in range(len(subclasses) - 1)] # 滑动窗口
#         for idx, subclass_range in enumerate(sliding_windows):
#             if test_mode == 'True':
#                 subclass_range = info['test_sliding_window'] # 手动指定 仅作测试
#                 color_spaces = info['test_color_spaces']
#             superclass, selected_subclasses = data_funcs.cifar100_pick_classes_from_superclass(data_funcs.cifar100_dict, superclass_name, subclass_range)
#             # todo 正常
#             for color_space in color_spaces:
#                 type_info = {
#                     'dataset_pick': dataset_pick,
#                     'train_dataset': train_dataset,
#                     'test_dataset': test_dataset,
#                     'train_map': train_map,
#                     'test_map': test_map,
#                     'resize': resize,
#                     'classes': num_classes,
#                     'linear_size': linear_size, # static
#                     'batch_size': batch_size, # static
#                     'lr': learning_rate, # static
#                     'epochs': epochs, # static
#                     'color_space': color_space,
#                     'superclass_name': superclass,
#                     'subclass_range': subclass_range,
#                     'emb_layer': emb_layer,
#                     'option_layer': option_layer
#                 }
#                 # use various environments to run the model
#                 start_time = datetime.now()
#                 train_epochs_loss, valid_epochs_loss, train_report_full, valid_report_full = run_model(type_info)
#                 end_time = datetime.now()
#                 run_time = (end_time - start_time).total_seconds()
#                 # ————————————————————————————————————————————————————————————
#                 if save_output == 'True':
#                     if test_mode == 'True':
#                         path_pick = './results/test_only'
#                     else:
#                         path_pick = f'./results/{dataset_pick}/{color_space}/emb-{emb_layer}-option-{option_layer}/{superclass}-{idx}'
#
#                     os.makedirs(path_pick, exist_ok=True)
#                     # 损失值、正确率保存为csv，便于panda绘图
#                     df = pd.DataFrame({
#                         'epoch': list(range(1, len(train_epochs_loss) + 1)),
#                         'train_loss': train_epochs_loss,
#                         'valid_loss': valid_epochs_loss,
#                         'train_accuracy': [report['accuracy'] for report in train_report_full],
#                         'valid_accuracy': [report['accuracy'] for report in valid_report_full]
#                     })
#                     df.to_csv(f'{path_pick}/training_metrics.csv', index=False)
#
#                     with open(f'{path_pick}/train_reports.json', 'w') as f:
#                         json.dump(train_report_full, f, indent=4)
#                     with open(f'{path_pick}/valid_reports.json', 'w') as f:
#                         json.dump(valid_report_full, f, indent=4)
#
#                     metadata = {'run_time': run_time}
#                     with open(f'{path_pick}/metadata.json', 'w') as f:
#                         json.dump(metadata, f)
#                     # /results/cifar100/rgb/emb-type1-option-type1/fruit and vegetables-0/scale-0_1
#                 # ————————————————————————————————————————————————————————————
#                 print(f"Completed {num_classes}-class task for {superclass} with {color_space},"
#                         f"subclasses: {selected_subclasses}, run time: {run_time}")
#                 print('--------')
