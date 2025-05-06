import random
import os
import re
import torch
import numpy as np
import scipy.io as sio
from torchvision import transforms
from PIL import Image
from torchvision.datasets import EuroSAT, MNIST
from torch.utils.data import Dataset

cifar100_dict = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}

# Step 1
# CIFAR-100 超类-子类映射处理
def cifar100_classes_map_create(input_dataset) -> dict:
    # todo 函数正常
    # 获取子类名称列表和样本标签
    classes = input_dataset.classes
    targets = input_dataset.targets

    # 创建子类到超类的映射
    subclass_to_superclass = {}
    for superclass, subclasses in cifar100_dict.items():
        for subclass in subclasses:
            subclass_to_superclass[subclass] = superclass

    # 初始化超类索引字典
    superclass_indices = {superclass: {subclass: [] for subclass in subclasses}
                          for superclass, subclasses in cifar100_dict.items()}
    # 遍历每个样本，记录索引
    for index, target in enumerate(targets):
        subclass_name = classes[target]
        superclass = subclass_to_superclass.get(subclass_name) # 找到子类对应的超类
        if superclass:  # 确保子类有对应的超类
            superclass_indices[superclass][subclass_name].append(index)

    return superclass_indices
    
def cifar100_show_classes(superclass_indices):
    # todo 函数正常
    # show the superclasses and their corresponding subclass indices 
    # limited to the first 5 indices 
    for superclass, subclasses in superclass_indices.items():
        print(f"{superclass}:")
        for subclass, indices in subclasses.items():
            print(f"  {subclass}: {indices[:5]}... total:{len(indices)}")

# Step 2
def cifar100_pick_classes_from_superclass(superclass_mapping, superclass_name, subclass_range):
    # todo 函数正常
    # 完整映射，需要执行的超类，
    subclasses = superclass_mapping.get(superclass_name, [])
    if len(subclasses) == 0:
        raise ValueError(f"Input superclass: '{superclass_name}' does not exist or not found subclasses.")
    if subclass_range[-1] >= len(subclasses):
        raise ValueError(f"Over range. Superclass: '{superclass_name}' only have {len(subclasses)} subclasses.")

    selected_subclasses = subclasses[subclass_range[0]:subclass_range[1] + 1]
    return superclass_name, selected_subclasses

# Step 3
def cifar100_pick_dataset_mix(dataset, superclass, classes, train_size=400, val_size=100, seed=None):
    if seed is not None:
        random.seed(seed)
    train_indices = []
    train_labels = []
    valid_indices = []
    valid_labels = []
    
    try:
        subclasses = dataset[superclass]
    except KeyError:
        raise ValueError(f"Superclass: '{superclass}' not found, check for input errors.")

    for label, subclass in enumerate(classes):
        try:
            subclass_indices = subclasses[subclass]  # 该子类的所有索引（500 张）
            if len(subclass_indices) == 500:
                # 随机打乱索引并划分
                indices_shuffled = random.sample(subclass_indices, len(subclass_indices))  # 打乱顺序
                train_subset = indices_shuffled[0:train_size]  # 前 400 张作为训练集
                train_indices.extend(train_subset)
                val_subset = indices_shuffled[train_size:train_size + val_size]  # 后 100 张作为验证集

                # 添加到训练集
                train_indices.extend(train_subset)
                train_labels.extend([label] * train_size)

                # 添加到验证集
                valid_indices.extend(val_subset)
                valid_labels.extend([label] * val_size)
        except KeyError:
            print(f"Warning: subclass: '{subclass}' is not in the superclass: '{superclass}'.")

    if not train_indices or not valid_indices:
        print(f"Warning: In superclass: '{superclass}' not found sufficient indices.")
        print("Check the code or input classes.")

    train_indices.extend(valid_indices)
    train_labels.extend(valid_labels)

    return train_indices, train_labels

def cifar100_pick_dataset(dataset, superclass, classes):
    # todo 函数正常
    indices = []
    labels = []

    try:
        subclasses = dataset[superclass] # 从完整映射字典选取超类，得到子类的映射字典
    except KeyError:
        raise ValueError(f"Superclass: '{superclass}' not found, check for input errors.")

    for label, subclass in enumerate(classes):
        # 0 mushroom, 1 orange
        try:
            subclass_indices = subclasses[subclass]
            # cifar-100不需要随机选取，因为训练集只有500张，测试集只有100张，刚好符合实验要求
            # indices_shuffled = random.sample(subclass_indices, len(subclass_indices))
            indices.extend(subclass_indices)  # add indices to the subclasses
            labels.extend([label] * len(subclass_indices))
        except KeyError:

            print(f"Warning: subclass: '{subclass}' is not in the superclass: '{superclass}'.")

    if not indices:
        print(f"Warning: In superclass: '{superclass}' not found any indices. Output indices could be empty!")
        print("Check the code, there have some problem.")
    return indices, labels

def general_pick_dataset(dataset, sliding_window, pick_size):
    try:
        dataset_name = dataset.root.split('/')[-1]
        if dataset_name == 'SVHN':
            all_classes = list(range(10))
            selected_classes = sliding_window
            targets = dataset.labels
        elif dataset_name in ['CIFAR10', 'SAT4', 'MNIST']:
            all_classes = dataset.classes
            selected_classes = [all_classes[i] for i in sliding_window]
            targets = dataset.targets
        else:
            raise ValueError(f"Unsupported dataset: '{dataset_name}'.")
    except AttributeError:
        raise ValueError(f"Input dataset: '{dataset.root}' is not a valid dataset.")

    indices = []
    labels = []

    for label, class_name in enumerate(selected_classes):
        # Find indices of samples belonging to this class
        class_indices = [i for i, target in enumerate(targets) if all_classes[target] == class_name]
        random.shuffle(class_indices)
        class_indices = class_indices[:pick_size]
        indices.extend(class_indices)
        labels.extend([label] * len(class_indices))  # Assign 0 or 1 based on position in sliding_window

    return indices, labels


def general_split_train_valid_dataset(dataset, sliding_window, train_size=500, val_size=100, seed=None) :
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 检查数据集类型并获取类别和目标
    try:
        dataset_name = dataset.root.split('/')[-1]
        if dataset_name == 'SVHN':
            all_classes = list(range(10))
            selected_classes = sliding_window
            targets = dataset.labels
        elif dataset_name in ['CIFAR10', 'SAT4', 'MNIST']:
            all_classes = dataset.classes
            selected_classes = [all_classes[i] for i in sliding_window]
            targets = dataset.targets
        else:
            raise ValueError(f"Unsupported dataset: '{dataset_name}'.")
    except AttributeError:
        raise ValueError(f"Input dataset: '{dataset.root}' is not a valid dataset.")

    total_size = train_size + val_size  # 每类总选取数量
    train_indices = []
    train_labels = []
    valid_indices = []
    valid_labels = []

    # 对每个选定类别进行处理
    for label, class_name in enumerate(selected_classes):
        # 获取该类的所有索引
        # if dataset_name == 'SVHN':
        #     class_indices = np.where(np.array(targets) == class_name)[0].tolist()
        # else:
        class_indices = [i for i, target in enumerate(targets) if all_classes[target] == class_name]

        # 检查可用样本数
        if len(class_indices) < total_size:
            raise ValueError(
                f"Class '{class_name}' has only {len(class_indices)} samples, but {total_size} are required.")

        # 高效随机采样
        sampled_indices = random.sample(class_indices, total_size)  # 一次性采样 total_size 个
        train_subset = sampled_indices[:train_size]  # 前 train_size 个作为训练集
        val_subset = sampled_indices[train_size:total_size]  # 后 val_size 个作为验证集

        # 添加到训练集
        train_indices.extend(train_subset)
        train_labels.extend([label] * train_size)

        # 添加到验证集
        valid_indices.extend(val_subset)
        valid_labels.extend([label] * val_size)

    return train_indices, train_labels, valid_indices, valid_labels

def special_pick_dataset(dataset, sliding_window, train_size=400, valid_size=100, test_size=100, seed=None):
    # 只用于本身未做处理数据集的划分，训练集、验证集、测试集
    # 目前支持 EuroSAT, targets完全有序, classes按顺序匹配。
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_classes = dataset.classes  # List of class names
    selected_classes = [all_classes[i] for i in sliding_window]  # Names of the two selected classes
    train_indices = []
    train_labels = []
    valid_indices = []
    valid_labels = []
    test_indices = []
    test_labels = []
    total_size = train_size + valid_size + test_size
    for label, class_name in enumerate(selected_classes):

        # EuroSAT有序的情况下，该方法为遍历搜索。
        class_indices = [i for i, target in enumerate(dataset.targets) if all_classes[target] == class_name]
        if len(class_indices) < total_size:
            raise ValueError(
                f"Class '{class_name}' has only {len(class_indices)} samples, but {total_size} are required.")
        random.shuffle(class_indices)
        # 划分训练集、验证集、测试集
        t_indices = class_indices[:train_size]  # 前 trange 个作为训练集
        v_indices = class_indices[train_size:train_size + valid_size]  # 接下来 vrange 个作为验证集
        test_indices_subset = class_indices[train_size + valid_size:train_size + valid_size + test_size]  # 最后 test_range 个作为测试集

        # 添加到相应列表
        train_indices.extend(t_indices)
        train_labels.extend([label] * train_size)
        valid_indices.extend(v_indices)
        valid_labels.extend([label] * valid_size)
        test_indices.extend(test_indices_subset)
        test_labels.extend([label] * test_size)

    return train_indices, train_labels, valid_indices, valid_labels, test_indices, test_labels


class EuroSAT2Data(EuroSAT):
    def __init__(self, root, transform=None, target_transform=None, download=False, cache_path='eurosat_data.npy'):
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.cache_path = os.path.join(root, cache_path)

        # 检查本地缓存是否存在
        if os.path.exists(self.cache_path):
            self.data = np.load(self.cache_path)
        else:
            # 创建data属性 (batch,h,w,c)，与CIFAR10数据集类似
            self.data = self._load_images()
            np.save(self.cache_path, self.data)

    def _load_images(self):
        """加载所有图像并返回 NumPy 数组"""
        data = []
        for img_path, _ in self.imgs: # list('path', label)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            data.append(img_array)
        # list使用stack堆叠 (27000, 64, 64, 3)
        return np.stack(data)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target

class SAT4(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ['barren land', 'trees', 'grassland', 'none']

        if self.train:
            cache_path = os.path.join(root, 'sat4_train.npy')
        else:
            cache_path = os.path.join(root, 'sat4_test.npy')

        if os.path.exists(cache_path):
            cached_data = np.load(cache_path, allow_pickle=True).item()
            self.data = torch.tensor(cached_data['data'], dtype=torch.float32)
            self.targets = torch.tensor(cached_data['targets'], dtype=torch.long)
            self.classes = cached_data['classes']
        else:
            data = sio.loadmat(os.path.join(root, 'sat-4-full.mat'))
            if self.train:
                images = data['train_x']  # Shape: (28, 28, 4, 400000)
                labels = data['train_y']  # Shape: (4, 400000)
            else:
                images = data['test_x']
                labels = data['test_y']

            # Preprocess images: (28, 28, 4, n) -> (n, 28, 28, 4)
            images = np.transpose(images, (3, 0, 1, 2))
            images = images[..., :3] # 剔除红外通道
            # Preprocess labels: (4, n) -> (n, 4) -> (n,) as indices
            labels = np.transpose(labels, (1, 0))  # [n, 4]
            labels = np.argmax(labels, axis=1) # 1000->0, 0100->1, 0010->2, 0001->3

            np.save(cache_path, {
                'data': images,
                'targets': labels,
                'classes': self.classes
            })

            self.data = torch.tensor(images, dtype=torch.float32)
            self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


# class FormatDataShape(Dataset):
#     def __init__(self, dataset_name, dataset):
#         self.dataset_name = dataset_name.lower()
#         # 根据数据集名称调整 data 格式
#         if self.dataset_name == "SVHN":
#             # SVHN: [n, c, h, w] -> [n, h, w, c]
#             self.data = np.transpose(dataset.data, (0, 2, 3, 1))
#             self.targets = dataset.labels  # 保持标签原样
#             self.root = dataset.root
#         elif self.dataset_name == "CIFAR10":
#             # CIFAR10: 已为 [n, h, w, c]，无需调整
#             self.data = dataset.data
#             self.targets = dataset.targets  # 保持标签原样
#         else:
#             raise ValueError(f"Unsupported dataset: {dataset_name}")
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         image = self.data[index]  # [h, w, c]
#         target = self.targets[index]  # 保持原样
#
#         return image, target


class MNIST2RGB(MNIST):
    def __init__(self, root, transform=None, target_transform=None, download=False, train=True):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

        cache_path = os.path.join(root, f'mnist2rgb_{"train" if train else "test"}.npy')

        if os.path.exists(cache_path):
            cached_data = np.load(cache_path, allow_pickle=True).item()
            self.data = cached_data["data"]
            self.targets = cached_data["targets"]
        else:
            self.data = self.data.unsqueeze(-1).repeat(1, 1, 1, 3)  # 复制灰度通道到 RGB 通道

            np.save(cache_path, {"data": self.data, "targets": self.targets})

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = transforms.ToPILImage()(img)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target


# def rename_tiny_imagenet():
#     base_dir = "./datasets/tiny-imagenet-200"
#     train_dir = os.path.join(base_dir, "train")
#     words_file = os.path.join(base_dir, "words.txt")
#
#     # 读取 words.txt 文件并构建映射字典
#     id_to_name = {}
#     with open(words_file, "r") as f:
#         for line in f:
#             parts = line.strip().split("\t")
#             if len(parts) == 2:
#                 n_id, class_name = parts
#                 class_name = class_name.split(",")[0].strip()
#                 id_to_name[n_id] = class_name
#
#     # 重命名 train 文件夹中的子文件夹
#     for folder_name in os.listdir(train_dir):
#         folder_path = os.path.join(train_dir, folder_name)
#         if os.path.isdir(folder_path):
#             if folder_name in id_to_name:
#                 new_name = id_to_name[folder_name]
#                 new_folder_path = os.path.join(train_dir, new_name)
#                 try:
#                     # 检查目标文件夹是否已存在
#                     if os.path.exists(new_folder_path):
#                         print(f"Warning: {new_name} already exists, skipping {folder_name}")
#                     else:
#                         os.rename(folder_path, new_folder_path)
#                         print(f"Renamed {folder_name} to {new_name}")
#                 except OSError as e:
#                     print(f"Error renaming {folder_name} to {new_name}: {e}")
#             else:
#                 print(f"No class name found for {folder_name}")
#
#     print("Folder renaming completed.")