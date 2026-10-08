"""
多标签数据加载器
直接从原始图像读取，供CNN和ResNet共用
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EuroSATDataset(Dataset):
    """EuroSAT遥感图像多标签数据集"""

    def __init__(self, data_dir, indices, labels, transform=None):
        """
        Args:
            data_dir: eurosat数据集根目录
            indices: numpy数组，指定使用的样本索引
            labels: numpy数组，所有样本的标签矩阵 (N, 17)
            transform: 数据增强/预处理
        """
        self.data_dir = data_dir
        self.indices = indices
        self.labels = labels
        self.transform = transform

        # 类别文件夹名到图像文件列表的映射
        self.class_folders = ['AnnualCrop', 'Forest', 'HerbaceousVegetation',
                              'Highway', 'Industrial', 'Pasture',
                              'PermanentCrop', 'Residential', 'River', 'SeaLake']

        # 构建所有图像路径列表（按类别文件夹顺序）
        self.image_paths = []
        for class_name in self.class_folders:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                files = sorted([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
                for f in files:
                    self.image_paths.append(os.path.join(class_dir, f))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.image_paths[real_idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # labels[idx] 直接索引第 idx 个训练样本的标签（不是 real_idx）
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def get_data_loaders(data_dir, indices_dir, labels_dir, batch_size=64, num_workers=0):
    """
    创建训练、验证、测试数据加载器

    Args:
        data_dir: eurosat数据集根目录
        indices_dir: R-stage1/data 目录（包含 indices_*.npy）
        labels_dir: labels_*.npy 文件所在目录
        batch_size: 批次大小
        num_workers: 数据加载线程数

    Returns:
        train_loader, val_loader, test_loader
    """
    # 加载索引和标签
    indices_train = np.load(os.path.join(indices_dir, 'indices_train.npy'))
    indices_val = np.load(os.path.join(indices_dir, 'indices_val.npy'))
    indices_test = np.load(os.path.join(indices_dir, 'indices_test.npy'))

    labels_train = np.load(os.path.join(labels_dir, 'labels_train.npy'))
    labels_val = np.load(os.path.join(labels_dir, 'labels_val.npy'))
    labels_test = np.load(os.path.join(labels_dir, 'labels_test.npy'))

    # 图像预处理
    # 训练时使用数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证和测试时不使用数据增强
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = EuroSATDataset(data_dir, indices_train, labels_train, train_transform)
    val_dataset = EuroSATDataset(data_dir, indices_val, labels_val, val_transform)
    test_dataset = EuroSATDataset(data_dir, indices_test, labels_test, val_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载器
    data_dir = 'd:/cxdownload/程序设计/data/eurosat'
    indices_dir = 'd:/cxdownload/程序设计/R-stage1/data'
    labels_dir = 'd:/cxdownload/程序设计/R-stage1/data'

    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir, indices_dir, labels_dir, batch_size=32
    )

    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")

    for images, labels in train_loader:
        print(f"图像批次形状: {images.shape}")
        print(f"标签批次形状: {labels.shape}")
        break
