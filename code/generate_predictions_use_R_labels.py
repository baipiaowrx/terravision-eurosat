"""
生成阶段三补充交付物：测试集预测概率
使用R提供的labels_test.npy作为标签来源
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

MAIN_LABELS = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

class EuroSATDataset(Dataset):
    """使用R提供的标签"""
    def __init__(self, data_dir, indices, labels, transform=None):
        self.data_dir = data_dir
        self.indices = indices
        self.labels = labels
        self.transform = transform

        self.image_paths = []
        for class_name in MAIN_LABELS:
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

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 使用R提供的标签
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

class CustomCNN(nn.Module):
    def __init__(self, num_labels=17):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_labels=17, pretrained=True):
        super(ResNet50MultiLabel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_labels)

    def forward(self, x):
        return self.resnet(x)

def get_predictions(model, data_loader, device):
    """获取模型在测试集上的预测概率"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()  # sigmoid转为概率
            all_probs.append(probs)
    return np.vstack(all_probs)

def main():
    data_dir = 'd:/cxdownload/程序设计/data/eurosat'
    indices_dir = 'd:/cxdownload/程序设计/R-stage1/data'
    save_dir = 'd:/cxdownload/程序设计/W-stage2'

    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    # 加载R提供的索引和标签
    indices_test = np.load(os.path.join(indices_dir, 'indices_test.npy'))
    labels_test = np.load(os.path.join(indices_dir, 'labels_test.npy'))

    print(f"测试集大小: {len(indices_test)}")
    print(f"标签形状: {labels_test.shape}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 使用R提供的标签
    test_dataset = EuroSATDataset(data_dir, indices_test, labels_test, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("\n=== 加载CNN模型并生成预测 ===")
    cnn_model = CustomCNN(num_labels=17).to(device)
    cnn_model.load_state_dict(torch.load(os.path.join(save_dir, 'results/cnn_best.pth'), map_location=device))
    cnn_probs = get_predictions(cnn_model, test_loader, device)
    print(f"CNN预测概率形状: {cnn_probs.shape}")
    print(f"CNN预测概率范围: [{cnn_probs.min():.4f}, {cnn_probs.max():.4f}]")
    np.save(os.path.join(save_dir, 'predictions/predictions_cnn_test.npy'), cnn_probs)

    print("\n=== 加载ResNet-50模型并生成预测 ===")
    resnet_model = ResNet50MultiLabel(num_labels=17, pretrained=False).to(device)
    resnet_model.load_state_dict(torch.load(os.path.join(save_dir, 'results/resnet50_best.pth'), map_location=device))
    resnet_probs = get_predictions(resnet_model, test_loader, device)
    print(f"ResNet-50预测概率形状: {resnet_probs.shape}")
    print(f"ResNet-50预测概率范围: [{resnet_probs.min():.4f}, {resnet_probs.max():.4f}]")
    np.save(os.path.join(save_dir, 'predictions/predictions_resnet_test.npy'), resnet_probs)

    print("\n=== 补充交付物生成完成 ===")
    print("文件列表:")
    print(f"- {save_dir}/predictions/predictions_cnn_test.npy")
    print(f"- {save_dir}/predictions/predictions_resnet_test.npy")

if __name__ == '__main__':
    main()