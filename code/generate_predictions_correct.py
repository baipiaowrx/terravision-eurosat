"""
生成阶段三补充交付物：测试集预测概率
使用正确的标签（从文件名解析）验证，确保与dl_results.csv一致
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import average_precision_score

MULTI_LABEL_MAP = {
    'AnnualCrop': ['Vegetation', 'OpenLand'],
    'Forest': ['Vegetation', 'NaturalArea'],
    'HerbaceousVegetation': ['Vegetation', 'OpenLand'],
    'Highway': ['Artificial', 'Transportation'],
    'Industrial': ['Artificial', 'Urban'],
    'Pasture': ['Vegetation', 'OpenLand'],
    'PermanentCrop': ['Vegetation', 'OpenLand'],
    'Residential': ['Artificial', 'Urban'],
    'River': ['Water', 'NaturalArea'],
    'SeaLake': ['Water', 'NaturalArea']
}

MAIN_LABELS = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
AUX_LABELS = ['Vegetation', 'OpenLand', 'Artificial', 'Urban', 'Transportation', 'Water', 'NaturalArea']
ALL_LABELS = MAIN_LABELS + AUX_LABELS

class EuroSATDataset(Dataset):
    """从文件名解析标签（与训练时一致）"""
    def __init__(self, data_dir, indices, transform=None):
        self.data_dir = data_dir
        self.indices = indices
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
        filename = os.path.basename(img_path)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 从文件名解析类别（与训练时一致）
        class_name = filename.split('_')[0]
        label = np.zeros(len(ALL_LABELS), dtype=np.float32)

        if class_name in MAIN_LABELS:
            main_idx = MAIN_LABELS.index(class_name)
            label[main_idx] = 1.0

            aux_labels = MULTI_LABEL_MAP.get(class_name, [])
            for aux_label in aux_labels:
                if aux_label in AUX_LABELS:
                    aux_idx = len(MAIN_LABELS) + AUX_LABELS.index(aux_label)
                    label[aux_idx] = 1.0

        return image, torch.tensor(label, dtype=torch.float32)

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

def get_predictions_and_labels(model, data_loader, device):
    """获取模型在测试集上的预测概率和真实标签"""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()  # sigmoid转为概率
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.vstack(all_probs), np.vstack(all_labels)

def calculate_metrics(probs, labels, threshold=0.5):
    """计算评估指标"""
    pred_labels = (probs > threshold).astype(int)
    
    # Hamming Loss
    hamming_loss = (pred_labels != labels).mean()
    
    # Subset Accuracy
    subset_acc = (pred_labels == labels).all(axis=1).mean()
    
    # Macro F1
    tp = (pred_labels * labels).sum(axis=0)
    fp = (pred_labels * (1 - labels)).sum(axis=0)
    fn = ((1 - pred_labels) * labels).sum(axis=0)
    
    precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    macro_f1 = f1.mean()
    
    # Micro F1
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # 正确计算mAP（使用sklearn）
    mAP = average_precision_score(labels, probs, average='macro')

    return {
        'hamming_loss': hamming_loss,
        'subset_accuracy': subset_acc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'mAP': mAP
    }

def main():
    data_dir = 'd:/cxdownload/程序设计/data/eurosat'
    indices_dir = 'd:/cxdownload/程序设计/R-stage1/data'
    save_dir = 'd:/cxdownload/程序设计/W-stage2'

    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    indices_test = np.load(os.path.join(indices_dir, 'indices_test.npy'))
    print(f"测试集大小: {len(indices_test)}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 使用从文件名解析的标签（与训练时一致）
    test_dataset = EuroSATDataset(data_dir, indices_test, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("\n=== 加载CNN模型并生成预测 ===")
    cnn_model = CustomCNN(num_labels=17).to(device)
    cnn_model.load_state_dict(torch.load(os.path.join(save_dir, 'results/cnn_best.pth'), map_location=device))
    cnn_probs, cnn_labels = get_predictions_and_labels(cnn_model, test_loader, device)
    print(f"CNN预测概率形状: {cnn_probs.shape}")
    print(f"CNN预测概率范围: [{cnn_probs.min():.4f}, {cnn_probs.max():.4f}]")
    
    # 验证CNN结果
    cnn_metrics = calculate_metrics(cnn_probs, cnn_labels)
    print(f"CNN验证结果: Hamming Loss={cnn_metrics['hamming_loss']:.4f}, Macro F1={cnn_metrics['macro_f1']:.4f}, Micro F1={cnn_metrics['micro_f1']:.4f}, mAP={cnn_metrics['mAP']:.4f}")
    
    np.save(os.path.join(save_dir, 'predictions/predictions_cnn_test.npy'), cnn_probs)

    print("\n=== 加载ResNet-50模型并生成预测 ===")
    resnet_model = ResNet50MultiLabel(num_labels=17, pretrained=False).to(device)
    resnet_model.load_state_dict(torch.load(os.path.join(save_dir, 'results/resnet50_best.pth'), map_location=device))
    resnet_probs, resnet_labels = get_predictions_and_labels(resnet_model, test_loader, device)
    print(f"ResNet-50预测概率形状: {resnet_probs.shape}")
    print(f"ResNet-50预测概率范围: [{resnet_probs.min():.4f}, {resnet_probs.max():.4f}]")
    
    # 验证ResNet结果
    resnet_metrics = calculate_metrics(resnet_probs, resnet_labels)
    print(f"ResNet-50验证结果: Hamming Loss={resnet_metrics['hamming_loss']:.4f}, Macro F1={resnet_metrics['macro_f1']:.4f}, Micro F1={resnet_metrics['micro_f1']:.4f}, mAP={resnet_metrics['mAP']:.4f}")
    
    np.save(os.path.join(save_dir, 'predictions/predictions_resnet_test.npy'), resnet_probs)

    print("\n=== 与dl_results.csv对比 ===")
    print("dl_results.csv中的结果:")
    csv_path = os.path.join(save_dir, 'results/dl_results.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            print(f.read().rstrip())
    else:
        print(f"未找到 {csv_path}")
    print("\n本次验证结果:")
    print(f"CNN: Hamming Loss={cnn_metrics['hamming_loss']:.4f}, Macro F1={cnn_metrics['macro_f1']:.4f}, Micro F1={cnn_metrics['micro_f1']:.4f}, mAP={cnn_metrics['mAP']:.4f}")
    print(f"ResNet50: Hamming Loss={resnet_metrics['hamming_loss']:.4f}, Macro F1={resnet_metrics['macro_f1']:.4f}, Micro F1={resnet_metrics['micro_f1']:.4f}, mAP={resnet_metrics['mAP']:.4f}")

    print("\n=== 补充交付物生成完成 ===")
    print("文件列表:")
    print(f"- {save_dir}/predictions/predictions_cnn_test.npy")
    print(f"- {save_dir}/predictions/predictions_resnet_test.npy")

if __name__ == '__main__':
    main()