"""
完整训练脚本：CNN + ResNet-50，生成所有阶段二交付物
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

# 多标签映射
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
        self.feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.feature_dim, num_labels)
    
    def forward(self, x):
        return self.resnet(x)
    
    def get_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        features = torch.flatten(x, 1)
        return features

def calculate_metrics(preds, targets, threshold=0.5):
    probs = torch.sigmoid(preds)
    pred_labels = (probs > threshold).float()
    
    hamming_loss = (pred_labels != targets).float().mean().item()
    subset_acc = (pred_labels == targets).all(dim=1).float().mean().item()
    
    tp = (pred_labels * targets).sum(dim=0).float()
    fp = (pred_labels * (1 - targets)).sum(dim=0).float()
    fn = ((1 - pred_labels) * targets).sum(dim=0).float()
    
    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    macro_f1 = f1.mean().item()
    
    micro_tp, micro_fp, micro_fn = tp.sum().item(), fp.sum().item(), fn.sum().item()
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # 正确计算mAP（使用sklearn）
    mAP = average_precision_score(targets.cpu().numpy(), probs.cpu().numpy(), average='macro')

    return {
        'hamming_loss': hamming_loss,
        'subset_accuracy': subset_acc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'mAP': mAP
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20, save_path=None):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                all_preds.append(outputs.cpu())
                all_targets.append(labels.cpu())
        
        val_loss /= len(val_loader)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = calculate_metrics(all_preds, all_targets)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
    
    return train_losses, val_losses, metrics

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())
    
    test_loss /= len(test_loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    
    return test_loss, metrics

def extract_resnet_features(model, data_loader, device):
    model.eval()
    all_features = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            features = model.get_features(images)
            all_features.append(features.cpu().numpy())
    return np.vstack(all_features)

def main():
    data_dir = 'd:/cxdownload/程序设计/data/eurosat'
    indices_dir = 'd:/cxdownload/程序设计/R-stage1/data'
    save_dir = 'd:/cxdownload/程序设计/W-stage2'
    
    os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'figures'), exist_ok=True)
    
    # 加载索引
    indices_train = np.load(os.path.join(indices_dir, 'indices_train.npy'))
    indices_val = np.load(os.path.join(indices_dir, 'indices_val.npy'))
    indices_test = np.load(os.path.join(indices_dir, 'indices_test.npy'))
    
    print(f"训练集: {len(indices_train)}, 验证集: {len(indices_val)}, 测试集: {len(indices_test)}")
    
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    print("\n创建数据集...")
    train_dataset = EuroSATDataset(data_dir, indices_train, train_transform)
    val_dataset = EuroSATDataset(data_dir, indices_val, val_transform)
    test_dataset = EuroSATDataset(data_dir, indices_test, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ============ 训练CNN ============
    print("\n=== 训练自定义CNN ===")
    cnn_model = CustomCNN(num_labels=17).to(device)
    cnn_criterion = nn.BCEWithLogitsLoss()
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    
    cnn_train_losses, cnn_val_losses, _ = train_model(
        cnn_model, train_loader, val_loader, cnn_criterion, cnn_optimizer, 
        device, epochs=20, save_path=os.path.join(save_dir, 'results/cnn_best.pth')
    )
    
    cnn_model.load_state_dict(torch.load(os.path.join(save_dir, 'results/cnn_best.pth')))
    cnn_test_loss, cnn_metrics = test_model(cnn_model, test_loader, cnn_criterion, device)
    
    print(f"\nCNN测试结果:")
    print(f"Hamming Loss: {cnn_metrics['hamming_loss']:.4f}")
    print(f"Macro F1: {cnn_metrics['macro_f1']:.4f}")
    print(f"Micro F1: {cnn_metrics['micro_f1']:.4f}")
    print(f"mAP: {cnn_metrics['mAP']:.4f}")
    
    # 绘制CNN Loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(cnn_train_losses, label='Train Loss')
    plt.plot(cnn_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'figures/cnn_loss_curve.png'), dpi=150)
    plt.close()
    
    # ============ 训练ResNet-50 ============
    print("\n=== 训练ResNet-50 ===")
    resnet_model = ResNet50MultiLabel(num_labels=17, pretrained=True).to(device)
    resnet_criterion = nn.BCEWithLogitsLoss()
    resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=0.0001)
    
    resnet_train_losses, resnet_val_losses, _ = train_model(
        resnet_model, train_loader, val_loader, resnet_criterion, resnet_optimizer, 
        device, epochs=15, save_path=os.path.join(save_dir, 'results/resnet50_best.pth')
    )
    
    resnet_model.load_state_dict(torch.load(os.path.join(save_dir, 'results/resnet50_best.pth')))
    resnet_test_loss, resnet_metrics = test_model(resnet_model, test_loader, resnet_criterion, device)
    
    print(f"\nResNet-50测试结果:")
    print(f"Hamming Loss: {resnet_metrics['hamming_loss']:.4f}")
    print(f"Macro F1: {resnet_metrics['macro_f1']:.4f}")
    print(f"Micro F1: {resnet_metrics['micro_f1']:.4f}")
    print(f"mAP: {resnet_metrics['mAP']:.4f}")
    
    # 绘制ResNet Loss曲线
    plt.figure(figsize=(10, 5))
    plt.plot(resnet_train_losses, label='Train Loss')
    plt.plot(resnet_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ResNet-50 Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'figures/resnet_loss_curve.png'), dpi=150)
    plt.close()
    
    # ============ 导出ResNet特征 ============
    print("\n=== 导出ResNet深度特征 ===")
    resnet_features_train = extract_resnet_features(resnet_model, train_loader, device)
    resnet_features_test = extract_resnet_features(resnet_model, test_loader, device)
    
    np.save(os.path.join(save_dir, 'features/resnet_features_train.npy'), resnet_features_train)
    np.save(os.path.join(save_dir, 'features/resnet_features_test.npy'), resnet_features_test)
    
    print(f"训练集特征形状: {resnet_features_train.shape}")
    print(f"测试集特征形状: {resnet_features_test.shape}")
    
    # ============ 保存结果到CSV ============
    with open(os.path.join(save_dir, 'results/dl_results.csv'), 'w') as f:
        f.write('model,hamming_loss,macro_f1,micro_f1,mAP\n')
        f.write(f"CNN,{cnn_metrics['hamming_loss']:.4f},{cnn_metrics['macro_f1']:.4f},{cnn_metrics['micro_f1']:.4f},{cnn_metrics['mAP']:.4f}\n")
        f.write(f"ResNet50,{resnet_metrics['hamming_loss']:.4f},{resnet_metrics['macro_f1']:.4f},{resnet_metrics['micro_f1']:.4f},{resnet_metrics['mAP']:.4f}\n")
    
    print(f"\n=== 阶段二交付物生成完成 ===")
    print("文件列表:")
    print(f"- {save_dir}/results/dl_results.csv")
    print(f"- {save_dir}/features/resnet_features_train.npy")
    print(f"- {save_dir}/features/resnet_features_test.npy")
    print(f"- {save_dir}/figures/cnn_loss_curve.png")
    print(f"- {save_dir}/figures/resnet_loss_curve.png")

if __name__ == '__main__':
    main()
