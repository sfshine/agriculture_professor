import os
import sys

# 添加项目根目录到Python路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
from collections import defaultdict
from src.data.dataset import AgriculturalDiseaseDataset, LABEL_MAP

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 对于 macOS
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def evaluate_model(model, dataloader, criterion, num_epochs=25, sample_ratio=0.1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    # Get label mapping from the model checkpoint if available
    checkpoint = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'agricultural_disease_model.pth'), map_location='cpu')
    label_to_index = checkpoint.get('label_to_index', {label: idx for idx, label in enumerate(sorted(LABEL_MAP.keys()))})
    index_to_label = checkpoint.get('index_to_label', {idx: label for label, idx in label_to_index.items()})
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # Remap labels to continuous range
            labels_remapped = torch.tensor([label_to_index[label.item()] for label in labels]).to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels_remapped)
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())  # Store original labels for reporting
            all_preds.extend(preds.cpu().numpy())    # Store predicted indices
            
    # 直接使用LABEL_MAP中的标签
    cm = confusion_matrix(all_labels, [index_to_label[pred] for pred in all_preds])
    plt.figure(figsize=(15, 12))
    
    # 获取标签名称
    unique_labels = sorted(set(all_labels))
    label_names = [LABEL_MAP[i] for i in unique_labels]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.title(f'Confusion Matrix (Epochs: {num_epochs}, Sample Ratio: {sample_ratio})', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_epochs_{num_epochs}_ratio_{sample_ratio}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算评估指标
    report = classification_report(all_labels, [index_to_label[pred] for pred in all_preds], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    # 将数字标签转换为可读的标签名称
    df_report.index = [LABEL_MAP.get(int(i), i) if i.isdigit() else i for i in df_report.index]
    print(f'Classification Report:\n{df_report}')
    
    accuracy = accuracy_score(all_labels, [index_to_label[pred] for pred in all_preds])
    precision = precision_score(all_labels, [index_to_label[pred] for pred in all_preds], average='weighted')
    recall = recall_score(all_labels, [index_to_label[pred] for pred in all_preds], average='weighted')
    f1 = f1_score(all_labels, [index_to_label[pred] for pred in all_preds], average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return accuracy, all_labels, all_preds

def main():
    try:
        # Get the project root directory (two levels up from this file)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        val_dir = os.path.join(base_dir, 'AgriculturalDisease_validationset')
        model_path = os.path.join(base_dir, 'agricultural_disease_model.pth')
        
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        label_map = checkpoint['label_map']
        num_classes = checkpoint['num_classes']
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_dataset = AgriculturalDiseaseDataset(
            txt_file=os.path.join(val_dir, 'ttest_list.txt'),
            root_dir=os.path.dirname(val_dir),
            label_map=label_map,
            transform=data_transform
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        criterion = nn.CrossEntropyLoss()
        print("Evaluating model on validation set...")
        accuracy, true_labels, pred_labels = evaluate_model(
            model,
            val_loader,
            criterion
        )
        print(f"Final Validation Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 