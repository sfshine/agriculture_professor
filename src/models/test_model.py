import os
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
from ..data.dataset import AgriculturalDiseaseDataset

# Label mapping with Chinese translations in comments
# 0: Apple-Healthy (苹果-健康)
# 3: Apple-Frogeye Spot (苹果-灰斑病)
# 5: Apple-Cedar Rust Serious (苹果-雪松锈病严重)
# 9: Corn-Healthy (玉米-健康)
# 13: Corn-Rust Serious (玉米-锈病严重)
# 15: Corn-Leaf Spot Serious (玉米-叶斑病严重)
# 17: Grape-Healthy (葡萄-健康)
# 21: Grape-Black Measles Serious (葡萄-轮斑病严重)
# 23: Grape-Leaf Blight Serious (葡萄-褐斑病严重)
LABEL_MAP = {
    0: "Apple-Healthy",
    3: "Apple-Frogeye Spot",
    5: "Apple-Cedar Rust Serious",
    9: "Corn-Healthy",
    13: "Corn-Rust Serious",
    15: "Corn-Leaf Spot Serious",
    17: "Grape-Healthy",
    21: "Grape-Black Measles Serious",
    23: "Grape-Leaf Blight Serious",
}

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 对于 macOS
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def evaluate_model(model, dataloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    # Create reverse mapping
    reverse_label_map = {v: k for k, v in dataloader.dataset.label_map.items()}
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Map predictions and true labels back to original labels
    original_labels = [reverse_label_map[label] for label in all_labels]
    original_preds = [reverse_label_map[pred] for pred in all_preds]
    
    cm = confusion_matrix(original_labels, original_preds)
    plt.figure(figsize=(15, 12))
    # Get English labels for display
    labels = [LABEL_MAP.get(int(i), f"Class {i}") for i in sorted(set(original_labels))]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate metrics using original labels
    report = classification_report(original_labels, original_preds, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.index = [LABEL_MAP.get(int(i), i) if i.isdigit() else i for i in df_report.index]
    print(f'Classification Report:\n{df_report}')
    
    accuracy = accuracy_score(original_labels, original_preds)
    precision = precision_score(original_labels, original_preds, average='weighted')
    recall = recall_score(original_labels, original_preds, average='weighted')
    f1 = f1_score(original_labels, original_preds, average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return accuracy, original_labels, original_preds

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