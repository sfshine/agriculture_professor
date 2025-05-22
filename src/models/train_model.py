import os
import sys

# 添加项目根目录到Python路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

# Configuration parameters
num_epochs = 10
sample_ratio = 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import time
import copy
from PIL import Image
from collections import defaultdict
from src.data.dataset import AgriculturalDiseaseDataset, LABEL_MAP
from src.models.test_model import evaluate_model

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Create a mapping from original labels to continuous indices (0 to num_classes-1)
    label_to_index = {label: idx for idx, label in enumerate(sorted(LABEL_MAP.keys()))}
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            phase_start_time = time.time()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                # Remap labels to continuous range
                labels = torch.tensor([label_to_index[label.item()] for label in labels]).to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.cpu().numpy())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.cpu().numpy())
            
            phase_time = time.time() - phase_start_time
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {phase_time:.2f}s')
            
            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1} completed in {epoch_time:.2f}s')
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs (Epochs: {}, Sample Ratio: {})'.format(num_epochs, sample_ratio))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs (Epochs: {}, Sample Ratio: {})'.format(num_epochs, sample_ratio))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics_epochs_{}_ratio_{}.png'.format(num_epochs, sample_ratio))
    plt.show()
    
    return model

def main():
    try:
        # Get the project root directory (two levels up from this file)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Data directories
        train_dir = os.path.join(base_dir, 'AgriculturalDisease_trainingset')
        val_dir = os.path.join(base_dir, 'AgriculturalDisease_validationset')
        
        # 检查目录是否存在
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
        
        # Data transformation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # Create datasets with specified sampling ratio for training set
        print("Loading training dataset...")
        train_dataset = AgriculturalDiseaseDataset(
            txt_file=os.path.join(train_dir, 'train_list.txt'),
            root_dir=os.path.dirname(train_dir),
            transform=data_transforms['train'],
            sample_ratio=sample_ratio
        )
        
        print("Loading validation dataset...")
        val_dataset = AgriculturalDiseaseDataset(
            txt_file=os.path.join(val_dir, 'ttest_list.txt'),
            root_dir=os.path.dirname(val_dir),
            transform=data_transforms['val']
        )
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
            'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        }
        
        dataset_sizes = {
            'train': len(train_dataset),
            'val': len(val_dataset)
        }
        
        print(f"Training set size: {dataset_sizes['train']}")
        print(f"Validation set size: {dataset_sizes['val']}")
        
        # 使用预定义的LABEL_MAP中的类别数量
        num_classes = len(LABEL_MAP)
        print(f"Training with {num_classes} classes from LABEL_MAP")
        print("Classes:")
        for label_id, label_name in LABEL_MAP.items():
            print(f"  {label_id}: {label_name}")
        
        # Initialize the model (ResNet50 with pretrained weights)
        print("Initializing model...")
        model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Define loss function, optimizer and learning rate scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Train the model
        print("Starting model training...")
        trained_model = train_model(
            model, 
            criterion, 
            optimizer, 
            exp_lr_scheduler,
            dataloaders, 
            dataset_sizes, 
            num_epochs=num_epochs
        )
        
        # Save the trained model with the label mapping
        print("Saving model...")
        label_to_index = {label: idx for idx, label in enumerate(sorted(LABEL_MAP.keys()))}
        index_to_label = {idx: label for label, idx in label_to_index.items()}
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'label_map': LABEL_MAP,
            'num_classes': num_classes,
            'label_to_index': label_to_index,
            'index_to_label': index_to_label
        }, 'agricultural_disease_model.pth')
        
        # Evaluate the model on the validation set
        print("\nEvaluating model on validation set:")
        accuracy, true_labels, pred_labels = evaluate_model(
            trained_model,
            dataloaders['val'],
            criterion,
            num_epochs=num_epochs,
            sample_ratio=sample_ratio
        )
        
        print(f"Final Validation Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 