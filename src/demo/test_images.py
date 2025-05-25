import os
import sys
import glob
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import requests
import json

# 添加项目根目录到Python路径
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from src.data.dataset import LABEL_MAP

def load_model(model_path):
    """加载训练好的模型"""
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型权重和参数
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    label_map = checkpoint.get('label_map', LABEL_MAP)
    
    # 创建模型架构
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # 获取标签映射
    label_to_index = checkpoint.get('label_to_index', {label: idx for idx, label in enumerate(sorted(LABEL_MAP.keys()))})
    index_to_label = checkpoint.get('index_to_label', {idx: label for label, idx in label_to_index.items()})
    
    return model, label_map, label_to_index, index_to_label, device

def process_image(image_path, transform):
    """处理单张图片"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    return image_tensor

def predict_image(model, image_tensor, index_to_label, label_map, device):
    """预测单张图片"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        
    # 获取预测的类别索引
    predicted_idx = preds.item()
    # 获取原始标签
    original_label = index_to_label[predicted_idx]
    # 获取可读的标签名称
    readable_label = label_map[original_label]
    
    return original_label, readable_label, outputs[0]

def main():
    try:
        # 获取项目根目录
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 模型路径
        model_path = os.path.join(base_dir, 'agricultural_disease_model.pth')
        
        # 验证集图片目录
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
        print(f"Images directory: {images_dir}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # 图片预处理
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        model, label_map, label_to_index, index_to_label, device = load_model(model_path)
        
        # 获取所有图片路径
        image_paths = glob.glob(os.path.join(images_dir, '*.*'))
        print(f"Found {len(image_paths)} images in directory.")
        
        # 测试每张图片
        for i, img_path in enumerate(image_paths):
            try:
                # 获取文件名
                img_name = os.path.basename(img_path)
                
                # 处理图片
                image_tensor = process_image(img_path, data_transform)
                
                # 预测
                predicted_class_id, predicted_label_name, outputs = predict_image(
                    model, image_tensor, index_to_label, label_map, device
                )
                
                # 获取前3个最可能的类别
                confidences, top_classes = torch.topk(outputs, 3)
                top_confidences = torch.softmax(confidences, dim=0).tolist()
                top_labels = [index_to_label[idx] for idx in top_classes.tolist()]
                top_readable = [label_map[label] for label in top_labels]
                
                # 打印结果
                print(f"\n[{i+1}/{len(image_paths)}] 图片: {img_name}")
                print(f"预测结果: {predicted_label_name} (标签ID: {predicted_class_id})")
                print("Top 3 预测:")
                for j in range(len(top_readable)):
                    print(f"  {j+1}. {top_readable[j]} - 置信度: {top_confidences[j]:.4f}")
                
                # Call API with prediction results
                try:
                    # Extract plant and disease from the readable label
                    label_parts = predicted_label_name.split('-')
                    if len(label_parts) == 2:
                        plant, disease = label_parts
                        payload = {
                            "plant": plant,
                            "disease": disease
                        }
                        api_url = "http://4zv48122tf42.vicp.fun/api/qa"
                        headers = {'Content-Type': 'application/json'}
                        response = requests.post(api_url, data=json.dumps(payload), headers=headers)
                        
                        if response.status_code == 200:
                            api_data = response.json()
                            print("\n=== Disease Information and Treatment ===")
                            print(api_data.get('answer', 'No detailed information available.'))
                            print("=====================================\n")
                        else:
                            print(f"\nAPI call failed with status code: {response.status_code}")
                    else:
                        print("\nCould not parse plant and disease from prediction label.")
                except Exception as api_error:
                    print(f"\nError calling API: {str(api_error)}")
                    
            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {str(e)}")
        
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 