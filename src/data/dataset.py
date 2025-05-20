import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
import numpy as np

# Label mapping with Chinese translations in comments
# 0: Apple-Healthy (苹果-健康)
# 3: Apple-Frogeye Spot (苹果-灰斑病)
# 4: Apple-Cedar Rust Serious (苹果-雪松锈病严重)
# 9: Corn-Healthy (玉米-健康)
# 13: Corn-Rust Serious (玉米-锈病严重)
# 15: Corn-Leaf Spot Serious (玉米-叶斑病严重)
# 17: Grape-Healthy (葡萄-健康)
# 21: Grape-Black Measles Serious (葡萄-轮斑病严重)
# 23: Grape-Leaf Blight Serious (葡萄-褐斑病严重)
LABEL_MAP = {
    0: "Apple-Healthy",
    3: "Apple-Frogeye Spot",
    4: "Apple-Cedar Rust",
    9: "Corn-Healthy",
    13: "Corn-Rust",
    15: "Corn-Leaf Spot",
    17: "Grape-Healthy",
    21: "Grape-Black Measles",
    23: "Grape-Leaf Blight",
}

class AgriculturalDiseaseDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, label_map=LABEL_MAP, sample_ratio=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            label_map (dict, optional): Mapping from original labels to new labels.
            sample_ratio (float, optional): If provided, randomly sample this ratio of data from each class.
        """
        self.annotations = []
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = label_map
        
        # Create reverse mapping from original labels to mapped labels
        self.reverse_label_map = {str(k): str(k) for k in self.label_map.keys()}
        
        # Load data with optional sampling
        if sample_ratio is not None:
            class_samples = defaultdict(list)
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            img_path, label = line.split()
                            # Only include images with labels in our label_map
                            if label in self.reverse_label_map:
                                img_path = img_path.replace('\\', os.sep)
                                mapped_label = int(self.reverse_label_map[label])
                                class_samples[mapped_label].append((img_path, mapped_label))
                        except (ValueError, KeyError) as e:
                            print(f"Warning: Skipping line due to {str(e)}: {line}")
                            continue
            
            # Sample from each class
            for label, samples in class_samples.items():
                n_samples = max(1, int(len(samples) * sample_ratio))
                sampled_indices = np.random.choice(len(samples), n_samples, replace=False)
                self.annotations.extend([samples[i] for i in sampled_indices])
        else:
            # Load all data without sampling
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            img_path, label = line.split()
                            # Only include images with labels in our label_map
                            if label in self.reverse_label_map:
                                img_path = img_path.replace('\\', os.sep)
                                mapped_label = int(self.reverse_label_map[label])
                                self.annotations.append((img_path, mapped_label))
                        except (ValueError, KeyError) as e:
                            print(f"Warning: Skipping line due to {str(e)}: {line}")
                            continue
                            
        print(f"Loaded {len(self.annotations)} images with valid labels from {txt_file}")
        # Print class distribution
        class_dist = defaultdict(int)
        for _, label in self.annotations:
            class_dist[label] += 1
        print("Class distribution:")
        for label, count in sorted(class_dist.items()):
            print(f"{self.label_map[label]}: {count} images")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, label = self.annotations[idx]
        
        # Remove the folder prefix from img_path if it's included
        if os.path.dirname(self.root_dir) in img_path:
            img_path = os.path.relpath(img_path, os.path.dirname(self.root_dir))
        
        img_path = os.path.join(self.root_dir, img_path)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise
        
        if self.transform:
            image = self.transform(image)
        
        return image, label 