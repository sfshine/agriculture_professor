from collections import Counter
import os

def analyze_labels(file_path):
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 每行的格式是: 图片路径 标签
            label = line.strip().split()[-1]
            labels.append(label)
    
    # 统计每个标签的数量
    label_counts = Counter(labels)
    
    # 打印结果
    print(f"\n分析文件: {os.path.basename(file_path)}")
    print(f"总样本数: {len(labels)}")
    print(f"不同标签数量: {len(label_counts)}")
    print("\n标签分布:")
    for label, count in sorted(label_counts.items(), key=lambda x: int(x[0])):
        print(f"标签 {label}: {count} 个样本")

def main():
    # 分析训练集
    train_file = "AgriculturalDisease_trainingset/train_list.txt"
    analyze_labels(train_file)
    
    # 分析测试集
    test_file = "AgriculturalDisease_validationset/ttest_list.txt"
    analyze_labels(test_file)

if __name__ == "__main__":
    main() 