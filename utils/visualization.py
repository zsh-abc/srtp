import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path="training_curves.png"):
    """
    绘制训练过程中的 loss 和 accuracy 曲线
    
    Args:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        val_losses: 验证损失列表
        val_accs: 验证准确率列表
        save_path: 保存路径
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 曲线
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy 曲线
    axes[1].plot(epochs, train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=6)
    axes[1].plot(epochs, val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 训练曲线已保存至: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path="confusion_matrix.png"):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        class_names: 类别名称列表（可选）
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # 如果没有提供类别名称，使用数字标签
    if class_names is None:
        num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 混淆矩阵已保存至: {save_path}")
    plt.close()


def plot_class_accuracy(y_true, y_pred, class_names=None, save_path="class_accuracy.png"):
    """
    绘制每个类别的准确率
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        class_names: 类别名称列表（可选）
        save_path: 保存路径
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    unique_classes = np.unique(y_true)
    class_accs = []
    
    for cls in unique_classes:
        mask = y_true == cls
        if mask.sum() > 0:
            acc = (y_pred[mask] == cls).sum() / mask.sum()
            class_accs.append(acc)
        else:
            class_accs.append(0.0)
    
    if class_names is None:
        class_names = [f"Class {int(c)}" for c in unique_classes]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(unique_classes)), class_accs, color='steelblue', alpha=0.7)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique_classes)), class_names, rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, class_accs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 类别准确率图已保存至: {save_path}")
    plt.close()


def decode_label_to_str(label_int):
    """
    将标签整数转换为字符串表示
    0~19 → "Arm X, Digit Y"
    """
    label_int = int(label_int)
    arm_id = label_int // 10 + 1
    digit = label_int % 10
    return f"Arm {arm_id}, Digit {digit}"


def plot_test_results_summary(y_true, y_pred, save_dir="results"):
    """
    生成测试结果的可视化摘要
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成类别名称（Arm X, Digit Y 格式）
    unique_labels = sorted(set(y_true + y_pred))
    class_names = [decode_label_to_str(label) for label in unique_labels]
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, class_names=class_names,
                         save_path=os.path.join(save_dir, "confusion_matrix.png"))
    
    # 绘制每个类别的准确率
    plot_class_accuracy(y_true, y_pred, class_names=class_names,
                       save_path=os.path.join(save_dir, "class_accuracy.png"))
    
    print(f"\n✅ 所有测试结果可视化图表已保存至: {save_dir}/")

