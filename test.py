import torch
from torch.utils.data import DataLoader
import os

from datasets.robo_dataset import RoboMNISTDataset
from models.multimodal_model import MultimodalModel
from utils.visualization import plot_test_results_summary
import config


def decode_label(label_int):
    """
    0~19 → (arm_id, digit)
    arm_id = label // 10 + 1
    digit = label % 10
    """
    label_int = int(label_int)
    arm_id = label_int // 10 + 1
    digit = label_int % 10
    return arm_id, digit


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    test_set = RoboMNISTDataset(config.data_root, num_frames=config.num_frames, mode="test")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Load model
    model = MultimodalModel().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    correct = 0
    all_true_labels = []
    all_pred_labels = []

    print("\n======= Test Results =======\n")

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inputs = {
                "front": batch["front"].to(device),
                "left": batch["left"].to(device),
                "right": batch["right"].to(device),
                "csi": batch["csi"].to(device),
            }

            true_label = batch["label"].item()
            pred = model(inputs).argmax(dim=1).item()

            # 记录所有预测结果用于可视化
            all_true_labels.append(true_label)
            all_pred_labels.append(pred)

            # Decode results
            true_arm, true_digit = decode_label(true_label)
            pred_arm, pred_digit = decode_label(pred)

            # Evaluate accuracy
            if pred == true_label:
                correct += 1

            print(f"Sample {idx+1}:")
            print(f"  ▶ True : Arm {true_arm}, Digit {true_digit}")
            print(f"  ▶ Pred : Arm {pred_arm}, Digit {pred_digit}")
            print("")

    acc = correct / len(test_set)
    print(f"\n🎯 FINAL TEST ACCURACY: {acc:.4f}\n")
    
    # 生成可视化图表
    print("\n📊 正在生成测试结果可视化...")
    plot_test_results_summary(all_true_labels, all_pred_labels, save_dir="test_results")


if __name__ == "__main__":
    main()