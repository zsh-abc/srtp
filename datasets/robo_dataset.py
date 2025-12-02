import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def complex_str_to_num(s):
    s = s.strip("()")
    return complex(s)


class RoboMNISTDataset(Dataset):
    """
    Each sample folder contains:
        cam_front.mp4
        cam_left.mp4
        cam_right.mp4
        csi.json
        label.json
    """

    def __init__(self, root_dir, num_frames=16, use_amplitude=True, mode="train"):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.use_amplitude = use_amplitude
        self.mode = mode

        # 读取 split.json
        split_file = os.path.join(self.root_dir, "split.json")
        with open(split_file, "r") as f:
            split = json.load(f)

        # 根据 mode 选择对应的列表（目录名）
        self.sample_list = split.get(mode, [])

        # 只保留 split.json 中且真实存在的样本目录
        self.samples = []
        for s in self.sample_list:
            folder = os.path.join(root_dir, s)
            if os.path.isdir(folder):
                self.samples.append(folder)
            else:
                print(f"[WARN] {mode} 样本在磁盘中不存在: {s}")

        # 按名称排序，保证可复现
        self.samples.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"[INFO] Loaded {len(self.samples)} samples ({mode}) from split.json")


    # ---------------- Video loader ---------------- #
    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()
        return torch.stack(frames)   # [T, 3, 224, 224]


    # ---------------- CSI loader ---------------- #
    def _load_csi(self, csi_path):
        with open(csi_path, "r") as f:
            data = json.load(f)

        module = data[0]
        rows = module["complex_csi"]

        processed = []
        for row in rows:
            complex_list = np.array([complex_str_to_num(x) for x in row])

            if self.use_amplitude:
                values = np.abs(complex_list)
            else:
                values = np.stack([complex_list.real, complex_list.imag], axis=1)
            processed.append(values)

        processed = np.array(processed)
        processed = (processed - processed.mean()) / (processed.std() + 1e-6)

        return torch.tensor(processed, dtype=torch.float32)


    # ---------------- Label loader ---------------- #
    def _load_label(self, label_path):
        with open(label_path, "r") as f:
            data = json.load(f)

        # Expect: {"digit": x, "arm_id": y}
        digit = int(data["digit"])
        arm_id = int(data["arm_id"])    # 1 or 2

        # Convert to 0~19
        final_label = (arm_id - 1) * 10 + digit

        return torch.tensor(final_label, dtype=torch.long)



    # ---------------- MAIN ---------------- #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]

        # Multi-view video input
        video_front = self._load_video(os.path.join(folder, "cam_front.mp4"))
        video_left  = self._load_video(os.path.join(folder, "cam_left.mp4"))
        video_right = self._load_video(os.path.join(folder, "cam_right.mp4"))

        csi = self._load_csi(os.path.join(folder, "csi.json"))
        label = self._load_label(os.path.join(folder, "label.json"))

        return {
            "front": video_front,
            "left": video_left,
            "right": video_right,
            "csi": csi,
            "label": label
        }


if __name__ == "__main__":
    dataset = RoboMNISTDataset("/home/zsh/Robominist/data")
    item = dataset[0]
    print("front video:", item["front"].shape)
    print("left video: ", item["left"].shape)
    print("right video:", item["right"].shape)
    print("csi shape:  ", item["csi"].shape)
    print("label:", item["label"])
