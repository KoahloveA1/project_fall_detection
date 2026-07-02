import os
from ultralytics import YOLO
import torch
import torch.nn as nn
from ai_edge_torch import convert

# -----------------------------------------
# 1. EXPORT YOLO-POSE SANG TFLITE
# -----------------------------------------
print("🚀 Đang xuất YOLO-Pose sang TFLite...")
yolo_model = YOLO("yolo26n-pose.pt")

# FP32
yolo_model.export(format="tflite", imgsz=[480, 640])
os.rename("yolo26n-pose_saved_model/yolo26n-pose_float32.tflite", "yolo_pose_fp32.tflite")

# FP16
yolo_model.export(format="tflite", half=True, imgsz=[480, 640])
os.rename("yolo26n-pose_saved_model/yolo26n-pose_float16.tflite", "yolo_pose_fp16.tflite")

# INT8
yolo_model.export(format="tflite", int8=True, data="coco8-pose.yaml", imgsz=[480, 640])
os.rename("yolo26n-pose_saved_model/yolo26n-pose_int8.tflite", "yolo_pose_int8.tflite")


# -----------------------------------------
# 2. EXPORT LSTM SANG TFLITE BẰNG AI-EDGE-TORCH
# -----------------------------------------
print("\n🚀 Đang xuất LSTM sang TFLite...")

class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size=37, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

lstm_model = FallDetectionLSTM()
lstm_model.load_state_dict(torch.load("fall_lstm_best.pth", map_location="cpu"))
lstm_model.eval()

# Dummy input cho LSTM (batch_size=1, sequence_length=15, input_size=37)
dummy_input = torch.randn(1, 15, 37)

# Chuyển đổi bằng Google ai-edge-torch (chính hãng cho TFLite)
edge_model = convert(lstm_model, (dummy_input,))
edge_model.export("lstm_fp32.tflite")

print("\n🎉 HOÀN TẤT! Các file .tflite đã được lưu.")
