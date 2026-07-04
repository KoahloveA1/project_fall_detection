import os
from ultralytics import YOLO
import torch
import torch.nn as nn
from litert_torch import convert

# 1. EXPORT YOLO-POSE SANG TFLITE (FLOAT32 và FLOAT16)
print("🚀 Đang xuất YOLO-Pose sang TFLite...")
yolo_model = YOLO("yolo26n-pose.pt")

# Xuất Float32 (16MB)
# img_size trong YOLO là [height, width]. Màn hình dọc 480x640 -> [640, 480]
yolo_model.export(format="tflite", imgsz=[640, 480])

# Xuất bản nhẹ hơn (w8a16 - weights 8bit, activations 16bit) thay cho FP16
yolo_model.export(format="tflite", quantize="w8a16", imgsz=[640, 480])


# 2. EXPORT LSTM V2 SANG TFLITE BẰNG LITERT-TORCH
print("\n🚀 Đang xuất LSTM V2 sang TFLite...")

# Định nghĩa lại kiến trúc của LSTM v2 (Input: 38, Seq_len: 30)
class FallDetectionLSTM_v2(nn.Module):
    def __init__(self, input_size=38, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

lstm_model = FallDetectionLSTM_v2()
# Đọc file LSTM v2 (đảm bảo bạn upload file này lên Colab)
lstm_model.load_state_dict(torch.load("fall_lstm_best_v2.pt", map_location="cpu"))
lstm_model.eval()

# Seq_len 30, Features 38
dummy_input = torch.randn(1, 30, 38)
edge_model = convert(lstm_model, (dummy_input,))

# Lưu TFLite Float32
edge_model.export("fall_lstm_v2_fp32.tflite")

# Lưu TFLite Float16
import litert_torch.quantization as q
q_model = convert(lstm_model, (dummy_input,), quantizer=q.Float16Quantizer())
q_model.export("fall_lstm_v2_fp16.tflite")

print("\n🎉 HOÀN TẤT XUẤT 4 FILE TFLITE!")
