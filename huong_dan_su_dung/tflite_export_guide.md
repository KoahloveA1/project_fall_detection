# Export Models to TFLite (INT8, FP16, FP32)

Hiện tại máy Mac của bạn đang chạy **Python 3.14** (phiên bản thử nghiệm rất mới). Các thư viện chuyển đổi TFLite của Google (như `tensorflow`, `litert-torch`) chưa hỗ trợ phiên bản Python này, dẫn đến lỗi khi xuất file TFLite ở bước trước.

Cách nhanh nhất và không bao giờ lỗi là chạy đoạn script sau trên **Google Colab** (môi trường Python chuẩn của Google).

### Bước 1: Mở Google Colab
Truy cập [Google Colab](https://colab.research.google.com/), tạo một Notebook mới.

### Bước 2: Tải file lên
Tải 2 file sau từ máy tính của bạn lên thư mục gốc của Colab:
- `yolo26n-pose.pt`
- `fall_lstm_best.onnx`

### Bước 3: Copy và chạy toàn bộ đoạn code sau vào 1 Cell trên Colab

```python
# 1. Cài đặt các thư viện cần thiết (Ép bản numpy và protobuf tương thích với Colab)
!pip install "numpy<2" "protobuf<5" ultralytics onnx2tf

import os
from ultralytics import YOLO
import tensorflow as tf

print("=========================================")
print("🚀 1. EXPORT YOLO-POSE SANG TFLITE")
print("=========================================")

# Load mô hình YOLO
model = YOLO("yolo26n-pose.pt")

# Xuất Float32 (Định dạng 480x640 cho điện thoại nằm ngang)
print("Đang xuất YOLO FP32...")
model.export(format="tflite", imgsz=[480, 640])
os.rename("yolo26n-pose_saved_model/yolo26n-pose_float32.tflite", "yolo_pose_fp32.tflite")

# Xuất Float16
print("Đang xuất YOLO FP16...")
model.export(format="tflite", half=True, imgsz=[480, 640])
os.rename("yolo26n-pose_saved_model/yolo26n-pose_float16.tflite", "yolo_pose_fp16.tflite")

# Xuất INT8 (cần tập dữ liệu mẫu để hiệu chỉnh - Calibration)
print("Đang xuất YOLO INT8...")
model.export(format="tflite", int8=True, data="coco8-pose.yaml", imgsz=[480, 640])
os.rename("yolo26n-pose_saved_model/yolo26n-pose_int8.tflite", "yolo_pose_int8.tflite")


print("\n=========================================")
print("🚀 2. EXPORT LSTM (ONNX) SANG TFLITE")
print("=========================================")

# Chuyển ONNX sang TensorFlow SavedModel bằng thư viện chuẩn onnx2tf
!onnx2tf -i fall_lstm_best.onnx -osd

saved_model_dir = "saved_model"

# Hàm Helper chuyển đổi bằng TFLiteConverter
def convert_tflite(saved_model_dir, output_name, optimization=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if optimization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif optimization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Tạo dummy dataset để calibrate cho INT8
        def representative_dataset():
            for _ in range(100):
                yield [tf.random.normal([1, 15, 37], dtype=tf.float32)]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_name, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Đã lưu {output_name}")

# Thực hiện chuyển đổi LSTM
print("Đang xuất LSTM FP32...")
convert_tflite(saved_model_dir, "lstm_fp32.tflite", optimization=None)

print("Đang xuất LSTM FP16...")
convert_tflite(saved_model_dir, "lstm_fp16.tflite", optimization="fp16")

print("Đang xuất LSTM INT8...")
convert_tflite(saved_model_dir, "lstm_int8.tflite", optimization="int8")

print("\n🎉 HOÀN TẤT! Hãy tải 6 file .tflite ở bên cột trái của Colab về máy!")
```

### Giải thích về các định dạng cho Android:
1. **FP32 (Float 32)**: Kích thước lớn nhất, độ chính xác cao nhất (như bản gốc), tốc độ trung bình.
2. **FP16 (Float 16)**: Kích thước giảm một nửa, độ chính xác gần như nguyên bản, **tốc độ rất nhanh trên GPU điện thoại**. (Khuyên dùng).
3. **INT8 (Integer 8)**: Kích thước nhỏ nhất (nhẹ bằng 1/4), tốc độ siêu nhanh (đặc biệt nếu điện thoại có chip NPU/Hexagon), nhưng độ chính xác có thể giảm nhẹ. Cần bộ dữ liệu hiệu chỉnh (Representative Dataset) khi convert.
