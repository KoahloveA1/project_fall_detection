import cv2
import numpy as np
import collections
import time
from ai_edge_litert.interpreter import Interpreter
from ultralytics import YOLO

# CẤU HÌNH MOBILE PIPELINE
TARGET_WIDTH = 480
TARGET_HEIGHT = 640
SEQ_LEN = 30
LSTM_THRESHOLD = 0.5
DROP_RATIO_THRESHOLD = 0.15  # Hông phải rơi > 15% chiều cao cơ thể mới tính là ngã

# 1. Khởi tạo Models
print("🚀 Khởi tạo Mobile Pipeline (480x640) với TFLite...")
yolo_model = YOLO("yolo26n-pose.tflite", task='pose')

# Load LSTM TFLite
lstm_interpreter = Interpreter(model_path="fall_lstm_v2_fp32.tflite")
lstm_interpreter.allocate_tensors()
lstm_input_details = lstm_interpreter.get_input_details()
lstm_output_details = lstm_interpreter.get_output_details()

def normalize_window(window_kpts):
    """
    HÀM CHUẨN HÓA DÀNH CHO MOBILE:
    Được thiết kế để chạy độc lập với kích thước màn hình. Dễ dàng viết lại bằng Java/Kotlin/Swift.
    """
    normalized = []
    prev_cx, prev_cy = None, None
    prev_scale = 1.0
    prev_head_y = None
    
    for i in range(len(window_kpts)):
        kpts = window_kpts[i]
        valid_mask = kpts[:, 2] > 0.2
        
        if np.sum(valid_mask) > 0:
            # 1. Tính tâm hông
            hip_mask = valid_mask[[11, 12]]
            if np.sum(hip_mask) == 2:
                cx = (kpts[11, 0] + kpts[12, 0]) / 2.0
                cy = (kpts[11, 1] + kpts[12, 1]) / 2.0
            elif valid_mask[11]:
                cx, cy = kpts[11, 0], kpts[11, 1]
            elif valid_mask[12]:
                cx, cy = kpts[12, 0], kpts[12, 1]
            else:
                cx = np.mean(kpts[valid_mask, 0])
                cy = np.mean(kpts[valid_mask, 1])
                
            # 2. Tính Scale và Aspect Ratio
            min_x = np.min(kpts[valid_mask, 0])
            max_x = np.max(kpts[valid_mask, 0])
            min_y = np.min(kpts[valid_mask, 1])
            max_y = np.max(kpts[valid_mask, 1])
            
            width = max_x - min_x
            height = max_y - min_y
            
            scale = max(height, width) + 1e-5
            aspect_ratio = width / (height + 1e-5)
            
            # 3. Tính Velocity (Vận tốc hông và đầu)
            if prev_cx is not None:
                delta_x = (cx - prev_cx) / prev_scale
                delta_y = (cy - prev_cy) / prev_scale
                if valid_mask[0] and prev_head_y is not None:
                    head_delta_y = (kpts[0, 1] - prev_head_y) / prev_scale
                else:
                    head_delta_y = 0.0
            else:
                delta_x = 0.0
                delta_y = 0.0
                head_delta_y = 0.0
                
            prev_cx, prev_cy = cx, cy
            prev_scale = scale
            if valid_mask[0]:
                prev_head_y = kpts[0, 1]
            
            # 4. Gom features
            frame_features = []
            for j in range(17):
                if valid_mask[j]:
                    norm_x = (kpts[j, 0] - cx) / scale
                    norm_y = (kpts[j, 1] - cy) / scale
                    frame_features.extend([norm_x, norm_y])
                else:
                    frame_features.extend([0.0, 0.0])
                    
            frame_features.extend([aspect_ratio, delta_x, delta_y, head_delta_y])
            normalized.append(frame_features)
        else:
            normalized.append([0.0] * 38)
            prev_cx, prev_cy = None, None
            prev_head_y = None
            
    return np.array(normalized, dtype=np.float32)

def compute_drop_ratio(window_kpts):
    """
    BỘ LỌC VẬN TỐC RƠI (Velocity Drop Filter)
    Tính tỷ lệ hông rơi xuống so với chiều cao cơ thể trong cửa sổ 30 frame.
    
    Trả về:
      - drop_ratio > 0 nếu hông rơi xuống (ngã thật)
      - drop_ratio ≈ 0 nếu hông không di chuyển (đang nằm sẵn)
      - drop_ratio < 0 nếu hông đi lên (đang đứng dậy)
    
    Trên Mobile, hàm này chỉ cần vài phép tính số học, rất nhẹ.
    """
    # Lấy hip_y trung bình ở 5 frame đầu và 5 frame cuối (robust hơn 1 frame)
    N_AVG = 5
    
    def get_hip_y(kpts_frame):
        """Lấy tọa độ Y của hông từ 1 frame keypoints."""
        l_hip, r_hip = kpts_frame[11], kpts_frame[12]
        if l_hip[2] > 0.2 and r_hip[2] > 0.2:
            return (l_hip[1] + r_hip[1]) / 2.0
        elif l_hip[2] > 0.2:
            return l_hip[1]
        elif r_hip[2] > 0.2:
            return r_hip[1]
        return None
    
    def get_body_height(kpts_frame):
        """Chiều cao cơ thể = khoảng cách Y từ điểm cao nhất đến thấp nhất."""
        valid = kpts_frame[kpts_frame[:, 2] > 0.2]
        if len(valid) < 2:
            return None
        return np.max(valid[:, 1]) - np.min(valid[:, 1])
    
    # Thu thập hip_y ở đầu và cuối window
    start_ys = []
    for i in range(min(N_AVG, len(window_kpts))):
        y = get_hip_y(window_kpts[i])
        if y is not None:
            start_ys.append(y)
    
    end_ys = []
    for i in range(max(0, len(window_kpts) - N_AVG), len(window_kpts)):
        y = get_hip_y(window_kpts[i])
        if y is not None:
            end_ys.append(y)
    
    if not start_ys or not end_ys:
        return 0.0  # Không đủ dữ liệu -> không chắc chắn -> không báo ngã
    
    hip_y_start = np.mean(start_ys)
    hip_y_end = np.mean(end_ys)
    
    # Chiều cao cơ thể: lấy max trong toàn bộ window (lúc đứng sẽ cao nhất)
    max_body_height = 0.0
    for kf in window_kpts:
        bh = get_body_height(kf)
        if bh is not None and bh > max_body_height:
            max_body_height = bh
    
    if max_body_height < 10:  # Quá nhỏ, không đáng tin
        return 0.0
    
    # drop_ratio > 0 nghĩa là hông rơi xuống (Y tăng = rơi xuống trong hệ tọa độ ảnh)
    drop_ratio = (hip_y_end - hip_y_start) / max_body_height
    return drop_ratio

def run_mobile_pipeline(video_path, out_dir=None):
    import os
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{name}_output.mp4")
    else:
        output_path = video_path.replace(ext, "_output.mp4")
        
    out = cv2.VideoWriter(output_path, fourcc, 30, (TARGET_WIDTH, TARGET_HEIGHT))
    
    track_history = collections.defaultdict(lambda: collections.deque(maxlen=SEQ_LEN))
    fall_timers = collections.defaultdict(int)  # Bộ đếm duy trì trạng thái ngã cho từng người
    
    print("Đang xử lý video giả lập (kích thước ép cứng 480x640)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Resize ép chuẩn Mobile Camera (giữ nguyên tỷ lệ bằng Letterbox để không bị méo người)
        h, w = frame.shape[:2]
        scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(frame, (new_w, new_h))
        padded = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        x_offset = (TARGET_WIDTH - new_w) // 2
        y_offset = (TARGET_HEIGHT - new_h) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        frame = padded
        
        # 2. YOLO-Pose tracking
        # Lưu ý: trên Mobile thường ta truyền bitmap, tracker tự triển khai
        results = yolo_model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        
        is_any_fall = False
        if len(results[0].boxes) > 0 and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            kpts = results[0].keypoints.data.cpu().numpy()
            
            for i in range(len(ids)):
                track_id = ids[i]
                raw_kpt = kpts[i]
                
                track_history[track_id].append(raw_kpt)
                
                # Vẽ người
                bbox = boxes.xyxy.cpu().numpy()[i]
                x1, y1, x2, y2 = map(int, bbox)
                
                color = (0, 255, 0)
                
                # Vẽ khớp xương
                for kpt in raw_kpt:
                    x, y, conf = kpt
                    if conf > 0.2:
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                
                # Vẽ đường nối xương
                skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
                for u, v in skeleton:
                    if raw_kpt[u, 2] > 0.2 and raw_kpt[v, 2] > 0.2:
                        cv2.line(frame, (int(raw_kpt[u, 0]), int(raw_kpt[u, 1])), (int(raw_kpt[v, 0]), int(raw_kpt[v, 1])), color, 1)
                
                if len(track_history[track_id]) == SEQ_LEN:
                    # 3. Normalize
                    seq_array = np.array(track_history[track_id], dtype=np.float32)
                    norm_seq = normalize_window(seq_array) # (30, 38)
                    
                    # 4. Chạy LSTM TFLite
                    input_data = np.expand_dims(norm_seq, axis=0).astype(np.float32) # (1, 30, 38)
                    lstm_interpreter.set_tensor(lstm_input_details[0]['index'], input_data)
                    lstm_interpreter.invoke()
                    outputs = lstm_interpreter.get_tensor(lstm_output_details[0]['index'])
                    
                    prob = outputs[0][0]
                    
                    # === BỘ LỌC 2 LỚP ===
                    # Lớp 1: LSTM Model (tư thế)
                    # Lớp 2: Velocity Drop Filter (vận tốc rơi)
                    drop_ratio = compute_drop_ratio(seq_array)
                    
                    lstm_says_fall = prob > LSTM_THRESHOLD
                    hip_actually_dropped = drop_ratio > DROP_RATIO_THRESHOLD
                    
                    if lstm_says_fall and hip_actually_dropped:
                        # Kích hoạt bộ đếm duy trì trạng thái ngã (ví dụ: 90 frame = 3 giây)
                        fall_timers[track_id] = 90
                        
                    if fall_timers[track_id] > 0:
                        fall_timers[track_id] -= 1
                        # TRẠNG THÁI NGÃ (Đang ngã hoặc dư âm ngã)
                        is_any_fall = True
                        color = (0, 0, 255)
                        cv2.putText(frame, f"FALL ({prob:.2f} D:{drop_ratio:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                        # Tô lại xương màu đỏ nếu ngã
                        for u, v in skeleton:
                            if raw_kpt[u, 2] > 0.2 and raw_kpt[v, 2] > 0.2:
                                cv2.line(frame, (int(raw_kpt[u, 0]), int(raw_kpt[u, 1])), (int(raw_kpt[v, 0]), int(raw_kpt[v, 1])), color, 2)
                    elif lstm_says_fall and not hip_actually_dropped:
                        # LSTM báo ngã nhưng hông không rơi -> Người đang nằm sẵn, BỎ QUA
                        cv2.putText(frame, f"LYING ({prob:.2f} D:{drop_ratio:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    else:
                        cv2.putText(frame, f"OK ({prob:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        main_color = (0, 0, 255) if is_any_fall else (0, 255, 0)
        cv2.putText(frame, "MOBILE PIPELINE (480x640)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, 2)
        out.write(frame)

    cap.release()
    out.release()
    print("✅ Hoàn tất giả lập Mobile Pipeline!")

if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "Test2.mp4"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    run_mobile_pipeline(video, out_dir)
