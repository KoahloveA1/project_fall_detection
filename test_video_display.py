import cv2
import torch
import numpy as np
import collections
import time
import sys
from train_lstm import FallDetectionLSTM, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS
from ultralytics import YOLO

SEQ_LEN = 15

from data_preparation import normalize_window

def test_video_display(video_path):
    print("Loading models...")
    # Tải YOLO-Pose
    yolo_model = YOLO("yolo26n-pose.pt")
    
    # Tải LSTM
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    lstm_model = FallDetectionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    lstm_model.load_state_dict(torch.load("/Users/ledangkhoa/do_an/fall_lstm_best.pt", map_location=device))
    lstm_model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}!")
        return

    # Buffer lưu trữ 30 frames liên tiếp
    sequence_buffer = collections.deque(maxlen=SEQ_LEN)
    
    print("Đang phát video... (Bấm phím 'q' trên cửa sổ video để thoát)")
    
    # Multi-person Tracking
    track_history = collections.defaultdict(lambda: collections.deque(maxlen=SEQ_LEN))
    alarm_cooldown = collections.defaultdict(int)
    
    delay = 1
    prev_time = time.time()
    out = None
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Đã phát hết video.")
            break
            
        # Thu nhỏ frame nếu video quá to để xem vừa trên màn hình
        height, width = frame.shape[:2]
        if height > 720:
            scale = 720 / height
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            
        # Chạy YOLO-Pose với ByteTrack để theo dõi nhiều người cùng lúc cực kỳ ổn định
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=device)
        
        is_any_fall = False
        
        if len(results[0].boxes) > 0 and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxys = boxes.xyxy.cpu().numpy()
            kpts = results[0].keypoints.data.cpu().numpy()
            
            for i in range(len(ids)):
                track_id = ids[i]
                raw_kpt = kpts[i]
                bbox = xyxys[i]
                
                # Đưa keypoints vào buffer của riêng ID này
                track_history[track_id].append(raw_kpt)
                
                fall_prob = 0.0
                is_fall = False
                
                # Khi gom đủ 15 frames cho người này thì chạy dự đoán LSTM
                if len(track_history[track_id]) == SEQ_LEN:
                    seq_array = np.array(track_history[track_id], dtype=np.float32)
                    norm_seq = normalize_window(seq_array) # Shape: (15, 37)
                    seq_tensor = torch.tensor(norm_seq).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = lstm_model(seq_tensor)
                        prob = torch.sigmoid(output).item()
                        fall_prob = prob
                        
                        # Bộ lọc Post-processing để loại bỏ nhiễu từ người ở xa hoặc đang đứng
                        bbox_w = bbox[2] - bbox[0]
                        bbox_h = bbox[3] - bbox[1]
                        aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
                        
                        # 1. Bỏ qua nếu người quá nhỏ (ở tít phía sau)
                        is_large_enough = bbox_h > (frame.shape[0] / 3.0) 
                        # 2. Bỏ qua nếu dáng người đang thẳng đứng (ngã thì phải nằm ngang/co cụm)
                        is_not_standing = aspect_ratio > 0.6
                        
                        if prob > 0.85 and is_large_enough and is_not_standing:
                            alarm_cooldown[track_id] = 30
                            print(f"🔥 FALL DETECTED on ID {track_id}! Prob: {prob:.4f}")
                
                if alarm_cooldown[track_id] > 0:
                    is_fall = True
                    is_any_fall = True
                    alarm_cooldown[track_id] -= 1
                    
                # Vẽ Box và Text cho từng người
                color = (0, 0, 255) if is_fall else (0, 255, 0)
                thickness = 4 if is_fall else 2
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                status_txt = "FALL" if is_fall else "NORMAL"
                cv2.putText(frame, f"ID:{track_id} {status_txt} ({fall_prob*100:.0f}%)", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Vẽ khớp xương
                for kpt in raw_kpt:
                    x, y, conf = kpt
                    if conf > 0.2:
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1)

        # Tính toán FPS
        curr_time = time.time()
        fps_display = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        # Hiển thị cảnh báo tổng
        main_color = (0, 0, 255) if is_any_fall else (0, 255, 0)
        main_text = "FALL DETECTED!" if is_any_fall else "ALL NORMAL"
        cv2.putText(frame, main_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, main_color, 3)
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if out is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = video_path.replace(".mp4", "_output.mp4")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            print(f"Saving output to {output_path}...")
            
        out.write(frame)
        
        # Mở cửa sổ hiển thị video
        # cv2.imshow("Video Fall Detection", frame)
        
        # Bấm phím 'q' để thoát, delay giúp video chạy đúng tốc độ thực
        # if cv2.waitKey(delay) & 0xFF == ord('q'):
        #     break

    cap.release()
    if out is not None:
        out.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "/Users/ledangkhoa/do_an/Test1.mp4"
    test_video_display(video_path)
