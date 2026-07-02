import cv2
import torch
import numpy as np
import collections
from train_lstm import FallDetectionLSTM, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS
from ultralytics import YOLO

SEQ_LEN = 15

from data_preparation import normalize_window

def test_pipeline(video_path, output_path):
    print("Loading models...")
    # Load YOLO-Pose (yolov8n-pose.pt will be downloaded automatically)
    yolo_model = YOLO("yolov8n-pose.pt")
    
    # Load LSTM
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    lstm_model = FallDetectionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    lstm_model.load_state_dict(torch.load("/Users/ledangkhoa/do_an/fall_lstm_best.pt", map_location=device))
    lstm_model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Rolling window    # Fall detection logic
    sequence_buffer = collections.deque(maxlen=SEQ_LEN)
    target_id = None
    
    print("Processing video...")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO
        results = yolo_model(frame, verbose=False)
        
        fall_prob = 0.0
        is_fall = False
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                # Chọn mục tiêu mới nếu chưa có hoặc mục tiêu cũ biến mất
                if target_id is None or target_id not in ids:
                    best_idx = np.argmax(confs)
                    target_id = ids[best_idx]
                
                # Tìm vị trí của mục tiêu đang theo dõi
                target_idx = np.where(ids == target_id)[0][0]
                raw_kpts = results[0].keypoints.data[target_idx].cpu().numpy()
            else:
                best_idx = torch.argmax(boxes.conf).item()
                raw_kpts = results[0].keypoints.data[best_idx].cpu().numpy()
            
            # Push to buffer
            sequence_buffer.append(raw_kpts)
            
            # Draw keypoints (basic)
            for kpt in raw_kpts:
                x, y, conf = kpt
                if conf > 0.2:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        else:
            # Append zeros if no person detected to keep the 30-frame sequence intact
            sequence_buffer.append(np.zeros((17, 3), dtype=np.float32))
            
        # Run LSTM if buffer is full
        if len(sequence_buffer) == SEQ_LEN:
            seq_array = np.array(sequence_buffer, dtype=np.float32)
            norm_seq = normalize_window(seq_array)
            seq_tensor = torch.tensor(norm_seq).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = lstm_model(seq_tensor)
                prob = torch.sigmoid(output).item()
                fall_prob = prob
                
                print(f"Frame {frame_count:03d} | Prob: {prob:.4f}")
                
                if prob > 0.5:
                    is_fall = True

        # Annotate Frame
        status_text = "FALL DETECTED" if is_fall else "NORMAL"
        color = (0, 0, 255) if is_fall else (255, 0, 0) # Red for Fall, Blue for Normal
        
        cv2.putText(frame, f"{status_text} (Prob: {fall_prob:.2f})", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"✅ Video processing complete! Saved to {output_path}")

if __name__ == "__main__":
    # Pick a random video from the dataset that has a fall
    # Let's use a Coffee_room_01 video which typically has a fall
    VIDEO_PATH = "tạo_cho_tôi_một_video_té_ngã_n.mp4"
    OUTPUT_PATH = "/Users/ledangkhoa/do_an/test_output.mp4"
    test_pipeline(VIDEO_PATH, OUTPUT_PATH)
