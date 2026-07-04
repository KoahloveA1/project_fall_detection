import cv2
import torch
import numpy as np
import time
import collections
import onnxruntime as ort
from ultralytics import YOLO

SEQ_LEN = 30
LSTM_THRESHOLD = 0.5
DROP_RATIO_THRESHOLD = 0.15

# Import hàm normalize và compute_drop_ratio từ mobile_pipeline_sim
from mobile_pipeline_sim import normalize_window, compute_drop_ratio

def test_webcam():
    print("Đang tải model...")
    device = "mps" if hasattr(__import__('torch').backends, 'mps') and __import__('torch').backends.mps.is_available() else "cpu"
    print(f"YOLO sẽ chạy trên: {device.upper()}")
    yolo_model = YOLO("yolo26n-pose.mlpackage", task="pose")

    lstm_session = ort.InferenceSession("tflite_models/fall_lstm_v2.onnx")
    lstm_input_name = lstm_session.get_inputs()[0].name
    lstm_output_name = lstm_session.get_outputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera!")
        return

    track_history = collections.defaultdict(lambda: collections.deque(maxlen=SEQ_LEN))

    skeleton = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),
                (5,6),(5,7),(6,8),(7,9),(8,10),(1,2),(0,1),(0,2),(1,3),(2,4),(3,5),(4,6)]

    prev_time = time.time()
    print("Camera đã bật! Bấm 'q' để thoát.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.time()
        # Sử dụng mô hình CoreML (.mlpackage) để có keypoint xịn và max FPS trên Mac
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml",
                                   verbose=False)
        t_yolo = time.time() - t0
        t_lstm = 0.0

        is_any_fall = False

        if len(results[0].boxes) > 0 and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            kpts = results[0].keypoints.data.cpu().numpy()

            for i in range(len(ids)):
                track_id = ids[i]
                raw_kpt = kpts[i]
                bbox = boxes.xyxy.cpu().numpy()[i]
                x1, y1, x2, y2 = map(int, bbox)

                track_history[track_id].append(raw_kpt)

                color = (0, 255, 0)
                label = ""

                if len(track_history[track_id]) == SEQ_LEN:
                    seq_array = np.array(track_history[track_id], dtype=np.float32)
                    norm_seq = normalize_window(seq_array)
                    input_data = np.expand_dims(norm_seq, axis=0).astype(np.float32)

                    t1 = time.time()
                    outputs = lstm_session.run([lstm_output_name], {lstm_input_name: input_data})
                    t_lstm = time.time() - t1
                    
                    prob = outputs[0][0][0]

                    drop_ratio = compute_drop_ratio(seq_array)

                    lstm_says_fall = prob > LSTM_THRESHOLD
                    hip_actually_dropped = drop_ratio > DROP_RATIO_THRESHOLD

                    if lstm_says_fall and hip_actually_dropped:
                        is_any_fall = True
                        color = (0, 0, 255)
                        label = f"FALL ({prob:.2f} D:{drop_ratio:.2f})"
                        print(f"🔥 FALL ID:{track_id} prob={prob:.2f} drop={drop_ratio:.2f}")
                    elif lstm_says_fall and not hip_actually_dropped:
                        color = (0, 165, 255)
                        label = f"LYING ({prob:.2f} D:{drop_ratio:.2f})"
                    else:
                        label = f"OK ({prob:.2f})"
                else:
                    label = f"Buffering {len(track_history[track_id])}/{SEQ_LEN}"
                    t_lstm = 0.0

                # Vẽ bbox
                thickness = 4 if color == (0, 0, 255) else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"ID:{track_id} {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Vẽ keypoints
                for kpt in raw_kpt:
                    x, y, conf = kpt
                    if conf > 0.2:
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1)

                # Vẽ skeleton
                for u, v in skeleton:
                    if raw_kpt[u, 2] > 0.2 and raw_kpt[v, 2] > 0.2:
                        cv2.line(frame,
                                 (int(raw_kpt[u, 0]), int(raw_kpt[u, 1])),
                                 (int(raw_kpt[v, 0]), int(raw_kpt[v, 1])),
                                 color, 1)

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        
        # Chỉ in log ra console mỗi 10 frame để không bị trôi
        if int(curr_time * 10) % 10 == 0:
            print(f"FPS: {fps:.1f} | YOLO: {t_yolo*1000:.1f}ms | LSTM: {t_lstm*1000:.1f}ms" if 't_yolo' in locals() else f"FPS: {fps:.1f}")

        # Header
        main_color = (0, 0, 255) if is_any_fall else (0, 255, 0)
        main_text = "⚠️  FALL DETECTED!" if is_any_fall else "ALL NORMAL"
        cv2.putText(frame, main_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, main_color, 3)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Fall Detection v2 | Press Q to quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Fall Detection - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Đã tắt camera.")

if __name__ == "__main__":
    test_webcam()
