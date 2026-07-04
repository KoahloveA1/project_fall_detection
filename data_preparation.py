import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_CSV_DIR = "/Users/ledangkhoa/do_an/results/extracted_keypoints"
ARCHIVE_DIR = "/Users/ledangkhoa/do_an/archive"
OUTPUT_DIR = "/Users/ledangkhoa/do_an/dataset_tensors"

SEQ_LEN = 30
# Augmentation multipliers
AUGMENT_COPIES = 3 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_annotation(video_name, subdir):
    # Try different possible paths for the annotation file
    # video_name: e.g., "video (1).csv" -> we need "video (1).txt"
    base_name = video_name.replace('.csv', '.txt')
    
    # Common LE2I annotation paths
    possible_paths = [
        os.path.join(ARCHIVE_DIR, subdir, subdir, 'Annotation_files', base_name),
        os.path.join(ARCHIVE_DIR, subdir, subdir, 'Annotations_files', base_name),
        os.path.join(ARCHIVE_DIR, subdir, 'Annotation_files', base_name)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    try:
                        start_fall = int(lines[0].strip())
                        end_fall = int(lines[1].strip())
                        return start_fall, end_fall
                    except ValueError:
                        pass
    return 0, 0 # No fall or annotation not found

def normalize_window(window_kpts):
    # window_kpts shape: (SEQ_LEN, 17, 3)
    # Output shape: (SEQ_LEN, 37)
    normalized = []
    
    prev_cx, prev_cy = None, None
    prev_scale = 1.0
    prev_head_y = None
    
    for i in range(len(window_kpts)):
        kpts = window_kpts[i]
        valid_mask = kpts[:, 2] > 0.2
        
        if np.sum(valid_mask) > 0:
            # 1. Tính trung tâm Hông (Pelvis)
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
            
            # 3. Tính Velocity (Delta X, Delta Y) chuẩn hóa theo scale của khung hình TRƯỚC
            # Lý do dùng scale trước: vận tốc là quãng đường trên scale gốc
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
                
            # Cập nhật prev
            prev_cx, prev_cy = cx, cy
            prev_scale = scale
            if valid_mask[0]:
                prev_head_y = kpts[0, 1]
            
            # 4. Chuẩn hóa 17 điểm theo tư thế hiện tại
            frame_features = []
            for j in range(17):
                if valid_mask[j]:
                    norm_x = (kpts[j, 0] - cx) / scale
                    norm_y = (kpts[j, 1] - cy) / scale
                    frame_features.extend([norm_x, norm_y])
                else:
                    frame_features.extend([0.0, 0.0])
                    
            # Thêm aspect_ratio, delta_x, delta_y, head_delta_y
            frame_features.extend([aspect_ratio, delta_x, delta_y, head_delta_y])
            normalized.append(frame_features)
        else:
            normalized.append([0.0] * 38)
            # Khởi tạo lại prev nếu bị mất tracking
            prev_cx, prev_cy = None, None
            prev_head_y = None
            
    return np.array(normalized, dtype=np.float32)

def augment_window(window):
    """ Apply Jitter and Flip to normalized window (SEQ_LEN, 37) """
    features = window.copy()
    
    # 1. Random Jitter (Noise)
    noise = np.random.normal(0, 0.02, size=(SEQ_LEN, 34))
    features[:, :34] += noise
    
    # 2. Random Horizontal Flip (50% chance)
    if np.random.rand() > 0.5:
        # Lật trục X (nhân -1)
        for k in range(17):
            features[:, k*2] *= -1
            
        # Đảo trái/phải
        swap_pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
        for left, right in swap_pairs:
            temp_x = features[:, left*2].copy()
            temp_y = features[:, left*2 + 1].copy()
            
            features[:, left*2] = features[:, right*2]
            features[:, left*2 + 1] = features[:, right*2 + 1]
            
            features[:, right*2] = temp_x
            features[:, right*2 + 1] = temp_y
            
        # Khi lật ngang, vận tốc ngang (Delta X) cũng phải đổi chiều!
        features[:, 35] *= -1
            
    return features

def process_dataset():
    subdirs = [d for d in os.listdir(DATA_CSV_DIR) if os.path.isdir(os.path.join(DATA_CSV_DIR, d))]
    
    # Chỉ định 2 thư mục này làm Validation (để tránh rò rỉ dữ liệu cùng phòng/cùng bối cảnh)
    VAL_DIRS = ['Coffee_room_02', 'Home_02']
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    
    for subdir in subdirs:
        csv_dir = os.path.join(DATA_CSV_DIR, subdir)
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        
        print(f"Processing {subdir}: {len(csv_files)} videos")
        is_val = subdir in VAL_DIRS
        
        for csv_file in tqdm(csv_files, leave=False):
            video_name = os.path.basename(csv_file)
            start_fall, end_fall = parse_annotation(video_name, subdir)
            
            df = pd.read_csv(csv_file)
            if len(df) < SEQ_LEN:
                continue
                
            # Extract keypoints (exclude frame_idx at column 0)
            raw_kpts = df.iloc[:, 1:].values # shape: (N, 51)
            
            # Sliding window
            # Step size: overlap more for falls to get more data
            for i in range(0, len(raw_kpts) - SEQ_LEN + 1, 5):
                # Get raw window
                raw_window = raw_kpts[i:i+SEQ_LEN].reshape(SEQ_LEN, 17, 3)
                # Normalize window
                window = normalize_window(raw_window) # shape (SEQ_LEN, 35)
                
                # Check labels for the window
                start_f = int(df.iloc[i]['frame_idx'])
                end_f = int(df.iloc[i+SEQ_LEN-1]['frame_idx'])
                window_frames = range(start_f, end_f + 1)
                
                fall_count = 0
                if start_fall > 0 and end_fall > 0:
                    fall_count = sum(1 for f in window_frames if start_fall <= f <= end_fall)
                
                if fall_count >= SEQ_LEN // 3:
                    label = 1
                else:
                    label = 0
                    
                if is_val:
                    X_val.append(window)
                    y_val.append(label)
                else:
                    X_train.append(window)
                    y_train.append(label)
                    
                    # Apply augmentation for Fall class to balance dataset (chỉ áp dụng cho tập Train)
                    if label == 1:
                        for _ in range(AUGMENT_COPIES):
                            aug_window = augment_window(window)
                            X_train.append(aug_window)
                            y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    
    print(f"Dataset generated!")
    print(f"TRAIN: {len(X_train)} samples (Normal: {np.sum(y_train == 0)}, Fall: {np.sum(y_train == 1)})")
    print(f"VAL  : {len(X_val)} samples (Normal: {np.sum(y_val == 0)}, Fall: {np.sum(y_val == 1)})")
    
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()
