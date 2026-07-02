import numpy as np

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def get_best_box_idx(prev_bbox, current_boxes, confs):
    if prev_bbox is None:
        return np.argmax(confs)
        
    ious = [compute_iou(prev_bbox, box) for box in current_boxes]
    best_idx = np.argmax(ious)
    
    # Nếu IoU quá thấp (mất dấu), chọn người rõ nhất
    if ious[best_idx] < 0.1:
        return np.argmax(confs)
    return best_idx
