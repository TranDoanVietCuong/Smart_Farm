import os
from ultralytics import YOLO

MODEL_PATH = "models/tomato_disease.pt"

# Kiểm tra sự tồn tại của file trọng số YOLOv11
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    model = None
    print(f"[Cảnh báo] Chưa tìm thấy file mô hình tại {MODEL_PATH}. Vui lòng bổ sung sau.")

def detect_tomato_disease(image_path: str) -> dict:
    """Sử dụng YOLOv11 phân loại/phát hiện bệnh lý qua ảnh lá cà chua."""
    if model is None:
        return {"disease": "Unknown", "confidence": 0.0}
    
    results = model(image_path)
    result = results[0]
    
    # Xử lý trường hợp mô hình là Classification (Phân loại bệnh lý)
    if hasattr(result, 'probs') and result.probs is not None:
        top_idx = result.probs.top1
        class_name = result.names[top_idx]
        confidence = float(result.probs.top1conf)
        return {"disease": class_name, "confidence": confidence}
    
    # Xử lý trường hợp mô hình là Object Detection (Phát hiện vùng bệnh)
    elif hasattr(result, 'boxes') and len(result.boxes) > 0:
        box = result.boxes[0]  # Lấy hộp có độ tin cậy cao nhất
        class_idx = int(box.cls[0])
        class_name = result.names[class_idx]
        confidence = float(box.conf[0])
        return {"disease": class_name, "confidence": confidence}
        
    return {"disease": "Tomato___healthy", "confidence": 1.0}