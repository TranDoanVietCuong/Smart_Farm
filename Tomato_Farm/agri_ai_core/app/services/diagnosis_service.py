from app.vision.vision_service import detect_tomato_disease
from app.rag.chain import ask_agriculture

# Bộ từ điển ánh xạ đã được nâng cấp khớp chuẩn với nhãn YOLOv11
DISEASE_MAP = {
    "Bacterial_spot": "Bệnh đốm vi khuẩn trên cây cà chua",
    "Early_blight": "Bệnh úa sớm (sương mai nội vi) trên cây cà chua",
    "Late_blight": "Bệnh sương mai muộn hoại tử trên cây cà chua",
    "Leaf_Mold": "Bệnh nấm mốc lá cà chua",
    "Septoria_leaf_spot": "Bệnh đốm lá Septoria",
    "Spider_mites Two-spotted_spider_mite": "Bệnh nhện đỏ tàn phá lá cà chua",
    "Target_Spot": "Bệnh đốm hình mục tiêu độc hại",
    "Tomato_Yellow_Leaf_Curl_Virus": "Bệnh virus xoăn lùn vàng lá cà chua",
    "Tomato_mosaic_virus": "Bệnh khảm virus thuốc lá hại cà chua",
    "powdery_mildew": "Bệnh phấn trắng trên cây cà chua",
    "healthy": "Cây cà chua hoàn toàn khỏe mạnh"
}

def run_combined_ai_workflow(image_path: str) -> dict:
    """Hợp nhất luồng Vision AI và RAG AI."""
    # Bước 1: Gọi mô hình thị giác máy tính nhận diện bệnh
    vision_res = detect_tomato_disease(image_path)
    raw_disease = vision_res["disease"]
    confidence = vision_res["confidence"]
    
    # Ánh xạ tên bệnh sang tiếng Việt để tra cứu tài liệu nông nghiệp
    vietnamese_disease = DISEASE_MAP.get(raw_disease, "Bệnh lạ trên cây cà chua")
    
    # Bước kiểm tra cây khỏe mạnh đã được đổi thành "healthy" cho đúng chuẩn YOLO
    if raw_disease == "healthy":
        return {
            "status": "Khỏe mạnh",
            "detected_disease": vietnamese_disease,
            "confidence": confidence,
            "treatment_plan": "Lá cây bình thường. Hãy duy trì lịch trình tưới nước, bón phân hữu cơ định kỳ theo đúng tiêu chuẩn VietGAP!"
        }
    
    # Bước 2: Tự động điều hướng, biến nhãn bệnh thành câu hỏi tối ưu hóa cấu trúc cho RAG
    rag_query = f"Trình bày chi tiết về nguyên nhân, cách xử lý khẩn cấp và biện pháp phòng ngừa triệt để đối với {vietnamese_disease}."
    
    # Bước 3: RAG quét Vector DB để LLM sinh phác đồ chuẩn khoa học
    treatment_plan = ask_agriculture(rag_query)
    
    return {
        "status": "Phát hiện bệnh hại",
        "detected_disease": vietnamese_disease,
        "confidence": round(confidence, 4),
        "treatment_plan": treatment_plan
    }