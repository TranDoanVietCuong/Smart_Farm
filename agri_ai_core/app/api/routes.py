import os
import shutil
from fastapi import APIRouter, UploadFile, File
from app.models.schemas import ChatRequest, ChatResponse, DiagnosisResponse
from app.rag.chain import ask_agriculture
from app.rag.ingest import ingest_pdf_pipeline
from app.services.diagnosis_service import run_combined_ai_workflow

router = APIRouter()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/chat", response_model=ChatResponse, summary="Chatbot RAG tư vấn nông nghiệp")
def api_chat(req: ChatRequest):
    """Endpoint tiếp nhận câu hỏi văn bản từ người nông dân."""
    answer = ask_agriculture(req.question)
    return ChatResponse(question=req.question, answer=answer)

@router.post("/upload-knowledge", summary="Nạp thêm tài liệu kỹ thuật nông nghiệp (PDF)")
def api_upload_knowledge(file: UploadFile = File(...)):
    """Endpoint cho phép quản trị viên hoặc kỹ sư nạp thêm cẩm nang PDF vào cơ sở tri thức."""
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Kích hoạt pipeline nạp dữ liệu
    total_chunks = ingest_pdf_pipeline(file_path)
    
    # Xóa file tạm sau khi nạp xong
    os.remove(file_path)
    
    return {
        "message": f"Nạp tài liệu '{file.filename}' thành công!",
        "processed_chunks": total_chunks
    }

@router.post("/diagnose-leaf", response_model=DiagnosisResponse, summary="Hợp nhất Nhận diện ảnh + Triết xuất phác đồ RAG")
async def api_diagnose_leaf(file: UploadFile = File(...)):
    """Endpoint nhận diện bệnh qua ảnh chụp lá và phản hồi phác đồ điều trị."""
    # Lưu file ảnh tạm thời từ người dùng
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        # Chạy luồng AI hợp nhất chuyên sâu
        result = run_combined_ai_workflow(file_path)
    finally:
        # Bảo đảm file ảnh tạm luôn được dọn dẹp sạch sẽ để tránh tràn bộ nhớ ổ đĩa
        if os.path.exists(file_path):
            os.remove(file_path)
            
    return DiagnosisResponse(
        status=result["status"],
        detected_disease=result["detected_disease"],
        confidence=result["confidence"],
        treatment_plan=result["treatment_plan"]
    )