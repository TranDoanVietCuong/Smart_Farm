# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as ai_router

app = FastAPI(
    title="Hệ thống Cốt lõi AI Nông Nghiệp Thông Minh (Cà Chua)",
    description="Backend API Production-Ready tích hợp RAG Tri thức và Thị giác máy tính YOLOv11 bảo vệ thực vật.",
    version="2.0.0"
)

# Cấu hình CORS để các ứng dụng Frontend (NextJS/Flutter) kết nối linh hoạt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong môi trường production thật, hãy thay bằng domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tích hợp cụm router điều hướng xử lý AI vào ứng dụng chính với prefix v1
app.include_router(ai_router, prefix="/api/v1")

@app.get("/", tags=["Root"])
def read_root():
    """Endpoint kiểm tra trạng thái hoạt động của Server."""
    return {
        "status": "Online", 
        "message": "Hệ thống AI Core hỗ trợ nông dân đã sẵn sàng hoạt động! 🍅🌱"
    }