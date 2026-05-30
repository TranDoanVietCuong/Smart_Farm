import requests
from app.rag.retriever import retriever
from app.rag.prompts import SYSTEM_PROMPT

def ask_agriculture(question: str) -> str:
    """Hàm xử lý logic hỏi đáp RAG sử dụng RIPT OpenAI Wrapper."""
    
    # Bước 1: Quét Vector DB (Qdrant) để lấy ra các đoạn tài liệu nông nghiệp liên quan nhất
    docs = retriever.invoke(question)
    
    # Nối các đoạn tài liệu lại thành một chuỗi văn bản dài
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Bước 2: Chuẩn bị nội dung gửi đi (Gộp ngữ cảnh và câu hỏi của nông dân)
    prompt_text = f"Ngữ cảnh (Context):\n{context}\n\nCâu hỏi của người nông dân:\n{question}"
    
    # URL và payload theo đúng chuẩn tài liệu Wrapper
    url = "https://sl-form-ai.ript.vn/api/v1/openai/chat"
    payload = {
        "text": prompt_text,
        "system_content": SYSTEM_PROMPT
    }
    
    # Bước 3: Gửi request gọi API
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status() # Bắt lỗi nếu server trả về 4xx, 5xx
        
        # Trích xuất kết quả trả về từ trường "result"
        return response.json().get("result", "Hệ thống không nhận được câu trả lời.")
        
    except Exception as e:
        return f"Xin lỗi, kết nối đến hệ thống AI đang gặp sự cố. Chi tiết lỗi: {str(e)}"