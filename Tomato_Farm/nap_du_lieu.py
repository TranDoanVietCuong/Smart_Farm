import sys
import os

# Mở đường cho Python vào lõi AI của cậu
sys.path.append("./agri_ai_core")

# Lôi hàm ingest_pdf_pipeline mà cậu đã viết sẵn trong agri_ai_core ra dùng
from agri_ai_core.app.rag.ingest import ingest_pdf_pipeline

# Tên file PDF cậu muốn dạy cho AI (nhớ đổi tên cho đúng file thực tế nhé)
tai_lieu_pdf = "cam_nang_ca_chua.pdf"

if not os.path.exists(tai_lieu_pdf):
    print(f"❌ Không tìm thấy file {tai_lieu_pdf}. Lloydz nhớ copy file PDF vào đây nha!")
else:
    print("⏳ AI đang đọc và học thuộc lòng cuốn cẩm nang... Quá trình này sẽ mất một lúc...")
    
    # Kích hoạt pipeline nạp dữ liệu!
    total_chunks = ingest_pdf_pipeline(tai_lieu_pdf)
    
    print(f"✅ HỌC XONG! Đã băm nhỏ và nạp thành công bộ nhớ vào Qdrant.")
    print("🚀 Từ giờ trở đi, cậu KHÔNG CẦN chạy lại file này nữa. Kiến thức đã lưu vĩnh viễn!")