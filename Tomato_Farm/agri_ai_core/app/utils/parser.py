import fitz  # PyMuPDF

def extract_pdf_text(pdf_path: str) -> str:
    """Trích xuất toàn bộ văn bản thô từ file tài liệu PDF tập huấn nông nghiệp."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text