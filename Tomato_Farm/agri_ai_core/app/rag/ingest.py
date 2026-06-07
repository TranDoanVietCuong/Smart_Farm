from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from app.config import OPENAI_API_KEY, QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
from app.utils.parser import extract_pdf_text

def ingest_pdf_pipeline(pdf_path: str):
    """Pipeline chia nhỏ tài liệu và đẩy dữ liệu vector vào QdrantDB."""
    # 1. Trích xuất text từ tài liệu nông nghiệp
    raw_text = extract_pdf_text(pdf_path)
    
    # 2. Chia nhỏ văn bản tối ưu cho ngữ nghĩa tiếng Việt
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       # Tăng dung lượng để chứa trọn vẹn 1 ý
    chunk_overlap=200,    # Tăng độ gối đầu lên để các câu không bị đứt đoạn
    separators=["\n\n", "\n", ".", " ", ""]
)
    chunks = splitter.split_text(raw_text)
    
    # 3. Đóng gói thành Document kèm Metadata lọc
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "crop": "cà chua",
                    "source": pdf_path,
                    "chunk_id": i
                }
            )
        )
    
    # 4. Khởi tạo mô hình Embedding tiếng Việt
    embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
    
    # 5. Lưu trữ trực tiếp vào Qdrant (ÉP TÊN KHO MỚI)
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url="http://localhost:6333", 
        collection_name=COLLECTION_NAME
    )
    return len(docs)