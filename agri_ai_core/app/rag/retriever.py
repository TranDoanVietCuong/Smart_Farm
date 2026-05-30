from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from app.config import OPENAI_API_KEY, QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

# Khởi tạo mô hình Embedding tiếng Việt
embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

# Kết nối tới Collection hiện có trên Qdrant DB
try:
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name=COLLECTION_NAME
    )
    # Gom luôn lệnh tạo retriever vào bên trong này
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) 
except Exception:
    print("⚠️ Cảnh báo: Kho Qdrant đang trống! Vui lòng nạp tài liệu PDF trước khi Chat.")
    vectorstore = None
    retriever = None