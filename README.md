# 🍅 Tomato Farm

Hệ thống Nông nghiệp Thông minh tích hợp AI & IoT chuyên biệt cho cây cà chua. Dự án là sự kết hợp giữa Deep Learning, Học tăng cường (Reinforcement Learning) và Hệ thống chuyên gia để tự động hóa và tối ưu hóa quy trình canh tác.



## 🌟 Tính năng nổi bật

* **Chẩn đoán bệnh thực thời (Vision AI):** Sử dụng **YOLOv11** để nhận diện 11 loại bệnh phổ biến trên lá cà chua với tốc độ xử lý nhanh và độ chính xác cao.
* **Trợ lý nông nghiệp chuyên sâu (RAG):** Kết hợp **Vietnamese-SBERT** và **Qdrant Vector Database** để xây dựng Chatbot RAG. Hệ thống tự động truy xuất phác đồ điều trị, nguyên nhân và cách phòng ngừa từ cẩm nang kỹ thuật nông nghiệp.
* **Điều khiển AI lai (Hybrid PPO Agent):** Sử dụng thuật toán **PPO (Proximal Policy Optimization)** để ra quyết định điều khiển phần cứng. Hệ thống được bảo vệ bởi **Guardrails** (luật cứng dựa trên dữ liệu thời tiết API) để đảm bảo an toàn vận hành.
* **Dự báo tưới tiêu (Time-series Forecasting):** Mô hình **LSTM** kết hợp với cơ sở dữ liệu **SQLite** để tính toán và dự báo tốc độ mất nước của đất trong 24 giờ tới.
* **Dashboard thời gian thực:** Giao diện Web trực quan với biểu đồ **Chart.js**, cập nhật trạng thái thiết bị và dữ liệu môi trường thông qua WebSocket (Socket.IO).

## 🏗️ Kiến trúc hệ thống

Dự án được xây dựng dựa trên kiến trúc phân lớp bất đồng bộ (Asynchronous Architecture):

1.  **Edge Layer:** ESP32 (mô phỏng Wokwi) thu thập cảm biến và thực thi lệnh qua MQTT.
2.  **Persistence Layer:** SQLite cho dữ liệu chuỗi thời gian; Qdrant cho Vector Tri thức.
3.  **AI Layer:** YOLOv11 (Vision) + RAG (NLP) + PPO (Control).
4.  **Backend Layer:** FastAPI Server (Async) quản lý luồng dữ liệu và API.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, Uvicorn, Socket.IO.
* **AI/ML:** PyTorch, Ultralytics (YOLOv11), Stable-Baselines3 (PPO), HuggingFace Transformers.
* **Database:** Qdrant (Vector DB), SQLite (Relational DB).
* **IoT:** MQTT (EMQX Broker).
* **Frontend:** Tailwind CSS, Chart.js, Marked.js.

## 🚀 Cài đặt & Chạy dự án

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.11+. Sau đó cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
pip install torch sentence-transformers torchvision
