import uvicorn
import socketio
import asyncio
import json
import os
import shutil
import sqlite3
from datetime import datetime
import paho.mqtt.client as mqtt
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys

# Đưa thư mục vào sys.path
sys.path.append("./agri_ai_core")
sys.path.append("./agri_ai_farmSMT")

# ================= 1. KẾT NỐI AI CORE =================
try:
    from agri_ai_core.app.services.diagnosis_service import run_combined_ai_workflow
    from agri_ai_core.app.rag.chain import ask_agriculture
    from agri_ai_farmSMT.tomato_server_hybrid import ppo_hybrid_decision
    
    def predict_24h_water_loss(db_path): return "Hệ thống LSTM đang thu thập dữ liệu..."
    print("✅ Đã kết nối thành công Siêu hệ thống Hybrid AI!")
except ImportError as e:
    print(f"⚠️ Đang dùng fallback: {e}")
    def run_combined_ai_workflow(img): return {"status": "Lỗi", "detected_disease": "Chưa kết nối AI", "confidence": 0, "treatment_plan": ""}
    def ask_agriculture(query): return "Hệ thống RAG đang khởi động..."
    def ppo_hybrid_decision(data): return {"pump": 0, "fan": 0, "light": 0, "scenario": "RL Agent đang quan sát..."}
    def predict_24h_water_loss(db_path): return "Đang thiết lập Model Hồi quy..."

# ================= 2. KHỞI TẠO SQLITE =================
DB_FILE = "farm_data.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, temp REAL, humidity REAL, soil_moist REAL, light TEXT)''')
    conn.commit(); conn.close()

def log_sensor_data(temp, hum, soil, light):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO sensor_logs (temp, humidity, soil_moist, light) VALUES (?, ?, ?, ?)", (temp, hum, soil, light))
    conn.commit(); conn.close()

# ================= 3. CẤU HÌNH MQTT & FASTAPI =================
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
TOPIC_SUBSCRIBE = "lloydz/tomato/sensors" 
TOPIC_PUBLISH = "lloydz/tomato/control"   

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
latest_sensor_data = {"do_am_dat": 0, "nhiet_do": 0, "do_am_kk": 0, "anh_sang": "0%"}
last_control_state = {"pump": -1, "fan": -1, "light": -1} # Biến lưu trạng thái toàn cục

def on_connect(client, userdata, flags, reason_code, properties): client.subscribe(TOPIC_SUBSCRIBE)
def on_message(client, userdata, msg): 
    global latest_sensor_data
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        latest_sensor_data = {"do_am_dat": data.get("soil", 0), "nhiet_do": data.get("temp", 0), "do_am_kk": data.get("hum", 0), "anh_sang": f"{data.get('light', 0)}%"}
        log_sensor_data(data.get("temp", 0), data.get("hum", 0), data.get("soil", 0), data.get("light", 0))
    except Exception: pass

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="Python-Backend-Lloydz-Master")
mqtt_client.on_connect = on_connect; mqtt_client.on_message = on_message

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db() 
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start() 
    task = asyncio.create_task(master_ai_loop())
    yield
    task.cancel(); mqtt_client.loop_stop(); mqtt_client.disconnect()

app = FastAPI(title="Tomato AIoT Farm Master", lifespan=lifespan)
sio_app = socketio.ASGIApp(sio, other_asgi_app=app)
templates = Jinja2Templates(directory="templates")

# ================= 4. API ROUTES =================
class Command(BaseModel): action: str
class ChatMessage(BaseModel): message: str

@app.get("/")
async def index(request: Request): return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/control")
async def control_device(cmd: Command):
    global last_control_state # Thêm global để cập nhật trạng thái ngay khi bấm nút
    payload = {}
    if cmd.action == "PUMP_ON": payload = {"pump": 1}
    elif cmd.action == "PUMP_OFF": payload = {"pump": 0}
    elif cmd.action == "FAN_ON": payload = {"fan": 1}
    elif cmd.action == "FAN_OFF": payload = {"fan": 0}
    elif cmd.action == "LIGHT_ON": payload = {"light": 1}
    elif cmd.action == "LIGHT_OFF": payload = {"light": 0}
        
    if payload:
        mqtt_client.publish(TOPIC_PUBLISH, json.dumps(payload))
        # Cập nhật ngay trạng thái để báo lên UI
        for key, value in payload.items():
            last_control_state[key] = value
        return {"status": "success"}
    return {"status": "error"}

@app.post("/api/chat")
async def chat_api(chat: ChatMessage):
    context_query = f"[Dữ liệu Farm hiện tại: Nhiệt độ {latest_sensor_data['nhiet_do']}C, Ẩm đất {latest_sensor_data['do_am_dat']}%. Nếu không cần thiết, bỏ qua dữ liệu này]. Câu hỏi từ nông dân: {chat.message}"
    return {"response": ask_agriculture(context_query)}

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    temp_dir = "agri_ai_core/temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = f"{temp_dir}/{file.filename}"
    with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    
    try:
        result = run_combined_ai_workflow(file_path)
        confidence_percent = round(result["confidence"] * 100, 2)
        if result["status"] == "Khỏe mạnh": res = f"**Kết quả:** {result['detected_disease']} (Độ tin cậy: {confidence_percent}%)\n\n{result['treatment_plan']}"
        else: res = f"**{result['status']}:** {result['detected_disease']} (Độ tin cậy: {confidence_percent}%)\n\n{result['treatment_plan']}"
    except Exception as e: res = f"**Lỗi hệ thống:** AI xử lý thất bại. Chi tiết: {str(e)}"
        
    if os.path.exists(file_path): os.remove(file_path) 
    return {"status": "success", "result": res}

# ================= 5. MASTER AI LOOP =================
async def master_ai_loop():
    global last_control_state
    while True:
        await asyncio.sleep(5) 
        decision = ppo_hybrid_decision(latest_sensor_data)
        water_forecast = predict_24h_water_loss(DB_FILE)
        
        if "pump" in decision:
            current_control = {"pump": decision["pump"], "fan": decision["fan"], "light": decision["light"]}
            if current_control != last_control_state:
                mqtt_client.publish(TOPIC_PUBLISH, json.dumps(current_control))
                last_control_state = current_control
                print(f"🤖 [AUTO-AI] Bắn lệnh MQTT xuống Wokwi: {current_control}")
        
        # CHỖ NÀY: Truyền thêm trạng thái thiết bị (device_state) lên Web
        await sio.emit('update_dashboard', {
            "sensors": latest_sensor_data, 
            "ai_status": {"scenario": decision.get("scenario", "RL Agent đang phân tích..."), "forecast": water_forecast},
            "device_state": last_control_state 
        })

if __name__ == '__main__':
    uvicorn.run("server:sio_app", host="0.0.0.0", port=8000, reload=True)