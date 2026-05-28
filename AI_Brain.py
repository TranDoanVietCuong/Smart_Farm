import paho.mqtt.client as mqtt
import json
import time
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ================= CẤU HÌNH HỆ THỐNG =================
MQTT_BROKER = "broker.hivemq.com"
TOPIC_SENSOR = "lloydz/tomato/sensors"
TOPIC_PUMP = "lloydz/tomato/pump_cmd"

OWM_API_KEY = "96f2481b97c204c1e1a8abca014bfe5b" # LLOYDZ ĐIỀN API KEY VÀO ĐÂY
CITY_NAME = "Hanoi"

POT_CAPACITY_ML = 2500       
PUMP_FLOW_RATE_ML_SEC = 40   
TARGET_MOISTURE = 65.0       

forecast_cache = {"data": None, "last_update": 0}
is_pumping = False

# ================= 1. XÂY DỰNG & HUẤN LUYỆN MÔ HÌNH MACHINE LEARNING =================
def generate_and_train_model():
    print("[ML] Đang khởi tạo tập dữ liệu lịch sử (2000 mẫu)...")
    X = [] # Features: [Độ ẩm hiện tại, Nhiệt độ hiện tại, Nhiệt độ dự báo (3h tới), Cảnh báo mưa (1=Có, 0=Không)]
    y = [] # Target: [Lượng nước cần bơm (ml)]
    
    # Tạo dữ liệu huấn luyện giả lập dựa trên tri thức nông nghiệp
    for _ in range(2000):
        current_moisture = np.random.uniform(10, 80)
        current_temp = np.random.uniform(20, 42)
        forecast_temp = current_temp + np.random.uniform(-5, 5)
        will_rain = np.random.choice([0, 1], p=[0.7, 0.3]) # 30% tỷ lệ có mưa
        
        # Nhãn (Label) lý tưởng để AI học theo:
        if will_rain == 1 and current_moisture > 40:
            water_ml = 0 # Sắp mưa, đất chưa quá khô -> Không tưới
        else:
            water_ml = (TARGET_MOISTURE - current_moisture) / 100.0 * POT_CAPACITY_ML
            if forecast_temp > 35: 
                water_ml += 600 # Bù bốc hơi nếu 3h tới nắng gắt
            if will_rain == 1:
                water_ml *= 0.4 # Sắp mưa nhưng đất rất khô -> Tưới 40% cầm chừng
                
        if water_ml < 0: water_ml = 0
        
        X.append([current_moisture, current_temp, forecast_temp, will_rain])
        y.append(water_ml)
        
    print("[ML] Đang huấn luyện mô hình Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    print("[ML] Huấn luyện thành công! AI đã sẵn sàng dự đoán.\n" + "="*50)
    return model

# Khởi tạo mô hình Toàn cục (Global)
ai_model = generate_and_train_model()

# ================= 2. API DỰ BÁO THỜI TIẾT (PREDICTIVE FORECAST) =================
def get_weather_forecast():
    current_time = time.time()
    if current_time - forecast_cache["last_update"] < 300 and forecast_cache["data"]:
        return forecast_cache["data"]
    
    try:
        # Dùng endpoint /forecast để lấy dự báo thời tiết trong các giờ tới
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY_NAME}&appid={OWM_API_KEY}&units=metric"
        response = requests.get(url, timeout=5).json()
        
        if response.get("cod") != "200":
            return {"forecast_temp": 25, "will_rain": 0}

        # Lấy mốc dự báo đầu tiên (Khoảng 3 tiếng tiếp theo)
        next_3h_data = response['list'][0]
        forecast_temp = next_3h_data['main']['temp']
        weather_id = next_3h_data['weather'][0]['id']
        
        # Phân loại mưa (Mã 2xx, 3xx, 5xx của OpenWeather)
        will_rain = 1 if weather_id < 600 else 0 
        
        result = {"forecast_temp": forecast_temp, "will_rain": will_rain}
        forecast_cache["data"] = result
        forecast_cache["last_update"] = current_time
        
        status = "CÓ MƯA" if will_rain else "NẮNG RÁO"
        print(f"[API Forecast] Dự báo 3h tới ở {CITY_NAME}: {forecast_temp}°C, {status}")
        return result
    except Exception as e:
        print(f"[API Lỗi] {e}")
        return {"forecast_temp": 25, "will_rain": 0}

# ================= 3. MQTT VÀ THỰC THI (PREDICTION LOOP) =================
def on_connect(client, userdata, flags, reason_code, properties):
    print(">>> AI BRAIN KẾT NỐI THÀNH CÔNG - ĐANG THEO DÕI WOKWI <<<")
    client.subscribe(TOPIC_SENSOR)

def on_message(client, userdata, msg):
    global is_pumping
    payload = json.loads(msg.payload.decode('utf-8'))
    current_moisture = payload.get("soil_moisture", 100)
    current_temp = payload.get("temperature", 25)
    
    # Nếu đang bơm, AI sẽ tập trung theo dõi để ép ngắt bơm (Vòng kín bảo vệ)
    if is_pumping:
        # THÊM DÒNG NÀY ĐỂ TERMINAL KHÔNG BỊ "ĐÓNG BĂNG"
        print(f"[Theo dõi Vòng kín] Đang bơm... Đất hiện tại: {current_moisture}% (Mục tiêu: {TARGET_MOISTURE}%)")
        
        if current_moisture >= TARGET_MOISTURE:
            print("[!] ĐÃ ĐẠT ĐỘ ẨM LÝ TƯỞNG! AI Ép dừng máy bơm.")
            client.publish(TOPIC_PUMP, json.dumps({"cmd": "PUMP_OFF"}))
            is_pumping = False
        return # Đang bơm thì không tính toán mô hình ML mới để tránh nhiễu

    # Chỉ dự đoán bằng Machine Learning khi đất bắt đầu khô (< 60%)
    if current_moisture < 60:
        forecast = get_weather_forecast()
        
        # CHUẨN BỊ FEATURES CHO MÔ HÌNH ML
        features = np.array([[current_moisture, current_temp, forecast["forecast_temp"], forecast["will_rain"]]])
        
        # AI PREDICTION (Dự đoán lượng nước)
        predicted_ml = ai_model.predict(features)[0]
        predicted_ml = int(predicted_ml)
        
        print(f"\n[Cảm biến thực tế] Đất: {current_moisture}% | Nhiệt: {current_temp}°C")
        
        if predicted_ml > 50: # Ngưỡng tối thiểu để bật bơm
            pump_time = int(predicted_ml / PUMP_FLOW_RATE_ML_SEC)
            print(f"[*] ML DỰ ĐOÁN: Cần bơm {predicted_ml}ml nước (Thời gian: {pump_time}s)")
            
            client.publish(TOPIC_PUMP, json.dumps({"cmd": "PUMP_ON", "vol_ml": predicted_ml, "pump_time_sec": pump_time}))
            is_pumping = True
        else:
            print(f"[*] ML DỰ ĐOÁN: Lượng nước yêu cầu quá ít ({predicted_ml}ml). Tạm hoãn tưới.")

if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, 1883, 60)
    client.loop_forever()