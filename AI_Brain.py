import paho.mqtt.client as mqtt
import json
import time
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ================= CẤU HÌNH HỆ THỐNG =================
MQTT_BROKER = "broker.hivemq.com"
TOPIC_SENSOR = "agrinova/sensors"
TOPIC_CMD = "agrinova/cmd"
OWM_API_KEY = "96f2481b97c204c1e1a8abca014bfe5b" 
CITY_NAME = "Dalat"

POT_CAPACITY_ML = 2500       
PUMP_FLOW_RATE_ML_SEC = 40   
TARGET_MOISTURE = 65.0       
is_pumping = False
forecast_cache = {"data": None, "last_update": 0}

# ================= 1. HỌC MÁY (ML REGRESSION) =================
def train_agrinova_model():
    print("[AI] Đang huấn luyện Mô hình ML phân tích môi trường đa biến...")
    X, y = [], []
    for _ in range(2000):
        c_moist = np.random.uniform(10, 80)
        c_temp = np.random.uniform(20, 42)
        c_air_hum = np.random.uniform(40, 95)
        f_temp = c_temp + np.random.uniform(-3, 6) # Dự báo nhiệt độ
        f_rain = np.random.choice([0, 1], p=[0.8, 0.2])
        
        water_ml = (TARGET_MOISTURE - c_moist) / 100.0 * POT_CAPACITY_ML
        
        # Rule học máy: Nồm ẩm / Mưa -> Giảm nước về 0
        if c_air_hum > 85 or f_rain == 1: 
            water_ml = 0 
        # Rule học máy: Nắng gắt -> Tăng mạnh lượng nước
        elif f_temp >= 36:
            water_ml += 800 
            
        if water_ml < 0: water_ml = 0
        X.append([c_moist, c_temp, c_air_hum, f_temp, f_rain])
        y.append(water_ml)
        
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    print("[AI] Sẵn sàng vận hành kịch bản Predictive Irrigation!\n" + "="*50)
    return model

ai_model = train_agrinova_model()

# ================= 2. TÍCH HỢP API THỜI TIẾT 12 GIỜ TỚI =================
def get_weather_forecast():
    current_time = time.time()
    if current_time - forecast_cache["last_update"] < 300 and forecast_cache["data"]:
        return forecast_cache["data"]
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY_NAME}&appid={OWM_API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        if res.get("cod") != "200": return {"max_temp": 25, "will_rain": 0}

        forecast_list = res['list'][:4] # Lấy 4 mốc (12 giờ tới)
        max_temp_12h = max(item['main']['temp'] for item in forecast_list)
        will_rain = 1 if any(item['weather'][0]['id'] < 600 for item in forecast_list) else 0
        
        result = {"max_temp": max_temp_12h, "will_rain": will_rain}
        forecast_cache["data"] = result
        forecast_cache["last_update"] = current_time
        return result
    except:
        return {"max_temp": 25, "will_rain": 0}

# ================= 3. ĐÁNH GIÁ & RA QUYẾT ĐỊNH KỊCH BẢN =================
def on_connect(client, userdata, flags, reason_code, properties):
    print(">>> KẾT NỐI BROKER THÀNH CÔNG <<<")
    client.subscribe(TOPIC_SENSOR)

def on_message(client, userdata, msg):
    global is_pumping
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        c_moist = payload.get("do_am_dat", 100)
        c_temp = payload.get("nhiet_do", 25)
        c_air_hum = payload.get("do_am_kk", 60)
        c_light = payload.get("anh_sang", 0)
        
        # Vòng kín bảo vệ ngập úng
        if is_pumping:
            if c_moist >= TARGET_MOISTURE:
                print("[!] VÒNG KÍN: Đạt độ ẩm, ra lệnh tắt bơm.")
                client.publish(TOPIC_CMD, json.dumps({"cmd": "PUMP_OFF", "net_cmd": "KEEP"}))
                is_pumping = False
            return

        if c_moist < 60:
            forecast = get_weather_forecast()
            f_temp = forecast["max_temp"]
            f_rain = forecast["will_rain"]
            
            print(f"\n[Dữ liệu] Đất:{c_moist}% | Ẩm KK:{c_air_hum}% | Nhiệt:{c_temp}°C | Sáng:{c_light}")
            print(f"[Dự báo 12h] Max Temp: {f_temp}°C | Mưa: {'CÓ' if f_rain else 'KHÔNG'}")
            
            net_command = "OPEN" 
            scenario = "Bình thường"

            # ĐÁNH GIÁ KỊCH BẢN THỰC TẾ
            if f_temp >= 36:
                scenario = "Nắng gắt cục bộ (Dự báo > 36°C) -> Chủ động che lưới!"
                net_command = "CLOSE"
            elif c_air_hum > 85 or f_rain == 1:
                scenario = "Nồm ẩm / Mưa lớn (Ẩm KK > 85%) -> Ngắt tưới chống nấm!"
                net_command = "OPEN"
            
            # Machine Learning dự đoán lượng nước
            features = np.array([[c_moist, c_temp, c_air_hum, f_temp, f_rain]])
            predicted_ml = int(ai_model.predict(features)[0])
            
            print(f"[*] Phân tích Kịch bản: {scenario}")
            
            if predicted_ml > 50:
                pump_time = int(predicted_ml / PUMP_FLOW_RATE_ML_SEC)
                print(f"[>] HÀNH ĐỘNG: BẬT BƠM {pump_time}s | KÉO LƯỚI: {net_command}")
                client.publish(TOPIC_CMD, json.dumps({"cmd": "PUMP_ON", "vol_ml": predicted_ml, "pump_time_sec": pump_time, "net_cmd": net_command}))
                is_pumping = True
            else:
                print(f"[>] HÀNH ĐỘNG: KHÔNG TƯỚI | KÉO LƯỚI: {net_command}")
                client.publish(TOPIC_CMD, json.dumps({"cmd": "PUMP_OFF", "net_cmd": net_command}))

    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, 1883, 60)
    client.loop_forever()
