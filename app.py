import paho.mqtt.client as mqtt
import json
import time
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

# ================= 1. KHỞI TẠO WEB SERVER =================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ================= 2. CẤU HÌNH HỆ THỐNG =================
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

# ================= 3. LOGIC TRÍ TUỆ NHÂN TẠO =================
def train_agrinova_model():
    print("[AI] Đang huấn luyện Mô hình ML...")
    X, y = [], []
    for _ in range(2000):
        c_moist = np.random.uniform(10, 80)
        c_temp = np.random.uniform(20, 42)
        c_air_hum = np.random.uniform(40, 95)
        f_temp = c_temp + np.random.uniform(-3, 6)
        f_rain = np.random.choice([0, 1], p=[0.8, 0.2])
        
        water_ml = (TARGET_MOISTURE - c_moist) / 100.0 * POT_CAPACITY_ML
        if c_air_hum > 85 or f_rain == 1: water_ml = 0 
        elif f_temp >= 36: water_ml += 800 
            
        if water_ml < 0: water_ml = 0
        X.append([c_moist, c_temp, c_air_hum, f_temp, f_rain])
        y.append(water_ml)
        
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    print("[AI] Huấn luyện xong. Sẵn sàng vận hành!")
    return model

ai_model = train_agrinova_model()

def get_weather_forecast():
    current_time = time.time()
    if current_time - forecast_cache["last_update"] < 300 and forecast_cache["data"]:
        return forecast_cache["data"]
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY_NAME}&appid={OWM_API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        if res.get("cod") != "200": return {"max_temp": 25, "will_rain": 0}

        forecast_list = res['list'][:4]
        max_temp_12h = max(item['main']['temp'] for item in forecast_list)
        will_rain = 1 if any(item['weather'][0]['id'] < 600 for item in forecast_list) else 0
        
        result = {"max_temp": max_temp_12h, "will_rain": will_rain}
        forecast_cache["data"] = result
        forecast_cache["last_update"] = current_time
        return result
    except:
        return {"max_temp": 25, "will_rain": 0}

# ================= 4. KẾT NỐI MQTT =================
def on_connect(client, userdata, flags, reason_code, properties):
    print(">>> KẾT NỐI MQTT BROKER THÀNH CÔNG <<<")
    client.subscribe(TOPIC_SENSOR)

def on_message(client, userdata, msg):
    global is_pumping
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        c_moist = payload.get("do_am_dat", 100)
        c_temp = payload.get("nhiet_do", 25)
        c_air_hum = payload.get("do_am_kk", 60)
        c_light = payload.get("anh_sang", 0)
        
        scenario = "Bình thường"
        predicted_ml = 0
        net_command = "OPEN"

        # Vòng kín bảo vệ ngập úng
        if is_pumping:
            if c_moist >= TARGET_MOISTURE:
                client.publish(TOPIC_CMD, json.dumps({"cmd": "PUMP_OFF", "net_cmd": "KEEP"}))
                is_pumping = False
        elif c_moist < 60:
            forecast = get_weather_forecast()
            f_temp = forecast["max_temp"]
            f_rain = forecast["will_rain"]
            
            if f_temp >= 36:
                scenario = "Nắng gắt cục bộ -> Chủ động che lưới!"
                net_command = "CLOSE"
            elif c_air_hum > 85 or f_rain == 1:
                scenario = "Nồm ẩm / Mưa lớn -> Ngắt tưới chống nấm!"
            
            features = np.array([[c_moist, c_temp, c_air_hum, f_temp, f_rain]])
            predicted_ml = int(ai_model.predict(features)[0])
            
            if predicted_ml > 50:
                pump_time = int(predicted_ml / PUMP_FLOW_RATE_ML_SEC)
                client.publish(TOPIC_CMD, json.dumps({"cmd": "PUMP_ON", "vol_ml": predicted_ml, "pump_time_sec": pump_time, "net_cmd": net_command}))
                is_pumping = True
            else:
                client.publish(TOPIC_CMD, json.dumps({"cmd": "PUMP_OFF", "net_cmd": net_command}))

        # Đẩy dữ liệu thời gian thực lên Website (Dashboard)
        socketio.emit('update_dashboard', {
            'sensors': payload,
            'ai_status': {
                'scenario': scenario,
                'predicted_water': predicted_ml,
                'pump_active': is_pumping
            }
        })
    except Exception as e:
        print(f"Lỗi: {e}")

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, 1883, 60)
mqtt_client.loop_start()

# ================= 5. ROUTER & API CỦA WEB =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/control', methods=['POST'])
def manual_control():
    data = request.json
    action = data.get('action')
    
    if action == "PUMP_ON":
        mqtt_client.publish(TOPIC_CMD, json.dumps({"cmd": "PUMP_ON", "vol_ml": 500, "pump_time_sec": 10, "net_cmd": "KEEP"}))
    elif action == "PUMP_OFF":
        mqtt_client.publish(TOPIC_CMD, json.dumps({"cmd": "PUMP_OFF", "net_cmd": "KEEP"}))
    elif action == "NET_OPEN":
        mqtt_client.publish(TOPIC_CMD, json.dumps({"cmd": "KEEP", "net_cmd": "OPEN"}))
    elif action == "NET_CLOSE":
        mqtt_client.publish(TOPIC_CMD, json.dumps({"cmd": "KEEP", "net_cmd": "CLOSE"}))
        
    return jsonify({"status": "success", "message": f"Đã gửi lệnh {action}"})

if __name__ == '__main__':
    print("[Web] Đang khởi động Server tại http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)