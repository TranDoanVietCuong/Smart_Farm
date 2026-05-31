import paho.mqtt.client as mqtt
import json
import numpy as np
import requests
import time
from stable_baselines3 import PPO

# ================= CẤU HÌNH HỆ THỐNG =================
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
TOPIC_SENSORS = "lloydz/tomato/sensors"
TOPIC_CONTROL = "lloydz/tomato/control"

# [!] Cậu có thể thay API Key của cậu vào đây sau. 
# Nếu hiện tại để trống hoặc Key sai, hệ thống sẽ TỰ ĐỘNG dùng dữ liệu giả lập để code chạy mượt!
OWM_API_KEY = "96f2481b97c204c1e1a8abca014bfe5b" 
CITY = "Vinh Long,VN"
OWM_URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={OWM_API_KEY}&units=metric"

# Biến toàn cục lưu trữ dữ liệu API
current_rain_prob = 0.0
current_max_temp = 25.0
last_api_call = 0

print("🧠 Đang nạp mô hình AI V3 (PPO)...")
try:
    model = PPO.load("tomato_ppo_v3_model")
    print("✅ Nạp mô hình thành công!\n")
except Exception as e:
    print(f"❌ Không tìm thấy file model. Cậu nhớ chạy file train_rl_v3.py trước để tạo bộ não nhé! Lỗi: {e}")
    exit()

def fetch_weather_forecast():
    """Hàm lấy dữ liệu thời tiết thực tế, có cơ chế chống sập (Fallback)"""
    global current_rain_prob, current_max_temp, last_api_call
    now = time.time()
    
    # Chỉ gọi API 30 phút/lần (1800 giây)
    if now - last_api_call < 1800 and last_api_call != 0:
        return
        
    try:
        res = requests.get(OWM_URL, timeout=5)
        if res.status_code == 200:
            data = res.json()['list'][0]
            current_rain_prob = data.get('pop', 0.0)
            current_max_temp = data['main']['temp_max']
            last_api_call = now
            print(f"☁️ [API THỰC TẾ] Cập nhật thành công: Mưa {current_rain_prob*100}%, Nhiệt Max {current_max_temp}°C")
        else:
            # Nếu API Key sai (401), tự động giả lập Lâm Đồng sắp mưa to để test
            current_rain_prob = 0.8  
            current_max_temp = 22.0
            print(f"⚠️ [API FALLBACK] Chưa có API Key hợp lệ. Đang dùng dữ liệu giả lập: Mưa {current_rain_prob*100}%, Nhiệt Max {current_max_temp}°C")
    except Exception as e:
        current_rain_prob = 0.8
        current_max_temp = 22.0
        print(f"❌ [API LỖI] Mất mạng! Tự động dùng dữ liệu giả lập: Mưa {current_rain_prob*100}%, Nhiệt Max {current_max_temp}°C")

def on_connect(client, userdata, flags, rc):
    print(f"🔗 Đã kết nối MQTT Broker thành công (Mã: {rc})")
    client.subscribe(TOPIC_SENSORS)
    print(f"📡 Đang lắng nghe dữ liệu Wokwi tại: {TOPIC_SENSORS}\n")

def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8")
    print("\n" + "="*70)
    print(f"📥 [SENSORS WOKWI] Nhận dữ liệu thô: {payload}")
    
    # 1. Cập nhật dữ liệu thời tiết từ API
    fetch_weather_forecast()
    
    try:
        data = json.loads(payload)
        temp = data.get("temp", 24.0)
        hum = data.get("hum", 50.0)
        soil = data.get("soil", 50.0)
        light = data.get("light", 50.0)
        
        # 2. IN BẢNG ĐIỀU KHIỂN (DASHBOARD) TỔNG HỢP CHO LLOYDZ
        print(f"📊 [DASHBOARD] Môi trường: {temp}°C | Đất: {soil}% | Sáng: {light}%  ||  API: Mưa {current_rain_prob*100}% | Max: {current_max_temp}°C")
        print("-" * 70)
        
        # 3. Để AI phán đoán (Black Box - Não phải)
        obs = np.array([temp, hum, soil, light, current_rain_prob, current_max_temp], dtype=np.float32)
        action, _states = model.predict(obs, deterministic=True)
        rl_pump, rl_fan, rl_light = int(action[0]), int(action[1]), int(action[2])
        
        # ================================================================
        # 🛡️ KIẾN TRÚC HYBRID (GUARDRAILS) - KẾT HỢP LUẬT CỨNG VÀ AI
        # ================================================================
        print("🔍 [EXPLAINABLE AI - Giải thích lý do ra quyết định]")
        
        # ---> BƠM (Thiết quân luật bằng Guardrail)
        if current_rain_prob > 0.6:
            final_pump = 0 
            print(f"   💧 BƠM:  [Guardrail] -> ÉP TẮT (API báo mưa {current_rain_prob*100}%, cấm AI bật bơm, để thiên nhiên lo!)")
        else:
            final_pump = rl_pump
            if final_pump == 1:
                 print(f"   💧 BƠM:  [AI V3] ---> BẬT (Đất cần nước và an toàn không có mưa to)")
            else:
                 print(f"   🚫 BƠM:  [AI V3] ---> TẮT (Đất đã đủ ẩm)")

        # ---> QUẠT (Luật V2 chặn họng thói quen bật quạt của AI)
        if temp < 26:
            final_fan = 0 
            print(f"   🌬️ QUẠT: [Guardrail] -> ÉP TẮT (Trời đang mát {temp}°C, cấm AI lãng phí điện!)")
        else:
            final_fan = rl_fan
            print(f"   🌬️ QUẠT: [AI V3] ---> {'BẬT' if rl_fan else 'TẮT'} (Theo thuật toán làm mát)")

        # ---> ĐÈN (Luật V2 chặn họng thói quen bật đèn)
        if light >= 40:
            final_light = 0 
            print(f"   💡 ĐÈN:  [Guardrail] -> ÉP TẮT (Trời đang đủ sáng {light}%, cấm AI lãng phí điện!)")
        else:
            final_light = rl_light
            print(f"   💡 ĐÈN:  [AI V3] ---> {'BẬT' if rl_light else 'TẮT'} (Bật để bù sáng cho cây)")
        # ================================================================

        # 4. Gửi lệnh cuối cùng xuống phần cứng Wokwi
        control_msg = {"pump": final_pump, "fan": final_fan, "light": final_light}
        client.publish(TOPIC_CONTROL, json.dumps(control_msg))
        print(f"📤 [FINAL COMMAND] Đã gửi lệnh xuống ESP32: {control_msg}")
        print("="*70)

    except Exception as e:
        print(f"❌ Lỗi xử lý JSON: {e}")

# Khởi chạy Server
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("🚀 Đang khởi động Tomato Ultimate Hybrid Server...")
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()