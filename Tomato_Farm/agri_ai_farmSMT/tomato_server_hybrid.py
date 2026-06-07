import numpy as np
import requests
import time
import os
from stable_baselines3 import PPO

# ================= CẤU HÌNH HỆ THỐNG =================
OWM_API_KEY = "96f2481b97c204c1e1a8abca014bfe5b" 
CITY = "Ha Noi,VN"
OWM_URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={OWM_API_KEY}&units=metric"

current_rain_prob = 0.0
current_max_temp = 25.0
last_api_call = 0

# Tải Model PPO (Sửa lại đường dẫn tương đối để server.py gọi được)
MODEL_PATH = "agri_ai_farmSMT/tomato_ppo_v3_model"
try:
    if os.path.exists(MODEL_PATH + ".zip"):
        model = PPO.load(MODEL_PATH)
        print("✅ [FarmSMT] Đã nạp thành công bộ não PPO V3!")
    else:
        model = None
        print(f"⚠️ [FarmSMT] Chưa tìm thấy {MODEL_PATH}.zip")
except Exception as e:
    print(f"❌ [FarmSMT] Lỗi nạp model: {e}")
    model = None

def fetch_weather_forecast():
    """Lấy dữ liệu thời tiết thực tế, cache trong 30 phút"""
    global current_rain_prob, current_max_temp, last_api_call
    now = time.time()
    
    if now - last_api_call < 1800 and last_api_call != 0:
        return
        
    try:
        res = requests.get(OWM_URL, timeout=5)
        if res.status_code == 200:
            data = res.json()['list'][0]
            current_rain_prob = data.get('pop', 0.0)
            current_max_temp = data['main']['temp_max']
            last_api_call = now
        else:
            current_rain_prob = 0.8  
            current_max_temp = 22.0
    except Exception:
        current_rain_prob = 0.8
        current_max_temp = 22.0

def ppo_hybrid_decision(sensor_data: dict) -> dict:
    """Hàm xuất khẩu quyết định AI cho server.py gọi"""
    fetch_weather_forecast()
    
    temp = sensor_data.get("nhiet_do", 24.0)
    hum = sensor_data.get("do_am_kk", 50.0)
    soil = sensor_data.get("do_am_dat", 50.0)
    
    # Ép kiểu ánh sáng từ chuỗi "50%" về số float 50.0
    light_str = str(sensor_data.get("anh_sang", "0")).replace("%", "")
    try:
        light = float(light_str)
    except:
        light = 0.0

    if model is None:
        return {"scenario": "⚠️ Lỗi: Không load được PPO Model."}

    # AI Phán đoán
    obs = np.array([temp, hum, soil, light, current_rain_prob, current_max_temp], dtype=np.float32)
    action, _states = model.predict(obs, deterministic=True)
    rl_pump, rl_fan, rl_light = int(action[0]), int(action[1]), int(action[2])
    
    scenario = f"API Thời tiết (Mưa {current_rain_prob*100}%, Nhiệt Max {current_max_temp}°C) -> "
    
    # 🛡️ KIẾN TRÚC GUARDRAILS
    if current_rain_prob > 0.6:
        final_pump = 0 
        scenario += "⛔ Cấm Bơm (Sắp mưa). "
    else:
        final_pump = rl_pump
        scenario += f"{'💧 Bật Bơm. ' if final_pump else '🚫 Tắt Bơm. '}"

    if temp < 24:
        final_fan = 0 
        scenario += "⛔ Cấm Quạt (Mát). "
    else:
        final_fan = rl_fan
        scenario += f"{'🌬️ Bật Quạt. ' if final_fan else '🚫 Tắt Quạt. '}"

    if light >= 40:
        final_light = 0 
        scenario += "⛔ Cấm Đèn (Đủ sáng). "
    else:
        final_light = rl_light
        scenario += f"{'💡 Bật Đèn.' if final_light else '🚫 Tắt Đèn.'}"

    return {
        "pump": final_pump,
        "fan": final_fan,
        "light": final_light,
        "scenario": scenario
    }