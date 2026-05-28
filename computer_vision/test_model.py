import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras import applications

# 1. Danh sách các bệnh (Bắt buộc phải xếp đúng thứ tự chữ cái như lúc Train)
class_names = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
    'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 
    'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato_mosaic_virus', 'healthy', 'powdery_mildew'
]

# 2. Đánh thức mô hình AI đã train
print("🤖 Đang gọi bác sĩ AI thức dậy...")
# Lưu ý: Đảm bảo file .keras nằm cùng thư mục với file code này
model = tf.keras.models.load_model(
    'd:/SMART_FARM/computer_vision/tomato_disease_mobilenet_v2.keras',
    custom_objects={'preprocess_input': applications.mobilenet_v2.preprocess_input},
    safe_mode=False
)
def diagnose_leaf(img_path):
    print(f"🔍 Đang phân tích hình ảnh: {img_path}...")
    
    # 3. Chuẩn bị ảnh (Phải đưa về đúng chuẩn 224x224 của MobileNetV2)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Thêm chiều batch: [1, 224, 224, 3]

    # 4. Đưa vào AI dự đoán
    predictions = model.predict(img_array)
    score = predictions[0] # Mạng nơ-ron xuất ra trực tiếp tỷ lệ phần trăm (do có lớp Softmax)

    # 5. Trích xuất kết quả cao nhất
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print("\n" + "="*40)
    print("🩺 KẾT QUẢ CHẨN ĐOÁN CỦA HỆ THỐNG:")
    print(f"🌿 Tên bệnh: {predicted_class}")
    print(f"🎯 Độ tin cậy: {confidence:.2f}%")
    print("="*40 + "\n")

    # 6. Bật cửa sổ hiển thị ảnh
    plt.imshow(img)
    plt.title(f"{predicted_class} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

# --- CHẠY THỬ THỰC TẾ ---
# Đường dẫn tới bức ảnh bạn vừa tải về
anh_can_test = 'D:\SMART_FARM\computer_vision\la_1.webp' 
diagnose_leaf(anh_can_test)