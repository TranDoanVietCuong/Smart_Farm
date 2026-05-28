import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
import matplotlib.pyplot as plt
import textwrap

# --- CẤU HÌNH FONT CHỮ TIMES NEW ROMAN ---
plt.rcParams['font.family'] = 'Times New Roman'

# 1. Danh sách tên bệnh tiếng Anh
class_names = [
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
    'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 
    'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato_mosaic_virus', 'healthy', 'powdery_mildew'
]

# 2. CƠ SỞ DỮ LIỆU IOT
medical_dict = {
    'Bacterial_spot': {
        'ten_tv': 'Đốm Vi Khuẩn',
        'nguyen_nhan': 'Vi khuẩn Xanthomonas bùng phát mạnh ở nhiệt độ 24-30°C và độ ẩm cao.',
        'phong_tranh': 'Luân canh cây trồng. Tuyệt đối không tưới phun mưa từ trên ngọn xuống làm ướt mặt lá.',
        'chua_tri': 'Cắt bỏ ngay lá bệnh. Phun thuốc bảo vệ thực vật gốc Đồng (Copper) kết hợp Mancozeb.',
        'cai_dat_iot': 'Tắt béc tưới phun sương. Chuyển sang tưới nhỏ giọt gốc. Bật quạt thông gió giảm độ ẩm < 70%.'
    },
    'Early_blight': {
        'ten_tv': 'Đốm Vòng (Sương Mai Sớm)',
        'nguyen_nhan': 'Nấm Alternaria tấn công khi thời tiết nóng ẩm.',
        'phong_tranh': 'Trồng thưa đón nắng. Phủ bạt nông nghiệp dưới gốc ngăn nấm từ đất bắn lên.',
        'chua_tri': 'Tỉa bỏ lá đốm đen vòng tròn. Phun thuốc chứa Mancozeb hoặc chế phẩm sinh học Bacillus subtilis.',
        'cai_dat_iot': 'Giảm tần suất tưới. Giữ độ ẩm đất ở mức 50-60%. Tăng tốc độ quạt lưu thông gió.'
    },
    'Late_blight': {
        'ten_tv': 'Mốc Sương (Bại Lụi)',
        'nguyen_nhan': 'Nấm Phytophthora. Lây lan cực nhanh khi trời lạnh và có sương mù.',
        'phong_tranh': 'Lên luống cao, thoát nước tốt. Thăm vườn liên tục khi thời tiết chuyển lạnh.',
        'chua_tri': 'Tiêu hủy cây bệnh nặng. Phun khẩn cấp thuốc đặc trị nấm nội hấp như Ridomil Gold.',
        'cai_dat_iot': 'BÁO ĐỘNG! Mở rèm đón nắng. Bật đèn sưởi nếu nhiệt độ < 20°C. Ngắt tạo ẩm.'
    },
    'Leaf_Mold': {
        'ten_tv': 'Nấm Mốc Lá',
        'nguyen_nhan': 'Nấm Passalora fulva phát triển trong môi trường độ ẩm không khí cực cao (>85%).',
        'phong_tranh': 'Tỉa bớt lá già phía dưới. Tưới nước vào sáng sớm để mặt lá khô trước khi trời tối.',
        'chua_tri': 'Phun thuốc chứa Chlorothalonil hoặc dùng nấm đối kháng Trichoderma tưới gốc.',
        'cai_dat_iot': 'Kích hoạt quạt hút ẩm chạy 100% công suất. Đưa độ ẩm không khí xuống < 75%.'
    },
    'Septoria_leaf_spot': {
        'ten_tv': 'Đốm Lá Septoria',
        'nguyen_nhan': 'Nấm Septoria ủ bệnh trong cỏ dại và lây lan qua giọt nước mưa bắn lên.',
        'phong_tranh': 'Làm sạch cỏ dại. Không cắt tỉa khi lá cây đang ướt sũng.',
        'chua_tri': 'Cắt bỏ lá gốc có đốm nhỏ. Phun định kỳ thuốc trị nấm Mancozeb.',
        'cai_dat_iot': 'Ngắt hệ thống tưới mặt. Kích hoạt tưới nhỏ giọt. Làm khô bề mặt luống.'
    },
    'Spider_mites Two-spotted_spider_mite': {
        'ten_tv': 'Nhện Đỏ',
        'nguyen_nhan': 'Nhện đỏ chích hút nhựa lá. Sinh sôi cực nhanh vào mùa KHÔ HANH, nắng nóng.',
        'phong_tranh': 'Duy trì độ ẩm, nuôi thiên địch (bọ rùa).',
        'chua_tri': 'Phun dầu Neem hoặc dung dịch tỏi ớt. Nếu nặng dùng thuốc Abamectin.',
        'cai_dat_iot': 'Bật béc tưới phun sương làm ướt mặt lá (nhện đỏ sợ nước). Giảm nhiệt độ nhà màng.'
    },
    'Target_Spot': {
        'ten_tv': 'Đốm Vòng Mục Tiêu',
        'nguyen_nhan': 'Nấm Corynespora xâm nhập qua vết xước cơ học trên lá, quả.',
        'phong_tranh': 'Tránh làm xước thân lá. Bổ sung Kali và Canxi để cây cứng cáp.',
        'chua_tri': 'Phun luân phiên thuốc diệt nấm chứa Azoxystrobin, Mancozeb.',
        'cai_dat_iot': 'Duy trì môi trường ổn định: Nhiệt độ 22-26°C, độ ẩm 65-75%.'
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'ten_tv': 'Virus Xoăn Vàng Lá',
        'nguyen_nhan': 'Virus TYLCV lây truyền duy nhất qua côn trùng chích hút (Bọ phấn trắng).',
        'phong_tranh': 'Treo bẫy dính màu vàng bắt côn trùng. Dọn sạch cỏ dại.',
        'chua_tri': 'VÔ PHƯƠNG CỨU CHỮA. Nhổ bỏ tiêu hủy. Phun thuốc trừ bọ phấn để bảo vệ cây khác.',
        'cai_dat_iot': 'Đóng kín rèm che côn trùng. Bật đèn bẫy bọ phấn vào ban đêm.'
    },
    'Tomato_mosaic_virus': {
        'ten_tv': 'Virus Khảm Lá',
        'nguyen_nhan': 'Virus ToMV lây qua tiếp xúc cơ học (Tay người, dụng cụ cắt tỉa, hạt giống).',
        'phong_tranh': 'Sát khuẩn dụng cụ bằng cồn. Không hút thuốc lá khi làm vườn.',
        'chua_tri': 'VÔ PHƯƠNG CỨU CHỮA. Nhổ bỏ đem đốt. Khử trùng đất.',
        'cai_dat_iot': 'Cảnh báo đỏ! Cô lập khu vực tưới của luống nhiễm bệnh để tránh lây theo nguồn nước.'
    },
    'powdery_mildew': {
        'ten_tv': 'Nấm Phấn Trắng',
        'nguyen_nhan': 'Nấm phấn trắng bùng phát khi ngày khô nóng, đêm lạnh có sương (chênh lệch nhiệt độ lớn).',
        'phong_tranh': 'Bón phân cân đối, tuyệt đối không bón thừa đạm (Nitơ).',
        'chua_tri': 'Phun Baking soda pha nước rửa chén, hoặc thuốc chứa Lưu huỳnh (Sulfur).',
        'cai_dat_iot': 'Bật sưởi/thông gió để giảm chênh lệch nhiệt độ Ngày - Đêm. Không để sương đọng.'
    },
    'healthy': {
        'ten_tv': 'Cây Khỏe Mạnh',
        'nguyen_nhan': 'Điều kiện sinh thái cân bằng, cây đang phát triển tốt nhất!',
        'phong_tranh': 'Bón phân định kỳ, bổ sung vi lượng.',
        'chua_tri': 'Không cần can thiệp. Chuẩn bị thu hoạch!',
        'cai_dat_iot': 'Duy trì: Nhiệt độ 20-28°C, Ẩm độ 60-70%, Chiếu sáng 8-10h/ngày.'
    }
}

# 3. Load mô hình
print("Đang gọi bác sĩ AI & Hệ thống IoT thức dậy...")
model = tf.keras.models.load_model(
    'd:/SMART_FARM/computer_vision/tomato_disease_mobilenet_v2.keras',
    custom_objects={'preprocess_input': applications.mobilenet_v2.preprocess_input},
    safe_mode=False
)

def diagnose_leaf(img_path):
    print(f"Đang phân tích hình ảnh: {img_path}...")
    
    # Chuẩn bị ảnh
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Dự đoán
    predictions = model.predict(img_array)
    score = predictions[0]

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    info = medical_dict[predicted_class]

    # --- IN RA TERMINAL ---
    print("\n" + "=".center(70, "="))
    print(f" TÊN BỆNH : {predicted_class} ({info['ten_tv'].upper()}) ")
    print(f" TIN CẬY  : {confidence:.2f}% ")
    print("=".center(70, "=") + "\n")

    # --- XUẤT RA GIAO DIỆN ẢNH (DASHBOARD) ---
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.2]})
    fig.canvas.manager.set_window_title('SMART FARM - AI DIAGNOSIS')
    
    # Cột 1: Hiển thị ảnh
    ax_img.imshow(img)
    ax_img.set_title(f"{info['ten_tv'].upper()}\nĐộ tin cậy: {confidence:.2f}%", fontsize=16, fontweight='bold', color='darkred')
    ax_img.axis('off')

    # Cột 2: Hiển thị văn bản báo cáo
    ax_text.axis('off') 
    
    def wrap(text, width=60):
        return "\n".join(textwrap.wrap(text, width=width))

    # Đã thay toàn bộ icon bằng dấu +)
    report = f"MÃ BỆNH: {predicted_class}\n\n"
    report += f"+) NGUYÊN NHÂN:\n{wrap(info['nguyen_nhan'])}\n\n"
    
    if predicted_class == 'healthy':
        report += f"+) TÌNH TRẠNG:\n{wrap(info['chua_tri'])}\n\n"
    else:
        report += f"+) ĐIỀU TRỊ:\n{wrap(info['chua_tri'])}\n\n"
        
    report += f"+) PHÒNG TRÁNH:\n{wrap(info['phong_tranh'])}\n\n"
    report += f"+) KHUYẾN NGHỊ:\n{wrap(info['cai_dat_iot'])}"

    # In chữ lên trục ax_text
    ax_text.text(0.05, 0.95, report, transform=ax_text.transAxes, fontsize=13, 
                 verticalalignment='top', linespacing=1.6,
                 bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.9))

    plt.tight_layout()
    plt.show()

# --- CHẠY THỬ THỰC TẾ ---
anh_can_test = 'd:/SMART_FARM/computer_vision/la4.jpg' 
diagnose_leaf(anh_can_test)
