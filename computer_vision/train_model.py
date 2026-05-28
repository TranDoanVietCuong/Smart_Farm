import tensorflow as tf
from tensorflow.keras import layers, models, applications

# --- 1. CẤU HÌNH ĐƯỜNG DẪN TRỰC TIẾP ---
IMG_SIZE = (224, 224) # Kích thước chuẩn bắt buộc của MobileNetV2
BATCH_SIZE = 32

# Điền đường dẫn trỏ thẳng tới 2 thư mục tương ứng của bạn
TRAIN_DIR = 'D:/DOWLOAD/train'
VALID_DIR = 'D:/DOWLOAD/valid'

# --- 2. TỰ ĐỘNG LOAD DATA TỪ HAI THƯ MỤC RIÊNG BIỆT ---
print("📂 Đang tải tập dữ liệu huấn luyện...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("📂 Đang tải tập dữ liệu kiểm định...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Lấy danh sách tên các loại bệnh (tương ứng với tên thư mục con)
class_names = train_ds.class_names
print("🌿 Các loại danh mục AI sẽ học nhận diện:", class_names)

# Tối ưu hóa bộ nhớ đệm để tăng tốc độ nạp ảnh từ ổ đĩa vào GPU
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. DATA AUGMENTATION (TĂNG CƯỜNG DỮ LIỆU CHỐNG HỌC VẸT) ---
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"), # Lật ảnh ngẫu nhiên
  layers.RandomRotation(0.2),                   # Xoay ảnh ngẫu nhiên góc nhỏ
  layers.RandomZoom(0.2),                       # Phóng to/thu nhỏ ngẫu nhiên
])

# --- 4. TẢI MÔ HÌNH XƯƠNG SỐNG MOBILENETV2 (TRANSFER LEARNING) ---
base_model = applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,          # Bỏ lớp phân loại cũ của ImageNet đi
    weights='imagenet'          # Sử dụng bộ trọng số đã học từ hàng triệu ảnh
)

# Đóng băng toàn bộ các lớp của mô hình gốc để giữ nguyên tri thức nền tảng
base_model.trainable = False 

# --- 5. XÂY DỰNG MẠNG NƠ-RON HOÀN CHỈNH ---
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,                                         # Bước 1: Biến đổi ảnh đa dạng
    layers.Lambda(applications.mobilenet_v2.preprocess_input), # Bước 2: Chuẩn hóa pixel theo chuẩn MobileNet
    base_model,                                                # Bước 3: Trích xuất đặc trưng hình ảnh
    layers.GlobalAveragePooling2D(),                           # Bước 4: Làm phẳng ma trận đặc trưng
    layers.Dropout(0.2),                                       # Bước 5: Ẩn ngẫu nhiên 20% nơ-ron để chống học vẹt
    layers.Dense(len(class_names), activation='softmax')       # Bước 6: Đưa ra xác suất của từng loại bệnh
])

# --- 6. BIÊN DỊCH MÔ HÌNH ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 7. TIẾN HÀNH TRAINING ---
epochs = 10 # Với MobileNetV2, chỉ cần khoảng 10 vòng lặp là độ chính xác đã cực kỳ cao
print("🚀 Hệ thống bắt đầu quá trình huấn luyện chuyển đổi tri thức...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# --- 8. LƯU KẾT QUẢ ---
model.save('tomato_disease_mobilenet_v2.keras')
print("✅ Quá trình huấn luyện hoàn tất! File model 'tomato_disease_mobilenet_v2.keras' đã sẵn sàng để dự đoán.")