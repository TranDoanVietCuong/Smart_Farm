#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ================= CẤU HÌNH WIFI & MQTT =================
const char* ssid = "Wokwi-GUEST"; // WiFi mặc định của Wokwi
const char* password = "";
const char* mqtt_server = "broker.emqx.io"; // Public broker để test

// Topic gửi dữ liệu cảm biến lên Server
const char* topic_publish = "lloydz/tomato/sensors"; 
// Topic nhận lệnh điều khiển từ AI Server
const char* topic_subscribe = "lloydz/tomato/control"; 

WiFiClient espClient;
PubSubClient client(espClient);

// ================= CẤU HÌNH CHÂN (PINS) =================
#define DHTPIN 15
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

#define SOIL_MOISTURE_PIN 34
#define LDR_PIN 35

#define PUMP_LED 25
#define FAN_LED 26
#define LIGHT_LED 27

// Variables for non-blocking timer
unsigned long lastMsg = 0;
const long interval = 5000; // Gửi dữ liệu mỗi 5 giây

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Đang kết nối WiFi: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi đã kết nối thành công!");
}

// Hàm callback xử lý khi nhận được lệnh từ AI (Python)
void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Nhận lệnh từ topic: ");
  Serial.println(topic);

  // Chuyển đổi payload thành chuỗi
  String messageTemp;
  for (int i = 0; i < length; i++) {
    messageTemp += (char)payload[i];
  }
  Serial.println("Nội dung: " + messageTemp);

  // Parse JSON nhận được
  StaticJsonDocument<200> doc;
  DeserializationError error = deserializeJson(doc, messageTemp);

  if (error) {
    Serial.print("Lỗi parse JSON: ");
    Serial.println(error.c_str());
    return;
  }

  // Đọc lệnh và điều khiển LED (1: Bật, 0: Tắt)
  if (doc.containsKey("pump")) {
    digitalWrite(PUMP_LED, doc["pump"] == 1 ? HIGH : LOW);
  }
  if (doc.containsKey("fan")) {
    digitalWrite(FAN_LED, doc["fan"] == 1 ? HIGH : LOW);
  }
  if (doc.containsKey("light")) {
    digitalWrite(LIGHT_LED, doc["light"] == 1 ? HIGH : LOW);
  }
}

void reconnect() {
  // Lặp lại cho đến khi kết nối lại MQTT thành công
  while (!client.connected()) {
    Serial.print("Đang thử kết nối MQTT Broker...");
    // Tạo client ID ngẫu nhiên
    String clientId = "ESP32Client-Lloydz-";
    clientId += String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println(" Đã kết nối MQTT!");
      // Đăng ký nhận lệnh điều khiển
      client.subscribe(topic_subscribe);
    } else {
      Serial.print(" Thất bại, rc=");
      Serial.print(client.state());
      Serial.println(" Thử lại sau 5 giây...");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  
  // Khởi tạo các chân Output
  pinMode(PUMP_LED, OUTPUT);
  pinMode(FAN_LED, OUTPUT);
  pinMode(LIGHT_LED, OUTPUT);
  
  // Tắt tất cả khi khởi động
  digitalWrite(PUMP_LED, LOW);
  digitalWrite(FAN_LED, LOW);
  digitalWrite(LIGHT_LED, LOW);

  dht.begin();
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  // Cần thiết để duy trì kết nối MQTT và nhận dữ liệu
  client.loop(); 

  unsigned long now = millis();
  if (now - lastMsg > interval) {
    lastMsg = now;

    // 1. Đọc dữ liệu cảm biến
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    
    // Đọc chiết áp (0-4095) và map về % (0-100%)
    int soil_raw = analogRead(SOIL_MOISTURE_PIN);
    int soil_percent = map(soil_raw, 0, 4095, 0, 100);
    
    // Đọc cảm biến ánh sáng (0-4095) và map về %
    int light_raw = analogRead(LDR_PIN);
    int light_percent = map(light_raw, 0, 4095, 0, 100);

    if (isnan(t) || isnan(h)) {
      Serial.println("Lỗi đọc cảm biến DHT22!");
      return;
    }

    // 2. Đóng gói dữ liệu thành JSON
    StaticJsonDocument<200> doc;
    doc["temp"] = t;
    doc["hum"] = h;
    doc["soil"] = soil_percent;
    doc["light"] = light_percent;

    char jsonBuffer[512];
    serializeJson(doc, jsonBuffer);

    // 3. Gửi lên MQTT Broker
    Serial.print("Publishing message: ");
    Serial.println(jsonBuffer);
    client.publish(topic_publish, jsonBuffer);
  }
}