#include <WiFi.h>
#include <HTTPClient.h>  
#include <mcp2515.h>

#define CAN_bitrate CAN_125KBPS
#define CAN_ID 0x001

// Настройки Wi-Fi
const char* ssid = "pentagon";
const char* password = "_omnyssy";

// URL сервера
const char* serverUrl = "http://192.168.43.212:8080/api/data"; // Или IP: http://192.168.1.100:5000/data

// Настройки CAN
MCP2515 mcp2515(5);  // CS на GPIO5
struct can_frame canRxMsg;

void setup() {
  Serial.begin(115200);

  // Инициализация CAN
  mcp2515.reset();
  mcp2515.setBitrate(CAN_bitrate, MCP_8MHZ);
  mcp2515.setNormalMode();
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(WiFi.status());
  }

  Serial.println("");
  Serial.println("WiFi connected..!");
  Serial.print("Got IP: ");  
  Serial.println(WiFi.localIP());
}

void loop() {
  // Чтение CAN-сообщения (с проверкой)
  if (mcp2515.readMessage(&canRxMsg) == MCP2515::ERROR_OK) {
    if (canRxMsg.can_id == CAN_ID) {
      int sensorValue = (canRxMsg.data[0] << 8) | canRxMsg.data[1];
      Serial.print("CAN данные: ");
      Serial.println(sensorValue);
    
      // Отправка на сервер (только если Wi-Fi подключен)
      sendToServer(sensorValue);
    }
  }
  
  uint8_t err = mcp2515.getErrorFlags();
  if (err) {
    Serial.print("CAN ошибка: 0x");
    Serial.println(err, HEX);
    
    if (err & 0x40) { // Если флаг переполнения
      mcp2515.clearRXnOVRFlags(); // Сброс
      Serial.println("Сброс буфера CAN!");
    }
    if (err & 0x15) { // Если 
      mcp2515.reset();
      mcp2515.setBitrate(CAN_bitrate, MCP_8MHZ);
    }
  }

  delay(100);
}

void sendToServer(int value) {
  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");

  String payload = "{\"value\":" + String(value) + "}";
  int httpCode = http.POST(payload);

  if (httpCode == HTTP_CODE_OK) {
    Serial.println("Данные отправлены!");
  } else {
    Serial.printf("Ошибка HTTP: %d\n", httpCode);
  }
  http.end();
}
