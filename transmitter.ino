#include <SPI.h>
#include <mcp2515.h>

#define CAN_bitrate CAN_125KBPS
#define CAN_ID 0x001

MCP2515 mcp2515(10); // CS на пине 10 (можно изменить)

struct can_frame canMsg;

void setup() {
  Serial.begin(115200);
  SPI.begin();
  
  mcp2515.reset();
  mcp2515.setBitrate(CAN_bitrate, MCP_8MHZ); // Настроки скорости
  mcp2515.setNormalMode();

  canMsg.can_id = CAN_ID; // ID сообщения (может быть от 0x000 до 0x7FF)
  canMsg.can_dlc = 2;    // Длина данных (2 байта)
}

int counter_error = 0;
int last_value = 0;

void loop() {
  int sensorValue = analogRead(A0); // Читаем аналоговый датчик (0-1023)
  if (last_value != sensorValue){
    last_value = sensorValue;

    // Разбиваем int на 2 байта
    canMsg.data[0] = sensorValue >> 8;   // Старший байт
    canMsg.data[1] = sensorValue & 0xFF; // Младший байт
    
    // Отправляем в CAN
    if (mcp2515.sendMessage(&canMsg)) {
      Serial.print("Sent: ");
      Serial.println(sensorValue);
    } else {
      Serial.println("CAN send error!");
      counter_error++;
    }
    if (counter_error >= 10){
      mcp2515.reset(); // Сброс модуля
      mcp2515.setBitrate(CAN_bitrate, MCP_8MHZ);
      Serial.println("CAN сброшен!");
      counter_error = 0;
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
    
    delay(1000); // Задержка между отправками
  }
}