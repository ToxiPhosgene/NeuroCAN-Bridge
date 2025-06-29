#define LEDpin 4

#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "ESP32";  
const char* password = "12345678";

IPAddress local_ip(192,168,1,1);
IPAddress gateway(192,168,1,1);
IPAddress subnet(255,255,255,0);

WebServer server(80);

bool LEDstatus = LOW;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(LEDpin, OUTPUT);
  digitalWrite(LEDpin, LOW);
  delay(1000);
  digitalWrite(LEDpin, HIGH);
  delay(1000);
  digitalWrite(LEDpin, LOW);

  WiFi.softAP(ssid, password);
  WiFi.softAPConfig(local_ip, gateway, subnet);
  delay(100);

  server.on("/", handle_OnConnect);
  server.on("/ledon", handle_ledon);
  server.on("/ledoff", handle_ledoff);
  server.onNotFound(handle_NotFound);

  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  // put your main code here, to run repeatedly:
   server.handleClient();
  if(LEDstatus)
  {digitalWrite(LEDpin, HIGH);}
  else
  {digitalWrite(LEDpin, LOW);}
}

void handle_OnConnect() {
  LEDstatus = LOW;
  Serial.println("GPIO4 Status: OFF");
  server.send(200, "text/html", "OK"); 
}

void handle_ledon() {
  LEDstatus = HIGH;
  Serial.println("GPIO4 Status: ON");
  server.send(200, "text/html", "OK"); 
}

void handle_ledoff() {
  LEDstatus = LOW;
  Serial.println("GPIO4 Status: OFF");
  server.send(200, "text/html", "OK"); 
}

void handle_NotFound(){
  server.send(404, "text/plain", "Not found");
}
