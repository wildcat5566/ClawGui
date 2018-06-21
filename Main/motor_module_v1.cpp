#include "motor_module_v1.h"

void Motor_Init(int ExtReset){
  pinMode(ExtReset, OUTPUT);
  digitalWrite(ExtReset, LOW);
  delay(10);
  digitalWrite(ExtReset, HIGH);
}

MotorObject::MotorObject(){
  rev = false;
}

void MotorObject::getMotorState(int* count, bool* collide){
  Wire.requestFrom(address, 6);
  uint8_t data[6];
  while (Wire.available()) {
    for(int i = 0; i < 6; i ++){
      data[i] = Wire.read();
    }
  }
  *collide = data[0];
  *count = hex2dec(data);
  delay(1);
}

void MotorObject::getMotorState(int* count){
  Wire.requestFrom(address, 6);
  uint8_t data[6];
  while (Wire.available()) {
    for(int i = 0; i < 6; i ++){
      data[i] = Wire.read();
    }
  }
  *count = hex2dec(data);
  delay(1);
}

void MotorObject::sendPwm(int pwm){
    Wire.beginTransmission(address);
    if(rev == true){
      pwm = pwm * (-1);
    }
    uint8_t* hex;
    hex = dec2hex(pwm);
    for(int i = 0; i < 3; i ++){
      Wire.write(hex[i]);
    }
    Wire.endTransmission();
    delay(1);
}
void MotorObject::setSlaveAddress(uint8_t addr){
  address = addr;
}

void MotorObject::Reverse(){
  rev = true;
}

int MotorObject::hex2dec(uint8_t *hex){
  int dec;
  dec = 16777216*hex[2] + 65536*hex[3] + 256*hex[4] + hex[5];
  if(hex[1]!=rev){
    dec = dec*(-1);
  }
  return dec;
}

uint8_t* MotorObject::dec2hex(int dec){
  static uint8_t hex[3];
  if(dec < 0){
    dec = dec * (-1);
    hex[0] = 1;
  }
  else{
    hex[0] = 0;
  }
  hex[1] = dec / 256;
  hex[2] = dec - hex[1]*256;
  return hex;
}

