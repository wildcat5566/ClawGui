#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"
MPU6050 mpu;
int16_t ax, ay, az, gx, gy, gz;
int acc[6] = {0,0,0,0,0,0};
int iterations = 10000;
void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();
  mpu.setRate(400);
  Serial.println(mpu.testConnection() ? "MPU6050 connection successful" : "MPU6050 connection failed");
  
  for(int i = 0; i < iterations; i++){
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    acc[0]+=ax;
    acc[1]+=ay;
    acc[2]+=az;
    acc[3]+=gx;
    acc[4]+=gy;
    acc[5]+=gz;
  }
  for(int i = 0; i < 6; i++){
    float avg = acc[i] / iterations;
    Serial.println(avg);
  }
}

void loop() {

/*Serial.print(ax);Serial.print(",");
Serial.print(ay);Serial.print(",");
Serial.print(az);Serial.print(",");
Serial.print(gx);Serial.print(",");
Serial.print(gy);Serial.print(",");
Serial.print(gz);Serial.println();*/
}
