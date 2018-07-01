#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"
#include "motor_module_v1.h"
#include <PID_v1.h>
#include "Kalman.h"
#include "pwm.h"
#include <Event.h>
#include <Timer.h>

#define RelayPin0 2
#define RelayPin1 4
#define ExtResetPin 23
#define WR_PWML1 10
#define WR_PWML2 11
#define WL_PWML1 12
#define WL_PWML2 13

int res = 113246;
byte relay = 0;
int angvel = 0;
int grad = 1;
int brightness = 0;

Kalman kalmanX;
Kalman kalmanY;
MPU6050 mpu;
MotorObject FR, FL, RR, RL, WR, WL;
DirectPWM WR_PWM(10,11);
DirectPWM WL_PWM(12,13);
Timer SerialCommTimer;

float roll_measure, pitch_measure;
double roll_predict, pitch_predict;

int count[6];
int theta_offset = int(res * 135 / 360);
int prev_count[4]= {0, 0, 0, 0};
float theta[4], dtheta[4];
float phi;
bool coll[4];
float imu[6];
double set[6] = {0, 0, 0, 0, 0, 0};
float pos_angle;

long int timestamp = 0;
int16_t ax, ay, az, gx, gy, gz;
float gyro_sen = 131.0;
float acc_sen = 16384.0;

/* 
//Red group
float ax_offset = 80.00;
float ay_offset = -375.00;
float az_offset = 17846.00 - acc_sen;
float gx_offset = -478.00;
float gy_offset = 250.00;
float gz_offset = 18.00;
*/

//Blue group
float ax_offset = -1188.00;
float ay_offset = -226.00;
float az_offset = 24623.00 - acc_sen;
float gx_offset = -178.00;
float gy_offset = 82.00;
float gz_offset = -48.00;

//******PID Settings******//
double Timer3_HZ = 100.0;

//////////////////////////////////////////////////////////
int PWMmax = 400;
int W_PWMmax = 100;
float P_Kp = 0.2, P_Ki = 0.5, P_Kd = 0;
/////////////////////////////////////////////////////////

//Feed & Set Settings
double FL_Feed = 0, FR_Feed = 0, RL_Feed = 0, RR_Feed = 0, WL_Feed = 0, WR_Feed = 0;
double FL_Set = 0, FR_Set = 0, RL_Set = 0, RR_Set = 0, WL_Set = 0, WR_Set = 0;
//Computed PWM requirement
double FL_PWMvalue = 0, FR_PWMvalue = 0, RL_PWMvalue = 0, RR_PWMvalue = 0, WL_PWMvalue = 0, WR_PWMvalue = 0;
//Control Variables
float FL_Angular_V_Tar = 0, FR_Angular_V_Tar = 0, RL_Angular_V_Tar = 0, RR_Angular_V_Tar = 0;
float Angular_V_Tar = 0;
float Angular_V_Add = 0.1;

PID FR_PID(&FR_Feed, &FR_PWMvalue, &set[0], 0, 0, 0, DIRECT);
PID FL_PID(&FL_Feed, &FL_PWMvalue, &set[1], 0, 0, 0, DIRECT);
PID RR_PID(&RR_Feed, &RR_PWMvalue, &set[2], 0, 0, 0, DIRECT);
PID RL_PID(&RL_Feed, &RL_PWMvalue, &set[3], 0, 0, 0, DIRECT);
PID WR_PID(&WR_Feed, &WR_PWMvalue, &set[4], 0, 0, 0, DIRECT);
PID WL_PID(&WL_Feed, &WL_PWMvalue, &set[5], 0, 0, 0, DIRECT);

void setup(){
  Serial.begin(115200);
  Motor_Init(ExtResetPin);
  Wire.begin();
  SerialCommTimer.every(50, SendViaSerial);

  pinMode(RelayPin0, OUTPUT);
  digitalWrite(RelayPin0, LOW);
  pinMode(RelayPin1, OUTPUT);
  digitalWrite(RelayPin1, LOW);
  
  mpu.initialize();
  mpu.setRate(400);
  if(!mpu.testConnection()){
    Serial.println("MPU6050 connection failed");
    while(1){} // Block progress
  }

  WL.setSlaveAddress(0x13); // Waist Left
  FL.setSlaveAddress(0x4B); // Front left
  WR.setSlaveAddress(0x14);  // Waist Right
  FR.setSlaveAddress(0x4D);  // Front Right
  RL.setSlaveAddress(0x4E); // Rear Right
  RR.setSlaveAddress(0x4F);  // Rear Left
  FR.Reverse(); WR.Reverse();

  int waist_pwmMax = 100;
  WL_PWM.PWM_Init(waist_pwmMax);
  WR_PWM.PWM_Init(waist_pwmMax);
  WR_PWM.Reverse();

  PID_init();

 byte start_msg[12]; 
  while(1){
    if (Serial.available()==12){
      for (int i = 0; i < 12; i++){
        start_msg[i] = Serial.read()-'0';
      }
     break;
    }
  }//wait for start signal
  
  // Decode Kalman covariances: "152" = 1.5e(-2)
  double Q_angle = (start_msg[0] + 0.1*start_msg[1])/pow(10, start_msg[2]);
  double Q_bias = (start_msg[3] + 0.1*start_msg[4])/pow(10, start_msg[5]);
  double R_measure = (start_msg[6] + 0.1*start_msg[7])/pow(10, start_msg[8]);
  kalmanX.setQangle(Q_angle);
  kalmanY.setQangle(Q_angle);
  kalmanX.setQbias(Q_bias);
  kalmanY.setQbias(Q_bias);
  kalmanX.setRmeasure(R_measure);
  kalmanY.setRmeasure(R_measure);

  angvel = start_msg[9]*100 + start_msg[10]*10 + start_msg[11];
  // Print acknowledge line
  Serial.print(Q_angle, 4);Serial.print(",");Serial.print(Q_bias, 4);Serial.print(",");Serial.print(R_measure, 4);Serial.print(",");Serial.println(angvel);
  
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  imu[0] = (ax - ax_offset) / acc_sen;
  imu[1] = (ay - ay_offset) / acc_sen;
  imu[2] = (az - az_offset) / acc_sen;
  roll_measure  = -atan2(imu[1], imu[2]) * RAD_TO_DEG;
  pitch_measure = atan(imu[0] / sqrt(imu[1] * imu[1] + imu[2] * imu[2])) * RAD_TO_DEG;

  delay(3000);
  digitalWrite(RelayPin0, HIGH);
  digitalWrite(RelayPin1, HIGH);

  // starting angle setting
  kalmanX.setAngle(roll_measure);
  kalmanY.setAngle(pitch_measure);
  timestamp = micros();
  
  int bts = 32;
  while(bts){
    Serial.read(); // Clear serial buffer
    bts--;
  }
}


void loop() {
  /* 1. Retrieve data */
  FR.getMotorState(&count[0], &coll[0]);
  FL.getMotorState(&count[1], &coll[1]);
  RR.getMotorState(&count[2], &coll[2]);
  RL.getMotorState(&count[3], &coll[3]);
  WR.getMotorState(&count[4]);  
  WL.getMotorState(&count[5]);
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  /* 2. Stopwatch */
  double dt = (double)(micros() - timestamp) / 1000000;
  timestamp = micros();
  
  /* 3. Unit transformation */
  pos_angle = abs((int((count[0] + count[1] + count[2] + count[3]) * 0.25 + theta_offset) % res) * 360.0 / res);
  phi = abs((( (count[4] + count[5])/2) % res) * 360.0 / res);

  imu[0] = (ax - ax_offset) / acc_sen;
  imu[1] = (ay - ay_offset) / acc_sen;
  imu[2] = (az - az_offset) / acc_sen;
  imu[3] = (gx - gx_offset) / gyro_sen;
  imu[4] = (gy - gy_offset) / gyro_sen;
  imu[5] = (gz - gz_offset) / gyro_sen;

  roll_measure  = -atan2(imu[1], imu[2]) * RAD_TO_DEG;
  pitch_measure = atan(imu[0] / sqrt(imu[1] * imu[1] + imu[2] * imu[2])) * RAD_TO_DEG;

  roll_predict = kalmanX.getAngle(roll_measure, imu[3], dt);
  pitch_predict = kalmanY.getAngle(pitch_measure, imu[4], dt);

  /* 4. Send robot state via serial */
  //SerialCommTimer.update();
  Serial.print(pos_angle);   Serial.print(",");
  Serial.print(roll_predict);Serial.print(",");Serial.print(pitch_predict);Serial.print(",");
  Serial.print(count[0]);Serial.print(",");
  Serial.print(count[1]);Serial.print(",");
  Serial.print(count[2]);Serial.print(",");
  Serial.print(count[3]);Serial.print(",");
  Serial.print(count[4]);Serial.print(",");
  Serial.print(count[5]);Serial.println(",");

  /* 5. Calculate PID */
  FR_Feed = count[0];
  FL_Feed = count[1];  
  RR_Feed = count[2];
  RL_Feed = count[3];
  WR_Feed = count[4];
  WL_Feed = count[5];

  if(grad < angvel){
      grad++;
  }

  set[0]  += grad;
  set[1]  += grad;
  set[2]  += grad;
  set[3]  += grad; 
  set[4] = 0; set[5] = 0;
  
  FR_PID.Compute();
  FL_PID.Compute();  
  RR_PID.Compute();
  RL_PID.Compute();
  WR_PID.Compute();
  WL_PID.Compute();

  /* 6. Send PWM to motors */
  FR.sendPwm(FR_PWMvalue);
  RR.sendPwm(RR_PWMvalue);
  RL.sendPwm(RL_PWMvalue);
  FL.sendPwm(FL_PWMvalue);
  WR_PWM.sendPwm(WR_PWMvalue);
  WL_PWM.sendPwm(WL_PWMvalue);
  
  Serial.print("WR: ");
  Serial.print(WR_PWMvalue);
  Serial.print(", WL: ");
  Serial.print(WL_PWMvalue);
  
  for(int i = 0; i < 4; i++){
    prev_count[i] = count[i];
  }
}


