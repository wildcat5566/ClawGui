#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"
#include "motor_module_v1.h"
#include <PID_v1.h>
#include "Kalman.h"
#include <LiquidCrystal.h>

const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

#define RelayPin0 6
#define RelayPin1 7
#define ExtResetPin 23
int res = 113246;

Kalman kalmanX;
Kalman kalmanY;
MPU6050 mpu;
MotorObject FR, FL, RR, RL, WR, WL;

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
 
//Red group
float ax_offset = 80.00;
float ay_offset = -375.00;
float az_offset = 17846.00 - acc_sen;
float gx_offset = -478.00;
float gy_offset = 250.00;
float gz_offset = 18.00;

/*
//Blue group
float ax_offset = -1188.00;
float ay_offset = -226.00;
float az_offset = 24623.00 - acc_sen;
float gx_offset = -178.00;
float gy_offset = 82.00;
float gz_offset = -48.00;*/

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
int vel = 0;

PID FR_PID(&FR_Feed, &FR_PWMvalue, &set[0], 0, 0, 0, DIRECT);
PID FL_PID(&FL_Feed, &FL_PWMvalue, &set[1], 0, 0, 0, DIRECT);
PID RR_PID(&RR_Feed, &RR_PWMvalue, &set[2], 0, 0, 0, DIRECT);
PID RL_PID(&RL_Feed, &RL_PWMvalue, &set[3], 0, 0, 0, DIRECT);
PID WR_PID(&WR_Feed, &WR_PWMvalue, &set[4], 0, 0, 0, DIRECT);
PID WL_PID(&WL_Feed, &WL_PWMvalue, &set[5], 0, 0, 0, DIRECT);

void setup() {
  Serial.begin(115200);
  lcd.begin(16, 2);
  lcd.print("hi");
  
  // Clear serial buffer
  if(Serial.available()){
    int bts = Serial.available();
    while(bts){
      Serial.read();
      bts--;
    }
  }
  
  Motor_Init(ExtResetPin);
  Wire.begin();

  pinMode(RelayPin0, OUTPUT);
  digitalWrite(RelayPin0, LOW);
  pinMode(RelayPin1, OUTPUT);
  digitalWrite(RelayPin1, LOW);
  
  mpu.initialize();
  mpu.setRate(400);
  //Serial.println(mpu.testConnection() ? "MPU6050 connection successful" : "MPU6050 connection failed");

  WL.setSlaveAddress(0x4A); // Waist Left
  FL.setSlaveAddress(0x4B); // Front left
  WR.setSlaveAddress(0x4C);  // Waist Right
  FR.setSlaveAddress(0x4D);  // Front Right
  RL.setSlaveAddress(0x4E); // Rear Right
  RR.setSlaveAddress(0x4F);  // Rear Left
  FR.Reverse(); WR.Reverse();

  PID_init();
  
  if(!mpu.testConnection()){
    Serial.println("MPU6050 connection failed");
    while(1){} // Block progress
  }
  
  byte cov[9]; 
  while(1){
    if (Serial.available()==9){
      for (int i = 0; i < 9; i++){
        cov[i] = Serial.read()-'0';
      }
     break;
    }
  }//waiting for start signal
  
  // Set Kalman covariances: "152" = 1.5e(-2)
  double Q_angle = (cov[0] + 0.1*cov[1])/pow(10, cov[2]);
  double Q_bias = (cov[3] + 0.1*cov[4])/pow(10, cov[5]);
  double R_measure = (cov[6] + 0.1*cov[7])/pow(10, cov[8]);
  kalmanX.setQangle(Q_angle);
  kalmanY.setQangle(Q_angle);
  kalmanX.setQbias(Q_bias);
  kalmanY.setQbias(Q_bias);
  kalmanX.setRmeasure(R_measure);
  kalmanY.setRmeasure(R_measure);

  Serial.print(Q_angle, 4);Serial.print(",");Serial.print(Q_bias, 4);Serial.print(",");Serial.println(R_measure, 4);
  if(Q_angle*Q_bias*R_measure==0){
    while(1){} // Block progress
  }

  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  imu[0] = (ax - ax_offset) / acc_sen;
  imu[1] = (ay - ay_offset) / acc_sen;
  imu[2] = (az - az_offset) / acc_sen;
  roll_measure  = -atan2(imu[1], imu[2]) * RAD_TO_DEG;
  pitch_measure = atan(imu[0] / sqrt(imu[1] * imu[1] + imu[2] * imu[2])) * RAD_TO_DEG;

  /*delay(3000);
  digitalWrite(RelayPin0, HIGH);
  digitalWrite(RelayPin1, HIGH);*/

  // starting angle setting
  kalmanX.setAngle(roll_measure);
  kalmanY.setAngle(pitch_measure);
  timestamp = micros();
}
int c = 0;
int grad = 0;
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
  //Serial.println(dt, 4);
  timestamp = micros();
  
  /* 3. Unit transformation */
  pos_angle = abs((int((count[0] + count[1] + count[2] + count[3]) * 0.25 + theta_offset) % res) * 360.0 / res);
  //pos_angle = abs((int(count[2] + theta_offset) % res) * 360.0 / res);
  
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

  /* 4. Send via serial */
  if(Serial.available()==4){
    int bts = Serial.available();

    byte comm[4];
    for (int i = 0; i < 4; i++){
      comm[i] = Serial.read()-'0';
    }
    vel = comm[0]*100 + comm[1]*10 + comm[2];
    lcd.setCursor(0,0); lcd.print(vel);
    if(comm[3]==0){
      digitalWrite(RelayPin0, LOW);
      digitalWrite(RelayPin1, LOW);
    }
    else{
      digitalWrite(RelayPin0, HIGH);
      digitalWrite(RelayPin1, HIGH);
    }

    /*Serial.print(count[0]);Serial.print(",");
    Serial.print(count[1]);Serial.print(",");
    Serial.print(count[2]);Serial.print(",");
    Serial.print(count[3]);Serial.print(",");
    Serial.print(count[4]);Serial.print(",");
    Serial.print(count[5]);Serial.print(",");*/
    Serial.print(pos_angle);   Serial.print(",");
    Serial.print(roll_predict);Serial.print(",");Serial.println(pitch_predict);
    
    while(bts){
      Serial.read(); //clear serial buffer
      bts--;
    }
 }
  
  /* 5. Calculate PID */
  FR_Feed = count[0];
  FL_Feed = count[1];  
  RR_Feed = count[2];
  RL_Feed = count[3];
  WR_Feed = count[4];
  WL_Feed = count[5];

  if(grad < 200){
    grad++;
  }
  
  set[0] += grad;
  set[1] += grad;
  set[2] += grad;
  set[3] += grad;  
  set[4] = 0; set[5] = 0;
  
  FR_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  FL_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  RR_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  RL_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  WR_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  WL_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  
  FR_PID.Compute();
  FL_PID.Compute();  
  RR_PID.Compute();
  RL_PID.Compute();
  WR_PID.Compute();
  WL_PID.Compute();
  
  /* 6. Send PWM to motors */
  FR.sendPwm(FR_PWMvalue);
  FL.sendPwm(FL_PWMvalue);
  RR.sendPwm(RR_PWMvalue);
  RL.sendPwm(RL_PWMvalue);
  WR.sendPwm(WR_PWMvalue);
  WL.sendPwm(WL_PWMvalue);

  for(int i = 0; i < 4; i ++){
    prev_count[i] = count[i];
  }

  c++;
  delay(1);
}

void PID_init(){
  //MANUAL: PID off; AUTOMATIC: PID on;
  FL_PID.SetSampleTime(1000 / Timer3_HZ);
  FL_PID.SetOutputLimits(0, PWMmax);
  FL_PID.SetMode(MANUAL);
  FR_PID.SetSampleTime(1000 / Timer3_HZ);
  FR_PID.SetOutputLimits(0, PWMmax);
  FR_PID.SetMode(MANUAL);
  RL_PID.SetSampleTime(1000 / Timer3_HZ);
  RL_PID.SetOutputLimits(0, PWMmax);
  RL_PID.SetMode(MANUAL);
  RR_PID.SetSampleTime(1000 / Timer3_HZ);
  RR_PID.SetOutputLimits(0, PWMmax);
  RR_PID.SetMode(MANUAL);
  
  WL_PID.SetSampleTime(1000 / Timer3_HZ);
  WL_PID.SetOutputLimits(-W_PWMmax, W_PWMmax);
  WL_PID.SetMode(MANUAL);
  WR_PID.SetSampleTime(1000 / Timer3_HZ);
  WR_PID.SetOutputLimits(-W_PWMmax, W_PWMmax);
  WR_PID.SetMode(MANUAL);

  FL_PID.SetMode(MANUAL);
  FR_PID.SetMode(MANUAL);
  RL_PID.SetMode(MANUAL);
  RR_PID.SetMode(MANUAL);  
  WL_PID.SetMode(MANUAL);
  WR_PID.SetMode(MANUAL);

  FL_PID.SetMode(AUTOMATIC);
  FR_PID.SetMode(AUTOMATIC);
  RL_PID.SetMode(AUTOMATIC);
  RR_PID.SetMode(AUTOMATIC);
  WL_PID.SetMode(AUTOMATIC);
  WR_PID.SetMode(AUTOMATIC);
} //PID_init()

