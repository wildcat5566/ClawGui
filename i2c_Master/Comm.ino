void SendViaSerial(){
  //if(Serial.available()){
  //  int bts = Serial.available();
  
    Serial.print(pos_angle);   Serial.print(",");
    Serial.print(roll_predict);Serial.print(",");Serial.print(pitch_predict);Serial.print(",");
    Serial.print(count[0]);Serial.print(",");
    Serial.print(count[1]);Serial.print(",");
    Serial.print(count[2]);Serial.print(",");
    Serial.print(count[3]);Serial.print(",");
    Serial.print(count[4]);Serial.print(",");
    Serial.print(count[5]);Serial.println(",");

   // while(bts){
   //   Serial.read();
   //   bts--;
   // }
  //}
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

  FR_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  FL_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  RR_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  RL_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  WR_PID.SetTunings(P_Kp, P_Ki, P_Kd);
  WL_PID.SetTunings(P_Kp, P_Ki, P_Kd);
} //PID_init()

void Breathe(){
  brightness = brightness + grad;
  if (brightness <= 0 || brightness >= 255) {
    grad = -grad ;
  }   
  set[0] += brightness;
  set[1] += brightness;
  set[2] += brightness;
  set[3] += brightness;
  set[4] = 0; set[5] = 0;
}

