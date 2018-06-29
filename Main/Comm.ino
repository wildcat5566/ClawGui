void SendViaSerial(){
  if(Serial.available()){
    int bts = Serial.available();
    Serial.print(pos_angle);   Serial.print(",");
    Serial.print(roll_predict);Serial.print(",");Serial.print(pitch_predict);Serial.print(",");
    Serial.print(count[0]);Serial.print(",");
    Serial.print(count[1]);Serial.print(",");
    Serial.print(count[2]);Serial.print(",");
    Serial.print(count[3]);Serial.print(",");
    Serial.print(count[4]);Serial.print(",");
    Serial.print(count[5]);Serial.println(",");

    while(bts){
      Serial.read();
      bts--;
    }
  }
}

void I2C_Right(){
   FR.sendPwm(FR_PWMvalue);
   WR.sendPwm(WR_PWMvalue);
   prev_count[0] = count[0];
   FR.getMotorState(&count[0], &coll[0]);
   WR.getMotorState(&count[4]);
}

void I2C_Rear(){
  RR.sendPwm(RR_PWMvalue);
  RL.sendPwm(RL_PWMvalue);
  prev_count[2] = count[2];
  prev_count[3] = count[3];
  RR.getMotorState(&count[2], &coll[2]);
  RL.getMotorState(&count[3], &coll[3]);
}

void I2C_Left(){
  FL.sendPwm(FL_PWMvalue);
  WL.sendPwm(WL_PWMvalue);
  prev_count[1] = count[1];
  FL.getMotorState(&count[1], &coll[1]);
  WL.getMotorState(&count[5]);
}

void Breathe(){
  brightness = brightness + grad;
  if (brightness <= 0 || brightness >= 255) {
    grad = -grad ;
  }   
  set[0] += brightness;
  set[1] += brightness;
  set[2] += brightness;
  set[3] += brightness;
}

