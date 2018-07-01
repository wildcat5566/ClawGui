#include "pwm.h"

DirectPWM::DirectPWM(int PWML1, int PWML2){
  rev = false;
  _PWML1 = PWML1;
  _PWML2 = PWML2;
}

void DirectPWM::PWM_Init(int pwmMax){
  analogWriteResolution(12);
  pinMode(_PWML1, OUTPUT);
  pinMode(_PWML2, OUTPUT);
  _max = pwmMax;
}

void DirectPWM::sendPwm(int pwm){
    if(rev == true){
      pwm = pwm * (-1);
    }
    pwm = constrain(pwm, -_max, _max);

    if (pwm >= 0){
      digitalWrite(_PWML1, HIGH);
      analogWrite(_PWML2, 4096 - pwm);
    }
    else{
      analogWrite(_PWML1, 4096 + pwm);
      digitalWrite(_PWML2, HIGH);
    }
}

void DirectPWM::Reverse(){
  rev = true;
}

