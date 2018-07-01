#ifndef _PWM_H_
#define _PWM_H_

#include "Arduino.h"

class DirectPWM{
public:
  DirectPWM(int PWML1, int PWML2);
  void PWM_Init(int pwmMax);
  void sendPwm(int pwm);                // Send PWM value to slave
  void Reverse();                                // Call to set flag & reverse default direction

private:
  bool rev;                                      // Direction reverse flag
  int _PWML1;
  int _PWML2;
  int _max;
};

#endif
