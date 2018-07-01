#include "main.h"
#include "i2c_slave.h"
#include "read_encoder.h"
#include "collision.h"
#include "pwm.h"
int count;
int collide;
int pwm_in;

int main(void){
	count = 0;
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC, ENABLE);
	i2c_init();
	timer_init();
	DelayInit();
	TIMx_PWM_Init();
	
	XA_config();
	XB_config();
	XC_config();
	
	vars_initialize();
	GPIOC->BSRR=(1<<13);
	
	while (1){
	  pwm_out(pwm_in);

		int timerValue = TIM_GetCounter(TIM2);
    if (timerValue == 0)
			collision_detect();
	}
}
