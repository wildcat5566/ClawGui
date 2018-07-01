#include "main.h"
#include "pwm.h"

void TIMx_PWM_Init()  {  
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIMx, ENABLE); 
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOx | RCC_APB2Periph_AFIO, ENABLE); 
	
	GPIO_InitTypeDef GPIO_InitStructure; 
	GPIO_InitStructure.GPIO_Pin = PWML1 | PWML2; 
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP; 
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;  
	GPIO_Init(GPIO_Port_PWM, &GPIO_InitStructure);
  
  TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
	TIM_TimeBaseStructure.TIM_Period = 1000;  // the resolution
	TIM_TimeBaseStructure.TIM_Prescaler = 0; 
	TIM_TimeBaseStructure.TIM_ClockDivision = 0;  
	TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;  
	TIM_TimeBaseInit(TIMx, &TIM_TimeBaseStructure);  

	TIM_OCInitTypeDef TIM_OCInitStructure;  
	TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM2; 
	TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable; 
	TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High; 
	TIM_OC3Init(TIMx, &TIM_OCInitStructure);  
	TIM_OC4Init(TIMx, &TIM_OCInitStructure);

	TIM_Cmd(TIMx, ENABLE);  
}

void pwm_out(int pwmval){
	if (pwmval >= 0){
		TIM_SetCompare3(TIMx,0);
		TIM_SetCompare4(TIMx,pwmval);
	}
	else{
		TIM_SetCompare3(TIMx,-pwmval);
		TIM_SetCompare4(TIMx,0);
	}
}
