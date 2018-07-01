#define PWML1 								GPIO_Pin_0
#define PWML2 								GPIO_Pin_1	
#define GPIO_Port_PWM 				GPIOB
#define RCC_APB1Periph_TIMx 	RCC_APB1Periph_TIM3
#define RCC_APB2Periph_GPIOx 	RCC_APB2Periph_GPIOB
#define TIMx 									TIM3
	
extern int pwm_in;
void TIMx_PWM_Init(void);
void pwm_out(int pwmval);
