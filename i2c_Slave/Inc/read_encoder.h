#define GPIO_Port_Encoder GPIOA
#define GPIO_Pin_A 				GPIO_Pin_3
#define GPIO_Pin_B			  GPIO_Pin_4
#define EXTI_LineA 				EXTI_Line3
#define EXTI_LineB 				EXTI_Line4
#define NVIC_IRQChannelA  EXTI3_IRQn;
#define NVIC_IRQChannelB  EXTI4_IRQn;

void XA_config(void);
void XB_config(void);
void vars_initialize(void);
