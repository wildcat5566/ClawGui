#define I2Cx_RCC				RCC_APB1Periph_I2C1
#define I2Cx						I2C1
#define GPIO_Port_I2C		GPIOB
#define I2C_PIN_SDA			GPIO_Pin_7
#define I2C_PIN_SCL			GPIO_Pin_6
#define I2Cx_EV_IRQn	  I2C1_EV_IRQn
#define bufferlength 		4
#define I2CSLAVE_ADDR		0x4B
#define I2C_CLOCK_FRQ   100000    // I2C-Frq in Hz (100 kHz)
#define SCL_read      	GPIO_ReadInputDataBit(GPIOB, GPIO_Pin_6) 
#define SDA_read      	GPIO_ReadInputDataBit(GPIOB, GPIO_Pin_7)

void i2c_init(void);
