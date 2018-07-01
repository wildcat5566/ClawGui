/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif 
	
#include "stm32f10x.h"
#include "stm32f10x_conf.h" 
#include <stdio.h>
#include <stdlib.h>
#include "delay.h"
	
extern int count;
extern int collide;
extern int pwm_in;
	
#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
