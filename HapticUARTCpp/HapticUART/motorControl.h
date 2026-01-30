#pragma once



extern unsigned char val[40]; //save motor's pushing value(for interpolate). 40 for 20 motors, u8type ult output to UART
extern int motorCurrentValue[20]; //up to 20 motors, int type
extern int motorBaseCurrentValue[20]; //0x01's value will in there, can only changed by command, and no + -
extern int last_motorBaseCurrentValue[20]; //save last motorBaseCurrentValue, for compare
extern float tilt_motorBaseCurrentValue[20]; //
extern int motorQ[20]; //save Motor's head(of course apporximately)
extern int DiffuseQ;
extern int MaxQ;
extern bool isCooling[20];
extern bool motorBaseSign[20]; // 记录每路基准电流的正负符号


void pushValue2Current();
int uchar_to_int(unsigned char high, unsigned char low);
void clearMotorCurrentValue();
void clearMotorBaseCurrentValue();


