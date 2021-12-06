#ifndef CNN_H
#define CNN_H

#include "ap_axi_sdata.h"
#include "ap_int.h"

struct my_float
{
	float data;
	ap_uint<1> last;
};

void test(my_float input[3][28][28], my_float output[6]);

#endif
