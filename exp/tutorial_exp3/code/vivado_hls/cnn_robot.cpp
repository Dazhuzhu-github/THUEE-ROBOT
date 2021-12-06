
#include "cnn_robot.h"
#include "network.h"


static float col1_out[10][24][24];
static float input_num[3][28][28];
void conv1()
{
	int out_size = 28 - 5 + 1;
	int out_sq = out_size * out_size;
	int ker_sq = 5 * 5;
	float convsum = 0;
	for (int i = 0; i < 10; i++)//out_ch
	{
		for (int x = 0; x < out_size; x++)
		{
			for (int y = 0; y < out_size; y++)
			{
				for (int j = 0; j < 3; j++) //in_ch
				{
					for (int i1 = 0; i1 < 5; i1++)//ker_size
					{
#pragma HLS PIPELINE
						for (int j1 = 0; j1 < 5; j1++)
						{
#pragma HLS UNROLL
							convsum += float(input_num[j][x + i1][y + j1] * conv1_weight[i][j*25+i1*5+j1]);
						}
					}
				}
				col1_out[i][x][y] = convsum + conv1_bias[i];
				convsum = 0;
			}
		}
	}

}

static float mp_relu1_out[10][12][12];
void mp_relu1()
{
	int out_size = 12;
	int out_sq = out_size * out_size;
	int ker_sq = 5 * 5;
	float max_temp=0;
	for (int i = 0; i < 10; i++)//out_ch
	{
		for (int x = 0; x < out_size; x++)
		{
#pragma HLS PIPELINE
			for (int y = 0; y < out_size; y++)
			{
#pragma HLS PIPELINE
				for (int i1 = 0; i1 < 2; i1++)//ker_size
				{
#pragma HLS UNROLL
					for (int j1 = 0; j1 < 2; j1++)
					{
#pragma HLS UNROLL
						max_temp = (max_temp > col1_out[i][x*2 + i1][y*2 + j1]) ? max_temp : col1_out[i][x*2 + i1][y*2 + j1];
					}
				}
				mp_relu1_out[i][x][y] = max_temp;
				max_temp = 0;
			}
		}
	}
}

static float col2_out[20][8][8] = {0};

void conv2()
{
	int out_size = 12 - 5 + 1;
	int out_sq = out_size * out_size;
	int ker_sq = 5 * 5;
	float convsum = 0;
	for (int i = 0; i < 20; i++)//out_ch
	{
		for (int x = 0; x < out_size; x++)
		{
			for (int y = 0; y < out_size; y++)
			{
				for (int j=0; j < 10; j++)
				{
					for (int i1 = 0; i1 < 5; i1++)//ker_size
					{
#pragma HLS PIPELINE
						for (int j1 = 0; j1 < 5; j1++)
						{
#pragma HLS UNROLL
							convsum += float(mp_relu1_out[j][x + i1][y + j1] * conv2_weight[i][j*25+i1*5+j1]);
						}
					}
				}
				col2_out[i][x][y] += convsum;
				convsum = 0;
			}
		}
	}
	for (int i = 0; i < 20; i++)
	{
		for (int x = 0; x < out_size; x++)
		{
#pragma HLS PIPELINE
			for (int y = 0; y < out_size; y++)
			{
#pragma HLS UNROLL
				col2_out[i][x][y] += conv2_bias[i];
			}
		}
	}
}

static float mp_relu2_out[20][4][4];
void mp_relu2()
{
	int out_size = 4;
	int out_sq = out_size * out_size;
	int ker_sq = 5 * 5;
	float max_temp = 0;
	for (int i = 0; i < 20; i++)//out_ch
	{

		for (int x = 0; x < out_size; x++)
		{
#pragma HLS PIPELINE
			for (int y = 0; y < out_size; y++)
			{
#pragma HLS PIPELINE
				for (int i1 = 0; i1 < 2; i1++)//ker_size
				{
#pragma HLS UNROLL
					for (int j1 = 0; j1 < 2; j1++)
					{
#pragma HLS UNROLL
						max_temp = (max_temp > col2_out[i][x * 2 + i1][y * 2 + j1]) ? max_temp : col2_out[i][x * 2 + i1][y * 2 + j1];
					}
				}
				mp_relu2_out[i][x][y] = max_temp;
				max_temp = 0;
			}
		}

	}

}

static float fc_out[6] = {0};
void fc()
{
	for (int i = 0; i < 6; i++)
	{
#pragma HLS UNROLL
		for (int j = 0; j < 320; j++)
		{
#pragma HLS PIPELINE
			fc_out[i] += *(**(mp_relu2_out) + j) * fc_weight[i][j];
		}
		fc_out[i] += fc_bias[i];
	}

}


void test(my_float input[3][28][28], my_float output[6])
{
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis register both port=input
#pragma HLS INTERFACE axis register both port=output
	for (int c=0; c<3; c++){
		for (int i=0;i<28;i++)
		{
#pragma HLS PIPELINE
			for (int j=0;j<28;j++)
			{
#pragma HLS PIPELINE
				input_num[c][i][j]=input[c][i][j].data;
			}
		}
	}
	conv1();
	mp_relu1();
	conv2();
	mp_relu2();
	fc();
	for (int i=0;i<6;i++)
	{
		output[i].data=fc_out[i];
		output[i].last=(i==5)?1:0;
	}
}

