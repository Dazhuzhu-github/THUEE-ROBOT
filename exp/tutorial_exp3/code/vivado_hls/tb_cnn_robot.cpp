#include "cnn_robot.h"
#include "picture.h"

#include <iostream>
using namespace std;

int main (void)
{
    my_float tb_input[3][28][28] = {0};
    my_float tb_out[6];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            for (int k = 0; k < 28; k++)
            {
                tb_input[i][j][k].data = pokeman[i][j][k];
            }
        }   
    }

    test(tb_input, tb_out);
    for (int i = 0; i < 6; i++)
    {
        cout << "tb_out[" << i <<"]: " << tb_out[i].data << " last: " << tb_out[i].last << endl;
    }
    
    return 0;
}
