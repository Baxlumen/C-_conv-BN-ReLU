#include <iostream>
#include <vector>
#include <math.h>
#include "omp.h"
using namespace std;

class Data
{
public:
    int channels;
    int height;
    int width;
    vector<float> data; // 一维向量存储，以列为主序

    Data(int m_channels, int m_height, int m_width)
    {
        channels = m_channels;
        height = m_height;
        width = m_width;
        data.resize(channels * height * width);
    }
};

class conv_param 
{
public:
    int pad;
    int stride;
    int kernel_size;
    int in_channels;
    int out_channels;
    // 卷积核数据,size = out_channels * in_channels * kernel_size * kernel_size
    // 一维向量存储，以列为主序
    vector<float> filter;
    conv_param(int p, int s, int k, int ic, int oc, vector<float> f) : pad(p), stride(s), kernel_size(k), in_channels(ic), out_channels(oc), filter(f) {}
};

class myFun
{
public:
    //便于计算，假设input的行列相等
    Data conv(int channels, int heigh, int width, Data input, conv_param conv_p)
    {
        int out_channel = conv_p.out_channels;
        int out_heigh, out_width;
        //输出数据行列  [(n+2p-f)/s+1],n->input数据行数，p->padding，f->卷积核行数，s->stride。结果向下取整
        out_heigh = floor(float((heigh + 2 * conv_p.pad - conv_p.kernel_size) / conv_p.stride) + 1);
        out_width = out_heigh;

        // 经过 padding 填充后的数据
        Data pad_input(channels, heigh + 2 * conv_p.pad, width + 2 * conv_p.pad);

        // #pragma omp parallel for num_threads(channels)
        for (int i = 0; i < channels; i++)
        {
            // #pragma omp parallel for num_threads(width)
            for (int j = 0; j < width; j++)
            {
                #pragma omp parallel for num_threads(heigh)
                for (int k = 0; k < heigh; k++)
                {
                    pad_input.data[i * (heigh + 2 * conv_p.pad) * (width + 2 * conv_p.pad) + (j + 1) * (heigh + 2 * conv_p.pad) + k + 1] = input.data[i * width * heigh + j * heigh + k];
                }
            }
        }
        heigh = heigh + 2 * conv_p.pad; // padding填充之后的行列
        width = width + 2 * conv_p.pad;
        // 定义输出
        Data output(out_channel, out_heigh, out_width);

        // #pragma omp parallel for num_threads(out_channel)
        for (int i = 0; i < out_channel; i++)
        {
            // #pragma omp parallel for num_threads(out_width)
            for (int j = 0; j < out_width; j++)
            {
                // #pragma omp parallel for num_threads(out_heigh)              
                for (int k = 0; k < out_heigh; k++)
                {
                    float sum = 0;
                    #pragma omp parallel for reduction(+:sum)
                    for (int kc = 0; kc < channels; kc++)
                    {
                        for (int kw = 0; kw < conv_p.kernel_size; kw++)
                        {
                            for (int kh = 0; kh < conv_p.kernel_size; kh++)
                            {
                                sum += pad_input.data[kc * heigh * width + (j * conv_p.stride + kw) * heigh + k * conv_p.stride + kh] * conv_p.filter[i * channels * conv_p.kernel_size * conv_p.kernel_size +
                                                                                                                                                      kc * conv_p.kernel_size * conv_p.kernel_size +
                                                                                                                                                      kw * conv_p.kernel_size +
                                                                                                                                                      kh];
                            }
                        }
                    }
                    output.data[i * out_heigh * out_width + j * out_heigh + k] = sum;
                }
            }
        }

        return output;
    }

    void BN_ReLU(vector<Data> &output, int batch)
    {
        // output 存储batch个data信息
        int num_ele = output[0].channels * output[0].height * output[0].width;
        float *mean = new float[num_ele];
        float *var = new float[num_ele];
        //计算均值
        #pragma omp parallel for
        for (int i = 0; i < num_ele; i++)
        {
            float sum = 0;
            for (int j = 0; j < batch; j++)
            {
                sum += output[j].data[i];
            }
            mean[i] = sum / batch;
        }

        //计算方差
        #pragma omp parallel for
        for (int i = 0; i < num_ele; i++)
        {
            float sum = 0;
            for (int j = 0; j < batch; j++)
            {
                sum += pow((output[j].data[i] - mean[i]), 2);
            }
            var[i] = sqrt(sum / batch);
        }

        // 归一化，ReLU：判断是否为负数，负数为0
        #pragma omp parallel for
        for (int i = 0; i < num_ele; i++)
        {
            for (int j = 0; j < batch; j++)
            {
                if (output[j].data[i] - mean[i] <= 0)
                {
                    output[j].data[i] = 0;
                }
                else
                {
                    output[j].data[i] = (output[j].data[i] - mean[i]) / var[i];
                }
            }
        }
        delete[] mean;
        delete[] var;
    }
};

int main()
{
    //omp_set_nested(0); 

    int batch = 2;
    Data input1(3, 5, 5); // 5*5大小的3通道数据
    Data input2(3, 5, 5);
    vector<Data> batch_input;
    vector<Data> batch_output;

    vector<float> tmp1 = {0, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1, 2, 2, 0, 2, 0, 0, 2, 2, 2, 2, 1, 0, 0, 0, 2, 2, 0, 1, 0, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 0, 0, 2, 0, 1, 1, 1, 2, 1, 2, 1, 0, 1, 2, 1, 0, 1, 0, 2, 1, 0, 0, 2, 1, 2, 1, 0, 1, 1, 0, 0, 0, 0, 1};
    vector<float> tmp2 = {0, 2, 0, 1, 1, 0, 2, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 2, 1, 0, 1, 0, 1, 0, 2, 0, 1, 2, 1, 1, 2, 1, 1, 0, 0, 2, 0, 1, 2, 1, 0, 1, 2, 2, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 2};
    input1.data.assign(tmp1.begin(), tmp1.end());
    input2.data.assign(tmp2.begin(), tmp2.end());
    batch_input.push_back(input1);
    batch_input.push_back(input2);

    // printf("Thread number = %d\n",omp_get_thread_num());
    vector<float> filter = {-1, 0, 1, 0, 0, -1, 1, 1, 1, -1, 1, 0, 0, -1, 1, 1, 1, 0, -1, 1, 0, 1, 1, -1, 1, 0, 0, 0, 0, 0, 1, -1, -1, -1, 0, 1, -1, 1, 1, 0, -1, -1, 0, 0, 0, -1, 0, 1, 1, -1, 0, -1, -1, 0};
    // 2个 3*3*3的卷积核
    // 输出channel 为 2，输入为3
    conv_param conv_p(1, 2, 3, 3, 2, filter);

    myFun fun;
    double starttime, endtime;
    starttime = omp_get_wtime();

    // 运行100次
    for (int p = 0; p < 99; p++)
    {
        for (int i = 0; i < batch; i++)
        {
            batch_output.push_back(fun.conv(batch_input[i].channels, batch_input[i].height, batch_input[i].width, batch_input[i], conv_p));
        }
        fun.BN_ReLU(batch_output, batch);

        for (int i = 0; i < batch; i++)
            batch_output.pop_back();
    }
    for (int i = 0; i < batch; i++)
    {
        batch_output.push_back(fun.conv(batch_input[i].channels, batch_input[i].height, batch_input[i].width, batch_input[i], conv_p));
    }
    fun.BN_ReLU(batch_output, batch);

    endtime = omp_get_wtime();
    // print 结果
    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < batch_output[b].channels; i++)
        {
            for (int j = 0; j < batch_output[b].height; j++)
            {
                for (int k = 0; k < batch_output[b].height; k++)
                {
                    cout << batch_output[b].data[i * batch_output[b].height * batch_output[b].height + j * batch_output[b].height + k] << " ";
                }
                cout << endl;
            }
            cout << "------------" << endl;
        }
        cout << "*************" << endl;
    }

    cout << "cost time :" << endtime - starttime <<endl;


    // 并行 100次时间：0.137
    // 串行 100次时间：0.00299978

    // 并行的时间更长。由于输入数据规模小，线程的初始化与销毁的时间更长，导致并行计算总的运行时间比串行的要慢

    return 0;
}
