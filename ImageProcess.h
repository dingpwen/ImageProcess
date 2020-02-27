//
// Created by wenpd on 2020/2/25.
//

#include <opencv2/opencv.hpp>
#include "log.h"

using namespace cv;

#ifndef OPENCV_APPLICATION_IMAGEPROCESS_H
#define OPENCV_APPLICATION_IMAGEPROCESS_H

class ImageProcess{
private:
    Mat mImg;

private:
    // delta^2 = delta0^2 + [sumRight(f^2) - sumLeft(f^2)]/N + M0^2 - M^2
    void getMeanFromPreviousCol(const Mat &img, Mat &mean, Mat &sigma, const Mat &sumM, const Mat &sumS, int m, int n, int pI, int pJ) {
        int total = (2*m + 1) * (2*n + 1);
        /*float M = 0;
        for(int i = pI - m; i <= pI + m; ++i) {
            M += (img.at<uchar>(i, pJ + n) - img.at<uchar>(i, pJ -n -1));
        }*/
        float M = sumM.at<float>(pI, pJ + n) - sumM.at<float>(pI, pJ -n -1);
        M /= total;
        mean.at<float>(pI, pJ) = M + mean.at<float>(pI, pJ -1);

        /*float S = 0;
        for(int i = pI - m; i <= pI + m; ++i) {
            S += (pow(img.at<uchar>(i, pJ + n), 2) - pow(img.at<uchar>(i, pJ -n -1), 2));
        }*/
        float S = sumS.at<float>(pI, pJ + n) - sumS.at<float>(pI, pJ -n -1);
        S /= total;
        S += pow(mean.at<float>(pI, pJ -1), 2) - pow(mean.at<float>(pI, pJ), 2);
        sigma.at<float>(pI, pJ) = S + sigma.at<float>(pI, pJ -1);
    }

    void getMeanFromPreviousRow(const Mat &img, Mat &mean, Mat &sigma, const vector<float> &sumM, const vector<float> &sumS,
            int m, int n, int pI, int pJ) {
        if(pJ != n) {
            LOGE("line0-something is wrong!with pos=(%d, %d)", pI, pJ);
            return;
        }
        int total = (2*m + 1) * (2*n + 1);
        /*float M = 0;
        for(int j = pJ - n; j <= pJ + n; ++j) {
            M += (img.at<uchar>(pI + n, j) - img.at<uchar>(pI - n -1, j));
        }*/
        float M = sumM[pI + n] - sumM[pI - n -1];
        M /= total;
        mean.at<float>(pI, pJ) = M + mean.at<float>(pI - 1, pJ);

        /*float S = 0;
        for(int j = pJ - n; j <= pJ + n; ++j) {
            S += (pow(img.at<uchar>(pI + n, j), 2) - pow(img.at<uchar>(pI - n -1, j), 2));
        }
        S /= total;*/
        float S = (sumS[pI + n] - sumS[pI - n -1])/total;
        S += pow(mean.at<float>(pI - 1, pJ), 2) - pow(mean.at<float>(pI, pJ), 2);
        sigma.at<float>(pI, pJ) = S + sigma.at<float>(pI - 1, pJ);
    }

    //Count col sum
    void preCountColSum(const Mat &img, Mat &sumM, Mat &sumS, int m) {
        int colSize = (m << 1) + 1;
        for(int j=0; j<img.cols; ++j) {
            float tM = 0;
            float tS = 0;
            for(int i=0; i<colSize; ++i) {
                tM += img.at<uchar>(i, j);
                tS += (float)pow(img.at<uchar>(i, j), 2);
            }
            sumM.at<float>(m, j) = tM;
            sumS.at<float>(m, j) = tS;
            for(int i=m+1; i<img.rows - m; ++i) {
                sumM.at<float>(i, j) = sumM.at<float>(i-1, j) + img.at<uchar>(i + m, j) - img.at<uchar>(i - m - 1, j);
                sumS.at<float>(i, j) = sumS.at<float>(i-1, j) + (float)pow(img.at<uchar>(i + m, j), 2) - (float)pow(img.at<uchar>(i - m - 1, j), 2);
            }
        }
    }

    void preCountRowSum(const Mat &img, vector<float> &sumM, vector<float> &sumS, int n) {
        int rowSize = (n << 1) + 1;
        for(int i=0; i<img.rows; ++i) {
            sumM[i] = 0;
            sumS[i] = 0;
            for(int j=0; j<rowSize; ++j) {
                sumM[i] += img.at<uchar>(i, j);
                sumS[i] += (float)pow(img.at<uchar>(i, j), 2);
            }
        }
    }

    void getMean(const Mat &img, Mat &mean, Mat &sigma, const Mat &sumM, const Mat &sumS, vector<float> &rowSumM, vector<float> &rowSumS,
            int m, int n, int pI, int pJ) {
        if(pI - m < 0 || pI + m >= img.rows || pJ - n < 0 || pJ + n >= img.cols) {
            mean.at<float>(pI, pJ) = -1;
            sigma.at<float>(pI, pJ) = 0;
            return ;
        }
        if(pJ == n && pI > m) {
            return getMeanFromPreviousRow(img, mean, sigma, rowSumM, rowSumS, m, n, pI, pJ);
        } else if(pJ > n) {
            return getMeanFromPreviousCol(img, mean, sigma, sumM, sumS, m, n, pI, pJ);
        }
        float M = 0;
        int total = (2*m + 1) * (2*n + 1);
        for(int i=pI -m; i<= pI + m; ++i) {
            for(int j=pJ - n; j <= pJ + n; ++j) {
                M += img.at<uchar>(i, j);
            }
        }
        M /= total;
        mean.at<float>(pI, pJ) = M;

        float S = 0;
        float diff;
        for(int i=pI -m; i<= pI + m; ++i) {
            for(int j=pJ - n; j <= pJ + n; ++j) {
                diff =  img.at<uchar>(i, j) - M;
                S += (diff * diff);
            }
        }
        S /= total;
        sigma.at<float>(pI, pJ) = S;
    }

    void getMean(const Mat &img, Mat &mean, Mat &sigma, int m, int n){
        Mat colSumM = Mat::zeros(img.rows, img.cols, CV_32FC1);
        Mat colSumS = Mat::zeros(img.rows, img.cols, CV_32FC1);
        preCountColSum(img, colSumM, colSumS, m);

        vector<float> rowSumM;
        vector<float> rowSumS;
        rowSumM.reserve(img.rows);
        rowSumS.reserve(img.rows);
        preCountRowSum(img, rowSumM, rowSumS, n);

        for(int i=0;i<mImg.rows; ++i) {
            //LOGI("line0-row:%d", i);
            for (int j = 0; j < mImg.cols; ++j) {
                getMean(img, mean, sigma, colSumM, colSumS, rowSumM, rowSumS, m, n, i, j);
            }
        }
        int total = (2*m + 1) * (2*n + 1);
        float value;
        for(int i=0;i<mImg.rows; ++i) {
            for (int j = 0; j < mImg.cols; ++j) {
                value = sigma.at<float>(i, j);
                sigma.at<float>(i, j) = sqrt(value);
            }
        }
    }

    float getMean(const Mat &mat) {
        Scalar mean;
        Scalar dev;
        meanStdDev(mat, mean, dev);

        return (float)mean.val[0];
    }


public:
    ImageProcess(const Mat &img) {
        mImg = img;
    }

    void adaptiveContrastEnhancement(int m, int n, int alpha, float maxG = 5) {
        Mat rgbImg;
        cvtColor(mImg, rgbImg, COLOR_RGBA2RGB);
        Mat ycc;
        cvtColor(rgbImg, ycc, COLOR_RGB2YCrCb);
        vector<Mat> channels(3);        //分离通道；
        split(ycc, channels);

        Mat meanMat(mImg.rows, mImg.cols, CV_32FC1);
        Mat sigmaMat(mImg.rows, mImg.cols, CV_32FC1);

       getMean(channels[0], meanMat, sigmaMat, m, n);
        Mat tmp = channels[0].clone();
        float M = getMean(tmp);
        Mat enhance(mImg.rows, mImg.cols, CV_8UC1);

        for(int i=0;i<mImg.rows; ++i) {
            for(int j=0; j<mImg.cols; ++j) {
                if(sigmaMat.at<float>(i, j) < (float)0.01) {
                    enhance.at<uchar>(i, j) = tmp.at<uchar>(i, j);
                } else {
                    float G = alpha * M / sigmaMat.at<float>(i, j);
                    if(G < 1) {
                        G = 1;
                    } else if(G > maxG) {
                        G = maxG;
                    }
                    float result = meanMat.at<float>(i, j) + G * (tmp.at<uchar>(i, j) - meanMat.at<float>(i, j));
                    if(result > 255) {
                        result = 255;
                    } else if(result < 0) {
                        result = 0;
                    }
                    enhance.at<uchar>(i, j) = (uchar)result;
                }
            }
        }

        channels[0] = enhance;   //合并通道，转换颜色空间回到RGB
        merge(channels, ycc);
        cvtColor(ycc, rgbImg, COLOR_YCrCb2RGB);
        cvtColor(rgbImg, mImg, COLOR_RGB2RGBA);
    }
};
#endif //OPENCV_APPLICATION_IMAGEPROCESS_H
