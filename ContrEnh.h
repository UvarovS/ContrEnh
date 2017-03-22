
#include <windows.h>
#include <iostream>
#include <fstream>
#include <conio.h>
#include <chrono>

#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;



class ContrEnh
{
        void ReadSettings(string FileName);
        void CreateGaborKernels();
        cv::Mat MyContrastEnhancement(cv::Mat GrayIm);
        cv::Mat ProcessIntensityImage(const cv::Mat Im);
        cv::Mat ProcessColorImage(const cv::Mat Im);
        cv::Mat ProcessGrayscaleImage(const cv::Mat ImGray);

        int AlgType = 0;
        double ThreshScale;
        double SoftMaxScale;
        vector<cv::Mat> GabKerBank;
    public:
        //ContrEnh(string FileName);
        ContrEnh::ContrEnh()
        {
            ReadSettings("settings.cfg");
            CreateGaborKernels();
            //AlgType = 2;
        }
        cv::Mat ProcessImage(cv::Mat Im);
        void ProcessImageFile(string FileName);
        void ProcessDB();
        double GetProcTime();
        void DrawHist(const cv::Mat GrayImg, string WndName);
};