// ContrEnh_1.cpp : Defines the entry point for the console application.
//
#include "ContrEnh.h"

void ContrEnh::ReadSettings(string FileName)
{
    ifstream CfgFile(FileName);
    string line;
    if (CfgFile.is_open())
    {
        while (getline(CfgFile, line))
        {
            cout << line << endl;
            if (line.find("Threshold scale:") != std::string::npos)
            {
                size_t pos = line.find(":");
                string tst = line.substr(pos + 1);
                ThreshScale = atof(tst.c_str());
            }
            if (line.find("SoftMax scale:") != std::string::npos)
            {
                size_t pos = line.find(":");
                string tst = line.substr(pos + 1);
                SoftMaxScale = atof(tst.c_str());
            }
            if (line.find("Algorithm") != std::string::npos)
            {
                size_t pos = line.find(":");
                string tst = line.substr(pos + 1);
                AlgType = atoi(tst.c_str());
            }
        }
    }
    else
    {
        cout << "ERROR: Can't open file 'settings.cfg' \n";
        cout << "Press any key to exit \n";
        _getch();
        exit(-1);
    }
}

void ContrEnh::DrawHist(const cv::Mat GrayImg, string WndName)
{
    int bins = 256;
    int histSize[] = { bins };
    float sranges[] = { 0, 256 };
    const float* ranges[] = { sranges };
    MatND hist;
    int channels[] = { 0 };
    calcHist(&GrayImg, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
    double maxVal = 0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);

    Mat histImg = Mat::zeros(bins, bins, CV_8UC1);
    float h255 = hist.at<float>(255);
    float h254 = hist.at<float>(254);
    float h253 = hist.at<float>(253);
    float h252 = hist.at<float>(252);

    for (int ii = 0; ii < bins; ii++)
    {
        float binVal = hist.at<float>(ii);
        int binHeight = cvRound(binVal * 255 / maxVal);
        histImg.at<unsigned char>(255 - binHeight, ii) = 255;
    }
    namedWindow(WndName, cv::WINDOW_NORMAL);
    cv::resizeWindow(WndName, 256, 256);
    imshow(WndName, histImg);
}

void ContrEnh::CreateGaborKernels()
{
    int ksize = 8; 
    double sigma = 5;
    double lambd = 16;
    double gamma = 2; 
    double psi = CV_PI*0.5;

    int levs = 3;
    int NumOfOrient = 4;
    double AngStep = 180.0 / (double)NumOfOrient;
    for (int ii = 0; ii < NumOfOrient; ii++)
    {
        double theta = (double)ii*AngStep;
        GabKerBank.push_back(getGaborKernel(cv::Size(ksize, ksize), sigma, theta, lambd, gamma, psi, CV_32F));
    }
    ksize = 8;
    sigma = 5;
    lambd = 8; 
    gamma = 2; 
    psi = 0;
    for (int ii = 0; ii < NumOfOrient; ii++)
    {        
        double theta = (double)ii*AngStep;
        GabKerBank.push_back(getGaborKernel(cv::Size(ksize, ksize), sigma, theta, lambd, gamma, psi, CV_32F));
    }
}

cv::Mat ContrEnh::MyContrastEnhancement(cv::Mat GrayIm)
{
    // CLAHE is used only to create sharpenning mask
    cv::Mat ImClahe;
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(GrayIm, ImClahe);
    clahe->collectGarbage();

    cv::Mat ImClahe_f;
    ImClahe.convertTo(ImClahe_f, CV_32F, 1.0 / 255, 0);

    //cv::namedWindow("Gabor Kernel", cv::WINDOW_NORMAL);
    //cv::resizeWindow("Gabor Kernel", 256, 256);

    //imshow("Gabor Kernel", GabKerBank[1]);
    // bank of Gabor filters to find features like "border curves"/lines
    cv::Mat MaxGabRes;
    cv::Mat GrayIm_f;
    GrayIm.convertTo(GrayIm_f, CV_32F, 1.0 / 255, 0);
    cv::filter2D(ImClahe_f, MaxGabRes, CV_32F, GabKerBank[0]);

    for (int ii = 1; ii < GabKerBank.size(); ii++)
    {
        cv::Mat ResIm;
        cv::filter2D(ImClahe_f - cv::mean(ImClahe_f), ResIm, -1, GabKerBank[ii]);
        ResIm = cv::abs(ResIm);
        cv::max(ResIm, MaxGabRes, MaxGabRes);
    }

    double minVal = 0;
    double maxVal = 0;
    minMaxLoc(MaxGabRes, &minVal, &maxVal, 0, 0);
    MaxGabRes = (MaxGabRes - minVal);

    cv::Mat MaxGabRes8U;
    MaxGabRes.convertTo(MaxGabRes8U, CV_8U, 255.0);
    cv::Mat ThreshGab;
    // auto threshold to distingush "what is border" and wat is not
    // only value of threshold is used
    double OtsuThr = cv::threshold(MaxGabRes8U, ThreshGab, 0, 255, THRESH_BINARY | THRESH_OTSU);
    OtsuThr /= 255.0;

    cv::Mat Mask;
    // softmax to control "softness/hardness of mask"
    cv::exp(-SoftMaxScale*(MaxGabRes - ThreshScale*OtsuThr), Mask);
    Mask = 1.0 / (1 + Mask);
    // mask bluring
    boxFilter(Mask, Mask, CV_32F, cv::Size(5,5));
    MaxGabRes = Mask;

    //cv::namedWindow("Gabor Filtered", cv::WINDOW_NORMAL);
    //cv::resizeWindow("Gabor Filtered", MaxGabRes.size().width, MaxGabRes.size().height);
    //cv::imshow("Gabor Filtered", Mask);

    //unsharp masking to create "high contrast image"
    cv::Mat BluredIm;
    int ksizeG = 32;
    double sigmaG = 5.0;
    double coe = 0.5;
    cv::Mat GaussKer = getGaussianKernel(ksizeG, sigmaG, CV_32F);
    cv::filter2D(GrayIm_f, BluredIm, -1, GaussKer);
    cv::Mat HighContrIm = GrayIm_f + coe*(GrayIm_f - BluredIm);

    //soft merging of original and high contrast images, correspondind to mask
    cv::Mat Part1;
    multiply(HighContrIm, MaxGabRes, Part1);
    cv::Mat Part2;
    multiply(GrayIm_f, (1.0 - MaxGabRes), Part2);
    cv::Mat ResIm = Part1 + Part2;
    minMaxLoc(ResIm, &minVal, &maxVal, 0, 0);
    ResIm.convertTo(ResIm, CV_8U, 255.0);
    return ResIm;
}

cv::Mat ContrEnh::ProcessIntensityImage(const cv::Mat Im)
{
    double minVal = 0;
    double maxVal = 0;
    minMaxLoc(Im, &minVal, &maxVal, 0, 0);
    cv::Mat ImX;
    ImX = Im;
    cv::Mat ResIm;
    switch (AlgType)
    {
    case 0:
    {
              cv::equalizeHist(ImX, ResIm);
    }
        break;
    case 1:
    {
              Ptr<CLAHE> clahe = createCLAHE();
              clahe->setClipLimit(2.0);
              clahe->apply(ImX, ResIm);
    }
        break;
    case 2:
    {
              ResIm = MyContrastEnhancement(ImX);
    }
        break;
    default:
        break;
    }

    //DrawHist(ImX, "Original Histogram");
    //DrawHist(ResIm, "Resulting Histogram");

    return ResIm;
}

cv::Mat ContrEnh::ProcessColorImage(const cv::Mat Im)
{
    cv::Mat ResIm;
    cv::Mat ImYUV;
    cv::cvtColor(Im, ImYUV, CV_BGR2Lab);
    vector<cv::Mat> spl;
    split(ImYUV, spl);
    spl[0] = ProcessIntensityImage(spl[0]);
    merge(spl, ResIm);
    cv::cvtColor(ResIm, ResIm, CV_Lab2BGR);
    return ResIm;
}

cv::Mat ContrEnh::ProcessGrayscaleImage(const cv::Mat ImGray)
{
    cv::Mat ResIm;
    ResIm = ProcessIntensityImage(ImGray);
    return ResIm;
}

cv::Mat ContrEnh::ProcessImage(cv::Mat Im)
{
    if (Im.channels() == 1)
        return ProcessGrayscaleImage(Im);
    if (Im.channels() >= 3)
        return ProcessColorImage(Im);

    cv::Mat Empty;
    return Empty;
}

void ContrEnh::ProcessImageFile(string FileName)
{
    cv::Mat image;
    image = cv::imread(FileName);
    if (image.empty())
    {
        cout << "ERROR: can't read image file '" << FileName << "'" << endl;
        cout << "Press any key to continue" << endl;
        _getch();
        return;
    }
    if ((image.size().width < 32) || (image.size().height < 32))
    {        
        cout << "ERROR: image '" << FileName << "'" << "is too small" << endl;
        cout << "Press any key to continue" << endl;
        _getch();
        return;
    }
    cv::namedWindow("Original window", cv::WINDOW_NORMAL);
    cv::namedWindow("Result 1 window", cv::WINDOW_NORMAL);
    //cv::namedWindow("Original window", cv::WINDOW_AUTOSIZE);
    //cv::namedWindow("Result 1 window", cv::WINDOW_AUTOSIZE);
    
    //cv::namedWindow("Difference window", cv::WINDOW_NORMAL);

    int WindSize = 1024;

    int MaxLinSize = max(image.size().width, image.size().height);
    double scale = 1.0;
    if (MaxLinSize > WindSize)
        scale = (double)WindSize / (double)MaxLinSize;

    int width = (int)(scale*image.size().width);
    int height = (int)(scale*image.size().height);

    //resize(image, image, cv::Size(width, height));

    cv::resizeWindow("Original window", width, height);
    cv::resizeWindow("Result 1 window", width, height);

    //cv::resizeWindow("Difference 1 window", width, height);

    //-------------- For grayscale result ------------//
    //cv::Mat ImGray;
    //cv::cvtColor(image, ImGray, CV_BGR2GRAY);
    //cv::imshow("Original window", ImGray);
    //cv::Mat ColImHistEq = ProcessGrayscaleImage(ImGray);
    //cv::imshow("Result 1 window", ColImHistEq);

    //-------------- For color result (if original is..) ------------//
    cv::imshow("Original window", image);
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    cv::Mat ColImHistEq = ProcessImage(image);
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
    cout << FileName << endl;
    cout << "Image size: " << image.size() << " Proc. time " << duration << " ms" << endl;
    cv::imshow("Result 1 window", ColImHistEq);
    //cv::imshow("Difference 1 window", 2*abs(image - ColImHistEq));

    cv::imwrite("Results/Enh_" + FileName, ColImHistEq);
}

int main(int argc, char** argv)
{
    if (argc <2)
    {
        cout << " Usage: " << endl;
        cout << "1) 'image_file' to process single image" << endl;
        cout << "2) '-L list_file' to process files from list " << endl;
        cout << "    list_file' - text file, each line is full name of file" << endl;
        cout << "Press any key to exit \n" << endl;
        _getch();
        return 0;
    }
    ContrEnh ContrEnhObj;
    if (argc == 2)
    {
        ContrEnhObj.ProcessImageFile(argv[1]);
        cv::waitKey(0);
        return 0;
    }
    
    if (argc == 3)
    {
        if (string(argv[1]) != "-L")
        {
            cout << "ERROR: wrong command line argument" << endl;
            cout << "Press any key to exit \n" << endl;
            _getch();
            return 0;
        }
        ifstream SampleList(argv[2]);
        string line;
        if (SampleList.is_open())
        {
            while (getline(SampleList, line))
            {
                ContrEnhObj.ProcessImageFile(line);
                char ch = cv::waitKey(0);
                if (ch == 27)
                   return 0;
            }
        }
        else
        {
            cout << "ERROR: unable to open file '" << argv[2] << "'" << endl;
            cout << "Press any key to exit \n" << endl;
            _getch();
        }
    }
    cv::waitKey(0);
    return 0;
}
