#ifndef CARSEGMENTATION_H
#define CARSEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <stack>
#include "bbox.h"

// Define here the utility functions
class CarSegmentation {
    private:
        cv::Ptr<cv::BackgroundSubtractor> backSub;
        void regionGrowing(cv::Mat frame, cv::Mat mask, cv::Mat &result, int threshold);
    public:
        CarSegmentation();
        ~CarSegmentation();
        //void computeHOG(cv::Mat &frame, cv::HOGDescriptor &hog, std::vector<float> &descriptors);
        //void trainSVM(const cv::Mat &trainingData, const cv::Mat &labels);
        void trainBackgroundSubtractor(std::vector<cv::Mat> empty_parkings);
        cv::Mat detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings);
        cv::Mat detectCarsTrue(cv::Mat &frame, cv::Mat &mask);
        cv::Mat classifyCars(cv::Mat &frame, std::vector<BBox> parkings);
};
#endif
