// Author: Eddy Frighetto, ID: 2119279

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
        double computeSimilarity(const cv::Mat& frame, const cv::Mat& background);
        cv::Mat selectClosestBackground(cv::Mat &frame, std::vector<cv::Mat> empty_parkings);
        cv::Mat refineForegroundMask(const cv::Mat &fgMask, int minArea, double minAspectRatio, double maxAspectRatio);
    public:
        CarSegmentation();
        ~CarSegmentation();
        void trainBackgroundSubtractor(std::vector<cv::Mat> empty_parkings);
        cv::Mat detectCars(cv::Mat frame, std::vector<cv::Mat> empty_parkings);
        cv::Mat detectCarsTrue(cv::Mat &frame, cv::Mat &mask);
        cv::Mat classifyCars(cv::Mat &frame, std::vector<BBox> parkings);
};
#endif
