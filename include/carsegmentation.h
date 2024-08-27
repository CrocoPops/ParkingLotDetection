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
        void regionGrowing(cv::Mat frame, cv::Mat mask, cv::Mat &result, int threshold);
    public:
        CarSegmentation();
        ~CarSegmentation();
        cv::Mat detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings);
        cv::Mat detectCarsTrue(cv::Mat &frame, cv::Mat &mask);
};
#endif
