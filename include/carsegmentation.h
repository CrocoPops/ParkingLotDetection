#ifndef CARSEGMENTATION_H
#define CARSEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "bbox.h"

// Define here the utility functions
class CarSegmentation {
    private:
    public:
        CarSegmentation();
        ~CarSegmentation();
        void detectCars(cv::Mat &frame, cv::Mat &mask);
};
#endif
