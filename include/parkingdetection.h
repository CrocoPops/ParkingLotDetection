#ifndef PARKINGDETECTION_H
#define PARKINGDETECTION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "bbox.h"

class ParkingDetection {
    private:
        std::vector<BBox> parkings;
    public:
        ParkingDetection(std::vector<BBox> parkings = {});
        void detect(cv::Mat &frame);
        int numberParkings();
        void draw(cv::Mat &frame);
        std::vector<BBox> getParkings();
};
#endif
