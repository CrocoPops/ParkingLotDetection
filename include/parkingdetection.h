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
        std::vector<BBox> detect(const cv::Mat &frame);
        std::vector<BBox> numberParkings(const std::vector<BBox> parkings);
        void draw(const cv::Mat &frame, const std::vector<BBox> parkings);
};
#endif
