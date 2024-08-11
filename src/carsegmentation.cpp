#include "carsegmentation.h"

CarSegmentation::CarSegmentation() {}

CarSegmentation::~CarSegmentation() {}

void CarSegmentation::detectCars(cv::Mat &frame, cv::Mat &mask) {
    cv::Mat parking;


    cv::imshow("Parking", frame);
    cv::imshow("Mask", mask);
    cv::waitKey(0);
}