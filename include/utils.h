#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "bbox.h"


// Define here the utility functions
void drawRotatedRectangle(cv::Mat& image, cv::RotatedRect rect, bool occupied, bool filled);
std::vector<BBox> parseParkingXML(const std::string& filename);

#endif
