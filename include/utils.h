#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "tinyxml2.h"
#include "bbox.h"

using namespace cv;
using namespace std;
using namespace tinyxml2;

// Define here the utility functions
void drawRotatedRectangle(Mat& image, RotatedRect rect, bool occupied);
BBox parseBBox(XMLElement* space);
vector<BBox> parseParkingXML(const string& filename);

#endif
