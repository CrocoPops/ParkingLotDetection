#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "bbox.h"

namespace parsers {
    std::vector<BBox> parseParkingXML(const std::string& filename);
}

namespace loaders {
    int loadAllFrames(std::vector<std::vector<cv::Mat>>& output);
    int loadAllBackgrounds(std::vector<cv::Mat>& output);
    int loadAllRealMasks(std::vector<std::vector<cv::Mat>>& output);
    int loadAllRealBBoxes(std::vector<std::vector<std::vector<BBox>>>& output);
}

#endif
