// Author: Eddy Frighetto, ID: 2119279

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "bbox.h"

namespace parsers {
    std::vector<BBox> parseParkingXML(const std::string& filename);
}

namespace loaders {
    int loadAllFrames(std::vector<std::vector<cv::Mat>>& output, std::vector<std::vector<std::string>>& dirs);
    int loadAllBackgrounds(std::vector<cv::Mat>& output);
    int loadAllRealMasks(std::vector<std::vector<cv::Mat>>& output);
    int loadAllRealBBoxes(std::vector<std::vector<std::vector<BBox>>>& output);
}

namespace savers {
    void saveImage(cv::Mat image, std::string path);
    void saveMetrics(float mAP, float mIoU, std::string path);
}

#endif
