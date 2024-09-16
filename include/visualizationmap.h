// Author: Eddy Frighetto, ID: 2119279

#ifndef VISUALIZATIONMAP_H
#define VISUALIZATIONMAP_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "bbox.h"
#include "utils.h"

class VisualizationMap {
    private:
        std::vector<std::vector<cv::Point2f>> bboxes;
        void addBBox(std::vector<cv::Point2f> points);
        std::vector<std::vector<cv::Point2f>> getBBoxes();
        void colorBBoxes(cv::Mat &map, std::vector<BBox> bboxes);
    public:
        cv::Mat drawParkingMap(cv::Mat &frame, std::vector<BBox> bboxes);
};
#endif
