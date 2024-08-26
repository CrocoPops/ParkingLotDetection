#ifndef METRICS_H
#define METRICS_H

#include <opencv2/opencv.hpp>
#include "bbox.h"

float computeIoU(const BBox& box1, const BBox& box2);
float computeAveragePrecision(const std::vector<float>& recalls, const std::vector<float>& precisions);
float computeMAP(const std::vector<BBox>& detections, const std::vector<BBox>& ground_truths, float iouThreshold);

#endif
