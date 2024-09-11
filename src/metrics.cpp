#include "metrics.h"

float computeIoU(const BBox& box1, const BBox& box2) {
    cv::RotatedRect rect1 = box1.getRotatedRect();
    cv::RotatedRect rect2 = box2.getRotatedRect();

    // Calculating intersection
    std::vector<cv::Point2f> inter_pts;
    int inter = cv::rotatedRectangleIntersection(rect1, rect2, inter_pts);

    if (inter == cv::INTERSECT_NONE)
        return 0.0f;  // No intersection
    
    double bbox_intersection = cv::contourArea(inter_pts);

    // Calculating union
    double area1 = rect1.size.area();
    double area2 = rect2.size.area();
    double bbox_union = (area1 + area2 - bbox_intersection);

    return bbox_intersection / bbox_union;
}

float computeIoU(const cv::Mat mask, const cv::Mat ground_truth) {
    cv::Mat inter_mask = mask & ground_truth;
    cv::Mat union_mask = mask | ground_truth;

    if (cv::countNonZero(union_mask) == 0)
        return 1.0f;

    int inter_area = cv::countNonZero(inter_mask);
    int union_area = cv::countNonZero(union_mask);

    return (float)inter_area / union_area;
}

float computeAveragePrecision(const std::vector<float>& recalls, const std::vector<float>& precisions) {

    // Since I want to calculate the area under the precision-recall curve, I will modify the vectors
    // so that the area is close for sure
    std::vector<float> modified_precisions(precisions);
    std::vector<float> modified_recalls(recalls);

    modified_recalls.insert(modified_recalls.begin(), 0.0f);
    modified_recalls.push_back(1.0f);
    modified_precisions.insert(modified_precisions.begin(), 0.0f);
    modified_precisions.push_back(0.0f);

    // Making the precision curve decreasing
    for (int i = modified_precisions.size() - 2; i >= 0; --i)
        modified_precisions[i] = std::max(modified_precisions[i], modified_precisions[i + 1]);

    // Calculating the effective AP (area below precision-recall curve)
    // in other words im calculating area of rectangles with base recall[i] - recall[i-1] and height precision[i]
    float ap = 0.0f;
    for (size_t i = 1; i < modified_recalls.size(); ++i) {
        if (modified_recalls[i] != modified_recalls[i - 1]) {
            ap += (modified_recalls[i] - modified_recalls[i - 1]) * modified_precisions[i];
        }
    }

    return ap;
}


float computeMAP(const std::vector<BBox>& detections, const std::vector<BBox>& ground_truths, float iouThreshold = 0.5) {
    int num_parkings = ground_truths.size();

    // I need a std::vector since it is cumulative
    std::vector<int> true_positives;

    for (const auto& detected_parking : detections) {
        float best_iou = 0.0f;

        for (const auto& ground_truth_parking : ground_truths) {

            // Compute intersection over union (in this case for each detected-real parking space)
            float iou = computeIoU(detected_parking, ground_truth_parking);

            // Since I have multiple parking space detections, I need to find the one
            // with the largest IoU to match the parking that I tried to detect
            if (iou > best_iou)
                best_iou = iou;

        }

        // Now that I found the best IoU, I can compare it with the threshold
        // This will append 0 if below threshold, 1 otherwise
        true_positives.push_back(best_iou > iouThreshold);
    }

    // If true_positives is empty, it means that we have no true positive so our mAP is 0
    if (true_positives.empty())
        return 0.0f;

    // Calculating the precisions and recalls
    int cum_true_positives = 0;
    std::vector<float> precisions, recalls;
    for (size_t i = 0; i < true_positives.size(); ++i) {
        cum_true_positives += true_positives[i];
        precisions.push_back((float)cum_true_positives / (i + 1));
        recalls.push_back((float)cum_true_positives / num_parkings);
    }

    // Calculating the AP
    float ap = computeAveragePrecision(recalls, precisions);
    return ap;
}
