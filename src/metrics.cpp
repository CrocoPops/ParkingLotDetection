// Author: Davide Ferrari, ID: 2122542

#include "metrics.h"

/**
 * Given two bounding boxes, return their insersection over union.
 * 
 * @param box1 First boundign box.
 * @param box2 Second bounding box.
 * @return Intersection over union (float).
 */
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

/**
 * Given a target color and a mask, return a new binary mask where only that color in the mask is white and other things are black.
 * 
 * @param mask Input mask.
 * @param color Target color.
 * @return Binary mask that is white only where the input mask is target color.
 */
cv::Mat convertMask(const cv::Mat mask, const cv::Scalar color) {

    cv::Mat new_mask;
    cv::inRange(mask, color, color, new_mask);

    return new_mask;
}

/**
 * Given two mask, return their mean intersection over union (mIoU).
 * 
 * @param mask Detected mask.
 * @param ground_truth Real mask.
 * @return mIoU score.
 */
float computeMIoU(const cv::Mat mask, const cv::Mat ground_truth) {

    // Per ogni classe, vogliamo il bianco dove ci importa

    float IoUs = 0;

    for (int i = 0; i < 3; i++) // Three classes (including background)
    {

        cv::Mat masked;
        cv::Mat masked_real;

        if(i == 0) { // Black
            masked = convertMask(mask, cv::Scalar(0,0,0));
            masked_real = convertMask(ground_truth, cv::Scalar(0,0,0));
        }
        else if (i == 1) // Green
        {
            masked = convertMask(mask, cv::Scalar(0, 255, 0));
            masked_real = convertMask(ground_truth, cv::Scalar(0, 255, 0));
        } else { // Red
            masked = convertMask(mask, cv::Scalar(0, 0, 255));
            masked_real = convertMask(ground_truth, cv::Scalar(0, 0, 255));

        }

        cv::Mat inter_mask = masked & masked_real;
        cv::Mat union_mask = masked | masked_real;

        if (cv::countNonZero(union_mask) == 0) {
            IoUs += 1.0f;
            continue;
        }

        int inter_area = cv::countNonZero(inter_mask);
        int union_area = cv::countNonZero(union_mask);

        IoUs += (float) inter_area / union_area;
    }
    
    return IoUs / 3.0f;
}

/**
 * Perform 11-interpolation on precision-recall curve.
 * 
 * @param recalls Vector of recalls.
 * @param precisions Vector of precisions.
 * @return Average precision obtained by 11-interpolation.
 */
float computeAveragePrecision(const std::vector<float>& recalls, const std::vector<float>& precisions) {

    // Define the 11-point recall levels
    std::vector<float> recall_levels = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    float ap = 0.0f;

    // Iterate over each recall level and find the maximum precision for that level
    for (float recall_level : recall_levels) {
        float max_precision = 0.0f;

        // Find the maximum precision where recall >= recall_level
        for (size_t i = 0; i < recalls.size(); ++i) {
            if (recalls[i] >= recall_level) {
                max_precision = std::max(max_precision, precisions[i]);
            }
        }

        // Accumulate the precision values for each recall level
        ap += max_precision;
    }

    // Average the precision over the 11 recall levels
    ap /= recall_levels.size();

    return ap;
}

/**
 * Given two vectors of bounding boxes and a threshold, return the mAP score.
 * 
 * @param detections Detected bounding boxes.
 * @param ground_truths Real bounding boxes.
 * @param iouThreshold Threshold for the IoU to be considered true positive or false positive.
 * @return mAP score.
 */
float computeMAP(const std::vector<BBox>& detections, const std::vector<BBox>& ground_truths, float iouThreshold = 0.5) {

    float aps = 0.0;

    for (int i = 0; i <= 1; i++) {
        std::vector<BBox> class_detected;
        std::vector<BBox> class_real;

        for(const BBox& detected : detections) {
            if(detected.isOccupied() == i) {
                class_detected.push_back(detected);
            }
        }

        
        for(const BBox& real : ground_truths) {
            if(real.isOccupied() == i) {
                class_real.push_back(real);
            }
        }


        int num_parkings = class_real.size();
        
        // I need a std::vector since it is cumulative
        std::vector<int> true_positives;

        for (const auto& detected_parking : class_detected) {
            float best_iou = 0.0f;

            for (const auto& ground_truth_parking : class_real) {

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
            aps += 1.0f;

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

        aps += ap;

    }
    
    return (float) aps / 2.0f;
}
