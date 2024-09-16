// Author: Eddy Frighetto, ID: 2119279

#include "carsegmentation.h"

CarSegmentation::CarSegmentation() {}

CarSegmentation::~CarSegmentation() {}

/**
 * Convert the ground truth mask to a colored mask.
 * 
 * @param frame Original frame.
 * @param mask Ground truth mask.
 * @return Colored mask.
 */
cv::Mat CarSegmentation::detectCarsTrue(cv::Mat &frame, cv::Mat &mask) {
    cv::Mat coloredMask = mask.clone();
    coloredMask.setTo(cv::Scalar(128, 128, 128), mask == 0);
    coloredMask.setTo(cv::Scalar(0, 0, 255), mask == 1);
    coloredMask.setTo(cv::Scalar(0, 255, 0), mask == 2);

    return coloredMask;
}

/**
 * Similarity based on the sum of absolute differences between the frame and the background.
 * 
 * @param frame Current frame.
 * @param background Background to compare with.
 * @return Similarity score.
 */
double CarSegmentation::computeSimilarity(const cv::Mat& frame, const cv::Mat& background) {
    cv::Mat diff;
    // Absolute difference between the frame and the background
    cv::absdiff(frame, background, diff);
    // Sum of the absolute differences between the frame and the background, for each channel
    cv::Scalar sum_diff = cv::sum(diff);
    // Return the sum of the absolute differences of the three channels
    return sum_diff[0] + sum_diff[1] + sum_diff[2];
}

// Return the background that is most similar to the current frame
/**
 * Background that is most similar to the current frame
 * 
 * @param frame Current frame.
 * @param empty_parkings Bucket of empty parkings available.
 * @return Best background.
 */
cv::Mat CarSegmentation::selectClosestBackground(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    double min_diff = std::numeric_limits<double>::max();
    cv::Mat best_background;

    for(const auto& empty_parking : empty_parkings) {
        double diff = computeSimilarity(frame, empty_parking);
        // If the difference is smaller than the minimum difference, update the minimum difference and the best background
        if(diff < min_diff) {
            min_diff = diff;
            best_background = empty_parking;
        }
    }

    return best_background;
}

/**
 * Refine the foreground mask by filtering contours based on area and by shape (aspect ratio).
 * 
 * @param fgMask Foreground mask.
 * @param minArea Minimum area of the contour.
 * @param minAspectRatio Minimum aspect ratio of the contour.
 * @param maxAspectRatio Maximum aspect ratio of the contour.
 * @return Refined mask.
 */
cv::Mat CarSegmentation::refineForegroundMask(const cv::Mat &fgMask, int minArea, double minAspectRatio, double maxAspectRatio) {
    cv::Mat refinedMask = fgMask.clone();

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(refinedMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat finalMask = cv::Mat::zeros(refinedMask.size(), CV_8UC1);

    // Filter contours based on area and by shape (aspect ratio)
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        cv::Rect bbox = cv::boundingRect(contour);
        double aspectRatio = static_cast<double>(bbox.width) / bbox.height;
        if (area > minArea && aspectRatio > minAspectRatio && aspectRatio < maxAspectRatio)
            cv::drawContours(finalMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(finalMask, finalMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(finalMask, finalMask, cv::MORPH_OPEN, kernel);

    return finalMask;
}

/**
 * Detect cars in the frame.
 * 
 * @param image Current frame.
 * @param empty_parkings Bucket of empty parkings available.
 * @return Mask containing the detected cars.
 */
cv::Mat CarSegmentation::detectCars(cv::Mat image, std::vector<cv::Mat> empty_parkings) {
    cv::Mat frame = image.clone();
    // Select the background that is most similar to the current frame from the bucket of empty parkings available
    cv::Mat best_background = selectClosestBackground(frame, empty_parkings);
    cv::Mat frame_gray, best_background_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(best_background, best_background_gray, cv::COLOR_BGR2GRAY);

    cv::Mat kernel3x3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat kernel5x5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat kernel7x7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::Mat kernel9x9 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));

    cv::Mat fgMask, thresh;
    // Absolute difference between the frame and the background
    cv::absdiff(frame_gray, best_background_gray, fgMask);
    cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel5x5);
    // Threshold the absolute difference
    cv::threshold(fgMask, thresh, 50, 255, cv::THRESH_BINARY);
    cv::inRange(fgMask, 50, 255, thresh);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel3x3);
    thresh = refineForegroundMask(thresh, 600, 0.5, 4.0);

    // Exploit the saturation channel to create a mask containing the green areas (trees, grass, etc.)
    // Later we will use this mask to filter out the green areas from the final mask
    cv::Mat background_hsv;
    cv::cvtColor(best_background, background_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> background_hsv_channels;
    cv::split(background_hsv, background_hsv_channels);

    cv::Mat green_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::threshold(background_hsv_channels[1], green_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, kernel5x5);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN, kernel7x7);
    cv::dilate(green_mask, green_mask, kernel9x9);
    
    cv::bitwise_not(green_mask, green_mask);

    // Exploit the Lab color space to detect 
    cv::Mat frame_lab;
    cv::cvtColor(frame, frame_lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels;
    cv::split(frame_lab, lab_channels);

    // Exploit the saturation channel to segment the red cars
    cv::Mat red_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::inRange(lab_channels[1], 140, 255, red_mask);
    cv::dilate(red_mask, red_mask, kernel3x3);
    red_mask = refineForegroundMask(red_mask, 600, 0.5, 4.0);

    // Exploit the lightness channel to segment the reflexes on the cars
    cv::Mat reflex_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::inRange(lab_channels[2], 0, 120, reflex_mask);
    reflex_mask = refineForegroundMask(reflex_mask, 100, 0, 15.0);

    // Combine all the masks
    cv::Mat final_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::bitwise_or(thresh, red_mask, final_mask);
    cv::bitwise_or(final_mask, reflex_mask, final_mask);
    cv::bitwise_and(final_mask, green_mask, final_mask);

    // Exploit the hue channel to segment the black cars
    cv::Mat frame_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(frame_hsv, hsv_channels);

    cv::Mat cars_details = cv::Mat::zeros(frame.size(), CV_8UC3);
    cv::inRange(hsv_channels[0], 80, 130, cars_details);
    cars_details = refineForegroundMask(cars_details, 600, 0.5, 4.0);

    // Connect components in final mask that are attached in black cars
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cars_details, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // On the background subtracted mask some cars details are not segmented really well.
    // We keep the components of cars_details that are attached to the components on the 
    // background subtracted mask
    cv::Mat refined_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int i = 0; i < contours.size(); i++) {
        cv::Mat component_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::drawContours(component_mask, contours, i, cv::Scalar(255), cv::FILLED);

        cv::Mat intersection;
        cv::bitwise_and(component_mask, final_mask, intersection);
        
        if(cv::countNonZero(intersection) > 0 && cv::contourArea(contours[i]) < 3000)
            cv::bitwise_or(refined_mask, component_mask, refined_mask);
    }

    cv::bitwise_or(final_mask, refined_mask, final_mask);

    // On the background subtracted mask the black cars are really influenced by illumination,
    // shadows and street color (cars difficult to segment due to the color of the street).
    // We apply a CLAHE filter to the frame and the background to enhance the contrast and
    // then we apply the same operations as before.
    cv::Mat frame_gray_clahe, best_background_gray_clahe;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->setTilesGridSize(cv::Size(8, 8));
    clahe->apply(frame_gray, frame_gray_clahe);
    clahe->apply(best_background_gray, best_background_gray_clahe);

    cv::Mat fgMask_clahe, thresh_clahe;
    cv::absdiff(frame_gray_clahe, best_background_gray_clahe, fgMask_clahe);
    cv::threshold(fgMask_clahe, thresh_clahe, 70, 255, cv::THRESH_BINARY);
    thresh_clahe = refineForegroundMask(thresh_clahe, 1000, 0.5, 3.0);

    std::vector<std::vector<cv::Point>> contours_clahe;
    cv::findContours(thresh_clahe, contours_clahe, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat final_mask_clahe = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int i = 0; i < contours_clahe.size(); i++) {
        cv::Mat component_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::drawContours(component_mask, contours_clahe, i, cv::Scalar(255), cv::FILLED);

        cv::Mat intersection;
        cv::bitwise_and(component_mask, final_mask, intersection);
        
        if(cv::countNonZero(intersection) > 0 && cv::contourArea(contours_clahe[i]) < 10000)
            cv::bitwise_or(final_mask_clahe, component_mask, final_mask_clahe);
    }

    cv::bitwise_or(final_mask, final_mask_clahe, final_mask);

    final_mask = refineForegroundMask(final_mask, 800, 0.5, 4.0);

    // Apply morphological operations to the final mask in order to fill the holes and remove the noise
    cv::Mat kernel25x25 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25));
    cv::morphologyEx(final_mask, final_mask, cv::MORPH_CLOSE, kernel25x25);

    final_mask = refineForegroundMask(final_mask, 800, 0.5, 4.0);

    return final_mask;
}


/**
 * Classify the cars in the frame.
 * 
 * @param mask Mask containing the detected cars.
 * @param parkings Parking spots.
 * @return Colored mask.
 */
cv::Mat CarSegmentation::classifyCars(cv::Mat &mask, std::vector<BBox> parkings) {
    cv::Mat colored_mask = mask.clone();
    cv::cvtColor(colored_mask, colored_mask, cv::COLOR_GRAY2BGR);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    // Initially, set all components to green
    colored_mask.setTo(cv::Scalar(0, 255, 0), mask == 255);
    
    for(const BBox& parking : parkings) {
        if(!parking.isOccupied()) // If the parking is not occupied go to the next one
            continue;
        
        int x = parking.getX();
        int y = parking.getY();
        int width = parking.getWidth();
        int height = parking.getHeight();
        int angle = parking.getAngle();

        cv::RotatedRect rect = cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(width, height), angle);
        
        cv::Point2f vertices[4];
        rect.points(vertices);
        cv::Point intVertices[4];
        for (int i = 0; i < 4; i++)
            intVertices[i] = cv::Point(static_cast<int>(vertices[i].x), static_cast<int>(vertices[i].y));

        // Create a mask for the parking BBox
        cv::Mat roiMask = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::fillConvexPoly(roiMask, intVertices, 4, cv::Scalar(255));
        // Iterate through each connected component
        for (int i = 1; i < nLabels; i++) {
            cv::Mat componentMask = (labels == i); 
            // Check if there is any intersection between the parking BBox and the component
            cv::Mat intersection;
            cv::bitwise_and(componentMask, roiMask, intersection);
            if (cv::countNonZero(intersection) > 0) {
                // If any part of the component is inside the BBox, set the whole component to red
                colored_mask.setTo(cv::Scalar(0, 0, 255), componentMask);
            }
        }
    }
    return colored_mask;
}