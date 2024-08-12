#include "carsegmentation.h"

CarSegmentation::CarSegmentation() {}

CarSegmentation::~CarSegmentation() {}

void CarSegmentation::detectCarsTrue(cv::Mat &frame, cv::Mat &mask) {
    cv::Mat coloredMask = mask.clone();
    coloredMask.setTo(cv::Scalar(128, 128, 128), mask == 0);
    coloredMask.setTo(cv::Scalar(0, 0, 255), mask == 1);
    coloredMask.setTo(cv::Scalar(0, 255, 0), mask == 2);

    cv::Mat result;
    cv::addWeighted(coloredMask, 0.7, frame, 1, 0, result);

    cv::imshow("Frame", frame);
    cv::imshow("Mask", mask);
    cv::imshow("Colored Mask", coloredMask);
    cv::imshow("Contours", result);
    cv::waitKey(0);
}

void CarSegmentation::detectCars(cv::Mat &frame, cv::Mat empty_parking) {
    cv::Mat frame_hsv, empty_parking_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(empty_parking, empty_parking_hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> frame_channels, empty_parking_channels;
    cv::split(frame_hsv, frame_channels);
    cv::split(empty_parking_hsv, empty_parking_channels);

    cv::Mat frame_eq, empty_parking_eq;
    cv::equalizeHist(frame_channels[2], frame_eq);
    cv::equalizeHist(empty_parking_channels[2], empty_parking_eq);

    frame_channels[2] = frame_eq;
    empty_parking_channels[2] = empty_parking_eq;
    cv::merge(frame_channels, frame_hsv);
    cv::merge(empty_parking_channels, empty_parking_hsv);

    cv::cvtColor(frame_hsv, frame_eq, cv::COLOR_HSV2BGR);
    cv::cvtColor(empty_parking_hsv, empty_parking_eq, cv::COLOR_HSV2BGR);

    cv::Ptr<cv::BackgroundSubtractor> backSub = cv::createBackgroundSubtractorMOG2();
    cv::Mat mask;

    backSub->apply(empty_parking_eq, mask, 0);
    backSub->apply(frame_eq, mask, 1);

    cv::Mat coloredMask;
    cv::cvtColor(mask, coloredMask, cv::COLOR_GRAY2BGR);
    coloredMask.setTo(cv::Scalar(0, 0, 255), mask == 255);

    cv::Mat result;
    cv::addWeighted(frame, 1, coloredMask, 0.7, 0, result);

    cv::imshow("Frame", frame);
    cv::imshow("Empty Parking", empty_parking);
    cv::imshow("Mask", mask);
    cv::imshow("Result", result);
    cv::waitKey(0);
}

/*
void CarSegmentation::detectCars(cv::Mat &frame, cv::Mat &mask, cv::Mat empty_parking) {
    cv::Mat diff, diffGray, subClosed, cars;
        
    // Compute the absolute difference between the current frame and the empty parking frame
    cv::absdiff(frame, empty_parking, diff);

    // Convert the difference image to grayscale
    cv::cvtColor(diff, diffGray, cv::COLOR_BGR2GRAY);

    // Apply a threshold to the grayscale difference image to create a binary mask
    cv::threshold(diffGray, mask, 50, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Apply morphological operations to remove noise and close gaps
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
    element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(33, 33));
    cv::morphologyEx(mask, subClosed, cv::MORPH_CLOSE, element);

    cv::bitwise_and(frame, frame, frame, subClosed);

    // Draw a rectangle around the cars
    cars = frame.clone();

    // Find contours on the closed mask to detect cars
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(subClosed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw rectangles around the detected cars
    for (const auto& contour : contours) {
        cv::Rect boundingBox = cv::boundingRect(contour);
        cv::rectangle(cars, boundingBox, cv::Scalar(0, 255, 0), 2);
    }
    
    // Display the results
    cv::imshow("Frame", frame);
    cv::imshow("Mask", mask);
    cv::imshow("Empty Parking", empty_parking);
    cv::imshow("Difference", diff);
    cv::imshow("Difference Gray", diffGray);
    cv::imshow("Sub Closed", subClosed);
    cv::imshow("Cars", cars);

    cv::waitKey(0);
}*/