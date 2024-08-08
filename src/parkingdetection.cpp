#include "parkingdetection.h"

ParkingDetection::ParkingDetection(std::vector<BBox> parkings) : parkings(parkings) {}

void ParkingDetection::detect(cv::Mat &frame) {
    cv::Mat parking_gray, parking_edges, parking_blurred, parking_bilateral, parking_eq;
    cv::cvtColor(frame, parking_gray, cv::COLOR_BGR2GRAY);

    // CLAHE (Contrast Limited Adaptive Histogram Equalization)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(parking_gray, parking_eq);

    // cv::GaussianBlur(parking_gray, parking_blurred, cv::Size(9, 9), 0);
    cv::bilateralFilter(parking_eq, parking_bilateral, 9, 30, 150, cv::BORDER_DEFAULT);
    std::vector<cv::Point> points = {};
    // cv::equalizeHist(parking_bilateral, parking_bilateral);
    // cv::bilateralFilter(parking_gray, parking_bilateral, 9, 60, 200, cv::BORDER_DEFAULT);
    cv::Canny(parking_bilateral, parking_edges, 50, 125);

    cv::threshold(parking_edges, parking_edges, 250, 255, cv::THRESH_BINARY);

    // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(1, 1));
    // cv::morphologyEx(parking_edges, parking_edges, cv::MORPH_CLOSE, element);

    /*std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(parking_edges, lines, 1, CV_PI/180, 50, 15, 25);

    // Draw the lines
    cv::Mat line_image = cv::Mat::zeros(frame.size(), CV_8UC3);
    for (const auto& line : lines) {
        // Keep the lines with a certain direction
        double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
        // std::cout << "Angle: " << angle << std::endl;
        //if((angle >= -65 && angle <= 0) || (angle >= 0 && angle <= 100)) 
        cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }*/

    cv::imshow("Original", frame); 
    cv::imshow("Histogram equalized", parking_eq);
    cv::imshow("Bilateral filter", parking_bilateral);
    cv::imshow("Edges", parking_edges);
    // cv::imshow("Final Result", line_image);
    cv::waitKey(0);
}

std::vector<BBox> ParkingDetection::getParkings(cv::Mat frame) {
    // TODO: Implement this method
    return parkings;
}

int ParkingDetection::numberParkings() {
    return parkings.size();
}

void ParkingDetection::draw(cv::Mat &frame) {
    // TODO: Implement this method
}

