#include "parkingdetection.h"

ParkingDetection::ParkingDetection(std::vector<BBox> parkings) : parkings(parkings) {}

void ParkingDetection::detect(cv::Mat &frame) {
    cv::Mat parking_gray, parking_bilateral, parking_blurred, parking_laplacian, parking_edges;
    
    // Convert to grayscale
    cv::cvtColor(frame, parking_gray, cv::COLOR_BGR2GRAY);

    // Apply bilateral filter
    cv::bilateralFilter(parking_gray, parking_bilateral, 5, 30, 50);
    // cv::equalizeHist(parking_bilateral, parking_bilateral);
    //cv::GaussianBlur(parking_gray, parking_blurred, cv::Size(3, 3), 0);
    // Apply Laplacian
    cv::Laplacian(parking_bilateral, parking_laplacian, CV_64F, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(parking_laplacian, parking_laplacian);
    cv::threshold(parking_laplacian, parking_laplacian, 100, 255, cv::THRESH_BINARY);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(parking_laplacian, parking_laplacian, cv::MORPH_CLOSE, element);
    //cv::inRange(temp, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 50), temp);
    

    // Apply Canny edge detection
    //cv::Canny(parking_laplacian, parking_edges, 10, 30, 3);

    // Hough Lines Transform on `parking_edges`
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(parking_laplacian, lines, 1, CV_PI/180, 50, 0, 5);

    // Keep the oblique edges
    std::vector<cv::Vec4i> oblique_lines;
    for(const auto& line : lines) {
        double angle = std::atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
        //std::cout << "Angle: " << angle << std::endl;
        //if ((std::abs(angle) > 0 && std::abs(angle) < 180) || (std::abs(angle) > 140 && std::abs(angle) < 180))
        //if(angle > -40 && angle < 20)
            oblique_lines.push_back(line);
    }
    std::cout << "Number of lines: " << oblique_lines.size() << std::endl;

    // Draw the lines on the original frame
    //cv::Mat line_image = frame.clone();
    // cv::Mat line_image = cv::Mat::zeros(frame.size(), CV_8UC3);
    cv::Mat line_image = frame.clone();
    for (const auto& line : oblique_lines)
        cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        
    
    //cv::addWeighted(frame, 0.8, temp, 1, 0, temp);

    // Display the results
    //cv::imshow("Original", frame);
    //cv::imshow("Blurred", parking_blurred);
    //cv::imshow("Gray", parking_gray);
    //cv::imshow("Bilateral", parking_bilateral);
    //cv::imshow("Laplacian", parking_laplacian);
    //cv::imshow("Edges", parking_edges);
    //cv::imshow("Lines", line_image);
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

