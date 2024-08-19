#include "parkingdetection.h"

ParkingDetection::ParkingDetection(std::vector<BBox> parkings) : parkings(parkings) {}

void deleteRegionsBySize(cv::Mat &img, int minSize, int maxSize) {
    // Ensure the image is binary
    cv::Mat binary;
    if (img.channels() > 1) {
        cv::cvtColor(img, binary, cv::COLOR_BGR2GRAY);
        cv::threshold(binary, binary, 128, 255, cv::THRESH_BINARY);
    } else {
        binary = img.clone();
    }

    // Find all contours in the binary image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Iterate over each contour and delete regions that are too big or too small
    for (const auto &contour : contours) {
        double area = cv::contourArea(contour);

        // Check if the area is outside the desired range
        if (area < minSize || area > maxSize) {
            // Draw over the contour with black (0) to delete the region
            cv::drawContours(img, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0), cv::FILLED);
        }
    }
}


void deleteWeakAreas(cv::Mat &img, int numAreas, int numPixels) {
    int rows = img.rows;
    int cols = img.cols;

    // Calculate the size of each square area
    int blockSizeX = cols / numAreas;
    int blockSizeY = rows / numAreas;

    // Iterate over each block
    for (int i = 0; i < numAreas; ++i) {
        for (int j = 0; j < numAreas; ++j) {
            // Define the region of interest (ROI) for the current block
            int startX = j * blockSizeX;
            int startY = i * blockSizeY;

            // Ensure the last block covers the remaining pixels (in case cols/numAreas or rows/numAreas is not a perfect division)
            int width = (j == numAreas - 1) ? cols - startX : blockSizeX;
            int height = (i == numAreas - 1) ? rows - startY : blockSizeY;

            cv::Rect roi(startX, startY, width, height);
            cv::Mat block = img(roi);

            // Draw the rectangle on the original image
            //cv::rectangle(img, roi, cv::Scalar(127), 1); 

            // Count the number of pixels with a value of 255 in the current block
            int count = cv::countNonZero(block == 255);

            // If the count is less than or equal to numPixels, set all pixels in the block to 0
            if (count <= numPixels) {
                img(roi).setTo(cv::Scalar(0));  
            }
        }
    }
}



void ParkingDetection::detect(cv::Mat &frame) {
cv::Mat parking_gray, parking_blurred, parking_laplacian, parking_edges;
    cv::Mat parking;
  
    // Convert to grayscale
    cv::cvtColor(frame, parking, cv::COLOR_BGR2HLS);
    cv::cvtColor(frame, parking, cv::COLOR_BGR2HLS);


   // Extract the L channel
    std::vector<cv::Mat> channels;
    cv::split(parking, channels);
    parking_gray = channels[1];

    cv::imshow("L", parking_gray);

    // Threshold on L
    cv::adaptiveThreshold(parking_gray, parking_gray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 3, 7);
    cv::imshow("Threshold", parking_gray);


    int count = cv::countNonZero(parking_gray);
    std::cout << "Count: " << count << std::endl;
    // Apply Laplacian
    cv::Laplacian(parking_gray, parking_laplacian, CV_64F, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Laplacian(parking_gray, parking_laplacian, CV_64F, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(parking_laplacian, parking_laplacian);
    cv::threshold(parking_laplacian, parking_laplacian, 100, 255, cv::THRESH_BINARY);
   
    
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(parking_laplacian, parking_laplacian, cv::MORPH_CLOSE, element);
    //cv::inRange(temp, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 50), temp);
    cv::imshow("Laplacian", parking_laplacian);

    // Apply Canny edge detection
    //cv::Canny(parking_laplacian, parking_edges, 10, 30, 3);

    // Hough Lines Transform on `parking_edges`
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(parking_laplacian, lines, 1, CV_PI/180, 50, count / 5200, count / 3500);

        
        

    // Draw the lines on the original frame
    cv::Mat line_image = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (const auto& line : lines)
    cv::Mat line_image = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (const auto& line : lines)
        cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        
    
    
    cv::Mat line_filtered = line_image.clone();
    // Delete weak areas
    deleteWeakAreas(line_filtered, 4, count / 5);

    // Delete weak areas 2

    cv::Mat line_filtered2 = line_filtered.clone();
    deleteWeakAreas(line_filtered2, 7, count / 20);

    // bitwise or
    cv::Mat bitwise_or;
    cv::bitwise_or(parking_gray, line_filtered2, bitwise_or);

    // Hough Lines Transform on `line_filtered2`
    std::vector<cv::Vec4i> lines2;
    cv::HoughLinesP(line_filtered2, lines2, 1, CV_PI/180, 50, count / 4000, count / 3000);

    // Draw the lines on the original frame
    cv::Mat line_image2 = frame.clone();
    for (const auto& line : lines2)
        cv::line(line_image2, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

    
    // Delete weak areas 3
    cv::Mat line_filtered3 = line_image2.clone();
    //deleteWeakAreas(line_filtered3, 3, 3000);
   
    // Display the results
    //cv::imshow("Edges", parking_edges);
    cv::imshow("Lines", line_image);
    cv::imshow("Lines filtered", line_filtered);
    cv::imshow("Lines filtered 2", line_filtered2);
    cv::imshow("Bitwise or", bitwise_or);
    cv::imshow("Lines 2", line_image2);
    //cv::imshow("Lines filtered 3", line_filtered3);





   
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

