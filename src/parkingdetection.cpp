#include "parkingdetection.h"
#include <opencv2/ximgproc.hpp>

ParkingDetection::ParkingDetection(std::vector<BBox> parkings) : parkings(parkings) {}


// Remove the isolated pixels that are far from the rest of the pixels
// neighbrhoodSize: size of the neighborhood to consider (rectangular)
// minPixels: minimum number of white pixels in the neighborhood to keep the pixel white

void removeIsolatedPixels(cv::Mat &img, int neighborhoodSize, int minPixels) {
    // Ensure the neighborhood size is odd to have a center pixel
    if (neighborhoodSize % 2 == 0) {
        neighborhoodSize += 1;
    }
    
    // Create a copy of the original image to modify
    cv::Mat output = img.clone();
    
    int offset = neighborhoodSize / 2;

    for (int y = offset; y < img.rows - offset; ++y) {
        for (int x = offset; x < img.cols - offset; ++x) {
            if (img.at<uchar>(y, x) == 255) { // Only consider white pixels
                // Define the neighborhood
                cv::Rect neighborhood(x - offset, y - offset, neighborhoodSize, neighborhoodSize);
                cv::Mat roi = img(neighborhood);

                // Count the number of white pixels in the neighborhood
                int whiteCount = cv::countNonZero(roi);

                // If the number of white pixels is less than the threshold, set the pixel to black
                if (whiteCount < minPixels) {
                    output.at<uchar>(y, x) = 0;
                }
            }
        }
    }

    img = output;
}

// Based on the region's area, delete the region if the area is between minSize and maxSize

void deleteAreasInRange(cv::Mat &img, int minSize, int maxSize) {
    // Ensure the image is binary
    if (img.type() != CV_8UC1) {
        std::cerr << "Image must be binary (CV_8UC1)" << std::endl;
        return;
    }

    // Find connected components
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(img, labels, stats, centroids);

    // Iterate over each connected component
    for (int i = 1; i < numComponents; ++i) { // Start from 1 to skip the background
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // If the area is between minSize and maxSize, delete it
        if (area >= minSize && area <= maxSize) {
            for (int y = 0; y < labels.rows; ++y) {
                for (int x = 0; x < labels.cols; ++x) {
                    if (labels.at<int>(y, x) == i) {
                        img.at<uchar>(y, x) = 0;
                    }
                }
            }
        }
    }
}

std::vector<cv::Vec4i> closest_neighbor_line(cv::Mat corners) {
    std::vector<cv::Vec4i> lines;
    
    // Iterate over all pixels in the image
    for(int i = 0; i < corners.rows; i++) {
        for(int j = 0; j < corners.cols; j++) {
            if(corners.at<uchar>(i, j) == 255) {
                // Found a white pixel, now search for the closest neighbor
                bool found = false;
                int radius = 1;
                
                while(!found) {
                    // Expand the search radius until a point is found
                    for(int k = i - radius; k <= i + radius; k++) {
                        for(int l = j - radius; l <= j + radius; l++) {
                            if(k >= 0 && k < corners.rows && l >= 0 && l < corners.cols) {
                                if(corners.at<uchar>(k, l) == 255 && !(k == i && l == j)) {
                                    // Found a different white pixel, create a line
                                    lines.push_back(cv::Vec4i(j, i, l, k));
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if(found) break;
                    }
                    radius++;
                }
            }
        }
    }
    return lines;
}


void enhanceWeakPointsNearStrongOnes(cv:: Mat &img, int neighborhoodSize, int minNumStrongPixels, int threshold) {
    
    // check if the threshold is valid
    if (threshold < 0 || threshold > 255) {
        std::cerr << "Threshold must be between 0 and 255" << std::endl;
        return;
    }
    
    
    cv::Mat output = cv::Mat::zeros(img.size(), CV_8UC1);

    int offset = neighborhoodSize / 2;

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (img.at<uchar>(y, x) != 0) { 
                int strongCount = 0;
                for(int i = -offset; i <= offset; i++) {
                    for(int j = -offset; j <= offset; j++) {
                        if (i == 0 && j == 0) continue;
                        if (y + i >= 0 && y + i < img.rows && x + j >= 0 && x + j < img.cols) {
                            if (img.at<uchar>(y + i, x + j) >= threshold)
                                strongCount++;
                        }
                    }
                }
                if (strongCount >= minNumStrongPixels || img.at<uchar>(y, x) >= threshold) {
                    output.at<uchar>(y, x) = 255;
                }
            }
        }
    }
    img = output;
}
                                    

/*
void ParkingDetection::detect(cv::Mat &frame) {
    cv::Mat frame_HSV;
    cv::cvtColor(frame, frame_HSV, cv::COLOR_BGR2HSV);
    
    // Divide the frame into its components
    cv::Mat frame_H, frame_S, frame_V;
    std::vector<cv::Mat> channels;
    cv::split(frame_HSV, channels);
    frame_H = channels[0];
    frame_S = channels[1];
    frame_V = channels[2];

    
    //Threshold on V
    cv::Mat frame_V_TH;
    cv::adaptiveThreshold(frame_V, frame_V_TH, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 5, 7);
    cv::imshow("Threshold V", frame_V_TH);




    // Apply Laplacian
    cv::Mat parking_laplacian;
    cv::Laplacian(frame_V_TH, parking_laplacian, CV_64F, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(parking_laplacian, parking_laplacian);
    cv::threshold(parking_laplacian, parking_laplacian, 100, 255, cv::THRESH_BINARY);
    
    // Closing
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(parking_laplacian, parking_laplacian, cv::MORPH_CLOSE, element);
    


    //Hough Lines Transform `
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(parking_laplacian, lines, 1, CV_PI/180, 50, 6, 10);

    // Draw the lines
    cv::Mat line_image = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (const auto& line : lines)
        cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

    cv::imshow("Lines", line_image);
    
    // Threshold on H and S for rexternal objects
    cv::Mat frame_H_TH, frame_S_TH;
    cv::threshold(frame_H, frame_H_TH, 30, 255, cv::THRESH_BINARY_INV);
    cv::threshold(frame_S, frame_S_TH, 150, 255, cv::THRESH_BINARY_INV);
    cv::imshow("Threshold H", frame_H_TH);
    cv::imshow("Threshold S", frame_S_TH);

    // bitwise and 
    cv::Mat bitwise_and;
    cv::bitwise_and(frame_H_TH, frame_S_TH, bitwise_and);
    cv::bitwise_and(line_image, bitwise_and, bitwise_and);
    cv::imshow("Bitwise and", bitwise_and);

    element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(bitwise_and, bitwise_and, element);


    // Hough Lines Transform on `bitwise_and`
    lines.clear();
    cv::HoughLinesP(bitwise_and, lines, 1, CV_PI/180, 50, 6, 10);

    // Draw the lines 
    line_image = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (const auto& line : lines)
        cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

    cv::imshow("Lines", line_image);

    //MSER
    cv::Ptr<cv::MSER> mser = cv::MSER::create();
    mser->setMinArea(40);
    mser->setMaxArea(600);
    std::vector<std::vector<cv::Point>> regions;
    std::vector<cv::Rect> bboxes;
    mser->detectRegions(frame_V, regions, bboxes);
    cv::Mat frame_mser = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (const auto& region : regions) {
        for (const auto& point : region) {
            frame_mser.at<uchar>(point) = 255;
        }
    }
    cv::imshow("MSER", frame_mser);

    // Filter MSER removing isolated pixels
    removeIsolatedPixels(frame_mser, 200, 400);
    cv::imshow("Filtered MSER", frame_mser);

    // dilate the filtered mask
    element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(70, 70));
    cv::dilate(frame_mser, frame_mser, element);
    cv::imshow("Dilated MSER", frame_mser);

    // Delete weak areas
    deleteAreasInRange(frame_mser, 5000, 100000);
    cv::imshow("Filtered MSER 2", frame_mser);

    // Bitwise and for having a more detailed mask without external objects
    cv::bitwise_and(frame_mser, line_image, frame_mser);
    element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(frame_mser, frame_mser, element);
    cv::imshow("Bitwise and 2", frame_mser);


    // Find corners using as mask the filtered MSER
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(frame_V, corners, 500, 0.0001, 10, frame_mser, 1);
    // Create an empty matrix to store the corner points
    cv::Mat frame_corner = cv::Mat::zeros(frame.size(), CV_8UC1);

    // Mark each corner with a single pixel
    for (size_t i = 0; i < corners.size(); i++) {
        // Convert floating-point corner coordinates to integer pixel coordinates
        cv::Point corner_point = cv::Point(cvRound(corners[i].x), cvRound(corners[i].y));

        // Set the pixel value at the corner position to 255
        frame_corner.at<uchar>(corner_point.y, corner_point.x) = 255;
    }



    std::vector<cv::Vec4i> lines_filtered;
    lines_filtered = closest_neighbor_line(frame_corner);
    


    // Draw the lines
    line_image = frame.clone();
    for (const auto& line : lines_filtered)
        cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    cv::imshow("Lines filtered", line_image);


    // Display the results
    cv::imshow("Corners", frame_corner);
    


   
    cv::waitKey(0);
}
*/

void ParkingDetection::detect(cv::Mat &frame) {

    // Convert the frame to HSV color space
    cv::Mat frame_backup = frame.clone();
    cv::Mat frame_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

    cv::Mat green_mask;
    std::vector<cv::Mat> hsv_channels;
    cv::split(frame_hsv, hsv_channels);
    cv::threshold(hsv_channels[1], green_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    cv::Mat frame_gray, frame_blurred, sub;
    cv::cvtColor(frame_hsv, frame_backup, cv::COLOR_HSV2BGR);
    cv::cvtColor(frame_backup, frame_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(frame_gray, frame_blurred, cv::Size(15, 15), 0);
    cv::subtract(frame_gray, frame_blurred, sub);

    // Delete parking lines from green_mask
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::erode(green_mask, green_mask, element);

    // Dilate the green mask
    element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
    cv::dilate(green_mask, green_mask, element);


    cv::bitwise_not(green_mask, green_mask);


    
    // Delete green mask from sub
    cv::bitwise_and(sub, green_mask, sub);
    cv::imshow("Sub", sub);

    // Maintain only the points over a specific threshold
    int threshold = 10;
    for (int i = 0; i < sub.rows; i++) {
        for (int j = 0; j < sub.cols; j++) {
            if (sub.at<uchar>(i, j) < threshold) {
                sub.at<uchar>(i, j) = 0;
            }
        }
    }

    
    cv::imshow("Sub threshold", sub);
    
    // Enhance weak points near strong ones
    enhanceWeakPointsNearStrongOnes(sub, 80, 10, 30);

    cv::imshow("Frame gray", frame_gray);
    cv::imshow("Green mask", green_mask);
    cv::imshow("Sub enhanced", sub);

    
    //MSER
    cv::Ptr<cv::MSER> mser = cv::MSER::create();
    mser->setMinArea(50);
    std::vector<std::vector<cv::Point>> regions;
    std::vector<cv::Rect> bboxes;
    mser->detectRegions(sub, regions, bboxes);
    cv::Mat frame_mser = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (const auto& region : regions) {
        for (const auto& point : region) {
            frame_mser.at<uchar>(point) = 255;
        }
    }
    cv::imshow("MSER", frame_mser);

    
    // Filter MSER removing isolated pixels
    removeIsolatedPixels(frame_mser, 200, 400);
    element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(frame_mser, frame_mser, cv::MORPH_CLOSE, element);
    deleteAreasInRange(frame_mser, 2000, 20000);
    deleteAreasInRange(frame_mser, 20, 50);
   // removeIsolatedPixels(frame_mser, 100, 150);
    cv::imshow("Filtered MSER", frame_mser);
    cv::imwrite("../Lab4/mser.jpg", frame_mser);
  
    // Hough Lines Transform
    std::vector<cv::Vec4i> lines;
    //cv::HoughLinesP(frame_mser, lines, 1, CV_PI/180, 30, 7, 10);
    cv::HoughLinesP(frame_mser, lines, 1, CV_PI/180, 30, 15, 10);
    // Draw the lines
    cv::Mat line_image = frame.clone();
    cv::Mat filtered_line_image = frame.clone();
    
    // Filter the lines based on the angle
    std::vector<cv::Vec4i> filtered_lines;
    std::vector<double> m;
    for (const auto& line : lines) {
        double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
        
        m.push_back((double)(line[3] - line[1]) / (double)(line[2] - line[0]));
    }

    double m_threshold = 1.0;

    for(int i = 0; i < m.size(); i++){
        if(m[i] <= -m_threshold || (m[i] >= 0 && m[i] <= 0.4)){
            filtered_lines.push_back(lines[i]);
        }
    }



    for (const auto& line : lines)
        cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    std::cout << "Number of lines: " << lines.size() << std::endl;
    std::cout << "Number of filtered lines: " << filtered_lines.size() << std::endl;

    for (const auto& line : filtered_lines)
        cv::line(filtered_line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    cv::imshow("Lines", line_image);
    cv::imshow("Filtered lines", filtered_line_image);

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

