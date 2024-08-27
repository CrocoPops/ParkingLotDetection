#include "parkingdetection.h"
#include <opencv2/ximgproc.hpp>

ParkingDetection::ParkingDetection(std::vector<BBox> parkings) : parkings(parkings) {}

/*
Remove the isolated pixels that are far from the rest of the pixels
PARAM:
    img: input image
    neighbrhoodSize: size of the neighborhood to consider (rectangular)
    minPixels: minimum number of white pixels in the neighborhood to keep the pixel white
*/

void removeIsolatedPixels(cv::Mat &img, int neighborhoodSize, int minPixels) {
    if (img.type() != CV_8UC1) {
        std::cerr << "Image must be binary (CV_8UC1)" << std::endl;
        return;
    }
        
    // Ensure the neighborhood size is odd to have a center pixel
    if (neighborhoodSize % 2 == 0)
        neighborhoodSize += 1;
    
    
    // Create a copy of the original image to modify
    cv::Mat output = img.clone();
    
    int offset = neighborhoodSize / 2;

    for (int y = offset; y < img.rows - offset; y++) {
        for (int x = offset; x < img.cols - offset; x++) {
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

/* 
Based on the region's area, delete the region if the area is between minSize and maxSize
PARAM:
    img: input image
    minSize: minimum area of the region to keep
    maxSize: maximum area of the region to keep
*/

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

/*
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

*/
/*
Set all the strong pixels and the weak pixels near the strong ones to 255; the others to 0
PARAM:
    img: input image
    neighborhoodSize: size of the neighborhood to consider (rectangular)
    minNumStrongPixels: minimum number of strong pixels in the neighborhood to keep the pixel white
    threshold: threshold to consider a pixel as strong
*/

void enhanceWeakPointsNearStrongOnes(cv:: Mat &img, int neighborhoodSize, int minNumStrongPixels, int threshold) {
    if (img.type() != CV_8UC1) {
        std::cerr << "Image must be binary (CV_8UC1)" << std::endl;
        return;
    }


    // Check if the threshold is valid
    if (threshold < 0 || threshold > 255) {
        std::cerr << "Threshold must be between 0 and 255" << std::endl;
        return;
    }

    // Ensure the neighborhood size is odd to have a center pixel
    if (neighborhoodSize % 2 == 0)
        neighborhoodSize += 1;
    
       
    
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
                if (strongCount >= minNumStrongPixels || img.at<uchar>(y, x) >= threshold) 
                    output.at<uchar>(y, x) = 255;
                else 
                    output.at<uchar>(y, x) = 0;
            }
        }
    }
    img = output;
}

/*
Calculate the distance between two lines
PARAM:
    line1: first line
    line2: second line
*/


double calculateDistance(cv::Vec4i line1, cv::Vec4i line2) {
    cv::Point p1(line1[0], line1[1]);
    cv::Point p2(line1[2], line1[3]);
    cv::Point p3(line2[0], line2[1]);
    cv::Point p4(line2[2], line2[3]);

    // Calculate the vector (p4 - p3)
    cv::Point line2_vector = p4 - p3;

    // Calculate the vector perpendicular to line2_vector
    cv::Point perpendicular_vector(-line2_vector.y, line2_vector.x);

    // Calculate the distance using the dot product formula
    double distance = std::abs((p1 - p3).dot(perpendicular_vector)) / cv::norm(perpendicular_vector);

    return distance;
}


/*
Calculate the angle of a line
PARAM:
    line: input line
*/

double calculateAngle(cv::Vec4i line) {
    cv::Point p1(line[0], line[1]);
    cv::Point p2(line[2], line[3]);
    return atan2(p2.y - p1.y, p2.x - p1.x) * 180 / CV_PI;
}

/*
Merge multiple lines into a single line
PARAM:
    lines: input lines
*/


/*

*/
cv::Vec4i mergeLines(const std::vector<cv::Vec4i>& lines) {
    // Compute the average of the points and create a mean line
    int x1_sum = 0, y1_sum = 0, x2_sum = 0, y2_sum = 0;

    for (const auto& line : lines) {
        x1_sum += line[0];
        y1_sum += line[1];
        x2_sum += line[2];
        y2_sum += line[3];
    }

    int n = lines.size();
    cv::Vec4i meanLine = cv::Vec4i(x1_sum / n, y1_sum / n, x2_sum / n, y2_sum / n);

    // Extend the line for better visualization
    cv::Point p1(meanLine[0], meanLine[1]);
    cv::Point p2(meanLine[2], meanLine[3]);

    cv::Point p1_new = p1 + 0.4 * (p1 - p2);
    cv::Point p2_new = p2 + 0.4 * (p2 - p1);

    return cv::Vec4i(p1_new.x, p1_new.y, p2_new.x, p2_new.y); 
}

std::vector<cv::Vec4i> unifySimilarLines(const std::vector<cv::Vec4i>& lines, double distanceThreshold, double angleThreshold) {
    std::vector<cv::Vec4i> result;
    std::vector<bool> merged(lines.size(), false);

    for (int i = 0; i < lines.size(); i++) {
        if (merged[i]) continue;

        std::vector<cv::Vec4i> group = {lines[i]};
        merged[i] = true;

        for (int j = i + 1; j < lines.size(); j++) {
            if (merged[j]) continue;

            double distance = calculateDistance(lines[i], lines[j]);
            double angleDiff = std::abs(calculateAngle(lines[i]) - calculateAngle(lines[j]));

            if (distance < distanceThreshold && angleDiff < angleThreshold) {
                group.push_back(lines[j]);
                merged[j] = true;
            }
        }

        result.push_back(mergeLines(group));
    }

    return result;
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
    cv::Mat frame_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

    cv::Mat green_mask;
    std::vector<cv::Mat> hsv_channels;
    cv::split(frame_hsv, hsv_channels);
    cv::threshold(hsv_channels[1], green_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    cv::Mat frame_gray, frame_blurred, sub;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
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

    
    // MSER
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

    cv::imshow("Filtered MSER", frame_mser);
  
    // Hough Lines Transform
    std::vector<cv::Vec4i> lines;
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

    
    // Unify the similar lines
    cv::Mat img = frame.clone();
    double distanceThreshold = 0.5; // Distance threshold to consider lines close
    double angleThreshold = 5.0;    // Angle threshold to consider lines similar (in degrees)

    std::vector<cv::Vec4i> unifiedLines = unifySimilarLines(filtered_lines, distanceThreshold, angleThreshold);

    // Draw the unified lines
    for (const auto& line : unifiedLines) {
        cv::line(img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    imshow("Unified Lines", img);
    
    // Filter the lines based on the distance between them
    
    cv::Mat bounding_box = frame.clone();
    
    int minDistanceThreshold = 15;
    int maxDistanceThreshold = 80;
    
    for (int i = 0; i < unifiedLines.size(); i++) {
        cv::Vec4i nearest_line;
        double min_distance = DBL_MAX; // Initialize with a large value

        for (int j = 0; j < unifiedLines.size(); j++) {
            if (i == j) continue; // Skip the same line

            double distance = calculateDistance(unifiedLines[i], unifiedLines[j]);

            // Find the nearest line pair that has a distance greater than minDistanceThreshold
            if (distance > minDistanceThreshold && distance < maxDistanceThreshold && distance < min_distance) {
                min_distance = distance;
                nearest_line = unifiedLines[j];
            }
        }

        // If a nearest line is found, draw the bounding box
        if (min_distance < DBL_MAX) {
            cv::Vec4i line1 = unifiedLines[i];
            cv::Vec4i line2 = nearest_line;

            // Calculate the four corners of the rectangle
            cv::Point top_left(line1[0], line1[1]);
            cv::Point top_right(line1[2], line1[3]);
            cv::Point bottom_left(line2[0], line2[1]);
            cv::Point bottom_right(line2[2], line2[3]);

            // Draw the rectangle using the corners
            cv::line(frame, top_left, top_right, cv::Scalar(0, 255, 0), 2);
            cv::line(frame, top_right, bottom_right, cv::Scalar(0, 255, 0), 2);
            cv::line(frame, bottom_right, bottom_left, cv::Scalar(0, 255, 0), 2);
            cv::line(frame, bottom_left, top_left, cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imshow("Bounding box", frame);

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

