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



// Utility function to compute the dot product of two vectors
double dotProduct(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.x + a.y * b.y;
}

// Utility function to compute the distance between two points
double distance(const cv::Point2f& a, const cv::Point2f& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// Utility function to project a point onto a line segment and clamp it to segment bounds
cv::Point2f projectPointOntoLineSegment(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
    cv::Point2f ab = b - a;
    double t = dotProduct(p - a, ab) / dotProduct(ab, ab);
    t = std::max(0.0, std::min(1.0, t));
    return a + t * ab;
}

/*
Calculate the distance between two lines
PARAM:
    line1: first line
    line2: second line
*/
double calculateDistance(const cv::Vec4i& line1, const cv::Vec4i& line2) {
    cv::Point2f p1(line1[0], line1[1]);
    cv::Point2f p2(line1[2], line1[3]);
    cv::Point2f p3(line2[0], line2[1]);
    cv::Point2f p4(line2[2], line2[3]);

    double minDist = std::numeric_limits<double>::max();

    // Check distance between each endpoint of the first segment to the second segment
    minDist = std::min(minDist, distance(p1, projectPointOntoLineSegment(p1, p3, p4)));
    minDist = std::min(minDist, distance(p2, projectPointOntoLineSegment(p2, p3, p4)));

    // Check distance between each endpoint of the second segment to the first segment
    minDist = std::min(minDist, distance(p3, projectPointOntoLineSegment(p3, p1, p2)));
    minDist = std::min(minDist, distance(p4, projectPointOntoLineSegment(p4, p1, p2)));

    return minDist;
}

double calculateLength(const cv::Vec4f& line) {
    cv::Point2f p1(line[0], line[1]);
    cv::Point2f p2(line[2], line[3]);

    return cv::norm(p1 - p2);
}


/*
Merge multiple lines into a single line
PARAM:
    lines: input lines
*/


/*

*/


/* 
Sort the line's vertices based on their position in the image
PARAM:
    lines: input lines
*/

std::vector<cv::Vec4i> sortLinesVertices(const std::vector<cv::Vec4i> lines) {
    std::vector<cv::Vec4i> sortedLines;

    for (const auto& line : lines) {
        cv::Point p1(line[0], line[1]);
        cv::Point p2(line[2], line[3]);

        // Sort the points based on their x-coordinate
        if (p1.x < p2.x) {
            sortedLines.push_back(cv::Vec4i(p1.x, p1.y, p2.x, p2.y));
        } else {
            sortedLines.push_back(cv::Vec4i(p2.x, p2.y, p1.x, p1.y));
        }
    }

    return sortedLines;
}


/*
Delete short lines
PARAM:
    lines: input lines
    minLength: minimum length of the line to keep
*/

std::vector<cv::Vec4f> deleteShortLines(const std::vector<cv::Vec4f> lines, double minLength) {
    std::vector<cv::Vec4f> result;

    for (const auto& line : lines) {
        cv::Point p1(line[0], line[1]);
        cv::Point p2(line[2], line[3]);

        double length = cv::norm(p1 - p2);

        if (length >= minLength) {
            result.push_back(line);
        }
    }

    return result;
}


/*
Compute the angle of a line
PARAM:
    line: input line
*/
double calculateAngle(const cv::Vec4f& line) {
    double dy = line[3] - line[1];
    double dx = line[2] - line[0];
    return std::atan2(dy, dx) * 180.0 / CV_PI; // Angle in degrees
}

/*
Find K mean values of similar lines by angle and filter them
PARAM:
lines: set of input lines
K = # groups of lines by angle
angleOffset = offset respect the mean for considering a line in a group
*/
std::vector<cv::Vec4f> filterLinesByKMeans(const std::vector<cv::Vec4f>& lines, int K, double angleOffset) {
    if (lines.empty() || K <= 0) {
        return {}; 
    }

    // Calculate the angle of each line
    std::vector<float> angles;
    for (const auto& line : lines) {
        angles.push_back(static_cast<float>(calculateAngle(line)));
    }

    // Apply K-Means clustering on angles
    cv::Mat labels, centers;
    cv::kmeans(angles, K, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    // Filter lines that are within the angleOffset of any cluster center
    std::vector<cv::Vec4f> filteredLines;
    for (int i = 0; i < lines.size(); i++) {
        float angle = angles[i];
        float center = centers.at<float>(labels.at<int>(i), 0);

        if (std::abs(angle - center) <= angleOffset) {
            filteredLines.push_back(lines[i]);
        }
    }

    return filteredLines;
}

cv::Vec4f mergeLines(const std::vector<cv::Vec4f>& lines) {
     if (lines.empty()) {
        return cv::Vec4f(); // Return an empty line if there are no lines in the input
    }

    // Initialize variables to track the maximum distance and corresponding points
    double maxDist = 0;
    cv::Point2f p1_max, p2_max;

    for (const auto& line1 : lines) {
        for (const auto& line2 : lines) {
            // Points from the first line
            cv::Point2f p1(line1[0], line1[1]);
            cv::Point2f p2(line1[2], line1[3]);

            // Points from the second line
            cv::Point2f p3(line2[0], line2[1]);
            cv::Point2f p4(line2[2], line2[3]);

            // Calculate distances between all combinations of endpoints
            double d1 = cv::norm(p1 - p3);
            double d2 = cv::norm(p1 - p4);
            double d3 = cv::norm(p2 - p3);
            double d4 = cv::norm(p2 - p4);

            // Find the maximum distance
            if (d1 > maxDist) {
                maxDist = d1;
                p1_max = p1;
                p2_max = p3;
            }
            if (d2 > maxDist) {
                maxDist = d2;
                p1_max = p1;
                p2_max = p4;
            }
            if (d3 > maxDist) {
                maxDist = d3;
                p1_max = p2;
                p2_max = p3;
            }
            if (d4 > maxDist) {
                maxDist = d4;
                p1_max = p2;
                p2_max = p4;
            }
        }
    }

    // Return the new line with the vertices that are farthest apart
    return cv::Vec4f(p1_max.x, p1_max.y, p2_max.x, p2_max.y);
}

std::vector<cv::Vec4f> unifySimilarLines(const std::vector<cv::Vec4f>& lines, double distanceThreshold, double angleThreshold, double lengthThreshold) {
    std::vector<cv::Vec4f> result;
    std::vector<bool> merged(lines.size(), false);

    for (int i = 0; i < lines.size(); i++) {
        if (merged[i]) continue;

        std::vector<cv::Vec4f> group = {lines[i]};
        merged[i] = true;

        for (int j = i + 1; j < lines.size(); j++) {
            if (merged[j]) continue;

            double distance = calculateDistance(lines[i], lines[j]);
            double angleDiff = std::abs(calculateAngle(lines[i]) - calculateAngle(lines[j]));
            double lengthDiff = std::abs(calculateLength(lines[i]) - calculateLength(lines[j]));
        
            if (distance < distanceThreshold && angleDiff < angleThreshold && lengthDiff < lengthThreshold) {
                group.push_back(lines[j]);
                merged[j] = true;
            }
        }
        if(group.size() > 1)
            result.push_back(mergeLines(group));
        // Else there are no lines to merge, maintain the original line
        else
            result.push_back(group[0]);
    }

    return result;
}

cv::Point2f closestPointOnSegment(const cv::Point2f &P, const cv::Vec4f &segment)
{
    cv::Point2f A(segment[0], segment[1]);
    cv::Point2f B(segment[2], segment[3]);
    cv::Point2f AP = P - A;
    cv::Point2f AB = B - A;
    float ab2 = AB.x * AB.x + AB.y * AB.y;
    float ap_ab = AP.x * AB.x + AP.y * AB.y;
    float t = ap_ab / ab2;
    t = std::min(std::max(t, 0.0f), 1.0f);
    return A + AB * t;
}

float distanceBetweenSegments(const cv::Vec4f &seg1, const cv::Vec4f &seg2)
{
    cv::Point2f A(seg1[0], seg1[1]);
    cv::Point2f B(seg1[2], seg1[3]);
    cv::Point2f C(seg2[0], seg2[1]);
    cv::Point2f D(seg2[2], seg2[3]);

    // Find the closest points on each segment to the other segment's endpoints
    cv::Point2f P1 = closestPointOnSegment(C, seg1);
    cv::Point2f P2 = closestPointOnSegment(D, seg1);
    cv::Point2f P3 = closestPointOnSegment(A, seg2);
    cv::Point2f P4 = closestPointOnSegment(B, seg2);

    // Compute the minimum distance
    float d1 = cv::norm(P1 - C);
    float d2 = cv::norm(P2 - D);
    float d3 = cv::norm(P3 - A);
    float d4 = cv::norm(P4 - B);

    return std::min({d1, d2, d3, d4});
}

float distanceBetweenSegments2(const cv::Vec4f &seg1, const cv::Vec4f &seg2)
{
    cv::Point2f mid1((seg1[0] + seg1[2]) / 2, (seg1[1] + seg1[3]) / 2); // Midpoint of seg1
    cv::Point2f mid2((seg2[0] + seg2[2]) / 2, (seg2[1] + seg2[3]) / 2); // Midpoint of seg2

    // Compute the Euclidean distance between the two midpoints
    float distance = cv::norm(mid1 - mid2);

    return distance;
}


double calculateLineAngle(const cv::Vec4f &line)
{
    return std::atan2(line[3] - line[1], line[2] - line[0]) * 180.0 / CV_PI;
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



/*
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

    
    // Sort the line's vertices based on their position in the image
    std::vector<cv::Vec4i> sortedLines = sortLinesVertices(lines);


    // Angle "K-means" clustering
    std::vector<cv::Vec4i> filtered_lines = filterLinesByKMeans(sortedLines, 4, 15);


    for (const auto& line : lines)
        cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);


    for (const auto& line : filtered_lines)
        cv::line(filtered_line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    cv::imshow("Lines", line_image);
    cv::imshow("Filtered lines", filtered_line_image);

   
    // Line Segment Detector
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD, 
    0.7, // scale 
    0.4, // sigma_scale 
    5.0, // quant
    20, // ang_th
    0.4, // log_eps
    0.8, // density_th
    1024 // n_bins
    );

   
    std::vector<cv::Vec4i> lsd_lines;
    lsd->detect(frame_gray, lsd_lines);

    cv::Mat line_image_lsd = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (int i = 0; i < lsd_lines.size(); i++) {
        cv::Vec4i line = lsd_lines[i];
        cv::line(line_image_lsd, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 1, cv::LINE_AA);
    }
    cv::imshow("LSD", line_image_lsd);


    cv::Mat filtered_line_image_lsd = frame.clone();
    // Sort the line's vertices based on their position in the image
    std::vector<cv::Vec4i> sortedLines_lsd = sortLinesVertices(lsd_lines);
    
    // Filter the lines based on the angle
    std::vector<cv::Vec4i> filtered_lines_lsd = filterLinesByKMeans(sortedLines_lsd, 4, 15);
  
  

    for (const auto& line : filtered_lines_lsd)
        cv::line(filtered_line_image_lsd, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    cv::imshow("Filtered lines LSD", filtered_line_image_lsd);


    // Unify the lines of filtered_lines and filtered_lines_lsd
    std::vector<cv::Vec4i> unifiedLines;
    unifiedLines.insert(unifiedLines.end(), filtered_lines.begin(), filtered_lines.end());
    unifiedLines.insert(unifiedLines.end(), filtered_lines_lsd.begin(), filtered_lines_lsd.end());

    
    // Delete short lines
    double minLength = 17;
    unifiedLines = deleteShortLines(unifiedLines, minLength);

    // Angle "K-means" clustering
    unifiedLines = filterLinesByKMeans(unifiedLines, 4, 15.0);
    

    // Draw the unified lines
    cv::Mat unifiedImg = frame.clone();

    for (const auto& line : unifiedLines) {
        cv::line(unifiedImg, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    
    imshow("Unified Lines  ", unifiedImg);

    
    
    // Unify the similar lines
    cv::Mat img = frame.clone();
    double distanceThreshold = 20; // Distance threshold to consider lines close
    double angleThreshold = 15.0;    // Angle threshold to consider lines similar (in degrees)

    unifiedLines = unifySimilarLines(unifiedLines, distanceThreshold, angleThreshold);

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
*/


bool areParallelAndClose(const cv::Vec4f &line1, const cv::Vec4f &line2, double angleThreshold, double distanceThreshold, bool mergeVertex = false)
{
    double angle1 = calculateLineAngle(line1);
    double angle2 = calculateLineAngle(line2);
    double vertexDistance = distanceBetweenSegments(line1, line2);

    return std::fabs(angle1 - angle2) < angleThreshold && vertexDistance < distanceThreshold;
}





cv::Vec4f mergeLineSegments(const cv::Vec4f &line_i, const cv::Vec4f &line_j)
{
    // Calculate the lengths of the lines
    double line_i_length = std::hypot(line_i[2] - line_i[0], line_i[3] - line_i[1]);
    double line_j_length = std::hypot(line_j[2] - line_j[0], line_j[3] - line_j[1]);

    // Calculate the centroid
    double Xg = line_i_length * (line_i[0] + line_i[2]) + line_j_length * (line_j[0] + line_j[2]);
    Xg /= 2 * (line_i_length + line_j_length);

    double Yg = line_i_length * (line_i[1] + line_i[3]) + line_j_length * (line_j[1] + line_j[3]);
    Yg /= 2 * (line_i_length + line_j_length);

    // Calculate the orientations
    double orientation_i = std::atan2(line_i[1] - line_i[3], line_i[0] - line_i[2]);
    double orientation_j = std::atan2(line_j[1] - line_j[3], line_j[0] - line_j[2]);
    double orientation_r;

    if (std::abs(orientation_i - orientation_j) <= CV_PI / 2)
    {
        orientation_r = (line_i_length * orientation_i + line_j_length * orientation_j) / (line_i_length + line_j_length);
    }
    else
    {
        orientation_r = (line_i_length * orientation_i + line_j_length * (orientation_j - CV_PI * orientation_j / std::abs(orientation_j))) / (line_i_length + line_j_length);
    }

    // Coordinate transformation
    double a_x_g = (line_i[1] - Yg) * std::sin(orientation_r) + (line_i[0] - Xg) * std::cos(orientation_r);
    double a_y_g = (line_i[1] - Yg) * std::cos(orientation_r) - (line_i[0] - Xg) * std::sin(orientation_r);

    double b_x_g = (line_i[3] - Yg) * std::sin(orientation_r) + (line_i[2] - Xg) * std::cos(orientation_r);
    double b_y_g = (line_i[3] - Yg) * std::cos(orientation_r) - (line_i[2] - Xg) * std::sin(orientation_r);

    double c_x_g = (line_j[1] - Yg) * std::sin(orientation_r) + (line_j[0] - Xg) * std::cos(orientation_r);
    double c_y_g = (line_j[1] - Yg) * std::cos(orientation_r) - (line_j[0] - Xg) * std::sin(orientation_r);

    double d_x_g = (line_j[3] - Yg) * std::sin(orientation_r) + (line_j[2] - Xg) * std::cos(orientation_r);
    double d_y_g = (line_j[3] - Yg) * std::cos(orientation_r) - (line_j[2] - Xg) * std::sin(orientation_r);

    // Orthogonal projections over the axis X
    double start_f = std::min({a_x_g, b_x_g, c_x_g, d_x_g});
    double end_f = std::max({a_x_g, b_x_g, c_x_g, d_x_g});
    double length_f = std::hypot(end_f - start_f, 0);

    // Compute the final merged line segment
    int start_x = static_cast<int>(Xg - start_f * std::cos(orientation_r));
    int start_y = static_cast<int>(Yg - start_f * std::sin(orientation_r));
    int end_x = static_cast<int>(Xg - end_f * std::cos(orientation_r));
    int end_y = static_cast<int>(Yg - end_f * std::sin(orientation_r));

    return cv::Vec4f(start_x, start_y, end_x, end_y);
}


cv::Vec4f mergeLineCluster(const std::vector<cv::Vec4f> &cluster)
{
    cv::Vec4f merged = cluster[0];
    for (size_t i = 1; i < cluster.size(); i++)
    {
        merged = mergeLineSegments(merged, cluster[i]);
    }

    return merged;
}



std::vector<cv::Vec4f> divideLongLines(const std::vector<cv::Vec4f> &lines, double max_length, double offset)
{
     std::vector<cv::Vec4f> dividedLines;

    for (const auto &line : lines) {
        
        cv::Point2f p1(line[0], line[1]); // Line start point
        cv::Point2f p2(line[2], line[3]); // Line end point
        
        double length = calculateLength(line);

        if (length > max_length) {
            // Calculate the midpoint of the line
            cv::Point2f midpoint = (p1 + p2) * 0.5;

            // Vector of the direction from p1 to p2
            cv::Point2f direction = (p2 - p1) / length;

            // Create two smaller lines with a hole (offset)
            cv::Point2f adjustedP1 = midpoint - direction * (offset / 2);
            cv::Point2f adjustedP2 = midpoint + direction * (offset / 2);

            // First part of the line
            dividedLines.push_back(cv::Vec4f(p1.x, p1.y, adjustedP1.x, adjustedP1.y));

            // Second part of the line
            dividedLines.push_back(cv::Vec4f(adjustedP2.x, adjustedP2.y, p2.x, p2.y));
        } else {
            // If the line is short enough, keep it as it is
            dividedLines.push_back(line);
        }
    }

    return dividedLines;
}


bool compareLines(const cv::Vec4f& line1, const cv::Vec4f& line2) {
    // Extract starting points
    cv::Point2f start1(line1[0], line1[1]);
    cv::Point2f start2(line2[0], line2[1]);

    // Sort primarily by y-coordinate, secondarily by x-coordinate
    if (start1.y < start2.y) return true;
    if (start1.y > start2.y) return false;
    return start1.x < start2.x;
}

std::vector<cv::Vec4f> sortLines(const std::vector<cv::Vec4f>& lines) {
    std::vector<cv::Vec4f> sorted_lines = lines;
    std::sort(sorted_lines.begin(), sorted_lines.end(), compareLines);
    return sorted_lines;
}



/*
cv::Mat drawBoundingBoxes(cv::Mat& frame, const std::vector<cv::Vec4f>& lines, int minDistanceThreshold, int maxDistanceThreshold, int maxAngleThreshold) {
    std::vector<cv::Vec4f> new_lines;
    std::vector<double> distances;
    for (size_t i = 0; i < lines.size(); i++)
    {
        double minD = maxDistanceThreshold;
        cv::Vec4f minDLine;

        for (size_t j = 0; j < lines.size(); j++)
        {
            if (i == j)
                continue;

            double d = distanceBetweenSegments2(lines[i], lines[j]);
            double a = std::fabs(calculateLineAngle(lines[i]) - calculateLineAngle(lines[j]));
            if (d < minD && a < maxAngleThreshold)
            {
                minD = d;
                minDLine = lines[j];
            }
        }

        if (distanceBetweenSegments(lines[i], minDLine) < maxDistanceThreshold)
        {
            new_lines.push_back(mergeLineSegments(lines[i], minDLine));
            distances.push_back(distanceBetweenSegments(lines[i], minDLine));
        }
    }

    cv::Mat n = frame.clone();
    for (size_t i = 0; i < new_lines.size(); i++)
    {
        cv::Vec4f line = new_lines[i];
        // cv::line(n, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 0, 0), 2);
        cv::RotatedRect rotatedRect(cv::Point((line[0] + line[2]) / 2, (line[1] + line[3]) / 2), cv::Size(calculateLength(line), distances[i]), calculateLineAngle(line));
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);

        // Draw the rotated rectangle using polylines
        for (int i = 0; i < 4; i++)
        {
            // Draw lines between consecutive vertices
            cv::line(n, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);
        }
        // cv::circle(n, cv::Point((line[0] + line[2]) / 2, (line[1] + line[3]) / 2), 10, cv::Scalar(0, 255, 0), 2);
    }

    return n;
}

*/

cv::Mat drawBoundingBoxes(cv::Mat& frame, const std::vector<cv::Vec4f>& lines, int minDistanceThreshold, int maxDistanceThreshold, int maxAngleThreshold, double minAreaThreshold, double maxAreaThreshold) {
    cv::Mat bounding_box = frame.clone();
    std::set<std::pair<int, int>> used_pairs;
    std::set<std::pair<int, int>> counter_lines;

    for (int i = 0; i < lines.size(); i++) {
        std::vector<std::pair<int, double>> candidates;  

        // Find valid candidate lines
        for (int j = 0; j < lines.size(); j++) {
            if (i == j) continue;

            if (used_pairs.count(std::make_pair(i, j)) || used_pairs.count(std::make_pair(j, i))) {
                continue;
            }

            double distance = distanceBetweenSegments2(lines[i], lines[j]);
            double angle1 = calculateLineAngle(lines[i]);
            double angle2 = calculateLineAngle(lines[j]);

            //int xDiff = std::abs(lines[i][0] - lines[j][0]);

            if (distance > minDistanceThreshold && distance < maxDistanceThreshold && std::abs(angle1 - angle2) < maxAngleThreshold) {
                candidates.push_back(std::make_pair(j, distance));
            }
        }

        bool found_valid_rectangle = false;
        int nearest_line_idx = -1;
        double min_distance = DBL_MAX;

        // Find the nearest candidate based on distance
        for (const auto& candidate : candidates) {
            if (candidate.second < min_distance && counter_lines.count(candidate) < 3) {
                nearest_line_idx = candidate.first;
                min_distance = candidate.second;
                counter_lines.insert(std::make_pair(i, nearest_line_idx));
            }
        }

        // If a nearest line is found, try to create a rectangle
        if (nearest_line_idx != -1) {
            cv::Vec4f line1 = lines[i];
            cv::Vec4f line2 = lines[nearest_line_idx];
            
            // Store the pair of lines as used
            used_pairs.insert(std::make_pair(i, nearest_line_idx));
            used_pairs.insert(std::make_pair(nearest_line_idx, i));


            cv::Vec4f line = mergeLineSegments(line1, line2);

            cv::RotatedRect rotatedRect(cv::Point((line[0] + line[2]) / 2, (line[1] + line[3]) / 2), cv::Size(calculateLength(line), distanceBetweenSegments(line1, line2)), calculateLineAngle(line));

            cv::Point2f vertices[4];
            rotatedRect.points(vertices);

            
            /*
            // Calculate the midpoints of each line
            cv::Point2f mid_point1((line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2);
            cv::Point2f mid_point2((line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2);

            // Find the midpoint of the rectangle
            cv::Point2f rect_center = (mid_point1 + mid_point2) / 2;

            // Calculate the angle between the lines (average angle, should be almost parallel)
            double angle = (calculateLineAngle(line1) + calculateLineAngle(line2)) / 2;

            // Calculate the length of the rectangle's width and height
            double width = cv::norm(cv::Point2f(line1[0], line1[1]) - cv::Point2f(line1[2], line1[3]));
            double height = cv::norm(mid_point1 - mid_point2);

            // Create the rectangle using the center, size, and angle
            cv::RotatedRect rect(rect_center, cv::Size2f(width, height), angle);
            */
            // Calculate the area of the rectangle
            double area = rotatedRect.size.area();

            // Check if the area is within the thresholds
            if (area < minAreaThreshold || area > maxAreaThreshold) {
                continue; // Skip this candidate and try the next one
            }

            // Draw the rotated rectangle using polylines
            for (int i = 0; i < 4; i++) {
                // Draw lines between consecutive vertices
                cv::line(bounding_box, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);
            }

            /*
            // Get the rectangle vertices
            cv::Point2f vertices[4];
            rect.points(vertices);

            // Draw the rectangle using the vertices
            cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
            for (int k = 0; k < 4; ++k) {
                cv::line(bounding_box, vertices[k], vertices[(k + 1) % 4], color, 2);
            }
            */
            found_valid_rectangle = true;
        }

        // If no valid rectangle was found for this line, continue to the next one
        if (!found_valid_rectangle) {
            continue;
        }
    }

    return bounding_box;
}

/*
cv::Mat drawBoundingBoxes(cv::Mat& frame, const std::vector<cv::Vec4f>& lines, int minDistanceThreshold, int maxDistanceThreshold, int maxAngleThreshold) {
    cv::Mat bounding_box = frame.clone();

    // Set to store line index pairs that have already been used to draw bounding boxes
    std::set<std::pair<int, int>> used_pairs;

    for (int i = 0; i < lines.size(); i++) {
        std::vector<std::pair<int, double>> candidates; // Store candidates with their distances

        for (int j = 0; j < lines.size(); j++) {
            if (i == j) continue; // Skip the same line

            // Ensure we don't process the same pair of lines again
            if (used_pairs.count(std::make_pair(i, j)) || used_pairs.count(std::make_pair(j, i))) {
                continue;
            }

            double distance = distanceBetweenSegments2(lines[i], lines[j]);

            // The lines have to be parallel
            double angle1 = calculateLineAngle(lines[i]);
            double angle2 = calculateLineAngle(lines[j]);

            // Store the candidate if it meets the distance and angle thresholds
            if (distance > minDistanceThreshold && distance < maxDistanceThreshold && std::abs(angle1 - angle2) < maxAngleThreshold) {
                candidates.push_back(std::make_pair(j, distance)); 
            }
        }

        // Now find the nearest candidate from the list
        if (!candidates.empty()) {
            int nearest_line_idx = -1;
            double min_distance = DBL_MAX;

            // Loop through the candidates to find the nearest line
            for (const auto& candidate : candidates) {
                if (candidate.second < min_distance) {
                    min_distance = candidate.second 
                // Calculate the four corners of the rectangle
                cv::Point top_left(line1[0], line1[1]);
                cv::Point top_right(line1[2], line1[3]);
                cv::Point bottom_left(line2[0], line2[1]);
                cv::Point bottom_right(line2[2], line2[3]);

                cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);

                // Draw the rectangle using the corners
                cv::line(bounding_box, top_left, top_right, color, 2);
                cv::line(bounding_box, top_right, bottom_right, color, 2);
                cv::line(bounding_box, bottom_right, bottom_left, color, 2);
                cv::line(bounding_box, bottom_left, top_left, color, 2);
            }
        }
    }

    return bounding_box;
}


*/
void ParkingDetection::detect(cv::Mat &frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat adaptive;
    cv::adaptiveThreshold(gray, adaptive, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 3, -8);

    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV, 0.6, 1);

    std::vector<cv::Vec4f> lines;
    lsd->detect(adaptive, lines);

    // Sort the line's vertices based on their position in the image
    for (cv::Vec4f &line : lines)
    {
        if (line[0] > line[2])
        {
            std::swap(line[0], line[2]);
            std::swap(line[1], line[3]);
        }
    }

    std::vector<int> labels;
    int numComponents = cv::partition(lines, labels, [&](const cv::Vec4i &l1, const cv::Vec4i &l2)
                                      { return areParallelAndClose(l1, l2, 5, 10); });

    cv::Mat c = cv::Mat::zeros(frame.size(), frame.type());


    std::vector<cv::Scalar> colors;
    for (int i = 0; i < numComponents; i++)
    {
        colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
    }

    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4f line = lines[i];
        cv::Point p1(line[0], line[1]);
        cv::Point p2(line[2], line[3]);
        int label = labels[i];
        cv::line(c, p1, p2, colors[label], 2);
    }

    std::vector<cv::Vec4f> merged_lines;
    for (int i = 0; i < numComponents; ++i)
    {
        std::vector<cv::Vec4f> cluster;
        for (size_t j = 0; j < labels.size(); ++j)
            if (labels[j] == i)
                cluster.push_back(lines[j]);

        merged_lines.push_back(mergeLineCluster(cluster));
    }


    // Remove short lines
    std::vector<cv::Vec4f> longLines = deleteShortLines(merged_lines, 15); 


    cv::Mat m = frame.clone();
    
    lsd->drawSegments(m, longLines);



    // Unify the similar lines
    cv::Mat img = frame.clone();
    double distanceThreshold = 15; // Distance threshold to consider lines close
    double angleThreshold = 30;    // Angle threshold to consider lines similar (in degrees)
    double lengthThreshold = 20; // Length threshold to consider lines similar

    std::vector<cv::Vec4f> unifiedLines = unifySimilarLines(longLines, distanceThreshold, angleThreshold, lengthThreshold);

    

    // Draw the unified lines
    for (const auto& line : unifiedLines) {
        cv::line(img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    longLines = deleteShortLines(unifiedLines, 20);
    cv::Mat l = frame.clone();
    lsd->drawSegments(l, longLines);

    
    // Kmeans clustering for filter the line based on the angle
    std::vector<cv::Vec4f> kmeans_lines = filterLinesByKMeans(longLines, 2, 30);
    
    cv::Mat k = frame.clone();
    lsd->drawSegments(k, kmeans_lines);


    // Divide in 2 parts the lines too long (They are lines that include two parkings)
    lines = divideLongLines(kmeans_lines, 200, 30);
    cv::Mat d = frame.clone();
    lsd->drawSegments(d, lines);
    

    // Sort the lines based in their position in the space 
    lines = sortLines(lines);
    
    int minDistanceThreshold = 20;
    int maxDistanceThreshold = 150;
    int maxAngleThreshold = 20;
    double minAreaThreshold = 100;
    double maxAreaThreshold = 20000;
    int xThreshold = 20;
    cv::Mat bounding_box = drawBoundingBoxes(frame, lines, minDistanceThreshold, maxDistanceThreshold, maxAngleThreshold, minAreaThreshold, maxAreaThreshold);


    imshow("Lines", frame);
    imshow("Cluster", c);
    imshow("Merged", m);
    imshow("Unified", img);
    imshow("Long lines", l);
    imshow("Kmeans", k);
    imshow("Divided", d);
    imshow("Bounding box", bounding_box);

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

