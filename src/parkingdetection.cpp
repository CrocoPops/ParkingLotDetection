#include "parkingdetection.h"
#include <opencv2/ximgproc.hpp>

ParkingDetection::ParkingDetection(std::vector<BBox> parkings) : parkings(parkings) {}


// Utility function to compute the dot product of two vectors
double ParkingDetection::dotProduct(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.x + a.y * b.y;
}

// Utility function to compute the distance between two points
double ParkingDetection::distance(const cv::Point2f& a, const cv::Point2f& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// Utility function to project a point onto a line segment and clamp it to segment bounds
cv::Point2f ParkingDetection::projectPointOntoLineSegment(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
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
double ParkingDetection::calculateDistance(const cv::Vec4i& line1, const cv::Vec4i& line2) {
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

/*
Calculate the length of a line
PARAM:
    line: input line
*/

double ParkingDetection::calculateLength(const cv::Vec4f& line) {
    cv::Point2f p1(line[0], line[1]);
    cv::Point2f p2(line[2], line[3]);

    return cv::norm(p1 - p2);
}

/*
Delete short lines
PARAM:
    lines: input lines
    minLength: minimum length of the line to keep
*/

std::vector<cv::Vec4f> ParkingDetection::deleteShortLines(const std::vector<cv::Vec4f> lines, double minLength) {
    std::vector<cv::Vec4f> result;

    for (const auto& line : lines) {

        double length = calculateLength(line);

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
double ParkingDetection::calculateAngle(const cv::Vec4f& line) {
    double dy = line[3] - line[1];
    double dx = line[2] - line[0];
    return std::atan2(dy, dx) * 180.0 / CV_PI; // Angle in degrees
}

/*
Find K mean values of similar lines by angle and filter them
PARAM:
    lines: set of input lines
    K: # groups of lines by angle
    angleOffset: offset respect the mean for considering a line in a group
*/
std::vector<cv::Vec4f> ParkingDetection::filterLinesByKMeans(const std::vector<cv::Vec4f>& lines, int K, double angleOffset) {
    if (lines.empty() || K <= 0) {
        return {}; 
    }

    // Calculate the angle of each line
    std::vector<float> angles;
    for (const auto& line : lines)
        angles.push_back(static_cast<float>(calculateAngle(line)));
    

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

/*
Merge two lines into a single line
PARAM:
    lines: vector containing the two lines to be merged
*/

cv::Vec4f ParkingDetection::mergeLines(const std::vector<cv::Vec4f>& lines) {
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

/*
Unify similar lines by distance, angle, and length
PARAM:
    lines: input lines
    distanceThreshold: maximum distance between two lines to be considered similar
    angleThreshold: maximum angle difference between two lines to be considered similar
    lengthThreshold: maximum length difference between two lines to be considered similar
*/

std::vector<cv::Vec4f> ParkingDetection::unifySimilarLines(const std::vector<cv::Vec4f>& lines, double distanceThreshold, double angleThreshold, double lengthThreshold) {
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

/*
Find the closest neighbor point of each line
PARAM:
    P: input point
    segments: input segments
*/

cv::Point2f ParkingDetection::closestPointOnSegment(const cv::Point2f &P, const cv::Vec4f &segment) {
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


/*
Compute the distance between two segments
PARAM:
    seg1: first segment
    seg2: second segment
*/

float ParkingDetection::distanceBetweenSegments(const cv::Vec4f &seg1, const cv::Vec4f &seg2) {
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


/*
Compute the distance between two segments, using the Euclidean distance between the midpoints
PARAM:
    seg1: first segment
    seg2: second segment
*/

float ParkingDetection::distanceBetweenSegments2(const cv::Vec4f &seg1, const cv::Vec4f &seg2) {
    cv::Point2f mid1((seg1[0] + seg1[2]) / 2, (seg1[1] + seg1[3]) / 2); // Midpoint of seg1
    cv::Point2f mid2((seg2[0] + seg2[2]) / 2, (seg2[1] + seg2[3]) / 2); // Midpoint of seg2

    // Compute the Euclidean distance between the two midpoints
    float distance = cv::norm(mid1 - mid2);

    return distance;
}



/*
Check if two lines are parallel and close
PARAM:
    line1: first line
    line2: second line
    angleThreshold: maximum angle difference to consider the lines parallel
    distanceThreshold: maximum distance between the lines to consider them close
*/

bool ParkingDetection::areParallelAndClose(const cv::Vec4f &line1, const cv::Vec4f &line2, double angleThreshold, double distanceThreshold) {
    double angle1 = calculateAngle(line1);
    double angle2 = calculateAngle(line2);
    double vertexDistance = distanceBetweenSegments(line1, line2);

    return std::fabs(angle1 - angle2) < angleThreshold && vertexDistance < distanceThreshold;
}


/*
Merge two line segments into a single line segment
PARAM:
    line_i: first line segment
    line_j: second line segment
*/

cv::Vec4f ParkingDetection::mergeLineSegments(const cv::Vec4f &line_i, const cv::Vec4f &line_j) {
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

/*
Merge a cluster of line segments into a single line segment
PARAM:
    cluster: vector of line segments to merge
*/

cv::Vec4f ParkingDetection::mergeLineCluster(const std::vector<cv::Vec4f> &cluster) {
    cv::Vec4f merged = cluster[0];
    for (int i = 1; i < cluster.size(); i++)
    {
        merged = mergeLineSegments(merged, cluster[i]);
    }

    return merged;
}


/*
Divide long lines into two shorter lines with a hole in the middle
PARAM:
    lines: input lines
    max_length: maximum length of the line
    offset: distance between the two smaller lines
*/


std::vector<cv::Vec4f> ParkingDetection::divideLongLines(const std::vector<cv::Vec4f> &lines, double max_length, double offset) {
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

/*
Compare two lines based on their starting points
PARAM:
    line1: first line
    line2: second line
*/

bool ParkingDetection::compareLines(const cv::Vec4f& line1, const cv::Vec4f& line2) {
    // Extract starting points
    cv::Point2f start1(line1[0], line1[1]);
    cv::Point2f start2(line2[0], line2[1]);

    // Sort primarily by y-coordinate, secondarily by x-coordinate
    if (start1.y < start2.y) return true;
    if (start1.y > start2.y) return false;
    return start1.x < start2.x;
}

/*
Sort lines based on their starting points
PARAM:
    lines: input lines
*/

std::vector<cv::Vec4f> ParkingDetection::sortLines(const std::vector<cv::Vec4f>& lines) {
    std::vector<cv::Vec4f> sorted_lines = lines;
    std::sort(sorted_lines.begin(), sorted_lines.end(), compareLines);
    return sorted_lines;
}


/*
Function to mirror line2 across line1 (as the axis of symmetry)
PARAM:
    axis: axis of symmetry (line1)
    line_to_mirror: line to mirror (line2)
*/

cv::Vec4f ParkingDetection::mirrorLineAcrossAxis(const cv::Vec4f& axis, const cv::Vec4f& line_to_mirror) {
    // Extract points from the line segments
    cv::Point2f A1(axis[0], axis[1]); // Start point of axis line
    cv::Point2f A2(axis[2], axis[3]); // End point of axis line
    cv::Point2f B1(line_to_mirror[0], line_to_mirror[1]); // Start point of line to mirror
    cv::Point2f B2(line_to_mirror[2], line_to_mirror[3]); // End point of line to mirror

    // Function to project a point onto a line and reflect it
    auto reflectPoint = [](const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
        // Compute direction vector of the axis
        cv::Point2f ab = b - a;

        // Normalize the direction vector
        double ab_length2 = ab.x * ab.x + ab.y * ab.y;
        cv::Point2f ab_norm = ab * (1.0 / std::sqrt(ab_length2));

        // Projection of point p onto the line defined by a and ab_norm
        double projection = (p - a).dot(ab_norm);
        cv::Point2f projection_point = a + ab_norm * projection;

        // Reflect p about the projection point
        cv::Point2f reflected_point = 2 * projection_point - p;

        return reflected_point;
    };

    // Reflect both points B1 and B2 across the axis
    cv::Point2f mirrored_B1 = reflectPoint(B1, A1, A2);
    cv::Point2f mirrored_B2 = reflectPoint(B2, A1, A2);

    // Return the new mirrored line as Vec4f
    return cv::Vec4f(mirrored_B1.x, mirrored_B1.y, mirrored_B2.x, mirrored_B2.y);
}


/*
Check if a bounding box is inside an image
PARAM:
    bbox: rotated rectangle representing the bounding box
    image_width: width of the image
    image_height: height of the image
*/

bool ParkingDetection::isBBoxInsideImage(const cv::RotatedRect& bbox, int image_width, int image_height) {
    // Get the four vertices of the rotated rectangle
    cv::Point2f vertices[4];
    bbox.points(vertices);

    // Check if all vertices are within the image boundaries
    for (int i = 0; i < 4; i++) {
        if (vertices[i].x < 0 || vertices[i].x >= image_width ||   // Check if x is within the image width
            vertices[i].y < 0 || vertices[i].y >= image_height) {  // Check if y is within the image height
            return false;  // If any vertex is outside the image, return false
        }
    }

    return true;  // If all vertices are inside, return true
}


/*
Check if the content inside two rectangles is equal
PARAM:
    mirroredRect: rotated rectangle representing the mirrored rectangle
    realRect: rotated rectangle representing the real rectangle
    frame: input frame
    threshold: threshold for the difference in mean pixel values
*/

bool ParkingDetection::isTheContentEqual(const cv::RotatedRect& mirroredRect, const cv::RotatedRect& realRect, const cv::Mat& frame, double threshold) {
    // Get bounding boxes for the two rectangles (mirrored and real)
    cv::Rect bbox_mirrored = mirroredRect.boundingRect();
    cv::Rect bbox_real = realRect.boundingRect();

    // Convert the frame to HSV color space
    cv::Mat frameHSV;
    cv::cvtColor(frame, frameHSV, cv::COLOR_BGR2HSV);

    // Split the HSV channels and select the Saturation channel (S)
    cv::Mat frameS;
    std::vector<cv::Mat> channels;
    cv::split(frameHSV, channels);
    frameS = channels[1];  // S channel represents the saturation

    // Ensure the bounding boxes are within image boundaries
    bbox_mirrored &= cv::Rect(0, 0, frame.cols, frame.rows); 
    bbox_real &= cv::Rect(0, 0, frame.cols, frame.rows);     

    // Extract ROI (Region of Interest) for both rectangles from the Saturation channel
    cv::Mat roi_mirrored = frameS(bbox_mirrored);
    cv::Mat roi_real = frameS(bbox_real);

    // Calculate the mean pixel values for both ROIs
    double mean_mirrored = cv::mean(roi_mirrored)[0];  
    double mean_real = cv::mean(roi_real)[0];          

    // Calculate the absolute difference between the mean values
    double difference = std::abs(mean_mirrored - mean_real);

    // Return true if the difference is below the threshold
    return difference < threshold;
}


/*
Create BBoxes from a set of lines
PARAM:
    frame: input frame
    lines: input lines
    minDistanceThreshold: minimum distance between two lines to consider them for a rectangle
    maxDistanceThreshold: maximum distance between two lines to consider them for a rectangle
    maxAngleThreshold: maximum angle difference between two lines to consider them for a rectangle
    minAreaThreshold: minimum area of the rectangle
    maxAreaThreshold: maximum area of the rectangle
*/

std::vector<BBox> ParkingDetection::createBoundingBoxes(cv::Mat frame, const std::vector<cv::Vec4f>& lines, int minDistanceThreshold, int maxDistanceThreshold, int maxAngleThreshold, double minAreaThreshold, double maxAreaThreshold) {
    std::vector<BBox> bounding_boxes;
    std::set<std::pair<int, int>> used_pairs;
    std::set<std::pair<int, int>> counter_lines;
    std::map<int, int> line_connections;  // To track how many connections each line has

    for (int i = 0; i < lines.size(); i++) {
        std::vector<std::pair<int, double>> candidates;

        // Find valid candidate lines
        for (int j = 0; j < lines.size(); j++) {
            if (i == j) continue;

            if (used_pairs.count(std::make_pair(i, j)) || used_pairs.count(std::make_pair(j, i))) {
                continue;
            }

            double distance = distanceBetweenSegments2(lines[i], lines[j]);
            double angle1 = calculateAngle(lines[i]);
            double angle2 = calculateAngle(lines[j]);

            if (distance > minDistanceThreshold && distance < maxDistanceThreshold && std::abs(angle1 - angle2) < maxAngleThreshold) {
                candidates.push_back(std::make_pair(j, distance));
            }
        }

        bool found_valid_rectangle = false;
        int nearest_line_idx = -1;
        double min_distance = DBL_MAX;

        // Find the nearest candidate based on distance
        for (const auto& candidate : candidates) {
            if (candidate.second < min_distance && counter_lines.count(candidate) < 2) {
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

            // Track the connections
            line_connections[i]++;
            line_connections[nearest_line_idx]++;

            cv::Vec4f merged_line = mergeLineSegments(line1, line2);
            cv::RotatedRect rotatedRect(cv::Point((merged_line[0] + merged_line[2]) / 2 - 10, (merged_line[1] + merged_line[3]) / 2 - 10), // Modify slightly the coordinates of the center
                                        cv::Size(calculateLength(merged_line), distanceBetweenSegments(line1, line2)),                 // For help then in parking classification
                                        calculateAngle(merged_line));

            // Calculate the area of the rectangle
            double area = rotatedRect.size.area();

            // Check if the area is within the thresholds
            if (area < minAreaThreshold || area > maxAreaThreshold)
                continue;

            // Add the new BBox to the vector
            BBox bbox(static_cast<int>(rotatedRect.center.x),
                      static_cast<int>(rotatedRect.center.y),
                      static_cast<int>(rotatedRect.size.width),
                      static_cast<int>(rotatedRect.size.height),
                      rotatedRect.angle,
                      false);  // Occupied is set to false

            bounding_boxes.push_back(bbox);

            found_valid_rectangle = true;
        }

        // If no valid rectangle was found for this line, continue to the next one
        if (!found_valid_rectangle)
            continue;
    }

    // Handle lines that are only connected to one other line
    for (const auto& connection : line_connections) {
        if (connection.second == 1) {
            int line_idx = connection.first;

            // Find the line it connects to
            for (const auto& pair : used_pairs) {
                if (pair.first == line_idx || pair.second == line_idx) {
                    int other_line_idx = (pair.first == line_idx) ? pair.second : pair.first;

                    // Create a specular bounding box mirrored across the line
                    cv::Vec4f line1 = lines[line_idx];       // Axis line
                    cv::Vec4f line2 = lines[other_line_idx]; // Line to mirror

                    // Use line1 as the axis and mirror line2 across it
                    cv::Vec4f mirrored_line = mirrorLineAcrossAxis(line1, line2);

                    // Check if the line to be mirrored is above the axis line (remember that the y-axis is inverted)
                    if (line2[1] < line1[1] && line2[3] < line1[3]) {
                        

                        // Merge the mirrored line with the axis
                        cv::Vec4f line = mergeLineSegments(line1, mirrored_line);
                        cv::Vec4f realLine = mergeLineSegments(line1, line2);

                        cv::RotatedRect realRect(cv::Point((realLine[0] + realLine[2]) / 2, (realLine[1] + realLine[3]) / 2),
                             cv::Size(calculateLength(realLine), distanceBetweenSegments(line1, line2)),
                             calculateAngle(realLine));

                        cv::RotatedRect mirroredRect(cv::Point((line[0] + line[2]) / 2 + 20, (line[1] + line[3]) / 2 + 10),  // Put down and right the center of the rectangle 
                                                     cv::Size(calculateLength(line), distanceBetweenSegments(line1, mirrored_line)), 
                                                     calculateAngle(line));

                        // If mirroredRect is inside the image and the content of the realBBox is similar to the new one (works only for the empty parking lot)
                        if (isBBoxInsideImage(mirroredRect, frame.cols, frame.rows) && isTheContentEqual(mirroredRect, realRect, frame)) {

                            // Add the mirrored BBox to the vector
                            BBox mirrored_bbox(static_cast<int>(mirroredRect.center.x),
                                            static_cast<int>(mirroredRect.center.y),
                                            static_cast<int>(mirroredRect.size.width),
                                            static_cast<int>(mirroredRect.size.height),
                                            mirroredRect.angle,
                                            false);  // Occupied is set to false

                            // Update connection counter
                            line_connections[line_idx]++;
                            line_connections[other_line_idx]++;
                            bounding_boxes.push_back(mirrored_bbox);
                        }
                    }
                }
            }
        }
    }

    return bounding_boxes;  // Return the vector of BBox objects
}


/*
Remove lines between others within a threshold distance (lines in the middle of two parking spots)
PARAM:
    lines: input lines
    xdistanceThreshold: maximum distance in x-axis
    ydistanceThreshold: maximum distance in y-axis
    lengthThreshold: minimum length of the line to keep
*/

std::vector<cv::Vec4f> ParkingDetection::removeLinesBetween(const std::vector<cv::Vec4f>& lines, double xdistanceThreshold, double ydistanceThreshold, double lengthThreshold) {
    std::vector<cv::Vec4f> filteredLines = lines;
    for(int i = 0; i < filteredLines.size(); i++) {
        for(int j = 0; j < filteredLines.size(); j++) {
            if(i == j) continue;

            double xDistance = std::abs(filteredLines[i][2] - filteredLines[j][0]);
            double yDistance = std::abs(filteredLines[i][3] - filteredLines[j][1]);
            double length = calculateLength(filteredLines[i]);
            if(xDistance < xdistanceThreshold && yDistance < ydistanceThreshold) {
                
                for(int z = 0; z < filteredLines.size(); z++) {
                    if(z == j) continue;
                    double xDistance2 = std::abs(filteredLines[j][2] - filteredLines[z][0]);
                    double yDistance2 = std::abs(filteredLines[j][3] - filteredLines[z][1]);
                    if(xDistance2 < xdistanceThreshold && yDistance2 < ydistanceThreshold || (xDistance < 15 && length > lengthThreshold)) {
                        filteredLines.erase(filteredLines.begin() + j);
                        break;
                    }
                }
            }
        }
    }
    return filteredLines;
}

/*
Check if two BBox objects are equal
PARAM:
    bbox1: first BBox
    bbox2: second BBox
*/

bool ParkingDetection::areBoxesEqual(const BBox& bbox1, const BBox& bbox2) {
    return bbox1.getX() == bbox2.getX() &&
           bbox1.getY() == bbox2.getY() &&
           bbox1.getWidth() == bbox2.getWidth() &&
           bbox1.getHeight() == bbox2.getHeight() &&
           std::abs(bbox1.getAngle() - bbox2.getAngle()) < 1e-5; // Using a small tolerance for angle comparison
}


/*
Remove duplicate BBox objects
PARAM:
    bboxes: input vector of BBox objects
*/


std::vector<BBox> ParkingDetection::getUniqueBoundingBoxes(const std::vector<BBox>& bboxes) {
    // Copy the input vector
    std::vector<BBox> uniqueBBoxes = bboxes;

    // Sort the vector to bring duplicates together (required by std::unique)
    std::sort(uniqueBBoxes.begin(), uniqueBBoxes.end(), [](const BBox& a, const BBox& b) {
        if (a.getX() != b.getX()) return a.getX() < b.getX();
        if (a.getY() != b.getY()) return a.getY() < b.getY();
        if (a.getWidth() != b.getWidth()) return a.getWidth() < b.getWidth();
        if (a.getHeight() != b.getHeight()) return a.getHeight() < b.getHeight();
        return a.getAngle() < b.getAngle();
    });

    // Use std::unique with the custom comparator to remove duplicates
    auto last = std::unique(uniqueBBoxes.begin(), uniqueBBoxes.end(), areBoxesEqual);

    // Resize the vector to remove the "extra" elements after std::unique
    uniqueBBoxes.erase(last, uniqueBBoxes.end());

    return uniqueBBoxes;
}

/*
Compute the area of intersection between two bounding boxes
PARAM:
    bbox1: first bounding box
    bbox2: second bounding box
*/

double ParkingDetection::getIntersectionArea(const BBox& bbox1, const BBox& bbox2) {
    // Create bounding boxes as Rects
    cv::Rect rect1(bbox1.getX(), bbox1.getY(), bbox1.getWidth(), bbox1.getHeight());
    cv::Rect rect2(bbox2.getX(), bbox2.getY(), bbox2.getWidth(), bbox2.getHeight());

    // Find intersection rectangle
    cv::Rect intersection = rect1 & rect2; // Intersection of two rects

    // Return the area of the intersection
    return intersection.area();
}

/*
Remove smaller BBox if intersection area exceeds threshold
PARAM:
    bboxes: input vector of BBox objects
    threshold: threshold for the intersection area
*/


std::vector<BBox> ParkingDetection::filterBoundingBoxesByIntersection(std::vector<BBox>& bboxes, double threshold) {
    std::vector<BBox> filteredBBoxes;
    
    std::vector<bool> to_remove(bboxes.size(), false); // Keep track of boxes to remove
    
    for (int i = 0; i < bboxes.size(); i++) {
        if (to_remove[i]) continue; // Skip if already marked for removal

        for (int j = i + 1; j < bboxes.size(); j++) {
            if (to_remove[j]) continue; // Skip if already marked for removal
            
            // Calculate intersection area
            double intersectionArea = getIntersectionArea(bboxes[i], bboxes[j]);

            // Calculate areas of both bounding boxes
            double area1 = bboxes[i].getWidth() * bboxes[i].getHeight();
            double area2 = bboxes[j].getWidth() * bboxes[j].getHeight();

            // Calculate the percentage of intersection relative to the smaller box
            double min_area = std::min(area1, area2);
            double overlap_ratio = intersectionArea / min_area;

            // If overlap exceeds threshold, remove the smaller BBox
            if (overlap_ratio >= threshold) {
                if (area1 > area2) {
                    to_remove[i] = true; 
                } else {
                    to_remove[j] = true; 
                }
            }
        }
    }

    // Collect remaining BBoxes
    for (int i = 0; i < bboxes.size(); i++) {
        if (!to_remove[i]) {
            filteredBBoxes.push_back(bboxes[i]);
        }
    }

    return filteredBBoxes;
}

std::vector<cv::Vec4f> ParkingDetection::enforceShortLines(std::vector<cv::Vec4f> lines, double threshold) {
    std::vector<cv::Vec4f> new_lines;
    for (const auto& line : lines){
        cv::Vec4f new_line = line;
        double length = calculateLength(line);
        if(length < threshold) {
            // Calculate the direction of the line (normalized)
            double dx = line[2] - line[0];  
            double dy = line[3] - line[1];  
            double scale = (threshold / length) / 2; 

            // Extend both ends of the line equally to double its length
            new_line[0] = line[0] - dx * scale; 
            new_line[1] = line[1] - dy * scale;
            new_line[2] = line[2] + dx * scale;
            new_line[3] = line[3] + dy * scale;
        }
        new_lines.push_back(new_line);
    }
    return new_lines;
}

std::vector<BBox> ParkingDetection::detect(const cv::Mat &frame) {
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


    // Unify the similar lines
    double distanceThreshold = 15; // Distance threshold to consider lines close
    double angleThreshold = 30;    // Angle threshold to consider lines similar (in degrees)
    double lengthThreshold = 20; // Length threshold to consider lines similar

    std::vector<cv::Vec4f> unifiedLines = unifySimilarLines(longLines, distanceThreshold, angleThreshold, lengthThreshold);

    longLines = deleteShortLines(unifiedLines, 17);

    
    // Kmeans clustering for filter the line based on the angle
    std::vector<cv::Vec4f> kmeans_lines = filterLinesByKMeans(longLines, 2, 30);


    // Divide in 2 parts the lines too long (They are lines that include two parkings)
    lines = divideLongLines(kmeans_lines, 200, 30);    

    // Sort the lines based in their position in the space 
    lines = sortLines(lines);


    // Remove xmiddle lines
    lines = removeLinesBetween(lines, 50, 17, 30);

    // Remove short lines 2
    lines = deleteShortLines(lines, 22);

    // Enforce the short lines
    lines = enforceShortLines(lines, 30.0);


    int minDistanceThreshold = 20;
    int maxDistanceThreshold = 150;
    int maxAngleThreshold = 20;
    double minAreaThreshold = 100;
    double maxAreaThreshold = 20000;

    std::vector<BBox> boundingBoxes = createBoundingBoxes(frame, lines, minDistanceThreshold, maxDistanceThreshold, maxAngleThreshold, minAreaThreshold, maxAreaThreshold);

    // Remove duplicate BBox objects
    boundingBoxes = getUniqueBoundingBoxes(boundingBoxes);

    // Filter the bounding boxes based on the intersection area
    double intersectionThreshold = 0.3;
    boundingBoxes = filterBoundingBoxesByIntersection(boundingBoxes, intersectionThreshold);
    
    
    // Reduce the size of the BBoxes
    for (BBox &bbox : boundingBoxes)
    {
        bbox.setWidth(bbox.getWidth() * 0.9);
        bbox.setHeight(bbox.getHeight() * 1.3);
    }

    
    return boundingBoxes;

}

std::vector<BBox> ParkingDetection::sortParkingsForFindId(const std::vector<BBox> parkings) {
    std::vector<BBox> parkings_copy;
    std::vector<BBox> parking_zone1;
    std::vector<BBox> parking_zone2;
    std::vector<BBox> parking_zone3;
    std::vector<BBox> parking_zone4;

    // Filter the BBox based on the parking line they belongs to.
    cv::Point p1(325, 35);
    cv::Point p2(860, 710);
 
    float slope1 = float(p2.y - p1.y) / float(p2.x - p1.x);
    float q1 = p1.y - slope1 * p1.x;

    
    cv::Point p3(500, 10);
    cv::Point p4(1240, 680);
 
    float slope2 = float(p4.y - p3.y) / float(p4.x - p3.x);
    float q2 = p3.y - slope2 * p3.x;


    cv::Point p5(660, 10);
    cv::Point p6(1270, 410);
 
    float slope3 = float(p6.y - p5.y) / float(p6.x - p5.x);
    float q3 = p5.y - slope3 * p5.x;

    for(const BBox& parking : parkings) {
        cv::Point center = cv::Point(parking.getX(), parking.getY());
        float y = slope1 * center.x + q1;
        if(center.y > y) {
            parking_zone1.push_back(parking);
        } else {
            y = slope2 * center.x + q2;
            if(center.y > y) {
                parking_zone2.push_back(parking);
            } else {
                y = slope3 * center.x + q3;
                if(center.y > y) {
                    parking_zone3.push_back(parking);
                } else {
                    parking_zone4.push_back(parking);
                }
            }
        }
    }

    // Sort each set of BBox in decreasing order of the y-coordinate
    std::sort(parking_zone1.begin(), parking_zone1.end(), [](const BBox& a, const BBox& b) { return a.getY() > b.getY(); });
    std::sort(parking_zone2.begin(), parking_zone2.end(), [](const BBox& a, const BBox& b) { return a.getY() > b.getY(); });
    std::sort(parking_zone3.begin(), parking_zone3.end(), [](const BBox& a, const BBox& b) { return a.getY() > b.getY(); });
    std::sort(parking_zone4.begin(), parking_zone4.end(), [](const BBox& a, const BBox& b) { return a.getY() > b.getY(); });

    // Merge the sorted BBox
    parkings_copy.insert(parkings_copy.end(), parking_zone1.begin(), parking_zone1.end());
    parkings_copy.insert(parkings_copy.end(), parking_zone2.begin(), parking_zone2.end());
    parkings_copy.insert(parkings_copy.end(), parking_zone3.begin(), parking_zone3.end());
    parkings_copy.insert(parkings_copy.end(), parking_zone4.begin(), parking_zone4.end());


    return parkings_copy;
}

std::vector<BBox> ParkingDetection::numberParkings(const std::vector<BBox> parkings) {
    std::vector<BBox> parkings_copy = sortParkingsForFindId(parkings);
    int i = 1;
    for(BBox& parking : parkings_copy) {
        parking.setId(i);
        i++;
    }

    return parkings_copy;
}

void ParkingDetection::draw(const cv::Mat &frame, const std::vector<BBox> parkings) {
    cv::Mat frame_copy = frame.clone();
    for (const BBox& parking : parkings) {
        cv::RotatedRect rotatedRect(cv::Point(parking.getX(), parking.getY()), cv::Size(parking.getWidth(), parking.getHeight()), parking.getAngle());
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);
        for (int i = 0; i < 4; i++)
            cv::line(frame_copy, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        
    }
    cv::imshow("Parking Detection", frame_copy);
    cv::waitKey(0);
    
}


cv::Mat ParkingDetection::drawColored(const cv::Mat &frame, const std::vector<BBox> parkings) {
    cv::Mat frame_copy = frame.clone();
    for (const BBox& parking : parkings) {
        cv::RotatedRect rotatedRect(cv::Point(parking.getX(), parking.getY()), cv::Size(parking.getWidth(), parking.getHeight()), parking.getAngle());
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);
        for (int i = 0; i < 4; i++) {
            cv::Scalar color;
            if(parking.isOccupied())
                color = cv::Scalar(0, 0, 255); // Red
            else
                color = cv::Scalar(255, 0, 0); // Blue
            cv::line(frame_copy, vertices[i], vertices[(i + 1) % 4], color, 2);
        }

        // Write in the center of the parking the Id of the BBox
        std::string text = std::to_string(parking.getId());
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, 0);
        int textX = parking.getX() - (textSize.width / 2);
        int textY = parking.getY() + (textSize.height / 2);
        if(parking.isOccupied())
            cv::putText(frame_copy, text, cv::Point(textX, textY), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        else
            cv::putText(frame_copy, text, cv::Point(textX, textY), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
    }
    cv::imshow("Parking classification", frame_copy);
    cv::waitKey(0);
    return frame_copy;
}

