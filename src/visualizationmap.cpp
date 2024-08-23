#include "visualizationmap.h"
#include <opencv2/opencv.hpp>

void VisualizationMap::addBBox(std::vector<cv::Point2f> points) {
    bboxes.push_back(points);
}

std::vector<std::vector<cv::Point2f>> VisualizationMap::getBBoxes() {
    return bboxes;
}

void VisualizationMap::colorBBoxes(cv::Mat &map, std::vector<BBox> bboxes) {
    for (int i = 0; i < bboxes.size(); i++) {
        // Get the 4 vertices of the rotated rectangle
        std::vector<cv::Point2f> vertices = getBBoxes()[i];
        bool occupied = bboxes[i].isOccupied();

        std::vector<cv::Point> vertices_int;
        for (int j = 0; j < 4; j++)
            vertices_int.push_back(cv::Point(vertices[j].x, vertices[j].y));

        cv::Scalar color = occupied ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
        cv::fillConvexPoly(map, vertices_int, color);

        cv::polylines(map, vertices_int, true, cv::Scalar(0, 0, 0), 2);
    }
}

cv::Mat VisualizationMap::drawParkingMap(cv::Mat &frame, std::vector<BBox> bboxes) {
    // Create a parking map (a smaller image)
    int mapHeight = frame.rows / 3;
    int mapWidth = frame.cols / 3;
    
    // Create a white background map
    cv::Mat map = cv::Mat::zeros(mapHeight, mapWidth, CV_8UC3);
    map.setTo(cv::Scalar(255, 255, 255));

    cv::Mat map_bbox = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
    map_bbox.setTo(cv::Scalar(255, 255, 255));

    // Define the size of each parking spot
    cv::Size2f rectSize(40, 20);

    // Offsets
    int rowOffset = 12;  // Vertical offset between rows
    int extraOffset = 20;  // Extra vertical offset between specific rows

    // Start position for the bottom-right parking spot (1st row, 10 spots)
    int startX = mapWidth - 50;
    int startY = mapHeight - 30;

    // Angle for the rotation (in degrees)
    float angle;

    // Draw parking spots row by row
    for (int row = 0; row < 5; row++) {
        int spotsInRow;
        if (row == 0 || row == 1) {
            spotsInRow = 10;  // First two rows have 10 spots each
        } else if (row == 2 || row == 3) {
            spotsInRow = 9;   // Next two rows have 9 spots each
        } else {
            spotsInRow = 2;   // Last row has 3 spots
        }

        // Adjust start position for each row
        int rowStartX = startX;  // Horizontal start position
        int rowStartY = startY - (row * (rectSize.height + rowOffset));  // Vertical start position with basic offset

        // Apply extra padding between specific rows
        if (row == 2) {
            rowStartY -= extraOffset;  // Extra padding before the third row
        } else if (row == 4) {
            rowStartY -= extraOffset * 2;  // Extra padding before the fifth row
        } else if (row == 3) {
            rowStartY -= extraOffset;  // Compensate for overlap between third and fourth row
        }


        if(row == 2 || row == 4){
            angle = 45.0f;
            rowStartX += 10;
        }
        else angle = -45.0f;


        for (int i = 0; i < spotsInRow; i++) {
            int x = rowStartX - i * (rectSize.width - 5);  // Adjust horizontal spacing between spots
            int y = rowStartY;

            // Create a rotated rectangle
            cv::RotatedRect rotatedRect(cv::Point2f(x, y), rectSize, angle);

            // Get the 4 vertices of the rotated rectangle
            cv::Point2f vertices[4];
            rotatedRect.points(vertices);

            std::vector<cv::Point2f> points(vertices, vertices + 4);
            addBBox(points);

            // Draw the rotated rectangle on the map
            for (int j = 0; j < 4; j++) {
                cv::line(map, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 0, 0), 3);
                // cv::line(map_bbox, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 0, 0), 3);
            }
        }
    }

    // Position the map in the bottom-left corner of the frame
    cv::Mat roi = frame(cv::Rect(20, frame.rows - mapHeight - 20, mapWidth, mapHeight));

    colorBBoxes(map, bboxes);
    
    // Blend or copy the map onto the ROI
    cv::add(roi, map, roi);

    // Display the result
    cv::imshow("Frame with Map", frame);
    cv::waitKey(0);
    
    return frame;
}
