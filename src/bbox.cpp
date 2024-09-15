// Author: Matteo Manuzzato, ID: 2119283

#include "bbox.h"

BBox::BBox(int x, int y, int width, int height, double angle, bool occupied) : x(x), y(y), width(width), height(height), angle(angle), occupied(occupied) {}

int BBox::getId() const {
    return id;
}

int BBox::getX() const {
    return x;
}

int BBox::getY() const {
    return y;
}

int BBox::getWidth() const {
    return width;
}

int BBox::getHeight() const {
    return height;
}

double BBox::getAngle() const {
    return angle;
}

std::vector<cv::Point> BBox::getContour() const {
    return contour;
}

cv::RotatedRect BBox::getRotatedRect() const {
    return cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(width, height), angle);
}

void BBox::setId(int id) {
    this->id = id;
}

void BBox::setX(int x) {
    this->x = x;
}

void BBox::setY(int y) {
    this->y = y;
}

void BBox::setWidth(int width) {
    this->width = width;
}

void BBox::setHeight(int height) {
    this->height = height;
}

void BBox::setAngle(double angle) {
    this->angle = angle;
}

bool BBox::isOccupied() const {
    return occupied;
}

void BBox::setOccupied(bool occupied) {
    this->occupied = occupied;
}

void BBox::setOccupiedfromObtainedMask(cv::Mat &mask) {
    int x = this->getX();
    int y = this->getY();
    int width = this->getWidth(); 
    int height = this->getHeight();
    int angle = this->getAngle();
    cv::RotatedRect rect = cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(width, height), angle);

    cv::Point2f vertices[4];
    rect.points(vertices);

    // Convert the vertices to integer
    cv::Point intVertices[4];
    for (int i = 0; i < 4; i++)
        intVertices[i] = cv::Point(static_cast<int>(vertices[i].x), static_cast<int>(vertices[i].y));

    // Create a mask in the ROI of the BBox considered
    cv::Mat roiMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::fillConvexPoly(roiMask, intVertices, 4, cv::Scalar(255));

    // Calculate the intersection of the ROI mask and the original mask
    cv::Mat intersection;
    cv::bitwise_and(mask, roiMask, intersection);

    // Calculate the white pixel count and total pixel count in the ROI
    int whitePixelCount = cv::countNonZero(intersection);
    int totalPixelCount = rect.size.area();
    
    double whitePixelRatio = static_cast<double>(whitePixelCount) / totalPixelCount;
    
    double whitePixelThreshold = 0.3;
    bool occupiedArea = (whitePixelRatio > whitePixelThreshold);

    // Now we check if the center of the bounding box is occupied
    uchar centerColor = mask.at<uchar>(y, x);
    bool occupiedCenter;
    if(centerColor == 0) {
        occupiedCenter = false;

        // Check the neighbors pixels from the center
        int Xradius = 10;
        int Yradius = 5;
        for (int dy = -Yradius; dy <= Yradius; dy++) {
            for (int dx = -Xradius; dx <= Xradius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < mask.cols && ny >= 0 && ny < mask.rows) {
                    uchar neighborColor = mask.at<uchar>(ny, nx);
                    if (neighborColor == 255) {
                        occupiedCenter = true;
                        break;
                    }
                    
                }
            }
        }
    }
    else
        occupiedCenter = true;

    if(occupiedArea || occupiedCenter)
        this->occupied = true;
    else
        this->occupied = false;
    
}

void BBox::setContour(std::vector<cv::Point> contour) {
    this->contour = contour;
}

void BBox::setContour(cv::Point vertex) {
    this->contour.push_back(vertex);
}

void BBox::toString() {
    int id, x, y, width, height;
        double angle;
        bool occupied;
    std::cout << "Id: " << this->id << "\nCenter: (" << this->x << ", " << this->y << ")\nWidth: " << this->width << "\nHeight: " << this->height << "\nAngle: " << this->angle << "\n\n";
} 