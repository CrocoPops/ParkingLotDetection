#include "dataobjects/bbox.h"

BBox::BBox(int x, int y, int width, int height, double angle, bool occupied) : x(x), y(y), width(width), height(height), angle(angle), occupied(occupied) {}

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

bool BBox::isOccupied() const {
    return occupied;
}

cv::RotatedRect BBox::getRotatedRect() const {
    return cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(width, height), angle);
}

