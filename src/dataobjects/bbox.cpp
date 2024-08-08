#include "dataobjects/bbox.h"

BBox::BBox(int x, int y, int width, int height, double angle) : x(x), y(y), width(width), height(height) {}

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
