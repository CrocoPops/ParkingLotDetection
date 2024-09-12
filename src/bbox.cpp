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

void BBox::setOccupied(cv::Mat &mask) {
    int x = this->getX();
    int y = this->getY();

    // Clone the mask to create a colored version for visualization purposes
    cv::Mat coloredMask = mask.clone();
    coloredMask.setTo(cv::Scalar(128, 128, 128), mask == 0);
    coloredMask.setTo(cv::Scalar(0, 0, 255), mask == 1);
    coloredMask.setTo(cv::Scalar(0, 255, 0), mask == 2);
   
    // Check if the pixel at the center is occupied
    cv::Scalar centerColor = coloredMask.at<cv::Vec3b>(y, x);
    if(centerColor == cv::Scalar(128, 128, 128)) {
        bool occupied = false;

        // Check the neighbors pixels from the center
        int radius = 3;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < mask.cols && ny >= 0 && ny < mask.rows) {
                    cv::Scalar neighborColor = coloredMask.at<cv::Vec3b>(ny, nx);
                    if (neighborColor != cv::Scalar(128, 128, 128)) {
                        occupied = true;
                        break;
                    }
                    
                }
            }
            if (occupied) {
                break;
            }
        }
        
        this->occupied = occupied;
    } else {
        // If the center pixel is not (128, 128, 128), it is occupied
        this->occupied = true;
    }
}


void BBox::setOccupiedfromObtainedMask(cv::Mat &mask) {
    int x = this->getX();
    int y = this->getY();

    // Check if the pixel at the center is occupied
    uchar centerColor = mask.at<uchar>(y, x);
    if(centerColor == 0) {
        bool occupied = false;

        // Check the neighbors pixels from the center
        int Xradius = 50;
        int Yradius = 15;
        for (int dy = -Yradius; dy <= Yradius; dy++) {
            for (int dx = -Xradius; dx <= Xradius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < mask.cols && ny >= 0 && ny < mask.rows) {
                    uchar neighborColor = mask.at<uchar>(ny, nx);
                    if (neighborColor == 255) {
                        occupied = true;
                        break;
                    }
                    
                }
            }
            if (occupied)  break;               
        }
        
        this->occupied = occupied;
    } else 
        // If the center pixel is not (0, 0, 0), it is occupied
        this->occupied = true;
    cv::imshow("mask", mask);

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