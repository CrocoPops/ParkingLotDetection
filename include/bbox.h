#ifndef BBOX_H
#define BBOX_H

#include <iostream>
#include <opencv2/opencv.hpp>

class BBox {
    private:
        int x, y, width, height;
        double angle;
        bool occupied;
    public:
        BBox(int x = 0, int y = 0, int width = 0, int height = 0, double angle = 0, bool occupied = false);

        int getX() const;
        int getY() const;
        int getWidth() const;
        int getHeight() const;
        double getAngle() const;
        bool isOccupied() const;
        void setOccupied(cv::Mat &mask);
        void setOccupiedfromObtainedMask(cv::Mat &mask);
        cv::RotatedRect getRotatedRect() const;
};
#endif
