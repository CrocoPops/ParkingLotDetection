#ifndef BBOX_H
#define BBOX_H

#include <iostream>
#include <opencv2/opencv.hpp>

class BBox {
    private:
        int id, x, y, width, height;
        double angle;
        bool occupied;
        std::vector<cv::Point> contour;
    public:
        BBox(int x = 0, int y = 0, int width = 0, int height = 0, double angle = 0, bool occupied = false);

        int getId() const;
        int getX() const;
        int getY() const;
        int getWidth() const;
        int getHeight() const;
        double getAngle() const;
        std::vector<cv::Point> getContour() const;
        cv::RotatedRect getRotatedRect() const;
        void setId(int id);
        void setX(int x);
        void setY(int y);
        void setWidth(int width);
        void setHeight(int height);
        void setAngle(double angle);
        bool isOccupied() const;
        void setOccupied(bool occupied);
        void setOccupied(cv::Mat &mask);
        void setOccupiedfromObtainedMask(cv::Mat &mask);
        void setContour(std::vector<cv::Point> contour);
        void setContour(cv::Point vertex);
        void toString();
};
#endif
