// Author: Matteo Manuzzato, ID: 2119283

#ifndef PARKINGDETECTION_H
#define PARKINGDETECTION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "bbox.h"

class ParkingDetection {
    private:
        std::vector<BBox> parkings;
        double dotProduct(const cv::Point2f& a, const cv::Point2f& b);
        double distance(const cv::Point2f& a, const cv::Point2f& b);
        cv::Point2f projectPointOntoLineSegment(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b);
        double calculateDistance(const cv::Vec4i& line1, const cv::Vec4i& line2);
        double calculateLength(const cv::Vec4f& line);
        std::vector<cv::Vec4f> deleteShortLines(const std::vector<cv::Vec4f> lines, double minLength);
        double calculateAngle(const cv::Vec4f& line);
        std::vector<cv::Vec4f> filterLinesByKMeans(const std::vector<cv::Vec4f>& lines, int K, double angleOffset);
        cv::Vec4f mergeLines(const std::vector<cv::Vec4f>& lines);
        std::vector<cv::Vec4f> unifySimilarLines(const std::vector<cv::Vec4f>& lines, double distanceThreshold, double angleThreshold, double lengthThreshold);
        cv::Point2f closestPointOnSegment(const cv::Point2f &P, const cv::Vec4f &segment);
        float distanceBetweenSegments(const cv::Vec4f &seg1, const cv::Vec4f &seg2);
        float distanceBetweenSegments2(const cv::Vec4f &seg1, const cv::Vec4f &seg2);
        bool areParallelAndClose(const cv::Vec4f &line1, const cv::Vec4f &line2, double angleThreshold, double distanceThreshold);
        cv::Vec4f mergeLineSegments(const cv::Vec4f &line_i, const cv::Vec4f &line_j);
        cv::Vec4f mergeLineCluster(const std::vector<cv::Vec4f> &cluster);
        std::vector<cv::Vec4f> divideLongLines(const std::vector<cv::Vec4f> &lines, double max_length, double offset);
        static bool compareLines(const cv::Vec4f& line1, const cv::Vec4f& line2);
        std::vector<cv::Vec4f> sortLines(const std::vector<cv::Vec4f>& lines);
        cv::Vec4f mirrorLineAcrossAxis(const cv::Vec4f& axis, const cv::Vec4f& line_to_mirror);
        bool isBBoxInsideImage(const cv::RotatedRect& bbox, int image_width, int image_height);
        bool isTheContentEqual(const cv::RotatedRect& mirroredRect, const cv::RotatedRect& realRect, const cv::Mat& frame, double threshold = 10.0);
        std::vector<BBox> createBoundingBoxes(cv::Mat frame, const std::vector<cv::Vec4f>& lines, int minDistanceThreshold, int maxDistanceThreshold, int maxAngleThreshold, double minAreaThreshold, double maxAreaThreshold);
        std::vector<cv::Vec4f> removeLinesBetween(const std::vector<cv::Vec4f>& lines, double xdistanceThreshold, double ydistanceThreshold, double lengthThreshold);
        static bool areBoxesEqual(const BBox& bbox1, const BBox& bbox2);
        std::vector<BBox> getUniqueBoundingBoxes(const std::vector<BBox>& bboxes);
        double getIntersectionArea(const BBox& bbox1, const BBox& bbox2);
        std::vector<BBox> filterBoundingBoxesByIntersection(std::vector<BBox>& bboxes, double threshold);
        std::vector<cv::Vec4f> reinforceShortLines(std::vector<cv::Vec4f> lines, double threshold);
        std::vector<BBox> sortParkingsForFindId(const std::vector<BBox> parkings);
        std::vector<BBox> numberParkings(const std::vector<BBox> parkings);
    public:
        std::vector<BBox> detect(const cv::Mat &frame);
        void draw(const cv::Mat &frame, const std::vector<BBox> parkings);
};
#endif
