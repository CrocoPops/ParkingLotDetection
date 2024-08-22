#include "parkingdetection.h"
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc.hpp>

ParkingDetection::ParkingDetection(std::vector<BBox> parkings) : parkings(parkings) {}


// Remove the isolated pixels that are far from the rest of the pixels
// neighbrhoodSize: size of the neighborhood to consider (rectangular)
// minPixels: minimum number of white pixels in the neighborhood to keep the pixel white

void removeIsolatedPixels(cv::Mat &img, int neighborhoodSize, int minPixels) {
    // Ensure the neighborhood size is odd to have a center pixel
    if (neighborhoodSize % 2 == 0) {
        neighborhoodSize += 1;
    }

    // Create a copy of the original image to modify
    cv::Mat output = img.clone();

    int offset = neighborhoodSize / 2;

    for (int y = offset; y < img.rows - offset; ++y) {
        for (int x = offset; x < img.cols - offset; ++x) {
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

// Based on the region's area, delete the region if the area is between minSize and maxSize

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


void deleteWeakAreas(cv::Mat &img, int numAreas, int numPixels) {
    int rows = img.rows;
    int cols = img.cols;

    // Calculate the size of each square area
    int blockSizeX = cols / numAreas;
    int blockSizeY = rows / numAreas;

    // Iterate over each block
    for (int i = 0; i < numAreas; ++i) {
        for (int j = 0; j < numAreas; ++j) {
            // Define the region of interest (ROI) for the current block
            int startX = j * blockSizeX;
            int startY = i * blockSizeY;

            // Ensure the last block covers the remaining pixels (in case cols/numAreas or rows/numAreas is not a perfect division)
            int width = (j == numAreas - 1) ? cols - startX : blockSizeX;
            int height = (i == numAreas - 1) ? rows - startY : blockSizeY;

            cv::Rect roi(startX, startY, width, height);
            cv::Mat block = img(roi);

            // Draw the rectangle on the original image
            //cv::rectangle(img, roi, cv::Scalar(127), 1); 

            // Count the number of pixels with a value of 255 in the current block
            int count = cv::countNonZero(block == 255);

            // If the count is less than or equal to numPixels, set all pixels in the block to 0
            if (count <= numPixels) {
                img(roi).setTo(cv::Scalar(0));  
            }
        }
    }
}
/*
cv::Mat imageQuantization(cv::Mat frame, int clusters, int iterations) {
    // Convert frame to a floating point format
    cv::Mat samples(frame.total(), 3, CV_32F);
    auto samples_ptr = samples.ptr<float>(0);

    // Prepare the data for k-means by flattening the image
    for(int x = 0; x < frame.rows; x++) {
        auto frame_begin = frame.ptr<uchar>(x);
        auto frame_end = frame_begin + frame.cols * frame.channels();

        for(auto frame_ptr = frame_begin; frame_ptr != frame_end; frame_ptr += frame.channels()) {
            samples_ptr[0] = static_cast<float>(frame_ptr[0]);
            samples_ptr[1] = static_cast<float>(frame_ptr[1]);
            samples_ptr[2] = static_cast<float>(frame_ptr[2]);
            samples_ptr += 3;
        }
    }

    // Perform k-means clustering
    cv::Mat labels, centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0);

    cv::kmeans(samples, clusters, labels, criteria, iterations, cv::KMEANS_PP_CENTERS, centers);

    // Create the clustered image
    cv::Mat clustered_image(frame.size(), frame.type());
    samples_ptr = samples.ptr<float>(0); // Reset the pointer to the start

    for(int x = 0; x < frame.rows; x++) {
        auto frame_ptr = clustered_image.ptr<uchar>(x);
        auto labels_ptr = labels.ptr<int>(x * frame.cols);

        for(int y = 0; y < frame.cols; y++, frame_ptr += frame.channels(), labels_ptr++) {
            int cluster_id = *labels_ptr;
            auto centers_ptr = centers.ptr<float>(cluster_id);

            frame_ptr[0] = static_cast<uchar>(centers_ptr[0]);
            frame_ptr[1] = static_cast<uchar>(centers_ptr[1]);
            frame_ptr[2] = static_cast<uchar>(centers_ptr[2]);
        }
    }

    return clustered_image;
}

*/



void ParkingDetection::detect(cv::Mat &frame) {
    // Gamma enhancement
    /*float gamma = 4.5;
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
        lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
    
    cv::Mat frame_gamma = frame.clone();
    cv::MatIterator_<cv::Vec3b> it, end;
    for(it = frame_gamma.begin<cv::Vec3b>(), end = frame_gamma.end<cv::Vec3b>(); it != end; it++) {
        (*it)[0] = lut[(*it)[0]];
        (*it)[1] = lut[(*it)[1]];
        (*it)[2] = lut[(*it)[2]];
    }
    */
    // Convert the frame to HSV color space
    cv::Mat frame_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

    cv::Mat green_mask;
    std::vector<cv::Mat> hsv_channels;
    cv::split(frame_hsv, hsv_channels);
    cv::threshold(hsv_channels[1], green_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::bitwise_not(green_mask, green_mask);
    cv::Mat frame_gray, frame_blurred, sub;
    cv::cvtColor(frame_hsv, frame, cv::COLOR_HSV2BGR);
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(frame_gray, frame_blurred, cv::Size(15, 15), 0);
    cv::subtract(frame_gray, frame_blurred, sub);

    cv::imshow("Frame gray", frame_gray);
    cv::imshow("Sub", sub);
    cv::imshow("Green mask", green_mask); 
    cv::waitKey(0);
}


/*
void ParkingDetection::detect(cv::Mat &frame) {
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    // Use Laplacian to detect the edges
    cv::Mat frame_laplacian;
    cv::Laplacian(frame_gray, frame_laplacian, CV_64F, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(frame_laplacian, frame_laplacian);

    // Apply threshold to the Laplacian image
    //cv::threshold(frame_laplacian, frame_laplacian, 100, 255, cv::THRESH_BINARY);
    cv::Mat mask_laplacian = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::threshold(frame_gray, mask_laplacian, 120, 255, cv::THRESH_BINARY);


    // SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(); // hessianThreshold adjusted
    std::vector<cv::KeyPoint> keypoints_frame;
    cv::Mat descriptors_frame;
    sift->detectAndCompute(frame_gray, cv::noArray(), keypoints_frame, descriptors_frame);
    cv::Mat frame_with_keypoints;
    cv::drawKeypoints(frame, keypoints_frame, frame_with_keypoints);

    cv::Mat mask_sift = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int i = 0; i < keypoints_frame.size(); i++)
        cv::circle(mask_sift, keypoints_frame[i].pt, 5, cv::Scalar(255), -1);
    
    //cv::dilate(mask_sift, mask_sift, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
    // Combine the mask and the frame, whether the color of the highlighted keypoints is 
    // between a given threshold (the detect the parking spots lines), then we keep them
    //cv::Mat temp;
    //cv::bitwise_and(frame, frame, temp, mask);
    //cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);
    //cv::threshold(temp, temp, 120, 255, cv::THRESH_BINARY);

    cv::Mat temp;
    cv::bitwise_and(mask_laplacian, mask_sift, temp);

    // ORB detector
    cv::Mat frame_bilateral;
    cv::bilateralFilter(frame_gray, frame_bilateral, 5, 30, 50);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(5000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(frame_bilateral, cv::noArray(), keypoints, descriptors);

    cv::Mat frame_with_keypoints_orb;
    cv::drawKeypoints(frame, keypoints, frame_with_keypoints_orb);

    // HSV color space
    cv::Mat frame_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

    // Split the channels
    std::vector<cv::Mat> hsv_channels;
    cv::split(frame_hsv, hsv_channels);

    cv::Mat v_channel = hsv_channels[2];

    sift->detectAndCompute(v_channel, cv::noArray(), keypoints_frame, descriptors_frame);

    cv::Mat frame_with_keypoints_v;
    cv::drawKeypoints(frame, keypoints_frame, frame_with_keypoints_v);

    cv::Mat mask_hsv = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int i = 0; i < keypoints.size(); i++)
        cv::circle(mask_hsv, keypoints[i].pt, 2, cv::Scalar(255), -1);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask_hsv, mask_hsv, cv::MORPH_CLOSE, element);

    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::bitwise_and(mask_hsv, mask_laplacian, mask);
    
    cv::imshow("Frame", frame);
    cv::imshow("Laplacian", frame_laplacian);
    cv::imshow("Mask Laplacian", mask_laplacian);
    cv::imshow("V Channel", v_channel);
    cv::imshow("Frame with keypoints V", frame_with_keypoints_v);
    cv::imshow("Mask HSV", mask_hsv);
    cv::imshow("Mask Laplacian", mask_laplacian);
    cv::imshow("Mask", mask);
    cv::waitKey(0);
}
*/
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

