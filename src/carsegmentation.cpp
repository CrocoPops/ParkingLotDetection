#include "carsegmentation.h"

CarSegmentation::CarSegmentation() {}

CarSegmentation::~CarSegmentation() {}

void CarSegmentation::regionGrowing(cv::Mat frame, cv::Mat mask, cv::Mat &result, int threshold) {
    result = cv::Mat::zeros(frame.size(), frame.type());

    std::stack<cv::Point> stack;
    
    for(int y = 0; y < mask.rows; y++) {
        for(int x = 0; x < mask.cols; x++) {
            if(mask.at<uchar>(y, x) == 255) {
                stack.push(cv::Point(x, y));
            }
        }
    }

    const cv::Point PointShift2D[8] = {
        cv::Point(-1, 0), cv::Point(1, 0), cv::Point(0, -1), cv::Point(0, 1),
        cv::Point(-1, -1), cv::Point(-1, 1), cv::Point(1, -1), cv::Point(1, 1)
    };

    while(!stack.empty()) {
        cv::Point center = stack.top();
        stack.pop();

        mask.at<uchar>(center) = 1;

        result.at<cv::Vec3b>(center) = cv::Vec3b(0, 0, 255);

        for(int i = 0; i < 8; i++) {
            cv::Point neighbor = center + PointShift2D[i];
 
            if(neighbor.x >= 0 && neighbor.x < frame.cols && neighbor.y >= 0 && neighbor.y < frame.rows) {

                int delta = int(std::pow(frame.at<cv::Vec3b>(center)[0] - frame.at<cv::Vec3b>(neighbor)[0], 2) +
                                std::pow(frame.at<cv::Vec3b>(center)[1] - frame.at<cv::Vec3b>(neighbor)[1], 2) +
                                std::pow(frame.at<cv::Vec3b>(center)[2] - frame.at<cv::Vec3b>(neighbor)[2], 2));

                if(result.at<cv::Vec3b>(neighbor) == cv::Vec3b(0, 0, 0) && mask.at<uchar>(neighbor) == 0 && delta < threshold) {
                    stack.push(neighbor);
                    mask.at<uchar>(neighbor) = 1;
                }
            }
        }
    }
}


cv::Mat CarSegmentation::detectCarsTrue(cv::Mat &frame, cv::Mat &mask) {
    cv::Mat coloredMask = mask.clone();
    coloredMask.setTo(cv::Scalar(128, 128, 128), mask == 0);
    coloredMask.setTo(cv::Scalar(0, 0, 255), mask == 1);
    coloredMask.setTo(cv::Scalar(0, 255, 0), mask == 2);

    cv::Mat result;
    cv::addWeighted(coloredMask, 0.7, frame, 1, 0, result);

    cv::imshow("Contours", result);
    cv::waitKey(0);

    return coloredMask;
}


cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    cv::Mat blurred;
    cv::pyrMeanShiftFiltering(frame, blurred, 20, 45, 2);

    cv::Mat frame_hsv;
    cv::cvtColor(blurred, frame_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(frame_hsv, hsv_channels);

    cv::Mat v_channel = hsv_channels[2];

    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int x = 0; x < v_channel.rows; x++) {
        for(int y = 0; y < v_channel.cols; y++) {
            if(v_channel.at<uchar>(x, y) > 200 || v_channel.at<uchar>(x, y) < 45) {
                mask.at<uchar>(x, y) = 255;
            }
        }
    }

    cv::Mat green_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::Mat parking_hsv;
    cv::cvtColor(empty_parkings[3], parking_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> parking_hsv_channels;
    cv::split(parking_hsv, parking_hsv_channels);
    cv::threshold(parking_hsv_channels[1], green_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat element5x5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat element7x7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::Mat element9x9 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::Mat element15x15 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN, element5x5);
    cv::dilate(green_mask, green_mask, element5x5);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, element5x5);
    cv::bitwise_not(green_mask, green_mask);

    cv::bitwise_and(mask, green_mask, mask);

    //cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element5x5);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element15x15);
    //cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element9x9);

    // Delete small connected components
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for(int i = 0; i < contours.size(); i++) {
        if(cv::contourArea(contours[i]) < 400) {
            cv::drawContours(mask, contours, i, cv::Scalar(0), cv::FILLED);
        }
    }

    cv::Mat result;
    cv::Mat blue_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
    blue_mask.setTo(cv::Scalar(255, 0, 0), mask == 255);
    cv::addWeighted(frame, 1, blue_mask, 0.7, 0, result);

    cv::imshow("Frame", frame);
    cv::imshow("Blurred", blurred);
    cv::imshow("HSV", frame_hsv);
    cv::imshow("Hue", hsv_channels[0]);
    cv::imshow("Saturation", hsv_channels[1]);
    cv::imshow("Value", hsv_channels[2]);
    cv::imshow("Mask", mask);
    cv::imshow("Green Mask", green_mask);
    cv::imshow("Result", result);
    cv::waitKey(0);
    return mask;  // Return the result mask
}


/*
// WORKS WELL
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    // Apply mean shift filtering to segment the image
    cv::Mat blurred;
    cv::pyrMeanShiftFiltering(frame, blurred, 20, 45, 1.5);

    // Initialize a mask with the same size as the blurred image
    cv::Mat mask = cv::Mat::zeros(blurred.size(), CV_8UC1);

    // Filter by intensity
    cv::Mat frame_gray;
    cv::cvtColor(blurred, frame_gray, cv::COLOR_BGR2GRAY);
    for(int x = 0; x < frame_gray.rows; x++) {
        for(int y = 0; y < frame_gray.cols; y++) {
            if(frame_gray.at<uchar>(x, y) > 150 || frame_gray.at<uchar>(x, y) < 60) {
                mask.at<uchar>(x, y) = 255;
            }
        }
    }
    
    for(int i = 0; i < blurred.rows; i++) {
        for(int j = 0; j < blurred.cols; j++) {
            cv::Vec3b pixel = blurred.at<cv::Vec3b>(i, j);
            if(pixel[0] > 200 && pixel[1] > 200 && pixel[2] > 200) {
                mask.at<uchar>(i, j) = 255;  // White
            } else if(pixel[0] < 40 && pixel[1] < 40 && pixel[2] < 40) {
                mask.at<uchar>(i, j) = 255;  // Black
            } else if(pixel[2] > 100 && pixel[0] < 60 && pixel[1] < 60) {
                mask.at<uchar>(i, j) = 255;  // Red
            }
        }
    }
    
    cv::Mat element3x3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat element5x5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat element7x7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::Mat element15x15 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::Mat element21x21 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element15x15);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element3x3);

    // Create mask to filter out green areas (trees)
    cv::Mat green_mask = cv::Mat::zeros(blurred.size(), CV_8UC1);
    cv::Mat parking_hsv;
    cv::cvtColor(empty_parkings[0], parking_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(parking_hsv, hsv_channels);
    cv::threshold(hsv_channels[1], green_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN, element15x15);
    cv::dilate(green_mask, green_mask, element15x15);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, element21x21);
    cv::bitwise_not(green_mask, green_mask);

    
    cv::Mat edges;
    cv::Canny(blurred, edges, 75, 225);
    cv::bitwise_and(edges, green_mask, edges);

    cv::bitwise_and(mask, green_mask, mask);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element5x5);
    //cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element21x21);
    // Delete small connected components
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for(int i = 0; i < contours.size(); i++)
        if(cv::contourArea(contours[i]) < 400)
            cv::drawContours(mask, contours, i, cv::Scalar(0), cv::FILLED);

    cv::Mat output = frame.clone();
    cv::Mat red_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
    red_mask.setTo(cv::Scalar(0, 0, 255), mask == 255);
    cv::addWeighted(frame, 1, red_mask, 0.7, 0, output);

    // Display the results
    cv::imshow("Frame", frame);
    cv::imshow("Mean Shift Segmentation", blurred);
    cv::imshow("Green Mask", green_mask);
    cv::imshow("Edges", edges);
    cv::imshow("Mean Shift Mask", mask);
    cv::imshow("Result", output);
    cv::waitKey(0);
    
    return cv::Mat();  // Return the result mask
}*/


/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Mat> empty_parking_subtracted;
    for(cv::Mat empty_parking : empty_parkings) {
        cv::Mat empty_parking_gray;
        cv::cvtColor(empty_parking, empty_parking_gray, cv::COLOR_BGR2GRAY);
        cv::Mat diff(frame.size(), CV_8UC1);
        for(int x = 0; x < frame_gray.rows; x++)
            for(int y = 0; y < frame_gray.cols; y++)
                diff.at<uchar>(x, y) = std::abs(frame_gray.at<uchar>(x, y) - empty_parking_gray.at<uchar>(x, y));
        empty_parking_subtracted.push_back(diff);
    }

    cv::Mat median = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int x = 0; x < frame_gray.rows; x++) {
        for(int y = 0; y < frame_gray.cols; y++) {
            std::vector<uchar> values;
            for(const cv::Mat& empty_parking_sub : empty_parking_subtracted) {
                values.push_back(empty_parking_sub.at<uchar>(x, y));
            }
            std::sort(values.begin(), values.end());
            median.at<uchar>(x, y) = values[values.size() / 2];
        }
    }

    cv::Mat mask;
    cv::threshold(median, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat edges;
    cv::Canny(frame_gray, edges, 50, 150);

    // MSER (Maximally Stable Extremal Regions) to detect RoIs
    cv::Ptr<cv::MSER> mser = cv::MSER::create();
    std::vector<std::vector<cv::Point>> mser_regions;
    std::vector<cv::Rect> mser_bboxes;
    mser->detectRegions(frame_gray, mser_regions, mser_bboxes);

    cv::Mat mser_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(const auto& contour : mser_regions)
        if(cv::contourArea(contour) > 1000)
            cv::drawContours(mser_mask, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), cv::FILLED);
    
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(mser_mask, mser_mask, cv::MORPH_CLOSE, element);

    // Delete small regions
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for(int i = 0; i < contours.size(); i++)
        if(cv::contourArea(contours[i]) < 250)
            cv::drawContours(mask, contours, i, cv::Scalar(0), cv::FILLED);

    cv::Mat output;
    cv::bitwise_and(mser_mask, mask, output);

    // Canny for contours detection
    cv::Mat edges_inv;
    cv::bitwise_not(edges, edges_inv);
    cv::bitwise_and(edges_inv, output, output);

    //cv::morphologyEx(output, output, cv::MORPH_CLOSE, cv::Mat::ones(7, 7, CV_8U));

       
    cv::morphologyEx(output, output, cv::MORPH_CLOSE, cv::Mat::ones(11, 11, CV_8U));

    cv::imshow("Frame", frame); 
    cv::imshow("Median", median);
    cv::imshow("Edges", edges);
    cv::imshow("Mask", mask);
    cv::imshow("MSER Mask", mser_mask);
    cv::imshow("Output", output);
    cv::waitKey(0);
    return cv::Mat();
}
*/


/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    // Convert the current frame to grayscale
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Mat> empty_parking_subtracted;

    // Subtract each empty parking lot image from the current frame
    for(cv::Mat empty_parking : empty_parkings) {
        cv::Mat empty_parking_gray;
        cv::cvtColor(empty_parking, empty_parking_gray, cv::COLOR_BGR2GRAY);

        cv::Mat diff;
        cv::absdiff(frame_gray, empty_parking_gray, diff); // Compute absolute difference
        empty_parking_subtracted.push_back(diff); // Store the difference image
    }

    // Calculate the median of the subtracted images
    cv::Mat median(frame.size(), CV_8UC1);
    for(int x = 0; x < median.rows; x++) {
        for(int y = 0; y < median.cols; y++) {
            std::vector<uchar> values;
            for(const cv::Mat& empty_parking_sub : empty_parking_subtracted) {
                values.push_back(empty_parking_sub.at<uchar>(x, y));
            }
            std::sort(values.begin(), values.end());
            median.at<uchar>(x, y) = values[values.size() / 2];
        }
    }

    // Display the subtracted images (for debugging)
    for(const cv::Mat& empty_parking_sub : empty_parking_subtracted) {
        cv::imshow("Subtracted", empty_parking_sub);
        cv::waitKey(0);
    }

    // Display the median image
    cv::imshow("Median", median);
    cv::waitKey(0);

    return median;
}
*/
/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) { 
    // Background subtraction with every empty parking lot, then take the median
    cv::Mat frame_yuv;
    cv::cvtColor(frame, frame_yuv, cv::COLOR_BGR2YUV);

    std::vector<cv::Mat> yuv_channels;
    cv::split(frame_yuv, yuv_channels);

    cv::Mat frame_y = yuv_channels[0];

    std::vector<cv::Mat> empty_parking_subtracted;
    for(const cv::Mat& empty_parking : empty_parkings) {
        cv::Mat empty_parking_yuv;
        cv::cvtColor(empty_parking, empty_parking_yuv, cv::COLOR_BGR2YUV);

        std::vector<cv::Mat> empty_parking_channels;
        cv::split(empty_parking_yuv, empty_parking_channels);

        cv::Mat empty_parking_y = empty_parking_channels[0];

        cv::Mat diff;
        cv::absdiff(frame_y, empty_parking_y, diff);
        empty_parking_subtracted.push_back(diff);
    }

    cv::Mat median(frame.size(), CV_8UC1);
    for(int x = 0; x < median.rows; x++) {
        for(int y = 0; y < median.cols; y++) {
            std::vector<uchar> values;
            for(const cv::Mat& empty_parking_sub : empty_parking_subtracted) {
                values.push_back(empty_parking_sub.at<uchar>(x, y));
            }
            std::sort(values.begin(), values.end());
            median.at<uchar>(x, y) = values[values.size() / 2];
        }
    }

    cv::imshow("Frame", frame);
    cv::imshow("Median", median);
    cv::waitKey(0);
    return cv::Mat();
}

/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) { 
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    // SIFT to detect keypoints and descriptors of the parking lot
    // In this part of the code, we are detecting keypoints and descriptors of the current frame
    // We will use these keypoints to filter out the keypoints that are also present in the empty parking lot
    // The keypoints that are not present in the empty parking lot are considered as RoIs in which probably there
    // is a car
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_frame;
    cv::Mat descriptors_frame;
    sift->detectAndCompute(frame_gray, cv::noArray(), keypoints_frame, descriptors_frame);

    cv::Mat sift_mask = cv::Mat::zeros(frame.size(), CV_8UC1);

    // Initialize matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    for(const cv::Mat& empty_parking : empty_parkings) {
        cv::Mat empty_parking_gray;
        cv::cvtColor(empty_parking, empty_parking_gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> keypoints_empty_parking;
        cv::Mat descriptors_empty_parking;
        sift->detectAndCompute(empty_parking_gray, cv::noArray(), keypoints_empty_parking, descriptors_empty_parking);

        // Match descriptors using KNN
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descriptors_frame, descriptors_empty_parking, knn_matches, 2);

        // Ratio test to filter matches
        const float ratio_thresh = 0.9f;
        std::vector<cv::DMatch> good_matches;
        std::vector<bool> matched_keypoints(keypoints_frame.size(), false);

        int matches = 0;
        for (int i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
                matched_keypoints[knn_matches[i][0].queryIdx] = true;
                matches++;
            }
        }     

        // Draw only the good matches
        cv::Mat img_matches;
        cv::drawMatches(frame, keypoints_frame, empty_parking, keypoints_empty_parking, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Display matches
        cv::imshow("Matches", img_matches);
        cv::waitKey(0);

        // Use matched_keypoints to filter out keypoints in the current frame
        for (int i = 0; i < keypoints_frame.size(); i++) {
            // If the keypoint is not matched, draw a rectangle around it
            if (!matched_keypoints[i]) {
                // Get the center of the keypoint
                cv::Point2f center = keypoints_frame[i].pt;
                // Define the size of the rectangle (7x7 as an example)
                int rect_size = 7;
                // Define the top-left and bottom-right points of the rectangle
                cv::Point2f top_left(center.x - rect_size, center.y - rect_size);
                cv::Point2f bottom_right(center.x + rect_size, center.y + rect_size);
                // Draw the rectangle on the mask
                cv::rectangle(sift_mask, top_left, bottom_right, cv::Scalar(255), -1);
            }
        }
    }

    cv::dilate(sift_mask, sift_mask, cv::Mat::ones(9, 9, CV_8U));
    //cv::morphologyEx(sift_mask, sift_mask, cv::MORPH_CLOSE, cv::Mat::ones(7, 7, CV_8U));

    // MSER (Maximally Stable Extremal Regions) to detect RoIs
    cv::Mat output, final_mask;
    cv::Ptr<cv::MSER> mser = cv::MSER::create();
    std::vector<std::vector<cv::Point>> mser_regions;
    std::vector<cv::Rect> mser_bboxes;
    mser->detectRegions(frame_gray, mser_regions, mser_bboxes);

    cv::Mat mser_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(const auto& contour : mser_regions)
        if(cv::contourArea(contour) > 1000)
            cv::drawContours(mser_mask, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), cv::FILLED);
    

    // Canny edge detection
    cv::Mat frame_blurred, frame_edges, thresh;
    cv::GaussianBlur(frame_gray, frame_blurred, cv::Size(11, 11), 0);
    cv::Canny(frame_blurred, frame_edges, 50, 150);
    cv::threshold(frame_edges, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::dilate(thresh, thresh, cv::Mat::ones(3, 3, CV_8U));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat canny_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int i = 0; i < contours.size(); i++)
        if(cv::contourArea(contours[i]) > 1000)
            cv::drawContours(canny_mask, contours, i, cv::Scalar(255), cv::FILLED);

    // Canny edge detection, compute the borders of the cars in order
    // to distinguish them from the background

    cv::dilate(frame_gray, frame_gray, cv::Mat::ones(7, 7, CV_8U));
    cv::Canny(frame_gray, frame_edges, 220, 660);

    


    cv::Mat edge_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    edge_mask.setTo(255, frame_edges == 0);

    // Combine the masks
    cv::Mat temp1, temp2;
    cv::bitwise_and(sift_mask, mser_mask, temp1);
    cv::bitwise_and(sift_mask, canny_mask, temp2);
    cv::bitwise_or(temp1, temp2, final_mask);
    cv::bitwise_and(final_mask, edge_mask, final_mask);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(final_mask, final_mask, cv::MORPH_CLOSE, element);
    
    //cv::bitwise_and(canny_mask, edge_mask, temp1);
    //cv::bitwise_and(mser_mask, edge_mask, temp2);
    //cv::bitwise_and(sift_mask, temp1, temp1);
    //cv::bitwise_and(sift_mask, temp2, temp2);
    //cv::bitwise_or(temp1, temp2, final_mask);


    frame.copyTo(output, final_mask);
    

    //cv::morphologyEx(final_mask, final_mask, cv::MORPH_CLOSE, cv::Mat::ones(11, 11, CV_8U));

    output.setTo(cv::Scalar(0, 0, 255), final_mask != 0);

    cv::Mat img;
    cv::addWeighted(frame, 1, output, 0.7, 0, img);
    
    // Display the final keypoints and masked output
    cv::imshow("Frame", frame);
    cv::imshow("SIFT Mask", sift_mask);
    cv::imshow("MSER Mask", mser_mask);
    cv::imshow("Canny Mask", canny_mask);
    cv::imshow("Canny edge mask", edge_mask);
    cv::imshow("SIFT + MSER", temp1);
    cv::imshow("SIFT + Canny", temp2);
    cv::imshow("Final Mask", final_mask);
    cv::imshow("Result", img);
    cv::waitKey(0);

    return output;
}
*/



/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) { 
    // SIFT to detect keypoints and descriptors of the parking lot
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_frame;
    cv::Mat descriptors_frame;
    sift->detectAndCompute(frame_gray, cv::noArray(), keypoints_frame, descriptors_frame);

    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    for(const auto& empty_parking : empty_parkings) {
        cv::Mat empty_parking_gray;
        cv::cvtColor(empty_parking, empty_parking_gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> keypoints_empty_parking;
        cv::Mat descriptors_empty_parking;
        sift->detectAndCompute(empty_parking_gray, cv::noArray(), keypoints_empty_parking, descriptors_empty_parking);

        std::vector<std::vector<cv::DMatch>> matches;
        matcher->knnMatch(descriptors_frame, descriptors_empty_parking, matches, 2);

        const float ratio_thresh = 0.75f;
        std::vector<bool> matched_keypoints(keypoints_frame.size(), false);

        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i].size() >= 2 && matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
                matched_keypoints[matches[i][0].queryIdx] = true;
            }
        }
        std::vector<cv::KeyPoint> car_keypoints;
        for (size_t i = 0; i < keypoints_frame.size(); i++) {
            if (!matched_keypoints[i]) {
                car_keypoints.push_back(keypoints_frame[i]);
            }
        }

        keypoints_frame = car_keypoints;
    }
    std::cout << "Here" << std::endl;

    cv::Mat output = frame.clone();
    cv::drawKeypoints(frame, keypoints_frame, output, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("Keypoints", output);
    cv::waitKey(0);
    
    return cv::Mat();
}
*/
/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    cv::Mat gray, stretched_gray, blurred, edges, thresh, mask, hsv_frame;

    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    //cv::dilate(gray, gray, cv::Mat::ones(5, 5, CV_8U));

    // Apply Gaussian Blur
    cv::GaussianBlur(gray, blurred, cv::Size(7, 7), 0);

    // Detect edges using Canny
    cv::Canny(blurred, edges, 50, 150);

    // Apply Otsu's thresholding
    cv::threshold(edges, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Dilate the thresholded image to close gaps
    cv::dilate(thresh, thresh, cv::Mat::ones(3, 3, CV_8U));
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a mask
    mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if(area > 1000)
            cv::drawContours(mask, contours, (int)i, cv::Scalar(255), cv::FILLED);
    } 

    // Discard if it doesn't 

    cv::Mat result;
    frame.copyTo(result, mask);

    // Display results
    cv::imshow("Frame", frame);
    cv::imshow("Segmented Cars", result);
    cv::imshow("Binary Mask", mask);
    cv::waitKey(0);

    return mask;
}
*/
/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    cv::Mat frame_gray, best_background, mask;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

    double min_mse = std::numeric_limits<double>::max();
    cv::Mat best_empty_parking;

    for(const auto& empty_parking : empty_parkings) {
        cv::Mat empty_parking_gray;
        cv::cvtColor(empty_parking, empty_parking_gray, cv::COLOR_BGR2GRAY);

        cv::Mat diff;
        cv::absdiff(frame_gray, empty_parking_gray, diff);
        cv::Mat squared_diff;
        cv::multiply(diff, diff, squared_diff);

        double mse = cv::sum(squared_diff)[0] / (frame_gray.rows * frame_gray.cols);
        if(mse < min_mse) {
            min_mse = mse;
            best_empty_parking = empty_parking;
        }
    }
    cv::Mat best_empty_parking_gray;
    cv::cvtColor(best_empty_parking, best_empty_parking_gray, cv::COLOR_BGR2GRAY);

    cv::absdiff(frame_gray, best_empty_parking_gray, mask);
    cv::threshold(mask, mask, 50, 255, cv::THRESH_BINARY);

    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat::ones(3, 3, CV_8U));
    
    cv::imshow("Frame", frame);
    cv::imshow("Best Empty Parking", best_empty_parking);
    cv::imshow("Foreground Mask", mask);
    cv::waitKey(0);

    return cv::Mat();
}
*/




/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, cv::Mat empty_parking) {

    cv::dilate(frame, frame, cv::Mat::ones(5, 5, CV_8U));
    static cv::Ptr<cv::BackgroundSubtractor> backSub = cv::createBackgroundSubtractorKNN();

    cv::Mat fgMask, frame_gray, frame_yuv;
    cv::cvtColor(frame, frame_yuv, cv::COLOR_BGR2YUV);

    // Apply background subtraction
    backSub->apply(frame, fgMask, 0.0001);

    // Split the YUV channels
    std::vector<cv::Mat> channels;
    cv::split(frame_yuv, channels);

    // Reset the shadow mask for each frame
    cv::Mat shadowMask = cv::Mat::zeros(frame.size(), CV_8UC1);

    // Shadow removal (adjust the threshold as necessary)
    shadowMask = (channels[0] > 75) & (fgMask == 255);

    // Set the shadow regions in the fgMask to 0
    fgMask.setTo(0, shadowMask);

    // Make sure the mask is binary
    cv::threshold(fgMask, fgMask, 127, 255, cv::THRESH_BINARY);

    // Apply morphological operations to clean up the mask
    cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
    cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, cv::Mat::ones(3, 3, CV_8U));
    cv::Mat result = frame.clone();
    cv::Mat mask_bgr;
    cv::cvtColor(fgMask, mask_bgr, cv::COLOR_GRAY2BGR);
    cv::addWeighted(frame, 1, mask_bgr, 0.7, 0, result);

    cv::imshow("Frame", frame);
    cv::imshow("Foreground Mask", fgMask);
    cv::imshow("Shadow Mask", shadowMask);
    cv::imshow("Foreground Mask without shadows", fgMask);
    cv::imshow("Result", result);
    cv::waitKey(0);

    return fgMask;
}
*/


/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, cv::Mat empty_parking) {
    cv::Mat frame_gray, empty_parking_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(empty_parking, empty_parking_gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints_frame, keypoints_empty_parking;
    cv::Mat descriptors_frame, descriptors_empty_parking;

    sift->detectAndCompute(frame_gray, cv::noArray(), keypoints_frame, descriptors_frame);
    sift->detectAndCompute(empty_parking_gray, cv::noArray(), keypoints_empty_parking, descriptors_empty_parking);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors_frame, descriptors_empty_parking, matches);

    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::Mat keypoint_density = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(const auto& kp : keypoints_frame) 
        if(kp.pt.x >= 0 && kp.pt.x < frame.cols && kp.pt.y >= 0 && kp.pt.y < frame.rows) 
            keypoint_density.at<float>(cv::Point(static_cast<int>(kp.pt.x), static_cast<int>(kp.pt.y))) += 1.0f;

    cv::Mat density_mask;
    cv::normalize(keypoint_density, density_mask, 0, 255, cv::NORM_MINMAX);
    cv::threshold(density_mask, density_mask, 50, 255, cv::THRESH_BINARY);

    cv::Mat morph_mask;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(density_mask, morph_mask, cv::MORPH_CLOSE, element);

    cv::Mat mask_3c;
    cv::cvtColor(morph_mask, mask_3c, cv::COLOR_GRAY2BGR);

    std::cout << frame.type() << std::endl;
    std::cout << mask_3c.type() << std::endl;
    frame.convertTo(frame, CV_8UC3);
    cv::Mat result;
    cv::addWeighted(frame, 1, mask_3c, 0.7, 0, result);

    cv::imshow("Keypoint Density Mask", density_mask);
    cv::imshow("Mask", morph_mask);
    cv::imshow("Result", result);
    cv::waitKey(0);

    return morph_mask; // Return the final mask of detected regions
}*/

/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, cv::Mat empty_parking) {
    cv::Mat frame_hsv, empty_parking_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(empty_parking, empty_parking_hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> frame_channels, empty_parking_channels;
    cv::split(frame_hsv, frame_channels);
    cv::split(empty_parking_hsv, empty_parking_channels);

    cv::Mat frame_value_eq, empty_parking_value_eq, frame_saturation_eq, empty_parking_saturation_eq;
    cv::equalizeHist(frame_channels[2], frame_value_eq);
    cv::equalizeHist(empty_parking_channels[2], empty_parking_value_eq);

    //cv::equalizeHist(frame_channels[1], frame_saturation_eq);
    //cv::equalizeHist(empty_parking_channels[1], empty_parking_saturation_eq);


    //frame_channels[1] = frame_saturation_eq;
    frame_channels[2] = frame_value_eq;
    //empty_parking_channels[1] = empty_parking_saturation_eq;
    empty_parking_channels[2] = empty_parking_value_eq;
    cv::merge(frame_channels, frame_hsv);
    cv::merge(empty_parking_channels, empty_parking_hsv);

    cv::Mat frame_eq, empty_parking_eq;
    //frame_eq = frame_hsv.clone();
    //empty_parking_eq = empty_parking_hsv.clone();
    cv::cvtColor(frame_hsv, frame_eq, cv::COLOR_HSV2BGR);
    cv::cvtColor(empty_parking_hsv, empty_parking_eq, cv::COLOR_HSV2BGR);

    cv::Ptr<cv::BackgroundSubtractor> backSub = cv::createBackgroundSubtractorMOG2();
    cv::Mat mask;

    backSub->apply(empty_parking_eq, mask, 0);
    backSub->apply(frame_eq, mask, 1);
    mask.setTo(0, mask != 255);

    //cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    //cv::dilate(mask, mask, element);

    cv::Mat result = frame.clone();
    result.setTo(cv::Scalar(255, 0, 0), mask == 255);
    //cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    //cv::dilate(mask, mask, element);

    //regionGrowing(frame, mask, result);

    // TODO: Region growing algorithm using mask as seed

    
    //cv::Mat coloredMask;
    //cv::cvtColor(mask, coloredMask, cv::COLOR_GRAY2BGR);
    //coloredMask.setTo(cv::Scalar(0, 0, 255), mask == 255);

    //cv::Mat result;
    //cv::addWeighted(frame, 1, coloredMask, 0.7, 0, result);
    
    cv::imshow("Frame", frame);
    cv::imshow("Empty Parking", empty_parking);
    cv::imshow("Mask", mask);
    cv::imshow("Equalized Frame", frame_eq);
    cv::imshow("Result", result);
    cv::waitKey(0);

    return mask;
}
*/


/*
void CarSegmentation::detectCars(cv::Mat &frame, cv::Mat &mask, cv::Mat empty_parking) {
    cv::Mat diff, diffGray, subClosed, cars;
        
    // Compute the absolute difference between the current frame and the empty parking frame
    cv::absdiff(frame, empty_parking, diff);

    // Convert the difference image to grayscale
    cv::cvtColor(diff, diffGray, cv::COLOR_BGR2GRAY);

    // Apply a threshold to the grayscale difference image to create a binary mask
    cv::threshold(diffGray, mask, 50, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Apply morphological operations to remove noise and close gaps
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
    element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(33, 33));
    cv::morphologyEx(mask, subClosed, cv::MORPH_CLOSE, element);

    cv::bitwise_and(frame, frame, frame, subClosed);

    // Draw a rectangle around the cars
    cars = frame.clone();

    // Find contours on the closed mask to detect cars
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(subClosed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw rectangles around the detected cars
    for (const auto& contour : contours) {
        cv::Rect boundingBox = cv::boundingRect(contour);
        cv::rectangle(cars, boundingBox, cv::Scalar(0, 255, 0), 2);
    }
    
    // Display the results
    cv::imshow("Frame", frame);
    cv::imshow("Mask", mask);
    cv::imshow("Empty Parking", empty_parking);
    cv::imshow("Difference", diff);
    cv::imshow("Difference Gray", diffGray);
    cv::imshow("Sub Closed", subClosed);
    cv::imshow("Cars", cars);

    cv::waitKey(0);
}*/