#include "carsegmentation.h"

CarSegmentation::CarSegmentation() {
    backSub = cv::createBackgroundSubtractorMOG2();
}

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

    // cv::Mat result;
    // cv::addWeighted(coloredMask, 0.7, frame, 1, 0, result);

    //cv::imshow("Contours", result);
    //cv::waitKey(0);

    return coloredMask;
}

void CarSegmentation::trainBackgroundSubtractor(std::vector<cv::Mat> empty_parkings) {
    std::cout << "Training background subtractor..." << std::endl;
    for(const auto& empty_parking : empty_parkings) {
        cv::Mat tmp;
        backSub->apply(empty_parking, tmp, 1.0);
    }
    std::cout << "Background subtractor trained successfully." << std::endl;
}

double computeSimilarity(const cv::Mat& frame, const cv::Mat& background) {
    cv::Mat diff;
    cv::absdiff(frame, background, diff);
    cv::Scalar sum_diff = cv::sum(diff);
    return sum_diff[0] + sum_diff[1] + sum_diff[2];
}

cv::Mat selectClosestBackground(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    double min_diff = std::numeric_limits<double>::max();
    cv::Mat best_background;

    for(const auto& empty_parking : empty_parkings) {
        double diff = computeSimilarity(frame, empty_parking);
        if(diff < min_diff) {
            min_diff = diff;
            best_background = empty_parking;
        }
    }

    return best_background;
}

cv::Mat refineForegroundMask(const cv::Mat &fgMask, int minArea, double minAspectRatio, double maxAspectRatio) {
    cv::Mat refinedMask = fgMask.clone();

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(refinedMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create a mask to hold the final result
    cv::Mat finalMask = cv::Mat::zeros(refinedMask.size(), CV_8UC1);

    // Filter contours based on area and by shape (aspect ratio)
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        cv::Rect bbox = cv::boundingRect(contour);
        double aspectRatio = static_cast<double>(bbox.width) / bbox.height;
        if (area > minArea && aspectRatio > minAspectRatio && aspectRatio < maxAspectRatio)
            cv::drawContours(finalMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
    }

    // Additional morphological operations if needed
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(finalMask, finalMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(finalMask, finalMask, cv::MORPH_OPEN, kernel);

    return finalMask;
}

cv::Mat CarSegmentation::detectCars(cv::Mat image, std::vector<cv::Mat> empty_parkings) {
    cv::Mat frame = image.clone();
    cv::Mat best_background = selectClosestBackground(frame, empty_parkings);
    cv::Mat frame_gray, best_background_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(best_background, best_background_gray, cv::COLOR_BGR2GRAY);

    cv::Mat kernel3x3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat kernel5x5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat kernel7x7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::Mat kernel9x9 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));

    cv::Mat fgMask, thresh;
    cv::absdiff(frame_gray, best_background_gray, fgMask);
    cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel5x5);
    cv::threshold(fgMask, thresh, 50, 255, cv::THRESH_BINARY);
    cv::inRange(fgMask, 50, 255, thresh);

    cv::Mat frame_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(frame_hsv, hsv_channels);

    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::threshold(hsv_channels[1], mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel3x3);

    // Keep only cars
    mask = refineForegroundMask(mask, 600, 0.5, 4.0);

    cv::Mat background_hsv;
    cv::cvtColor(best_background, background_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> background_hsv_channels;
    cv::split(background_hsv, background_hsv_channels);

    cv::Mat green_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::threshold(background_hsv_channels[1], green_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, kernel5x5);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN, kernel7x7);
    cv::dilate(green_mask, green_mask, kernel9x9);
    
    cv::bitwise_not(green_mask, green_mask);
    //cv::bitwise_and(mask, green_mask, mask);
    //cv::bitwise_or(mask, thresh, mask);

    //mask = refineForegroundMask(mask);
    //cv::dilate(thresh, thresh, kernel5x5);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel3x3);
    thresh = refineForegroundMask(thresh, 600, 0.5, 4.0);

    cv::Mat frame_lab;
    cv::cvtColor(frame, frame_lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels;
    cv::split(frame_lab, lab_channels);

    // Red cars
    cv::Mat red_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::inRange(lab_channels[1], 140, 255, red_mask);
    cv::dilate(red_mask, red_mask, kernel3x3);
    red_mask = refineForegroundMask(red_mask, 600, 0.5, 4.0);

    // Black cars
    cv::Mat reflex_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::inRange(lab_channels[2], 0, 120, reflex_mask);
    reflex_mask = refineForegroundMask(reflex_mask, 100, 0, 15.0);

    cv::Mat final_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::bitwise_or(thresh, red_mask, final_mask);
    cv::bitwise_or(final_mask, reflex_mask, final_mask);
    cv::bitwise_and(final_mask, green_mask, final_mask);

    cv::Mat black_cars = cv::Mat::zeros(frame.size(), CV_8UC3);
    cv::inRange(hsv_channels[0], 80, 130, black_cars);
    black_cars = refineForegroundMask(black_cars, 600, 0.5, 4.0);

    // Connect components in final mask that are attached in black cars
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(black_cars, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat refined_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int i = 0; i < contours.size(); i++) {
        cv::Mat component_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::drawContours(component_mask, contours, i, cv::Scalar(255), cv::FILLED);

        cv::Mat intersection;
        cv::bitwise_and(component_mask, final_mask, intersection);
        
        if(cv::countNonZero(intersection) > 0 && cv::contourArea(contours[i]) < 3000)
            cv::bitwise_or(refined_mask, component_mask, refined_mask);
    }

    cv::bitwise_or(final_mask, refined_mask, final_mask);

    cv::Mat frame_gray_clahe, best_background_gray_clahe;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->setTilesGridSize(cv::Size(8, 8));
    clahe->apply(frame_gray, frame_gray_clahe);
    clahe->apply(best_background_gray, best_background_gray_clahe);

    cv::Mat fgMask_clahe, thresh_clahe;
    cv::absdiff(frame_gray_clahe, best_background_gray_clahe, fgMask_clahe);
    cv::threshold(fgMask_clahe, thresh_clahe, 70, 255, cv::THRESH_BINARY);
    thresh_clahe = refineForegroundMask(thresh_clahe, 1000, 0.5, 3.0);

    std::vector<std::vector<cv::Point>> contours_clahe;
    cv::findContours(thresh_clahe, contours_clahe, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat final_mask_clahe = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int i = 0; i < contours_clahe.size(); i++) {
        cv::Mat component_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::drawContours(component_mask, contours_clahe, i, cv::Scalar(255), cv::FILLED);

        cv::Mat intersection;
        cv::bitwise_and(component_mask, final_mask, intersection);
        
        if(cv::countNonZero(intersection) > 0 && cv::contourArea(contours_clahe[i]) < 10000)
            cv::bitwise_or(final_mask_clahe, component_mask, final_mask_clahe);
    }

    cv::bitwise_or(final_mask, final_mask_clahe, final_mask);

    final_mask = refineForegroundMask(final_mask, 800, 0.5, 4.0);

    cv::Mat kernel25x25 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25));
    cv::morphologyEx(final_mask, final_mask, cv::MORPH_CLOSE, kernel25x25);

    final_mask = refineForegroundMask(final_mask, 800, 0.5, 4.0);

    /*cv::imshow("Frame", frame);
    cv::imshow("Threshold", thresh);
    cv::imshow("H Channel", hsv_channels[0]);
    cv::imshow("Black Cars", black_cars);
    cv::imshow("Refined Mask", refined_mask);
    cv::imshow("Final Mask", final_mask);
    cv::imshow("Output", output);
    cv::waitKey(0);*/
    return final_mask;
}



cv::Mat CarSegmentation::classifyCars(cv::Mat &mask, std::vector<BBox> parkings) {
    cv::Mat colored_mask = mask.clone();
    cv::cvtColor(colored_mask, colored_mask, cv::COLOR_GRAY2BGR);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    // Initially, set all components to green
    colored_mask.setTo(cv::Scalar(0, 255, 0), mask == 255);
    
    for(const BBox& parking : parkings) {
        if(!parking.isOccupied()) // If the parking is not occupied go to the next one
            continue;
        
        int x = parking.getX();
        int y = parking.getY();
        int width = parking.getWidth();
        int height = parking.getHeight();
        int angle = parking.getAngle();

        cv::RotatedRect rect = cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(width, height), angle);
        
        cv::Point2f vertices[4];
        rect.points(vertices);
        cv::Point intVertices[4];
        for (int i = 0; i < 4; i++)
            intVertices[i] = cv::Point(static_cast<int>(vertices[i].x), static_cast<int>(vertices[i].y));

        // Create a mask for the parking BBox
        cv::Mat roiMask = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::fillConvexPoly(roiMask, intVertices, 4, cv::Scalar(255));
        // Iterate through each connected component
        for (int i = 1; i < nLabels; i++) {
            cv::Mat componentMask = (labels == i); 
            // Check if there is any intersection between the parking BBox and the component
            cv::Mat intersection;
            cv::bitwise_and(componentMask, roiMask, intersection);
            if (cv::countNonZero(intersection) > 0) {
                // If any part of the component is inside the BBox, set the whole component to red
                colored_mask.setTo(cv::Scalar(0, 0, 255), componentMask);
            }
        }
    }
    return colored_mask;
}



/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    cv::Mat best_background = selectClosestBackground(frame, empty_parkings);

    cv::Mat frame_gray, best_background_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(best_background, best_background_gray, cv::COLOR_BGR2GRAY);

    cv::Mat fgMask;
    cv::absdiff(frame_gray, best_background_gray, fgMask);
    cv::threshold(fgMask, fgMask, 50, 255, cv::THRESH_BINARY);
    
    cv::Mat kernel3x3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat kernel5x5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat kernel7x7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::Mat kernel15x15 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel5x5);
    cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel3x3);
    cv::dilate(fgMask, fgMask, kernel3x3);

    // Green mask to filter out trees
    cv::Mat green_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::Mat empty_parking_hsv;
    cv::cvtColor(empty_parkings[0], empty_parking_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> empty_parking_hsv_channels;
    cv::split(empty_parking_hsv, empty_parking_hsv_channels);

    cv::threshold(empty_parking_hsv_channels[1], green_mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, kernel5x5);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN, kernel7x7);
    cv::erode(green_mask, green_mask, kernel15x15);
    //cv::dilate(green_mask, green_mask, kernel15x15);

    cv::bitwise_and(fgMask, green_mask, fgMask);

    // Refine the foreground mask
    cv::Mat refinedMask = refineForegroundMask(fgMask);

    // Red cars: this system has some problems in detecting red cars
    // Now we use another method to detect red cars
    cv::Mat frame_hsv;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(frame_hsv, hsv_channels);

    cv::Mat thresh;
    cv::threshold(hsv_channels[1], thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::bitwise_and(thresh, green_mask, thresh);
    thresh = refineForegroundMask(thresh);

    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int x = 0; x < hsv_channels[2].rows; x++) 
        for(int y = 0; y < hsv_channels[2].cols; y++)
            if(hsv_channels[2].at<uchar>(x, y) > 200 || hsv_channels[2].at<uchar>(x, y) < 45)
                mask.at<uchar>(x, y) = 255;

    cv::dilate(mask, mask, kernel7x7);
    cv::Mat result;
    cv::bitwise_or(refinedMask, thresh, result);
    cv::bitwise_or(result, mask, result);
    cv::bitwise_and(result, green_mask, result);

    result = refineForegroundMask(result);

    // Only for visualization purposes, not needed in the final implementation
    cv::Mat red_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
    red_mask.setTo(cv::Scalar(0, 0, 255), result == 255);

    cv::Mat output = frame.clone();
    cv::addWeighted(output, 1, red_mask, 0.7, 0, output);

    cv::imshow("Frame", frame);
    cv::imshow("Result", result);
    cv::imshow("Foreground Mask", fgMask);
    cv::imshow("Green Mask", green_mask);
    cv::imshow("Output", output);
    cv::waitKey(0);
    return result;
}
*/

/*
void CarSegmentation::computeHOG(cv::Mat &frame, std::vector<float> &descriptors) {
    cv::HOGDescriptor hog(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
    cv::resize(frame, frame, cv::Size(64, 128)); // Resize image to the fixed window size
    hog.compute(frame, descriptors);
}

void CarSegmentation::trainSVM(const std::vector<std::vector<float>> &car_features, const std::vector<std::vector<float>> &non_car_features) {
    std::cout << "Training SVM..." << std::endl;
    std::vector<std::vector<float>> train_data;
    std::vector<int> labels;

    train_data.insert(train_data.end(), car_features.begin(), car_features.end());
    labels.insert(labels.end(), car_features.size(), 1);

    train_data.insert(train_data.end(), non_car_features.begin(), non_car_features.end());
    labels.insert(labels.end(), non_car_features.size(), 0);

    cv::Mat train_data_mat(train_data.size(), static_cast<int>(train_data[0].size()), CV_32FC1);
    for(int i = 0; i < train_data.size(); i++)
        for(int j = 0; j < train_data[i].size(); j++)
            train_data_mat.at<float>(i, j) = train_data[i][j];

    cv::Mat labels_mat(labels.size(), 1, CV_32SC1);
    for(int i = 0; i < labels.size(); i++)
        labels_mat.at<int>(i, 0) = labels[i];
    
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setC(0.01);
    svm->train(train_data_mat, cv::ml::ROW_SAMPLE, labels_mat);
    std::cout << "SVM trained successfully." << std::endl;
    svm->save("car_detector.xml");
}

cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    std::cout << "Detecting cars..." << std::endl;
    cv::Mat frame_copy = frame.clone();
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("car_detector.xml");
    cv::Size window_size(64, 128);

    for(int y = 0; y <= frame.rows - window_size.height; y += 8) {
        for(int x = 0; x <= frame.cols - window_size.width; x += 8) {
            cv::Rect window(x, y, window_size.width, window_size.height);
            cv::Mat window_frame = frame(window).clone();

            std::vector<float> descriptors;
            computeHOG(window_frame, descriptors);

            cv::Mat descriptors_mat(1, static_cast<int>(descriptors.size()), CV_32FC1);
            for(int i = 0; i < descriptors.size(); i++)
                descriptors_mat.at<float>(0, i) = descriptors[i];

            float response = svm->predict(descriptors_mat);
            std::cout << "Response: " << response << std::endl;
            if(response == 1) {
                cv::rectangle(frame_copy, window, cv::Scalar(0, 0, 255), 2);
            }
        }
    }
    
    cv::imshow("Frame", frame_copy);
    cv::waitKey(0);
    return cv::Mat();
}*/


/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    cv::Mat blurred;
    cv::pyrMeanShiftFiltering(frame, blurred, 20, 45, 1);

    cv::Mat frame_hsv;
    cv::cvtColor(blurred, frame_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(frame_hsv, hsv_channels);

    cv::Mat v_channel = hsv_channels[2].clone();

    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    for(int x = 0; x < v_channel.rows; x++) 
        for(int y = 0; y < v_channel.cols; y++)
            if(v_channel.at<uchar>(x, y) > 200 || v_channel.at<uchar>(x, y) < 45)
                mask.at<uchar>(x, y) = 255;

    cv::Mat green_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::Mat parking_hsv;
    cv::cvtColor(empty_parkings[3], parking_hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> parking_hsv_channels;
    cv::split(parking_hsv, parking_hsv_channels);
    cv::threshold(parking_hsv_channels[1], green_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat element3x3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat element5x5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat element7x7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::Mat element9x9 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::Mat element15x15 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN, element5x5);
    cv::dilate(green_mask, green_mask, element15x15);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, element5x5);
    cv::bitwise_not(green_mask, green_mask);

    cv::bitwise_and(mask, green_mask, mask);

    //cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element5x5);
    //cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element15x15);
    //cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element9x9);

    // Delete small connected components
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for(int i = 0; i < contours.size(); i++) {
        if(cv::contourArea(contours[i]) < 200) {
            cv::drawContours(mask, contours, i, cv::Scalar(0), cv::FILLED);
        }
    }

    cv::Mat result;
    cv::Mat blue_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
    blue_mask.setTo(cv::Scalar(255, 0, 0), mask == 255);
    cv::addWeighted(frame, 1, blue_mask, 0.7, 0, result);

    cv::Mat h_channel = hsv_channels[0].clone();
    cv::Mat s_channel = hsv_channels[1].clone();
    cv::threshold(h_channel, h_channel, 100, 120, cv::THRESH_BINARY);
    //cv::threshold(s_channel, s_channel, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(s_channel, s_channel, 100, 255, cv::THRESH_BINARY);

    //h_channel.setTo(255, h_channel > 0);
    /*
    cv::Mat temp;
    //cv::bitwise_or(mask, s_channel, temp);
    //cv::bitwise_or(temp, h_channel, temp);
    //cv::bitwise_and(temp, green_mask, temp);

    cv::morphologyEx(temp, temp, cv::MORPH_CLOSE, element7x7);

    std::vector<std::vector<cv::Point>> contours2;
    cv::findContours(temp, contours2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for(int i = 0; i < contours2.size(); i++) {
        if(cv::contourArea(contours2[i]) > 5000) {
            cv::drawContours(temp, contours2, i, cv::Scalar(0), cv::FILLED);
        }
    }
    
    //cv::morphologyEx(temp, temp, cv::MORPH_OPEN, element3x3);
    
    cv::Mat frame_lab;
    cv::cvtColor(frame, frame_lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels;
    cv::split(frame_lab, lab_channels);

    cv::Mat a_channel = lab_channels[1];

    // Red cars
    for(int x = 0; x < a_channel.rows; x++)
        for(int y = 0; y < a_channel.cols; y++)
            if(a_channel.at<uchar>(x, y) > 150)
                mask.at<uchar>(x, y) = 255;

   



    cv::imshow("Frame", frame);
    //cv::imshow("Blurred", blurred);
    //cv::imshow("HSV", frame_hsv);
    cv::imshow("Hue", hsv_channels[0]);
    cv::imshow("Saturation", hsv_channels[1]);
    cv::imshow("Value", hsv_channels[2]);
    cv::imshow("Mask", mask);
    cv::imshow("L", lab_channels[0]);
    cv::imshow("a", lab_channels[1]);
    cv::imshow("b", lab_channels[2]);
    //cv::imshow("Y Channel", ycbcr_channels[0]);
    //cv::imshow("Cr Channel", ycbcr_channels[1]);
    //cv::imshow("Cb Channel", ycbcr_channels[2]);
    //cv::imshow("Green Mask", green_mask);
    //cv::imshow("Result", result);
    //cv::imshow("H Channel", h_channel);
    //cv::imshow("S Channel", s_channel);
    //cv::imshow("Temp", temp);
    //cv::imshow("a", a_channel);
    cv::waitKey(0);
    return mask;  // Return the result mask
}
*/

/*
cv::Mat CarSegmentation::detectCars(cv::Mat &frame, std::vector<cv::Mat> empty_parkings) {
    cv::Mat frame_gray, blurred;
    cv::pyrMeanShiftFiltering(frame, blurred, 20, 45, 2);
    cv::dilate(blurred, blurred, cv::Mat::ones(3, 3, CV_8U));
    cv::cvtColor(blurred, frame_gray, cv::COLOR_BGR2GRAY);

    cv::Mat grad_x, grad_y;
    cv::Sobel(frame_gray, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(frame_gray, grad_y, CV_16S, 0, 1, 3);

    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    cv::Mat grad;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    cv::Mat binary;
    cv::threshold(grad, binary, 50, 255, cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    //cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

    cv::Mat dist;
    cv::distanceTransform(binary, dist, cv::DIST_L2, 3);
    cv::normalize(dist, dist, 0, 1, cv::NORM_MINMAX);

    cv::threshold(dist, dist, 0.4, 1, cv::THRESH_BINARY);

    cv::Mat sure_fg;
    dist.convertTo(sure_fg, CV_8U);

    cv::Mat sure_bg;
    cv::dilate(binary, sure_bg, kernel, cv::Point(-1, -1), 3);

    cv::Mat unknown;
    cv::subtract(sure_bg, sure_fg, unknown);

    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);

    markers = markers + 1;

    markers.setTo(0, unknown == 255);

    cv::watershed(frame, markers);

    cv::Mat result = cv::Mat::zeros(markers.size(), CV_8UC3);
    for(int x = 0; x < markers.rows; x++) {
        for(int y = 0; y < markers.cols; y++) {
            if(markers.at<int>(x, y) == -1) {
                result.at<cv::Vec3b>(x, y) = cv::Vec3b(0, 0, 255);
            }
        }
    }

    cv::imshow("Frame", frame);
    cv::imshow("Blurred", blurred);
    cv::imshow("Gradient", grad);
    cv::imshow("Binary", binary);
    cv::imshow("Distance", dist);
    cv::imshow("Sure FG", sure_fg);
    cv::imshow("Sure BG", sure_bg);
    cv::imshow("Unknown", unknown);
    cv::imshow("Watershed", result);
    cv::waitKey(0);

    return cv::Mat();
}*/


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
