#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <filesystem>
#include "utils.h"
#include "parkingdetection.h"
#include "carsegmentation.h"
#include "visualizationmap.h"
#include "metrics.h"

using namespace cv;
using namespace std;

namespace fs = std::filesystem;

int main(int argc, char** argv) {

    // Image loading
    string path;
    std::vector<std::vector<cv::Mat>> parkingImages;

    for (int i = 0; i <= 5; i++) {
        path = "dataset/sequence" + std::to_string(i) + "/frames";
        if (fs::exists(path) && fs::is_directory(path)) {
            std::vector<cv::Mat> images;
            std::vector<std::string> imagePaths;

            // Collect all image paths
            for (const auto& entry : fs::directory_iterator(path)) {
                imagePaths.push_back(entry.path().string());
            }

            // Sort image paths by filename
            std::sort(imagePaths.begin(), imagePaths.end());

            // Load images in sorted order
            for (const auto& imagePath : imagePaths) {
                cv::Mat image = cv::imread(imagePath);
                if (!image.empty()) {
                    images.push_back(image);
                } else {
                    std::cerr << "Failed to load image: " << imagePath << std::endl;
                }
            }

            parkingImages.push_back(images);
        } else {
            std::cerr << "Directory does not exist or is not a directory: " << path << std::endl;
        }
    }

    std::vector<cv::Mat> empty_parkings_aug;
    path = "dataset/backgrounds";
    if(fs::exists(path) && fs::is_directory(path)) {
        for(const auto& entry : fs::directory_iterator(path)) {
            cv::Mat empty_parking = cv::imread(entry.path().string());
            if(!empty_parking.empty()) {
                empty_parkings_aug.push_back(empty_parking);
            } else {
                std::cerr << "Failed to load image: " << entry.path().string() << std::endl;
            }
        }
    } else {
        std::cerr << "Directory does not exist or is not a directory: " << path << std::endl;
    }

    // Masks loading
    std::vector<std::vector<cv::Mat>> parkingMasks;
    for (int i = 1; i <= 5; i++) {
        path = "dataset/sequence" + std::to_string(i) + "/masks";
        if (fs::exists(path) && fs::is_directory(path)) {
            std::vector<cv::Mat> masks;
            std::vector<std::string> maskPaths;

            // Collect all mask paths
            for (const auto& entry : fs::directory_iterator(path)) {
                maskPaths.push_back(entry.path().string());
            }

            // Sort the paths by filename
            std::sort(maskPaths.begin(), maskPaths.end());

            // Load and store the masks in sorted order
            for (const auto& maskPath : maskPaths) {
                cv::Mat mask = cv::imread(maskPath);
                if (!mask.empty()) {
                    masks.push_back(mask);
                } else {
                    std::cerr << "Warning: Unable to load mask from " << maskPath << std::endl;
                }
            }
            
            parkingMasks.push_back(masks);
        } else {
            std::cerr << "Directory does not exist or is not a directory." << std::endl;
        }
    }

    // Real bounding boxes paths
    
    std::vector<std::vector<string>> bboxesPaths;
    for(int i = 0; i <= 5; i++) {
        path = "dataset/sequence" + std::to_string(i) + "/bounding_boxes";
        if (fs::exists(path) && fs::is_directory(path)) {
            std::vector<string> bboxes;
            for (auto& entry : fs::directory_iterator(path)) {
                bboxes.push_back(entry.path().string());
            }
            // Sort the bounding box paths by filename
            std::sort(bboxes.begin(), bboxes.end());
            bboxesPaths.push_back(bboxes);
        } else {
            std::cerr << "File does not exist." << std::endl;
        }
    }
    



    
    // PARKING DETECTION & CLASSIFICATION REAL
    
    ParkingDetection pd;
    std::vector<BBox> real_bboxes;
    std::vector<std::vector<BBox>> all_detected_bboxes;
    for(int i = 0; i < parkingImages[0].size(); i++) {
        cv::Mat parking = parkingImages[0][i];
        if (parking.empty()) {
            std::cerr << "Invalid input" << std::endl;
            return -1;
        }
        
        // Show real image bounding box
        Mat realBBoxes = parking.clone();
        real_bboxes = parseParkingXML(bboxesPaths[0][i]);

        std::vector<BBox> detected_bboxes = pd.detect(parking);
        all_detected_bboxes.push_back(detected_bboxes);
        pd.draw(parking, detected_bboxes);
        
        // mAP
        std::cout << "METRICS: " << std::endl;
        std::cout << "mAP: " << computeMAP(detected_bboxes, real_bboxes, 0.5) << std::endl;

        waitKey(0);
        cv::destroyAllWindows();
    }
    
    
    // CAR SEGMENTATION

    // Extracting HOG features for SVM
    // Decomment only if you want to retrain the SVM, heavy operation
    // Took only 100 images for each class
    /*
    std::vector<int> labels;
    std::vector<cv::Mat> car_images;
    std::vector<cv::Mat> non_car_images;
    std::vector<std::vector<float>> car_features;
    std::vector<std::vector<float>> non_car_features;
    int max_samples = 100;
    path = "dataset/cars";
    if (fs::exists(path) && fs::is_directory(path)) {
        int count = 0;
        for (const auto& entry : fs::directory_iterator(path)) {
            if (count >= max_samples) {
                break;
            }
            cv::Mat frame = cv::imread(entry.path().string());
            //std::cout << "Car image: " << entry.path().string() << std::endl;
            if(!frame.empty()) {
                cv::resize(frame, frame, cv::Size(64, 128));
                car_images.push_back(frame);
                count++;
            }
        }
    } else {
        std::cerr << "Directory does not exist or is not a directory." << std::endl;
    }

    path = "dataset/non-cars";
     if (fs::exists(path) && fs::is_directory(path)) {
        int count = 0;
        for (const auto& entry : fs::directory_iterator(path)) {
            if (count >= max_samples) {
                break;
            }
            cv::Mat frame = cv::imread(entry.path().string());
            //std::cout << "Non-car image: " << entry.path().string() << std::endl;
            if(!frame.empty()) {
                cv::resize(frame, frame, cv::Size(64, 128));
                non_car_images.push_back(frame);
                count++;
            }
        }
    } else {
        std::cerr << "Directory does not exist or is not a directory." << std::endl;
    }
    */
    CarSegmentation cs;
    /*
    cv::HOGDescriptor hog(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
    std::cout << "(Main) Extracting HOG features for cars..." << std::endl;
    for(cv::Mat& car_image : car_images) {
        std::vector<float> descriptors;
        cs.computeHOG(car_image, hog, descriptors);
        car_features.push_back(descriptors);
    }
    for(cv::Mat& non_car_image : non_car_images) {
        std::vector<float> descriptors;
        cs.computeHOG(non_car_image, hog, descriptors);
        non_car_features.push_back(descriptors);
    }
    std::cout << "(Main) Training SVM..." << std::endl;
    cv::Mat trainingData;
    //std::vector<int> labels;
    for(const auto& car_feature : car_features) {
        trainingData.push_back(cv::Mat(car_feature).reshape(1, 1));
        labels.push_back(1);
    }
    for(const auto& non_car_feature : non_car_features) {
        trainingData.push_back(cv::Mat(non_car_feature).reshape(1, 1));
        labels.push_back(0);
    }
    trainingData.convertTo(trainingData, CV_32F);
    cv::Mat labelsMat(labels, true);
    cs.trainSVM(trainingData, labelsMat);
    */

    
    // Array of vector, in position 0 there are the masks related of sequence 1
    // p_i = masks_i-1

    // SEMPLIFICATION
    cv::Point p1(870, 0);
    cv::Point p2(1275, 225);

    float slope = float(p2.y - p1.y) / float(p2.x - p1.x);
    float q = p1.y - slope * p1.x;
    
    
    std::vector<std::vector<cv::Mat>> car_segmentation;
    std::vector<std::vector<float>> all_ious;
    //cs.trainBackgroundSubtractor(empty_parkings_aug);
    for(int i = 0; i <= 5; i++) {
        std::vector<cv::Mat> cars;
        std::vector<float> ious;
        std::cout << "Sequence " << i << ":" << std::endl;
        for(int j = 0; j < parkingImages[i].size(); j++) {
        //for(int i = 0; i < 1; i++) {
            cv::Mat parking = parkingImages[i][j];
            cv::Mat parking_mask;
            if(i != 0)
                parking_mask = parkingMasks[i - 1][j];
            else
                parking_mask = cv::Mat::zeros(parking.size(), parking.type());

            if (parking.empty()) {
                std::cerr << "Invalid input" << std::endl;
                return -1;
            }
            if(parking_mask.empty()){
                std::cerr << "Invalid mask" << std::endl;
                return -1;
            }
            cv::Mat mask = cs.detectCars(parking, empty_parkings_aug);
            cv::Mat true_mask = cs.detectCarsTrue(parking, parking_mask);
            cars.push_back(mask);
            
            // Real mask
            cv::cvtColor(true_mask, true_mask, cv::COLOR_BGR2GRAY);
            true_mask.setTo(255, true_mask != 128);
            true_mask.setTo(0, true_mask == 128);
            // Semplification: cars in the upper-right part of the image are not considered
            // Above the line connecting the two points (0,870), (225, 1275) all black
            for(int x = 0; x < mask.cols; x++) {
                int y = slope * x + q;
                for(int i = 0; i < y && y < mask.rows; i++) {
                    mask.at<uchar>(i, x) = 0;
                    true_mask.at<uchar>(i, x) = 0;
                }
            }
            //cv::imshow("Real mask", true_mask);
            //cv::imshow("Obtained mask", mask);
            //cv::waitKey(0);
            float iou = computeIoU(mask, true_mask);
            std::cout << "IoU: " << iou << std::endl;
            ious.push_back(iou);
        }
        car_segmentation.push_back(cars);
        all_ious.push_back(ious);
    }

    // mIoU
    float global_iou = 0.0f;
    int size = 0;
    for(std::vector<float> ious: all_ious) {
        float mean_iou = 0.0f;
        for(const auto& iou : ious)
            mean_iou += iou;
        global_iou += mean_iou;
        mean_iou /= ious.size();
        size += ious.size();
        std::cout << "\nmIoU: " << mean_iou << std::endl;
    }
    global_iou /= size;
    std::cout << "\nGlobal mIoU: " << global_iou << std::endl;

    /*float mean_iou = 0.0f;
    for(const auto& iou : ious)
        mean_iou += iou;
    mean_iou /= ious.size();
    std::cout << "\nmIoU: " << mean_iou << std::endl;*/

    
    
    
    
    
    // PARKING DETECTION & CLASSIFICATION OUR MASKS
    // Set as BBoxes the best BBoxes obtained from the previous step
    std::vector<BBox> bboxes = all_detected_bboxes[0];
    std::vector<std::vector<cv::Mat>> parking_classification;
    for(int i = 0; i <= 5; i++) {
        std::vector<cv::Mat> temp;
        for(int j = 0; j < parkingImages[i].size(); j++) {
            cv::Mat parking = parkingImages[i][j];
            if (parking.empty()) {
                std::cerr << "Invalid input" << std::endl;
                return -1;
            }
            
            for (auto& bbox : bboxes) 
                bbox.setOccupiedfromObtainedMask(car_segmentation[i][j]);
            temp.push_back(pd.drawColored(parking, bboxes));
            cv::Mat output = temp[j];
            cv::Mat colored_mask = cv::Mat::zeros(output.size(), CV_8UC3);
            colored_mask.setTo(cv::Scalar(0, 0, 255), car_segmentation[i][j] == 255);
            cv::addWeighted(output, 1, colored_mask, 0.7, 0, output);
            cv::imshow("Parking", output);
            cv::waitKey(0);
            // PARKING DETECTION
            //ParkingDetection pd;
            //pd.detect(parking);
            //pd.draw(parking);
            
            cv::destroyAllWindows();
        }
        parking_classification.push_back(temp);
    }
    // TODO: PARKING SPACES CLASSIFICATION
    // Categories: 0 - Empty Space, 1 - Occupied Space

    // TODO: CAR SEGMENTATION
    // Categories: 1 - Car correctly parked, 2 - Car out of place

    // 2D TOP-VIEW VISUALIZATION MAP
    /*
    for(int i = 1; i <= 5; i++) {
        for(int j = 0; j < parkingImages[i].size(); j++) {
        //for(int i = 0; i < 1; i++) {
            cv::Mat parking = parkingImages[i][j];
            cv::Mat parking_mask = parkingMasks[i - 1][j];
            if (parking.empty()) {
                std::cerr << "Invalid input" << std::endl;
                return -1;
            }
            if(parking_mask.empty()){
                std::cerr << "Invalid mask" << std::endl;
                return -1;
            }

            std::vector<BBox> bboxes = parseParkingXML(bboxesPaths[i][j]);
            VisualizationMap map;
            map.drawParkingMap(parking, bboxes);
        }
    }
    */


    return 0;
}
/*
// 3td try
int main(int argc, char** argv) {
    cv::Mat parking = cv::imread("dataset/sequence0/frames/2013-02-24_15_10_09.jpg");
    if (parking.empty()) {
        std::cerr << "Invalid input" << std::endl;
        return 1;
    }

    cv::Mat parking_gray, parking_edges;
    cv::cvtColor(parking, parking_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(parking_gray, parking_gray, cv::Size(5, 5), 0);
    cv::Canny(parking_gray, parking_edges, 50, 125);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(parking_edges, parking_edges, cv::MORPH_CLOSE, element);
    cv::Mat edge_mask = (parking_edges == 255);
    cv::Mat gray_mask = (parking_gray >= 150) & (parking_gray <= 255);
    cv::Mat combined_mask = edge_mask & gray_mask;
    cv::Mat result = cv::Mat::zeros(parking.size(), parking.type());
    parking.copyTo(result, combined_mask);
    cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(result, result, cv::Size(5, 5), 0);
    cv::Canny(result, result, 50, 125);
    // Hough Line Transform
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(result, lines, 5, CV_PI/180, 50, 15, 15);
    cv::Mat line_image = cv::Mat::zeros(parking.size(), CV_8UC3);
    for (const auto& line : lines) {
        cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    std::cout << "Number of lines: " << lines.size() << std::endl;

    cv::imshow("Parking", parking);
    cv::imshow("Parking Edges", parking_edges);
    cv::imshow("Result", result);
    cv::imshow("Parking Lines", line_image);
    cv::waitKey(0);

    return 0;
}
*/

/*
2nd try
int main(int argc, char** argv) {
    cv::Mat parking = cv::imread("dataset/sequence0/frames/2013-02-24_15_10_09.jpg");
    if (parking.empty()) {
        std::cerr << "Invalid input" << std::endl;
        return 1;
    }
    cv::Mat parking_gray, edges, white_mask, combined_mask, result;
    cv::cvtColor(parking, parking_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(parking_gray, parking_gray, cv::Size(5, 5), 0);
    cv::Canny(parking_gray, edges, 75, 125);

    cv::threshold(edges, edges, 0, 255, cv::THRESH_BINARY);
    cv::Mat mask = edges.clone();
    cv::Mat mask_3ch;
    cv::cvtColor(mask, mask_3ch, cv::COLOR_GRAY2BGR);
    parking.copyTo(result, mask);

    // Define the white color threshold
    const cv::Scalar white_min(125, 125, 125); // Minimum white value
    const cv::Scalar white_max(255, 255, 255); // Maximum white value

    // Iterate through each pixel and discard pixels not close to white
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            cv::Vec3b color = result.at<cv::Vec3b>(y, x);
            if (!(color[0] >= white_min[0] && color[0] <= white_max[0] &&
                  color[1] >= white_min[1] && color[1] <= white_max[1] &&
                  color[2] >= white_min[2] && color[2] <= white_max[2])) {
                result.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // Set to black
            }
        }
    }

    cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, element);
    cv::Mat skel;
    cv::ximgproc::thinning(result, skel, 1);
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, element);
    // Hough Line Transform
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(skel, lines, 5, CV_PI/180, 50, 25, 20);
    cv::Mat line_image = cv::Mat::zeros(skel.size(), CV_8UC3);
    for (const auto& line : lines) {
        cv::line(line_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    std::cout << "Number of lines: " << lines.size() << std::endl;
    cv::imshow("Parking", skel);
    cv::imshow("Image lines", line_image);
    cv::waitKey(0);

    return 0;
}
*/
/*
// 1st try
cv::Mat parking, parking_gray;
int main(int argc, char** argv) {
    parking = cv::imread("dataset/sequence0/frames/2013-02-24_15_10_09.jpg");
    if (parking.empty()) {
        std::cerr << "Invalid input" << std::endl;
        return 1;
    }
    cv::GaussianBlur(parking, parking, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT);
    cv::cvtColor(parking, parking_gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad;
    cv::Sobel(parking_gray, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(parking_gray, grad_y, CV_16S, 0, 1, 3);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // Threshold the gradient image to create a mask
    cv::Mat mask;
    cv::threshold(grad, mask, 50, 255, cv::THRESH_BINARY);
    
    // Use the mask to keep only the lines in the original BGR image
    cv::Mat result;
    parking.copyTo(result, mask);

    // Further filter to keep only the pixels close to white
    cv::Mat white_mask;
    cv::inRange(result, cv::Scalar(150, 150, 150), cv::Scalar(255, 255, 255), white_mask);

    // cv::erode(white_mask, white_mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1)), cv::Point(-1, -1), 1);

    // Apply the white_mask to the result image
    cv::Mat final_result;
    result.copyTo(final_result, white_mask);

    cv::imshow("Parking", final_result);
    cv::waitKey(0);

    return 0;
}*/
