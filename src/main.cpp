#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "utils.h"
#include "parkingdetection.h"
#include "carsegmentation.h"
#include "visualizationmap.h"
#include "metrics.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    
    // Loading all frames
    std::vector<std::vector<cv::Mat>> images;    
    std::cout << "Loading all frames..." << std::endl;
    int framesCount = loaders::loadAllFrames(images);
    std::cout << "Done!" << std::endl << "Total frames loaded: " << framesCount << std::endl << std::endl;

    // Loading all backgrounds used in segmentation
    std::vector<cv::Mat> backgrounds;
    std::cout << "Loading all backgrounds..." << std::endl;
    int backgroundsCount = loaders::loadAllBackgrounds(backgrounds);
    std::cout << "Done!" << std::endl << "Total backgrounds loaded: " << backgroundsCount << std::endl << std::endl;

    // Loading all ground truth masks
    std::vector<std::vector<cv::Mat>> realMasks;
    std::cout << "Loading all ground truth masks..." << std::endl;
    int realMasksCount = loaders::loadAllRealMasks(realMasks);
    std::cout << "Done!" << std::endl << "Total masks loaded: " << realMasksCount << std::endl << std::endl;

    // Loading all ground truth bboxes
    std::vector<std::vector<std::vector<BBox>>> realBBoxes;
    std::cout << "Loading all ground truth bboxes..." << std::endl;
    int realBBoxCount = loaders::loadAllRealBBoxes(realBBoxes);
    std::cout << "Done!" << std::endl << "Total bboxes loaded: " << realBBoxCount << std::endl << std::endl;
    
    // Finding parkings from sequence 0
    std::vector<std::vector<BBox>> detected;
    std::vector<float> mAPs;
    for (int i = 0; i < 5; i++)
    {
        cv::Mat current_image = images[0][i].clone();
        ParkingDetection pd;

        std::vector<BBox> current_detected = pd.detect(current_image);
        pd.draw(current_image, current_detected);
        detected.push_back(current_detected);

        float current_mAP = computeMAP(current_detected, realBBoxes[0][i], 0.5);
        mAPs.push_back(current_mAP);

        std::cout << "mAP for sequence0 frame " << i + 1 << std::endl;
        std::cout << current_mAP << std::endl << std::endl;

        cv::imshow("Parking detection", current_image);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
    
    int bestIndex = std::distance(mAPs.begin(), std::max_element(mAPs.begin(), mAPs.end())); // Argmax
    std::cout << "Best parkin detection was done on frame " << bestIndex + 1 << " (sequence0). Using it now on." << std::endl;

    std::vector<BBox> bestDetectedBBoxes = detected[bestIndex];
    
    std::vector<cv::Mat> detectedMasks;
    for (int i = 0; i < images.size(); i++)
    {
        for (int j = 0; j < images[i].size(); j++)
        {
            cv::Mat current_image = images[i][j].clone();

            CarSegmentation cs;
            
            // Segmenting
            cv::Mat mask = cs.detectCars(current_image, backgrounds);

            // Apply occupied based on segmentation
            for(auto& bbox : bestDetectedBBoxes)
                bbox.setOccupiedfromObtainedMask(mask);
            
            // Make the mask colored (green, red, black)
            mask = cs.classifyCars(mask, bestDetectedBBoxes);
            detectedMasks.push_back(mask);
            
            cv::Mat parkingsWithMap = current_image.clone();

            // Drawing colored segmentation into original image
            cv::Mat maskedImage;
            cv::addWeighted(current_image, 1, mask, 0.7, 0, maskedImage);

            // Drawing colored parkings
            ParkingDetection pd;
            pd.draw(current_image, bestDetectedBBoxes);

            // Drawing parkings + segmentation
            cv::Mat maskedParkingsImage;
            cv::addWeighted(current_image, 1, mask, 0.7, 0, maskedParkingsImage);

            VisualizationMap vm;
            vm.drawParkingMap(parkingsWithMap, bestDetectedBBoxes);

            std::cout << "Metrics for sequence" << i << " frame " << j + 1 << std::endl;
            std::cout << "mAP: " << computeMAP(bestDetectedBBoxes, realBBoxes[i][j], 0.5) << std::endl;
            std::cout << "mIoU: " << computeIoU(mask, realMasks[i][j]) << std::endl << std::endl;
            
            cv::imshow("Parkings", current_image);
            cv::imshow("Minimap", parkingsWithMap);
            cv::imshow("Segmented original", maskedImage);
            cv::imshow("Parkings + segmentation", maskedParkingsImage);
            cv::waitKey(0);


        }
        
    }
    

    return 0;
}
