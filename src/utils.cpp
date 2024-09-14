#include "utils.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <filesystem>
#include <string>
#include "bbox.h"

// Implement here the utility functions
void drawRotatedRectangle(cv::Mat& image, cv::RotatedRect rect, bool occupied, bool filled = false)
{   
    // 0 -> empty
    // 1 -> occupied

    cv::Scalar color;

    if(occupied)
        color = cv::Scalar(0, 0, 255); //red
        
    else
        color = cv::Scalar(255, 0, 0); //blue

    cv::Point2f vertices2f[4];
    rect.points(vertices2f);

    cv::Point vertices[4];    
    for(int i = 0; i < 4; i++)
        vertices[i] = vertices2f[i];

    if(filled) {
        std::vector<cv::Point> points(vertices, vertices + 4);
        cv::fillPoly(image, std::vector<std::vector<cv::Point>>{points}, color);
    }
    else {
        for (int i = 0; i < 4; i++)
            cv::line(image, vertices[i], vertices[(i+1)%4], color, 2);
    }
}

std::vector<BBox> parseParkingXML(const std::string& filename) {
    std::vector<BBox> spaces;
    std::ifstream file(filename);
    if(!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return std::vector<BBox>();
    }

    std::string line;
    BBox bbox;
    cv::Point point;
    // Without using tinyxml2
    while(std::getline(file, line)) {
        if(line.find("<space") != std::string::npos) {
            std::size_t id_pos = line.find("id=\"");
            std::size_t occ_pos = line.find("occupied=\"");
            if(id_pos != std::string::npos) 
                bbox.setId(std::stoi(line.substr(id_pos + 4, line.find("\"", id_pos + 4) - id_pos - 4)));
            if(occ_pos != std::string::npos) {
                int occ = std::stoi(line.substr(occ_pos + 10, line.find("\"", occ_pos + 10) - occ_pos - 10));
                if(occ == 0) bbox.setOccupied(false);
                else bbox.setOccupied(true);
            }
        }
        else if(line.find("<center") != std::string::npos) {
            std::size_t x_pos = line.find("x=\"");
            std::size_t y_pos = line.find("y=\"");
            bbox.setX(std::stoi(line.substr(x_pos + 3, line.find("\"", x_pos + 3) - x_pos - 3)));
            bbox.setY(std::stoi(line.substr(y_pos + 3, line.find("\"", y_pos + 3) - y_pos - 3)));
        }
        else if(line.find("<size") != std::string::npos) {
            std::size_t w_pos = line.find("w=\"");
            std::size_t h_pos = line.find("h=\"");
            bbox.setWidth(std::stoi(line.substr(w_pos + 3, line.find("\"", w_pos + 3) - w_pos - 3)));
            bbox.setHeight(std::stoi(line.substr(h_pos + 3, line.find("\"", h_pos + 3) - h_pos - 3)));
        }
        else if(line.find("<angle") != std::string::npos) {
            std::size_t d_pos = line.find("d=\"");
            bbox.setAngle(std::stod(line.substr(d_pos + 3, line.find("\"", d_pos + 3) - d_pos - 3)));
        }
        else if(line.find("<point") != std::string::npos) {
            std::size_t x_pos = line.find("x=\"");
            std::size_t y_pos = line.find("y=\"");
            point.x = std::stoi(line.substr(x_pos + 3, line.find("\"", x_pos + 3) - x_pos - 3));
            point.y = std::stoi(line.substr(y_pos + 3, line.find("\"", y_pos + 3) - y_pos - 3));
            bbox.setContour(point);
            point = cv::Point();
        }
        else if(line.find("</space>") != std::string::npos) {
            spaces.push_back(bbox);
            bbox = BBox();
        }
    }
    file.close();
    return spaces;
}
namespace loaders {

    int loadAllFrames(std::vector<std::vector<cv::Mat>>& output) {
        int framesCount = 0;

        // Image loading
        std::string path;

        for (int i = 0; i <= 5; i++) {
            path = "dataset/sequence" + std::to_string(i) + "/frames";
            if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
                std::vector<cv::Mat> images;
                std::vector<std::string> imagePaths;

                // Collect all image paths
                for (const auto& entry : std::filesystem::directory_iterator(path)) {
                    imagePaths.push_back(entry.path().string());
                }

                // Sort image paths by filename
                std::sort(imagePaths.begin(), imagePaths.end());

                // Load images in sorted order
                for (const auto& imagePath : imagePaths) {
                    cv::Mat image = cv::imread(imagePath);
                    if (!image.empty()) {
                        images.push_back(image);
                        framesCount++;
                    } else {
                        std::cerr << "Failed to load image: " << imagePath << std::endl;
                    }
                }

                output.push_back(images);
            } else {
                std::cerr << "Directory does not exist or is not a directory: " << path << std::endl;
            }
        }

        return framesCount;
    }

    int loadAllBackgrounds(std::vector<cv::Mat>& output) {
        std::string path = "dataset/backgrounds";
        if(std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
            for(const auto& entry : std::filesystem::directory_iterator(path)) {
                cv::Mat empty_parking = cv::imread(entry.path().string());
                if(!empty_parking.empty()) {
                    output.push_back(empty_parking);
                } else {
                    std::cerr << "Failed to load image: " << entry.path().string() << std::endl;
                }
            }
        } else {
            std::cerr << "Directory does not exist or is not a directory: " << path << std::endl;
        }

        return output.size();
    }


    int loadAllRealMasks(std::vector<std::vector<cv::Mat>>& output) {

        int masksCount = 0;

        // Adding black mask to sequence0 index (no file in the dataset)
        std::vector<cv::Mat> sequence0;
        cv::Mat exampleImage = cv::imread("dataset/sequence0/frames/2013-02-24_10_05_04.jpg");
        for (int i = 0; i < 5; i++) {
            sequence0.push_back(cv::Mat::zeros(exampleImage.size(), exampleImage.type()));
        }
        output.push_back(sequence0);
        

        for (int i = 1; i <= 5; i++) {
            std::string path = "dataset/sequence" + std::to_string(i) + "/masks";
            if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
                std::vector<cv::Mat> masks;
                std::vector<std::string> maskPaths;

                // Collect all mask paths
                for (const auto& entry : std::filesystem::directory_iterator(path)) {
                    maskPaths.push_back(entry.path().string());
                }

                // Sort the paths by filename
                std::sort(maskPaths.begin(), maskPaths.end());

                // Load and store the masks in sorted order
                for (const auto& maskPath : maskPaths) {
                    cv::Mat mask = cv::imread(maskPath);
                    if (!mask.empty()) {
                        cv::Mat coloredMask = mask.clone();
                        coloredMask.setTo(cv::Scalar(0, 0, 0), mask == 0);
                        coloredMask.setTo(cv::Scalar(0, 0, 255), mask == 1);
                        coloredMask.setTo(cv::Scalar(0, 255, 0), mask == 2);

                        masks.push_back(coloredMask);
                        masksCount++;
                    } else {
                        std::cerr << "Warning: Unable to load mask from " << maskPath << std::endl;
                    }
                }
                
                output.push_back(masks);
            } else {
                std::cerr << "Directory does not exist or is not a directory." << std::endl;
            }
        }

        return masksCount;
    }

    int loadAllRealBBoxes(std::vector<std::vector<std::vector<BBox>>>& output) {
        
        int bboxesCount = 0;

        for(int i = 0; i <= 5; i++) {
            std::vector<std::vector<BBox>> sequence;
            std::string path = "dataset/sequence" + std::to_string(i) + "/bounding_boxes";
            if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
                std::vector<std::vector<BBox>> sequence;
                for (auto& entry : std::filesystem::directory_iterator(path)) {
                    std::vector<BBox> bboxes = parseParkingXML(entry.path().string());
                    bboxesCount += bboxes.size();

                    cv::Point p1(870, 0);
                    cv::Point p2(1275, 225);

                    float slope = float(p2.y - p1.y) / float(p2.x - p1.x);
                    float q = p1.y - slope * p1.x;

                    for(int z = 0; z < bboxes.size(); z++) {
                        cv::RotatedRect bbox = bboxes[z].getRotatedRect();
                        cv::Point2f vertices[4];
                        bbox.points(vertices);
                        for(int j = 0; j < 4; j++) {
                            if(vertices[j].y < slope * vertices[j].x + q) {
                                bboxes.erase(bboxes.begin() + z);
                                z--; // I have to decrement the index because I have deleted an element
                                break;
                            }
                        }
                    }  
    
                    sequence.push_back(bboxes);
                }

                output.push_back(sequence);
            } else {
                std::cerr << "File does not exist." << std::endl;
            }


        }

        return bboxesCount;
    }
}
