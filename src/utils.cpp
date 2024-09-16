// Author: Eddy Frighetto, ID: 2119279

#include "utils.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <filesystem>
#include <string>
#include <fstream>
#include "bbox.h"

namespace parsers {

    /**
     * Custom XML parser.
     * 
     * @param filename File path of the xml file.
     * @return Parsed vector of BBox.
     */
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
}
namespace loaders {

    /**
     * Load all the frames in the dataset folder.
     * 
     * @param output 2D vector to load the frames in (vec[sequence][frame]).
     * @return Number of frames loaded.
     */
    int loadAllFrames(std::vector<std::vector<cv::Mat>>& output, std::vector<std::vector<std::string>>& dirs) {

        int framesCount = 0;

        // Image path that is currently loading
        std::string path;

        for (int i = 0; i <= 5; i++) {
            path = "dataset/sequence" + std::to_string(i) + "/frames";
            std::vector<std::string> sequence;
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

                    // Saving frame name
                    std::filesystem::path path_string(imagePath);
                    sequence.push_back(path_string.stem());

                    cv::Mat image = cv::imread(imagePath);
                    if (!image.empty()) {
                        images.push_back(image);
                        framesCount++;
                    } else {
                        std::cerr << "Failed to load image: " << imagePath << std::endl;
                    }
                }
                dirs.push_back(sequence);
                output.push_back(images);
            } else {
                std::cerr << "Directory does not exist or is not a directory: " << path << std::endl;
            }
        }

        return framesCount;
    }

    /**
     * Load all the empty backgrounds used to train background subtractor.
     * 
     * @param output Vector to load the real masks in (vec[frame]).
     * @return Number of backgrounds loaded.
     */
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

    /**
     * Load all the real masks from the dataset folder.
     * 
     * @param output 2D vector to load the real masks in (vec[sequence][frame]).
     * @return Number of masks loaded.
     */
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

    /**
     * Load all the real bounding boxes from the dataset folder.
     * 
     * @param output 3D vector to load the real bounding boxes in (vec[sequence][frame][bbox]).
     * @return Number of bounding boxes loaded.
     */
    int loadAllRealBBoxes(std::vector<std::vector<std::vector<BBox>>>& output) {
        
        int bboxesCount = 0;

        for(int i = 0; i <= 5; i++) {
            std::vector<std::vector<BBox>> sequence;
            std::string path = "dataset/sequence" + std::to_string(i) + "/bounding_boxes";
            if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
                std::vector<std::vector<BBox>> sequence;
                for (auto& entry : std::filesystem::directory_iterator(path)) {

                    // BBoxes are saved as .xml, need to parse it
                    std::vector<BBox> bboxes = parsers::parseParkingXML(entry.path().string());

                    // Removing BBoxes in the top right corner
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

                    bboxesCount += bboxes.size();
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

namespace savers{

    /**
     * Saves the image in the path.
     * 
     * @param image Image to save.
     * @param path Path where to save the image.
     */
    void saveImage(cv::Mat image, std::string path) {

        std::filesystem::path fs_path(path);

        // Creating the folders if they dont exist
        try {
            std::filesystem::create_directories(fs_path.remove_filename().string());
        } catch (const std::filesystem::filesystem_error e) {
            std::cerr << "Something went wron creating directories!" << std::endl;
        }

        // Saving the image
        cv::imwrite(path, image);
    }

    /**
     * Saves the metrics in the path.
     * 
     * @param mAP mAP score to save.
     * @param mIoU mIoU score to save.
     * @param path Path where to save the metrics.
     */
    void saveMetrics(float mAP, float mIoU, std::string path) {
        std::filesystem::path fs_path(path);

        // Creating folders if they dont exist
        try {
            std::filesystem::create_directories(fs_path.remove_filename().string());
        } catch (const std::filesystem::filesystem_error e) {
            std::cerr << "Something went wron creating directories!" << std::endl;
        }

        // Writing the txt file
        std::ofstream file(path);
        if (file.is_open()) {
            file << "mAP: " << mAP << std::endl;
            file << "mIoU: " << mIoU << std::endl;
            file.close();
        } else {
            std::cerr << "Unable to open file for writing." << std::endl;
        }
    }
}
