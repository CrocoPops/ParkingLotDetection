#include "utils.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
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
    std::cout << "Reading file " << filename << std::endl;
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