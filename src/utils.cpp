#include "utils.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include "tinyxml2.h"
#include "bbox.h"

using namespace cv;
using namespace std;
using namespace tinyxml2;

// Implement here the utility functions
void drawRotatedRectangle(Mat& image, RotatedRect rect)
{
    Scalar color = Scalar(0.0, 0.0, 255.0);

    Point2f vertices2f[4];
    rect.points(vertices2f);

    Point vertices[4];    
    for(int i = 0; i < 4; ++i)
        vertices[i] = vertices2f[i];

    for (int i = 0; i < 4; i++)
        line(image, vertices[i], vertices[(i+1)%4], color, 2);
}

BBox parseBBox(XMLElement* space) {
    int id = space->IntAttribute("id");
    bool occupied = space->IntAttribute("occupied") != 0;

    XMLElement* rotatedRectElement = space->FirstChildElement("rotatedRect");

    XMLElement* center = rotatedRectElement->FirstChildElement("center");
    float centerX = center->FloatAttribute("x");
    float centerY = center->FloatAttribute("y");

    XMLElement* size = rotatedRectElement->FirstChildElement("size");
    float width = size->FloatAttribute("w");
    float height = size->FloatAttribute("h");

    XMLElement* angle = rotatedRectElement->FirstChildElement("angle");
    float angleD = angle->DoubleAttribute("d");
    BBox parkingSpace(centerX, centerY, width, height, angleD, occupied);
    return parkingSpace;
}

vector<BBox> parseParkingXML(const string& filename) {
    XMLDocument doc;
    
    if (doc.LoadFile(filename.c_str()) != XML_SUCCESS) {
        cerr << "Error loading XML file!" << endl;
        return {};
    }

    XMLElement* root = doc.FirstChildElement("parking");
    vector<BBox> spaces;

    XMLElement* space = root->FirstChildElement("space");
    // Loop until it can find more <space> tags
    for (; space != nullptr; space = space->NextSiblingElement("space")) {
        BBox bbox = parseBBox(space);
        spaces.push_back(bbox);
    }

    return spaces;
}

