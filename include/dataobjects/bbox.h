#ifndef BBOX_H
#define BBOX_H

#include <iostream>

class BBox {
    private:
        int x, y, width, height;
        double angle;
    public:
        BBox(int x = 0, int y = 0, int width = 0, int height = 0, double angle = 0);

        int getX() const;
        int getY() const;
        int getWidth() const;
        int getHeight() const;
        double getAngle() const;
};
#endif
