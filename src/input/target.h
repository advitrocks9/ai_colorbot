// AimbotTarget — represents a selected detection target with a computed aim pivot.
#ifndef TARGET_H
#define TARGET_H

#include <opencv2/opencv.hpp>
#include <vector>

class AimbotTarget
{
public:
    int x, y, w, h;
    int classId;

    // Computed aim point in detection-space coordinates
    double pivotX;
    double pivotY;

    AimbotTarget(int x, int y, int w, int h, int classId,
                 double pivotX = 0.0, double pivotY = 0.0);
};

// Select the closest target to screen centre, preferring head boxes unless
// disableHeadshot is true. Returns nullptr if no valid target is found.
AimbotTarget* sortTargets(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>&      classes,
    int  screenWidth,
    int  screenHeight,
    bool disableHeadshot
);

#endif // TARGET_H
