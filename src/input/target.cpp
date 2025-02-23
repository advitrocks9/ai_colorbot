// Target selection — picks the detection box closest to the screen centre,
// applies head/body Y offsets from config, and computes the aim pivot point.
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>

#include "app.h"
#include "input/target.h"
#include "config/config.h"

AimbotTarget::AimbotTarget(int x_, int y_, int w_, int h_, int cls,
                           double px, double py)
    : x(x_), y(y_), w(w_), h(h_), classId(cls), pivotX(px), pivotY(py)
{
}

AimbotTarget* sortTargets(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>&      classes,
    int  screenWidth,
    int  screenHeight,
    bool disableHeadshot)
{
    if (boxes.empty() || classes.empty())
        return nullptr;

    cv::Point center(screenWidth / 2, screenHeight / 2);
    double minDistance = std::numeric_limits<double>::max();
    int    nearestIdx  = -1;
    int    targetY     = 0;

    // Priority 1: head boxes (closest to centre)
    if (!disableHeadshot)
    {
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (classes[i] != config.class_head) continue;
            int headOffsetY = static_cast<int>(boxes[i].height * config.head_y_offset);
            cv::Point tp(boxes[i].x + boxes[i].width / 2, boxes[i].y + headOffsetY);
            double dist = std::pow(tp.x - center.x, 2.0) + std::pow(tp.y - center.y, 2.0);
            if (dist < minDistance) {
                minDistance = dist;
                nearestIdx  = static_cast<int>(i);
                targetY     = tp.y;
            }
        }
    }

    // Priority 2: body targets (player, bot, hideout targets)
    if (disableHeadshot || nearestIdx == -1)
    {
        minDistance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (disableHeadshot && classes[i] == config.class_head) continue;
            if (classes[i] != config.class_player       &&
                classes[i] != config.class_bot          &&
                classes[i] != config.class_hideout_target_human &&
                classes[i] != config.class_hideout_target_balls)
                continue;

            int offsetY = static_cast<int>(boxes[i].height * config.body_y_offset);
            cv::Point tp(boxes[i].x + boxes[i].width / 2, boxes[i].y + offsetY);
            double dist = std::pow(tp.x - center.x, 2.0) + std::pow(tp.y - center.y, 2.0);
            if (dist < minDistance) {
                minDistance = dist;
                nearestIdx  = static_cast<int>(i);
                targetY     = tp.y;
            }
        }
    }

    if (nearestIdx == -1) return nullptr;

    int finalY;
    if (classes[nearestIdx] == config.class_head)
    {
        int headOffsetY = static_cast<int>(boxes[nearestIdx].height * config.head_y_offset);
        finalY = boxes[nearestIdx].y + headOffsetY - boxes[nearestIdx].height / 2;
    }
    else
    {
        finalY = targetY - boxes[nearestIdx].height / 2;
    }

    int finalX = boxes[nearestIdx].x;
    int finalW = boxes[nearestIdx].width;
    int finalH = boxes[nearestIdx].height;
    int finalClass = classes[nearestIdx];

    double pivotX = finalX + finalW / 2.0;
    double pivotY = finalY + finalH / 2.0;

    return new AimbotTarget(finalX, finalY, finalW, finalH, finalClass, pivotX, pivotY);
}
