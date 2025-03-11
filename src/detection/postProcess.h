// YOLO postprocessing: bounding box decoding and NMS.
#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "detection/detector.h"

struct Detection
{
    cv::Rect box{};
    float    confidence = 0.0f;
    int      classId    = -1;
};

void NMS(std::vector<Detection>& detections, float nmsThreshold);

// Decode YOLO10 output (centre-format [cx, cy, x2, y2, conf, classId] per row).
std::vector<Detection> postProcessYolo10(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold);

// Decode YOLO8/9/11/12 output (transposed [4 + numClasses] × numAnchors).
std::vector<Detection> postProcessYolo11(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold);

#endif // POSTPROCESS_H
