// YOLO bounding box decoding and NMS.
#include <algorithm>
#include <numeric>
#include <iostream>

#include "detection/postProcess.h"
#include "app.h"
#include "detection/detector.h"

void NMS(std::vector<Detection>& detections, float nmsThreshold)
{
    if (detections.empty()) return;

    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<bool> suppress(detections.size(), false);
    std::vector<Detection> result;
    result.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppress[i]) continue;
        result.push_back(detections[i]);

        const cv::Rect& boxI  = detections[i].box;
        const float     areaI = static_cast<float>(boxI.area());

        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppress[j]) continue;
            const cv::Rect& boxJ        = detections[j].box;
            const cv::Rect intersection = boxI & boxJ;
            if (intersection.width > 0 && intersection.height > 0)
            {
                float interArea = static_cast<float>(intersection.area());
                float unionArea = areaI + static_cast<float>(boxJ.area()) - interArea;
                if (interArea / unionArea > nmsThreshold)
                    suppress[j] = true;
            }
        }
    }

    detections = std::move(result);
}

// YOLO10 output layout: [batch, numDetections, 6]
// Each row: [cx, cy, x2, y2, confidence, classId]  (in 640-space coordinates)
std::vector<Detection> postProcessYolo10(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold)
{
    std::vector<Detection> detections;
    int64_t numDetections = shape[1];

    for (int64_t i = 0; i < numDetections; ++i)
    {
        const float* det = output + i * shape[2];
        float confidence = det[4];
        if (confidence <= confThreshold) continue;

        int   classId = static_cast<int>(det[5]);
        float cx = det[0], cy = det[1], dx = det[2], dy = det[3];

        Detection d;
        d.box = cv::Rect(
            static_cast<int>(cx * detector.img_scale),
            static_cast<int>(cy * detector.img_scale),
            static_cast<int>((dx - cx) * detector.img_scale),
            static_cast<int>((dy - cy) * detector.img_scale)
        );
        d.confidence = confidence;
        d.classId    = classId;
        detections.push_back(d);
    }

    NMS(detections, nmsThreshold);
    return detections;
}

// YOLO8/9/11/12 output layout: [batch, 4 + numClasses, numAnchors]  (transposed)
// Each column: [cx, cy, w, h, class_score_0 ... class_score_N]
std::vector<Detection> postProcessYolo11(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold)
{
    if (shape.size() != 3) {
        std::cerr << "[postProcess] Unsupported output shape" << std::endl;
        return {};
    }

    int rows = static_cast<int>(shape[1]);
    int cols = static_cast<int>(shape[2]);

    if (rows < 4 + numClasses) {
        std::cerr << "[postProcess] Row count too small for " << numClasses
                  << " classes" << std::endl;
        return {};
    }

    std::vector<Detection> detections;
    detections.reserve(static_cast<size_t>(cols));

    cv::Mat det_output(rows, cols, CV_32F, const_cast<float*>(output));
    const float imgScale = detector.img_scale;

    for (int i = 0; i < cols; ++i)
    {
        cv::Mat classScores = det_output.col(i).rowRange(4, 4 + numClasses);
        cv::Point classIdPt;
        double score;
        cv::minMaxLoc(classScores, nullptr, &score, nullptr, &classIdPt);
        if (score <= confThreshold) continue;

        float cx = det_output.at<float>(0, i);
        float cy = det_output.at<float>(1, i);
        float ow = det_output.at<float>(2, i);
        float oh = det_output.at<float>(3, i);
        float halfW = 0.5f * ow;
        float halfH = 0.5f * oh;

        Detection d;
        d.box = cv::Rect(
            static_cast<int>((cx - halfW) * imgScale),
            static_cast<int>((cy - halfH) * imgScale),
            static_cast<int>(ow * imgScale),
            static_cast<int>(oh * imgScale)
        );
        d.confidence = static_cast<float>(score);
        d.classId    = classIdPt.y;
        detections.push_back(d);
    }

    if (!detections.empty())
        NMS(detections, nmsThreshold);

    return detections;
}
