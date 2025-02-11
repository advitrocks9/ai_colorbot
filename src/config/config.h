// Runtime configuration — all tuneable parameters loaded from config/config.ini.
#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

class Config
{
public:
    // Capture
    int   detection_resolution;
    int   capture_fps;
    int   monitor_idx;
    bool  capture_use_cuda;
    bool  circle_mask;

    // Target
    bool  disable_headshot;
    float body_y_offset;
    float head_y_offset;

    // Mouse / PID
    int    dpi;
    float  sensitivity;
    int    fovX;
    int    fovY;
    double pid_kp;
    double pid_ki;
    double pid_kd;

    // AI / TensorRT
    std::string ai_model;
    float       confidence_threshold;
    float       nms_threshold;
    int         max_detections;
    std::string postprocess;
    int         export_precision;

    // CUDA optimisations
    bool use_cuda_graph;
    bool use_pinned_memory;

    // Per-class indices (custom-trained label set)
    int class_player;
    int class_bot;
    int class_weapon;
    int class_outline;
    int class_dead_body;
    int class_hideout_target_human;
    int class_hideout_target_balls;
    int class_head;
    int class_smoke;
    int class_fire;
    int class_third_person;
    std::vector<bool> detect; // whether to include each class in output
    std::vector<bool> color;  // whether to apply HSV color filter per class

    // Debug visualisation
    bool show_window;
    bool show_fps;
    int  window_size;
    bool always_on_top;
    bool verbose;

    // HSV color filter (applied per-class when color[classId] == true)
    int   hsv_lower_h;
    int   hsv_lower_s;
    int   hsv_lower_v;
    int   hsv_upper_h;
    int   hsv_upper_s;
    int   hsv_upper_v;
    float contour_area; // minimum contour area to accept a color match

    bool loadConfig(const std::string& filename = "config.ini");
    bool saveConfig(const std::string& filename = "config.ini");
    std::string joinStrings(const std::vector<std::string>& vec,
                            const std::string& delimiter = ",");

private:
    std::vector<std::string> splitString(const std::string& str, char delimiter = ',');
};

#endif // CONFIG_H
