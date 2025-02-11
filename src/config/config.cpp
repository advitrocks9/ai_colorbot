// Configuration loading and saving — reads/writes all parameters from/to config.ini
// using the SimpleIni header-only library.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>

#include "config/config.h"
#include "SimpleIni.h"

std::vector<std::string> Config::splitString(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter))
    {
        while (!item.empty() && isspace(static_cast<unsigned char>(item.front())))
            item.erase(item.begin());
        while (!item.empty() && isspace(static_cast<unsigned char>(item.back())))
            item.pop_back();
        tokens.push_back(item);
    }
    return tokens;
}

std::string Config::joinStrings(const std::vector<std::string>& vec,
                                const std::string& delimiter)
{
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i) oss << delimiter;
        oss << vec[i];
    }
    return oss.str();
}

bool Config::loadConfig(const std::string& filename)
{
    if (!std::filesystem::exists(filename))
    {
        std::cerr << "[Config] No config found, creating defaults: " << filename << "\n";

        detection_resolution = 320;
        capture_fps          = 100;
        monitor_idx          = 0;
        capture_use_cuda     = true;
        circle_mask          = false;

        disable_headshot     = false;
        body_y_offset        = 0.25f;
        head_y_offset        = 0.60f;

        dpi                  = 1000;
        sensitivity          = 0.48f;
        fovX                 = 50;
        fovY                 = 50;
        pid_kp               = 0.50;
        pid_ki               = 0.000;
        pid_kd               = 0.0000;

        ai_model             = "yolo10.engine";
        confidence_threshold = 0.25f;
        nms_threshold        = 0.50f;
        max_detections       = 100;
        postprocess          = "yolo10";
        export_precision     = 8;

        use_cuda_graph       = false;
        use_pinned_memory    = true;

        class_player = 0; class_bot = 1; class_weapon = 2; class_outline = 3;
        class_dead_body = 4; class_hideout_target_human = 5;
        class_hideout_target_balls = 6; class_head = 7;
        class_smoke = 8; class_fire = 9; class_third_person = 10;
        detect = { true,false,false,false,false,false,false,true,false,false,false };
        color  = detect;

        show_window   = true;
        show_fps      = true;
        window_size   = 80;
        always_on_top = true;
        verbose       = false;

        hsv_lower_h  = 140; hsv_lower_s  = 40;  hsv_lower_v  = 100;
        hsv_upper_h  = 160; hsv_upper_s  = 255; hsv_upper_v  = 255;
        contour_area = 10.0f;

        saveConfig(filename);
        return true;
    }

    CSimpleIniA ini;
    ini.SetUnicode();
    if (ini.LoadFile(filename.c_str()) < 0) {
        std::cerr << "[Config] Failed to parse " << filename << "\n";
        return false;
    }

    auto getString = [&](const char* key, const char* def) {
        return std::string(ini.GetValue("", key, def));
    };
    auto getBool   = [&](const char* key, bool def)   { return ini.GetBoolValue("", key, def); };
    auto getLong   = [&](const char* key, long def)   { return static_cast<int>(ini.GetLongValue("", key, def)); };
    auto getDouble = [&](const char* key, double def) { return ini.GetDoubleValue("", key, def); };

    detection_resolution = getLong("detection_resolution", 320);
    capture_fps          = getLong("capture_fps", 100);
    monitor_idx          = getLong("monitor_idx", 0);
    capture_use_cuda     = getBool("capture_use_cuda", true);
    circle_mask          = getBool("circle_mask", false);

    disable_headshot = getBool("disable_headshot", false);
    body_y_offset    = static_cast<float>(getDouble("body_y_offset", 0.25));
    head_y_offset    = static_cast<float>(getDouble("head_y_offset", 0.60));

    dpi         = getLong("dpi", 800);
    sensitivity = static_cast<float>(getDouble("sensitivity", 0.40));
    fovX        = getLong("fovX", 50);
    fovY        = getLong("fovY", 50);
    pid_kp      = getDouble("pid_kp", 0.050);
    pid_ki      = getDouble("pid_ki", 0.000);
    pid_kd      = getDouble("pid_kd", 0.0000);

    ai_model             = getString("ai_model", "yolo10.engine");
    confidence_threshold = static_cast<float>(getDouble("confidence_threshold", 0.25));
    nms_threshold        = static_cast<float>(getDouble("nms_threshold", 0.50));
    max_detections       = getLong("max_detections", 100);
    postprocess          = getString("postprocess", "yolo10");
    export_precision     = getLong("export_precision", 8);

    use_cuda_graph    = getBool("use_cuda_graph", true);
    use_pinned_memory = getBool("use_pinned_memory", true);

    class_player               = getLong("class_player", 0);
    class_bot                  = getLong("class_bot", 1);
    class_weapon               = getLong("class_weapon", 2);
    class_outline              = getLong("class_outline", 3);
    class_dead_body            = getLong("class_dead_body", 4);
    class_hideout_target_human = getLong("class_hideout_target_human", 5);
    class_hideout_target_balls = getLong("class_hideout_target_balls", 6);
    class_head                 = getLong("class_head", 7);
    class_smoke                = getLong("class_smoke", 8);
    class_fire                 = getLong("class_fire", 9);
    class_third_person         = getLong("class_third_person", 10);

    {
        auto d = splitString(getString("detect", ""));
        detect.clear();
        for (auto& s : d) detect.push_back(s == "true");
    }
    {
        auto c = splitString(getString("color", ""));
        color.clear();
        for (auto& s : c) color.push_back(s == "true");
    }

    show_window   = getBool("show_window", true);
    show_fps      = getBool("show_fps", true);
    window_size   = getLong("window_size", 80);
    always_on_top = getBool("always_on_top", true);
    verbose       = getBool("verbose", false);

    hsv_lower_h  = getLong("hsv_lower_h", 140);
    hsv_lower_s  = getLong("hsv_lower_s", 40);
    hsv_lower_v  = getLong("hsv_lower_v", 100);
    hsv_upper_h  = getLong("hsv_upper_h", 160);
    hsv_upper_s  = getLong("hsv_upper_s", 255);
    hsv_upper_v  = getLong("hsv_upper_v", 255);
    contour_area = static_cast<float>(getDouble("contour_area", 4.0));

    return true;
}

bool Config::saveConfig(const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[Config] Cannot write " << filename << "\n";
        return false;
    }

    file << "# Capture\n"
         << "detection_resolution = " << detection_resolution << "\n"
         << "capture_fps = "          << capture_fps          << "\n"
         << "monitor_idx = "          << monitor_idx          << "\n"
         << "capture_use_cuda = "     << (capture_use_cuda ? "true" : "false") << "\n"
         << "circle_mask = "          << (circle_mask      ? "true" : "false") << "\n\n";

    file << "# Target\n"
         << "disable_headshot = " << (disable_headshot ? "true" : "false") << "\n"
         << std::fixed << std::setprecision(2)
         << "body_y_offset = "    << body_y_offset    << "\n"
         << "head_y_offset = "    << head_y_offset    << "\n\n";

    file << "# Mouse\n"
         << "dpi = "         << dpi         << "\n"
         << std::fixed << std::setprecision(1)
         << "sensitivity = " << sensitivity << "\n"
         << std::setprecision(0)
         << "fovX = "        << fovX        << "\n"
         << "fovY = "        << fovY        << "\n"
         << std::fixed << std::setprecision(4)
         << "pid_kp = "      << pid_kp      << "\n"
         << "pid_ki = "      << pid_ki      << "\n"
         << "pid_kd = "      << pid_kd      << "\n\n";

    file << "# AI\n"
         << "ai_model = "           << ai_model             << "\n"
         << std::fixed << std::setprecision(2)
         << "confidence_threshold = " << confidence_threshold << "\n"
         << "nms_threshold = "        << nms_threshold        << "\n"
         << std::setprecision(0)
         << "max_detections = "       << max_detections       << "\n"
         << "postprocess = "          << postprocess          << "\n"
         << "export_precision = "     << export_precision     << "\n\n";

    file << "# CUDA\n"
         << "use_cuda_graph = "    << (use_cuda_graph    ? "true" : "false") << "\n"
         << "use_pinned_memory = " << (use_pinned_memory ? "true" : "false") << "\n\n";

    file << "# Custom Classes\n"
         << "class_player = "               << class_player               << "\n"
         << "class_bot = "                  << class_bot                  << "\n"
         << "class_weapon = "               << class_weapon               << "\n"
         << "class_outline = "              << class_outline              << "\n"
         << "class_dead_body = "            << class_dead_body            << "\n"
         << "class_hideout_target_human = " << class_hideout_target_human << "\n"
         << "class_hideout_target_balls = " << class_hideout_target_balls << "\n"
         << "class_head = "                 << class_head                 << "\n"
         << "class_smoke = "                << class_smoke                << "\n"
         << "class_fire = "                 << class_fire                 << "\n"
         << "class_third_person = "         << class_third_person         << "\n\n";

    file << "# Detect & color\n";
    {
        std::vector<std::string> detectStr;
        detectStr.reserve(detect.size());
        for (bool b : detect) detectStr.push_back(b ? "true" : "false");
        file << "detect = " << joinStrings(detectStr, ",") << "\n";
    }
    {
        std::vector<std::string> colorStr;
        colorStr.reserve(color.size());
        for (bool b : color) colorStr.push_back(b ? "true" : "false");
        file << "color = " << joinStrings(colorStr, ",") << "\n\n";
    }

    file << "# Debug window\n"
         << "show_window = "   << (show_window   ? "true" : "false") << "\n"
         << "show_fps = "      << (show_fps      ? "true" : "false") << "\n"
         << "window_size = "   << window_size                        << "\n"
         << "always_on_top = " << (always_on_top ? "true" : "false") << "\n"
         << "verbose = "       << (verbose       ? "true" : "false") << "\n\n";

    file << "# Colorbot\n"
         << "hsv_lower_h = "  << hsv_lower_h  << "\n"
         << "hsv_lower_s = "  << hsv_lower_s  << "\n"
         << "hsv_lower_v = "  << hsv_lower_v  << "\n"
         << "hsv_upper_h = "  << hsv_upper_h  << "\n"
         << "hsv_upper_s = "  << hsv_upper_s  << "\n"
         << "hsv_upper_v = "  << hsv_upper_v  << "\n"
         << std::fixed << std::setprecision(1)
         << "contour_area = " << contour_area << "\n\n";

    file.close();
    return true;
}
