#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <vector>
#include <string>
#include "imgui/imgui.h"
#include "app.h"
#include "overlay/overlay.h"
#include "util/otherTools.h"

void drawCapture()
{
    // ── Capture Settings ────────────────────────────────────────────
    ImGui::TextUnformatted("Capture Settings");
    ImGui::SliderInt("Detection Resolution", &config.detection_resolution, 50, 1280);
    if (config.detection_resolution >= 400)
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "WARNING: Large capture size may hurt performance.");

    ImGui::SliderInt("Capture FPS", &config.capture_fps, 0, 240);
    if (config.capture_fps == 0)
    {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Disabled");
    }
    else if (config.capture_fps >= 120)
    {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "WARNING: High FPS may hurt performance.");
    }

    ImGui::Checkbox("Use CUDA in Capture", &config.capture_use_cuda);
    ImGui::Checkbox("Circle Mask",         &config.circle_mask);
    {
        int monitors = getActiveMonitors();
        std::vector<std::string> monitorNames;
        if (monitors == -1) {
            monitorNames.push_back("Monitor 1");
        } else {
            for (int i = -1; i < monitors; ++i)
                monitorNames.push_back("Monitor " + std::to_string(i + 1));
        }
        std::vector<const char*> items;
        for (auto& name : monitorNames)
            items.push_back(name.c_str());

        if (ImGui::Combo("Capture Monitor", &config.monitor_idx,
                         items.data(), static_cast<int>(items.size())))
        {
            config.saveConfig();
        }
    }

    ImGui::Separator();

    // ── Debug Window ────────────────────────────────────────────────
    ImGui::TextUnformatted("Debug Window");
    ImGui::Checkbox  ("Show Window",            &config.show_window);
    ImGui::Checkbox  ("Show FPS",               &config.show_fps);
    ImGui::SliderInt ("Window Size",            &config.window_size, 10, 350);
    ImGui::Checkbox  ("Always On Top",          &config.always_on_top);
    ImGui::Checkbox  ("Verbose Console Output", &config.verbose);
}
