#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "imgui/imgui.h"
#include "app.h"

void drawAim()
{
    static double kpMin = 0.00, kpMax = 0.30;
    static double kiMin = 0.00, kiMax = 0.05;
    static double kdMin = 0.00, kdMax = 0.02;

    ImGui::TextUnformatted("Mouse Settings");
    ImGui::SliderInt  ("DPI",         &config.dpi,         400,   1600);
    ImGui::SliderFloat("Sensitivity", &config.sensitivity, 0.10f, 1.0f, "%.2f");
    ImGui::SliderInt  ("FOV X",       &config.fovX,        10,    320);
    ImGui::SliderInt  ("FOV Y",       &config.fovY,        10,    320);

    ImGui::Separator();

    ImGui::TextUnformatted("PID Controller");
    ImGui::SliderScalar("Kp", ImGuiDataType_Double, &config.pid_kp, &kpMin, &kpMax, "%.3f");
    ImGui::SliderScalar("Ki", ImGuiDataType_Double, &config.pid_ki, &kiMin, &kiMax, "%.3f");
    ImGui::SliderScalar("Kd", ImGuiDataType_Double, &config.pid_kd, &kdMin, &kdMax, "%.4f");

    ImGui::Separator();
    ImGui::TextUnformatted("Targeting");
    ImGui::Checkbox   ("Disable Headshots", &config.disable_headshot);
    ImGui::SliderFloat("Body Y Offset",     &config.body_y_offset, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Head Y Offset",     &config.head_y_offset, 0.0f, 1.0f, "%.2f");
}
