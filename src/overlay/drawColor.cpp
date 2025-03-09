#include "imgui/imgui.h"
#include "app.h"

void drawColor()
{
    // ── HSV Range ───────────────────────────────────────────────────
    ImGui::TextUnformatted("Colorbot HSV Range");
    ImGui::SliderInt("Lower Hue",        &config.hsv_lower_h, 0, 180, "%d");
    ImGui::SliderInt("Lower Saturation", &config.hsv_lower_s, 0, 255, "%d");
    ImGui::SliderInt("Lower Value",      &config.hsv_lower_v, 0, 255, "%d");
    ImGui::Separator();
    ImGui::SliderInt("Upper Hue",        &config.hsv_upper_h, 0, 180, "%d");
    ImGui::SliderInt("Upper Saturation", &config.hsv_upper_s, 0, 255, "%d");
    ImGui::SliderInt("Upper Value",      &config.hsv_upper_v, 0, 255, "%d");

    // ── Contour Filter ──────────────────────────────────────────────
    ImGui::Separator();
    ImGui::TextUnformatted("Contour Filter");
    ImGui::SliderFloat("Min Contour Area", &config.contour_area, 1.0f, 30.0f, "%.1f");
}
