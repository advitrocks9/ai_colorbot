// Win32 layered window + OpenGL + ImGui overlay with live config propagation.
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <tchar.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <GL/gl.h>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_opengl2.h"

#include "overlay/overlay.h"
#include "overlay/drawSettings.h"
#include "config/config.h"
#include "app.h"
#include "capture/capture.h"
#include "util/otherTools.h"
#include "detection/detector.h"

HWND g_hwnd      = nullptr;
HDC  g_hDC       = nullptr;
HGLRC g_glContext = nullptr;
int  overlayWidth  = 680;
int  overlayHeight = 480;

extern Config       config;
extern std::mutex   configMutex;
extern std::atomic<bool> shouldExit;
extern CaptureThread*    globalCaptureThread;
extern MouseThread*      globalMouseThread;

static LRESULT WINAPI wndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

static bool createOverlayWindow()
{
    DWORD exStyle = WS_EX_LAYERED | WS_EX_TOPMOST;
    WNDCLASSEX wc{ sizeof(wc), CS_CLASSDC, wndProc, 0, 0,
                   GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                   _T("OverlayClass"), NULL };
    RegisterClassEx(&wc);

    g_hwnd = CreateWindowEx(
        exStyle,
        wc.lpszClassName, _T("Overlay"),
        WS_POPUP,
        0, 0, overlayWidth, overlayHeight,
        NULL, NULL, wc.hInstance, NULL);
    if (!g_hwnd) return false;

    SetLayeredWindowAttributes(g_hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);

    g_hDC = GetDC(g_hwnd);
    PIXELFORMATDESCRIPTOR pfd = {
        sizeof(pfd), 1,
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        PFD_TYPE_RGBA, 32,
        0,0,0,0,0,0,0,0,0,0,0,0,
        24, 8, 0, PFD_MAIN_PLANE, 0, 0, 0, 0, 0
    };
    int pf = ChoosePixelFormat(g_hDC, &pfd);
    SetPixelFormat(g_hDC, pf, &pfd);
    g_glContext = wglCreateContext(g_hDC);
    wglMakeCurrent(g_hDC, g_glContext);
    return true;
}

static void setupImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplOpenGL2_Init();
    ImGui::StyleColorsDark();
}

static void cleanupOverlay()
{
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(g_glContext);
    ReleaseDC(g_hwnd, g_hDC);
    DestroyWindow(g_hwnd);
    UnregisterClass(_T("OverlayClass"), GetModuleHandle(NULL));
}

void overlayThread()
{
    if (!createOverlayWindow()) return;
    setupImGui();

    int   prev_detection_resolution  = config.detection_resolution;
    int   prev_capture_fps           = config.capture_fps;
    bool  prev_capture_use_cuda      = config.capture_use_cuda;
    bool  prev_circle_mask           = config.circle_mask;
    bool  prev_disable_headshot      = config.disable_headshot;
    float prev_body_y_offset         = config.body_y_offset;
    float prev_head_y_offset         = config.head_y_offset;
    int   prev_dpi                   = config.dpi;
    float prev_sensitivity           = config.sensitivity;
    int   prev_fovX                  = config.fovX;
    int   prev_fovY                  = config.fovY;
    float prev_pid_kp                = static_cast<float>(config.pid_kp);
    float prev_pid_ki                = static_cast<float>(config.pid_ki);
    float prev_pid_kd                = static_cast<float>(config.pid_kd);
    std::string prev_ai_model        = config.ai_model;
    float prev_confidence_thresh     = config.confidence_threshold;
    float prev_nms_thresh            = config.nms_threshold;
    int   prev_max_detections        = config.max_detections;
    std::string prev_postprocess     = config.postprocess;
    int   prev_export_precision      = config.export_precision;
    bool  prev_cuda_graph            = config.use_cuda_graph;
    bool  prev_pinned_memory         = config.use_pinned_memory;
    std::vector<bool> prev_detect    = config.detect;
    std::vector<bool> prev_color     = config.color;
    bool  prev_show_window           = config.show_window;
    bool  prev_show_fps              = config.show_fps;
    int   prev_window_size           = config.window_size;
    bool  prev_verbose               = config.verbose;
    int   prev_h_l = config.hsv_lower_h, prev_h_s = config.hsv_lower_s;
    int   prev_h_v = config.hsv_lower_v;
    int   prev_H_l = config.hsv_upper_h, prev_H_s = config.hsv_upper_s;
    int   prev_H_v = config.hsv_upper_v;
    float prev_contour_area          = config.contour_area;

    MSG msg;
    ZeroMemory(&msg, sizeof(msg));

    while (!shouldExit)
    {
        while (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) { shouldExit = true; return; }
        }

        // Toggle overlay visibility with backslash key
        if (GetAsyncKeyState(VK_OEM_5) & 0x8000)
        {
            ShowWindow(g_hwnd, IsWindowVisible(g_hwnd) ? SW_HIDE : SW_SHOW);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        if (IsWindowVisible(g_hwnd))
        {
            ImGui_ImplOpenGL2_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(static_cast<float>(overlayWidth),
                                            static_cast<float>(overlayHeight)),
                                     ImGuiCond_Always);
            ImGui::Begin("Options", nullptr,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

            {
                std::lock_guard<std::mutex> lock(configMutex);
                if (ImGui::BeginTabBar("Options tab bar"))
                {
                    if (ImGui::BeginTabItem("Aim"))     { drawAim();     ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Capture")) { drawCapture(); ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("AI"))      { drawAi();      ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Color"))   { drawColor();   ImGui::EndTabItem(); }
                    ImGui::EndTabBar();
                }

                // Capture changes
                if (prev_detection_resolution != config.detection_resolution) {
                    prev_detection_resolution = config.detection_resolution;
                    detection_resolution_changed.store(true);
                    globalCaptureThread->updateConfig(config.detection_resolution,
                        config.capture_use_cuda, config.capture_fps, config.monitor_idx);
                    config.saveConfig();
                }
                if (prev_capture_use_cuda != config.capture_use_cuda) {
                    prev_capture_use_cuda = config.capture_use_cuda;
                    globalCaptureThread->updateConfig(config.detection_resolution,
                        config.capture_use_cuda, config.capture_fps, config.monitor_idx);
                    config.saveConfig();
                }
                if (prev_capture_fps != config.capture_fps) {
                    prev_capture_fps = config.capture_fps;
                    capture_fps_changed.store(true);
                    globalCaptureThread->updateConfig(config.detection_resolution,
                        config.capture_use_cuda, config.capture_fps, config.monitor_idx);
                    config.saveConfig();
                }

                // Target changes
                if (prev_disable_headshot != config.disable_headshot ||
                    prev_body_y_offset    != config.body_y_offset    ||
                    prev_head_y_offset    != config.head_y_offset)
                {
                    prev_disable_headshot = config.disable_headshot;
                    prev_body_y_offset    = config.body_y_offset;
                    prev_head_y_offset    = config.head_y_offset;
                    config.saveConfig();
                }

                // Mouse / PID changes
                if (prev_dpi != config.dpi           ||
                    prev_sensitivity != config.sensitivity ||
                    prev_fovX != config.fovX          ||
                    prev_fovY != config.fovY          ||
                    prev_pid_kp != static_cast<float>(config.pid_kp) ||
                    prev_pid_ki != static_cast<float>(config.pid_ki) ||
                    prev_pid_kd != static_cast<float>(config.pid_kd))
                {
                    prev_dpi         = config.dpi;
                    prev_sensitivity = config.sensitivity;
                    prev_fovX        = config.fovX;
                    prev_fovY        = config.fovY;
                    prev_pid_kp      = static_cast<float>(config.pid_kp);
                    prev_pid_ki      = static_cast<float>(config.pid_ki);
                    prev_pid_kd      = static_cast<float>(config.pid_kd);
                    globalMouseThread->updateConfig(config.detection_resolution,
                        config.dpi, config.sensitivity, config.fovX, config.fovY,
                        config.pid_kp, config.pid_ki, config.pid_kd, config.capture_fps);
                    config.saveConfig();
                }

                // AI changes
                if (prev_ai_model != config.ai_model) {
                    prev_ai_model = config.ai_model;
                    detector_model_changed.store(true);
                    config.saveConfig();
                }
                if (prev_confidence_thresh != config.confidence_threshold ||
                    prev_nms_thresh        != config.nms_threshold        ||
                    prev_max_detections    != config.max_detections        ||
                    prev_postprocess       != config.postprocess           ||
                    prev_export_precision  != config.export_precision)
                {
                    prev_confidence_thresh = config.confidence_threshold;
                    prev_nms_thresh        = config.nms_threshold;
                    prev_max_detections    = config.max_detections;
                    prev_postprocess       = config.postprocess;
                    prev_export_precision  = config.export_precision;
                    detector_model_changed.store(true);
                    config.saveConfig();
                }

                // CUDA options
                if (prev_cuda_graph    != config.use_cuda_graph ||
                    prev_pinned_memory != config.use_pinned_memory)
                {
                    prev_cuda_graph    = config.use_cuda_graph;
                    prev_pinned_memory = config.use_pinned_memory;
                    config.saveConfig();
                }

                // Per-class detect/color
                if (prev_detect != config.detect || prev_color != config.color) {
                    prev_detect = config.detect;
                    prev_color  = config.color;
                    config.saveConfig();
                }

                // Debug window
                if (prev_show_window != config.show_window ||
                    prev_show_fps    != config.show_fps    ||
                    prev_window_size != config.window_size ||
                    prev_verbose     != config.verbose)
                {
                    prev_show_window = config.show_window;
                    prev_show_fps    = config.show_fps;
                    prev_window_size = config.window_size;
                    prev_verbose     = config.verbose;
                    show_window_changed.store(true);
                    config.saveConfig();
                }

                // Colorbot HSV
                if (prev_h_l != config.hsv_lower_h || prev_h_s != config.hsv_lower_s ||
                    prev_h_v != config.hsv_lower_v  || prev_H_l != config.hsv_upper_h ||
                    prev_H_s != config.hsv_upper_s  || prev_H_v != config.hsv_upper_v ||
                    prev_contour_area != config.contour_area)
                {
                    prev_h_l = config.hsv_lower_h; prev_h_s = config.hsv_lower_s;
                    prev_h_v = config.hsv_lower_v; prev_H_l = config.hsv_upper_h;
                    prev_H_s = config.hsv_upper_s; prev_H_v = config.hsv_upper_v;
                    prev_contour_area = config.contour_area;
                    config.saveConfig();
                }
            }

            ImGui::End();
            ImGui::Render();

            glViewport(0, 0, overlayWidth, overlayHeight);
            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
            SwapBuffers(g_hDC);
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    cleanupOverlay();
}

static LRESULT WINAPI wndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return TRUE;
    switch (msg)
    {
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED) {
            overlayWidth  = LOWORD(lParam);
            overlayHeight = HIWORD(lParam);
            glViewport(0, 0, overlayWidth, overlayHeight);
        }
        return 0;
    case WM_DESTROY:
        shouldExit = true;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}
