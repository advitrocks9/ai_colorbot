// Overlay — Win32/OpenGL settings window with ImGui tabs.
#ifndef OVERLAY_H
#define OVERLAY_H

#include <Windows.h>
#include <atomic>
#include "app.h"
#include "imgui/imgui_impl_win32.h"

// Change-notification flags: written by the overlay UI, read by pipeline threads.
// Declared inline (C++17) so they can be safely included in multiple translation units.
inline std::atomic<bool> detection_resolution_changed{false};
inline std::atomic<bool> capture_fps_changed{false};
inline std::atomic<bool> detector_model_changed{false};
inline std::atomic<bool> show_window_changed{false};
inline std::atomic<bool> capture_cuda_changed{false};

extern HWND g_hwnd;
extern int  overlayWidth;
extern int  overlayHeight;

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg,
                                              WPARAM wParam, LPARAM lParam);

// Main overlay thread — creates the window, runs the ImGui loop, and monitors
// config values for changes to propagate to capture/detection threads.
void overlayThread();

#endif // OVERLAY_H
