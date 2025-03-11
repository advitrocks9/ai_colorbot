// PID-controlled mouse movement via Makcu serial device.
#ifndef MOUSE_H
#define MOUSE_H

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <mutex>
#include <atomic>
#include <chrono>
#include <vector>
#include <utility>
#include <queue>
#include <thread>
#include <condition_variable>

#include "input/target.h"
#include "input/makcuConnection.h"

class MouseThread
{
private:
    double screen_width;
    double screen_height;
    double dpi;
    double mouse_sensitivity;
    double fov_x;
    double fov_y;
    double center_x;
    double center_y;

    // PID state
    double kp, ki, kd;
    double integral_x, integral_y;
    double prev_error_x, prev_error_y;
    std::chrono::steady_clock::time_point last_pid_time;
    double dtMin; // minimum dt clamp (= 1 / capture_fps)

    std::atomic<bool> mouse_pressed{false};

    MakcuConnection* makcu;

    // Sub-pixel remainder accumulator for smooth fractional moves
    double remainder_x = 0.0;
    double remainder_y = 0.0;

    // Move queue decouples PID compute from serial write latency
    struct Move { int dx; int dy; };
    std::queue<Move>        moveQueue;
    std::mutex              queueMtx;
    std::condition_variable queueCv;
    const size_t            queueLimit = 5;
    std::thread             moveWorker;
    std::atomic<bool>       workerStop{false};

    void moveWorkerLoop();
    void queueMove(int dx, int dy);
    void sendMovementToDriver(int dx, int dy);
    void pidMove(double targetX, double targetY);

public:
    std::mutex input_method_mutex;

    MouseThread(
        int    resolution,
        int    dpi,
        double sensitivity,
        int    fovX,
        int    fovY,
        double pidKp,
        double pidKi,
        double pidKd,
        int    captureFps,
        MakcuConnection* makcuConnection = nullptr
    );
    ~MouseThread();

    void updateConfig(int resolution, double dpi, double sensitivity,
                      int fovX, int fovY,
                      double pidKp, double pidKi, double pidKd,
                      int captureFps);

    void moveMousePivot(double pivotX, double pivotY);
    void moveMouse(const AimbotTarget& target);
    void pressMouse(const AimbotTarget& target);
    void releaseMouse();
    void setMakcuConnection(MakcuConnection* newMakcu);
    void setTargetDetected(bool detected) { (void)detected; }
    void resetPID();
};

#endif // MOUSE_H
