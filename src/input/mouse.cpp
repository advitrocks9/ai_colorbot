// PID controller: converts detection-space targets to pixel deltas, accumulates
// sub-pixel remainders, queues moves to a worker thread for serial dispatch.
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <vector>

#include "input/mouse.h"
#include "input/target.h"

MouseThread::MouseThread(
    int    resolution,
    int    dpi,
    double sensitivity,
    int    fovX,
    int    fovY,
    double pidKp,
    double pidKi,
    double pidKd,
    int    captureFps,
    MakcuConnection* makcuConnection)
    : screen_width(resolution),
      screen_height(resolution),
      dpi(dpi),
      mouse_sensitivity(sensitivity),
      fov_x(fovX),
      fov_y(fovY),
      center_x(resolution / 2.0),
      center_y(resolution / 2.0),
      kp(pidKp),
      ki(pidKi),
      kd(pidKd),
      integral_x(0.0),
      integral_y(0.0),
      prev_error_x(0.0),
      prev_error_y(0.0),
      last_pid_time(std::chrono::steady_clock::now()),
      dtMin((1000.0 / captureFps) * 1e-3),
      makcu(makcuConnection)
{
    moveWorker = std::thread(&MouseThread::moveWorkerLoop, this);
}

void MouseThread::updateConfig(
    int    resolution,
    double dpiVal,
    double sensitivity,
    int    fovX,
    int    fovY,
    double pidKp,
    double pidKi,
    double pidKd,
    int    captureFps)
{
    screen_width = screen_height = resolution;
    dpi              = dpiVal;
    mouse_sensitivity = sensitivity;
    fov_x = fovX; fov_y = fovY;
    center_x = center_y = resolution / 2.0;
    kp = pidKp; ki = pidKi; kd = pidKd;
    dtMin = (1000.0 / captureFps) * 1e-3;
    integral_x = integral_y = 0.0;
    prev_error_x = prev_error_y = 0.0;
    last_pid_time = std::chrono::steady_clock::now();
}

MouseThread::~MouseThread()
{
    workerStop = true;
    queueCv.notify_all();
    if (moveWorker.joinable()) moveWorker.join();
}

void MouseThread::queueMove(int dx, int dy)
{
    std::lock_guard<std::mutex> lg(queueMtx);
    if (moveQueue.size() >= queueLimit) moveQueue.pop();
    moveQueue.push({ dx, dy });
    queueCv.notify_one();
}

void MouseThread::moveWorkerLoop()
{
    while (!workerStop)
    {
        std::unique_lock<std::mutex> ul(queueMtx);
        queueCv.wait(ul, [&] { return workerStop || !moveQueue.empty(); });
        while (!moveQueue.empty())
        {
            Move m = moveQueue.front();
            moveQueue.pop();
            ul.unlock();
            sendMovementToDriver(m.dx, m.dy);
            ul.lock();
        }
    }
}

void MouseThread::sendMovementToDriver(int dx, int dy)
{
    if (makcu) makcu->move(dx, dy);
}

void MouseThread::pidMove(double targetX, double targetY)
{
    double errorX = targetX - center_x;
    double errorY = targetY - center_y;

    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_pid_time).count();
    if (dt < dtMin) dt = dtMin;
    last_pid_time = now;

    integral_x += errorX * dt;
    integral_y += errorY * dt;

    double derivX = (errorX - prev_error_x) / dt;
    double derivY = (errorY - prev_error_y) / dt;

    prev_error_x = errorX;
    prev_error_y = errorY;

    double outputX = kp * errorX + ki * integral_x + kd * derivX;
    double outputY = kp * errorY + ki * integral_y + kd * derivY;

    // Empirically fitted sensitivity→physical-pixel scale factor
    double sensFactor = 1.07437623 * std::pow(mouse_sensitivity, -0.9936827126);
    double dpiScale   = 800.0 / dpi;

    remainder_x += outputX * sensFactor * dpiScale;
    remainder_y += outputY * sensFactor * dpiScale;

    int moveX = static_cast<int>(remainder_x);
    int moveY = static_cast<int>(remainder_y);

    remainder_x -= moveX;
    remainder_y -= moveY;

    queueMove(moveX, moveY);
}

void MouseThread::moveMousePivot(double pivotX, double pivotY)
{
    std::lock_guard<std::mutex> lg(input_method_mutex);
    pidMove(pivotX, pivotY);
}

void MouseThread::moveMouse(const AimbotTarget& target)
{
    std::lock_guard<std::mutex> lg(input_method_mutex);
    pidMove(target.x + target.w / 2.0, target.y + target.h / 2.0);
}

void MouseThread::pressMouse(const AimbotTarget& /*target*/)
{
    std::lock_guard<std::mutex> lg(input_method_mutex);
    if (!mouse_pressed) {
        makcu->press();
        mouse_pressed = true;
    } else {
        makcu->release();
        mouse_pressed = false;
    }
}

void MouseThread::releaseMouse()
{
    std::lock_guard<std::mutex> lg(input_method_mutex);
    if (mouse_pressed) {
        makcu->release();
        mouse_pressed = false;
    }
}

void MouseThread::setMakcuConnection(MakcuConnection* newMakcu)
{
    makcu = newMakcu;
}

void MouseThread::resetPID()
{
    std::lock_guard<std::mutex> lg(input_method_mutex);
    integral_x = integral_y = 0.0;
    prev_error_x = prev_error_y = 0.0;
    last_pid_time = std::chrono::steady_clock::now();
    remainder_x = remainder_y = 0.0;
}
