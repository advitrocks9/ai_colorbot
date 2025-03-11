// Serial link to Makcu ESP32 HID emulator.
// Handles baudrate negotiation (115200 to 4 Mb/s) and async button reporting.
#ifndef MAKCU_CONNECTION_H
#define MAKCU_CONNECTION_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include "serial/serial.h"

class MakcuConnection
{
public:
    MakcuConnection();
    ~MakcuConnection();

    bool isOpen() const;

    void        write(const std::string& data);
    std::string read();

    // HID mouse commands
    void press();
    void release();
    void move(int x, int y);

    // Button state (updated asynchronously by buttonListener)
    bool isLeftPressed()   const;
    bool isRightPressed()  const;
    bool isMiddlePressed() const;
    bool isSide1Pressed()  const;
    bool isSide2Pressed()  const;

private:
    void sendCommand(const std::string& command);
    bool checkVersion();
    void sendBaudSwitchCommand();
    void buttonListener();
    void flushBuffers();
    std::string findPort();

    serial::Serial    serial_;
    std::atomic<bool> is_open_{false};
    std::atomic<bool> listening_{false};
    std::thread       listening_thread_;
    std::mutex        write_mutex_;

    std::atomic<bool> left_{false};
    std::atomic<bool> right_{false};
    std::atomic<bool> middle_{false};
    std::atomic<bool> side1_{false};
    std::atomic<bool> side2_{false};
};

#endif // MAKCU_CONNECTION_H
