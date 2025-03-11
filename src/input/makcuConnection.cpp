// Makcu serial connection: baudrate negotiation and command protocol.
// Commands: km.move(x,y), km.left(1/0), km.buttons(1), km.version()
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <iostream>
#include <string>
#include <thread>
#include <exception>
#include <mutex>
#include "input/makcuConnection.h"
#include "app.h"

void MakcuConnection::flushBuffers()
{
    serial_.flush();
    Sleep(100);
}

MakcuConnection::MakcuConnection() : is_open_(false), listening_(false)
{
    std::string portStr = findPort();
    if (portStr.empty()) {
        std::cerr << "[Makcu] No COM port found.\n";
        return;
    }
    serial_.setPort(portStr);

    // Attempt direct 4 Mb/s connection first (device may already be at high baud)
    try {
        serial_.setBaudrate(4000000);
        serial_.open();
        Sleep(100);
        flushBuffers();
        if (checkVersion()) {
            std::cerr << "[Makcu] Connected directly at 4 Mb/s.\n";
            is_open_ = true;
        } else {
            std::cerr << "[Makcu] No response at 4 Mb/s, falling back to 115200.\n";
            if (serial_.isOpen()) serial_.close();
        }
    } catch (const std::exception& e) {
        std::cerr << "[Makcu] 4 Mb/s open failed: " << e.what() << ". Trying 115200.\n";
        if (serial_.isOpen()) serial_.close();
    }

    // Fallback: connect at 115200, send baud-switch packet, re-connect at 4 Mb/s
    if (!is_open_)
    {
        try {
            serial_.setBaudrate(115200);
            if (serial_.isOpen()) serial_.close();
            serial_.open();
            Sleep(100);
            flushBuffers();

            if (!checkVersion()) {
                std::cerr << "[Makcu] No response at 115200.\n";
                if (serial_.isOpen()) serial_.close();
                return;
            }

            std::cerr << "[Makcu] Connected at 115200. Sending baud-switch command.\n";
            sendBaudSwitchCommand();
            Sleep(300);

            serial_.setBaudrate(4000000);
            flushBuffers();
            Sleep(100);

            if (!checkVersion()) {
                std::cerr << "[Makcu] No response at 4 Mb/s after switch.\n";
                if (serial_.isOpen()) serial_.close();
                return;
            }

            std::cerr << "[Makcu] Switched to 4 Mb/s successfully.\n";
            is_open_ = true;
        } catch (const std::exception& e) {
            std::cerr << "[Makcu] Connection error: " << e.what() << "\n";
            if (serial_.isOpen()) serial_.close();
            return;
        }
    }

    if (is_open_) {
        listening_ = true;
        listening_thread_ = std::thread(&MakcuConnection::buttonListener, this);
    } else {
        std::cerr << "[Makcu] All connection attempts failed.\n";
    }
}

MakcuConnection::~MakcuConnection()
{
    listening_ = false;
    if (listening_thread_.joinable())
        listening_thread_.join();
    if (serial_.isOpen())
        serial_.close();
}

bool MakcuConnection::isOpen() const { return is_open_; }

void MakcuConnection::write(const std::string& data)
{
    std::lock_guard<std::mutex> lk(write_mutex_);
    if (is_open_) {
        try { serial_.write(data); }
        catch (...) { is_open_ = false; }
    }
}

std::string MakcuConnection::read()
{
    if (!is_open_) return {};
    try {
        auto line = serial_.readline(65536, "\n");
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return line;
    } catch (...) {
        is_open_ = false;
        return {};
    }
}

void MakcuConnection::move(int x, int y)
{
    sendCommand("km.move(" + std::to_string(x) + "," + std::to_string(y) + ")");
}

void MakcuConnection::press()   { sendCommand("km.left(1)"); }
void MakcuConnection::release() { sendCommand("km.left(0)"); }

void MakcuConnection::sendCommand(const std::string& cmd)
{
    std::lock_guard<std::mutex> lk(write_mutex_);
    if (!serial_.isOpen()) return;
    try { serial_.write(cmd + "\r"); }
    catch (...) { is_open_ = false; }
}

std::string MakcuConnection::findPort()
{
    auto ports = serial::list_ports();
    return ports.empty() ? std::string() : ports[0].port;
}

bool MakcuConnection::checkVersion()
{
    flushBuffers();
    sendCommand("km.version()");
    Sleep(300);
    std::string buf;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(600);
    while (std::chrono::steady_clock::now() < deadline)
    {
        size_t n = serial_.available();
        if (n > 0) {
            buf += serial_.read(n);
            if (buf.find("MAKCU") != std::string::npos) return true;
        }
        Sleep(10);
    }
    std::cerr << "[Makcu] Unexpected version response: " << buf << "\n";
    return false;
}

void MakcuConnection::sendBaudSwitchCommand()
{
    // Binary packet that instructs the Makcu firmware to switch to 4 Mb/s
    uint8_t pkt[] = { 0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00 };
    try { serial_.write(pkt, sizeof(pkt)); }
    catch (...) { std::cerr << "[Makcu] Error sending baud-switch packet.\n"; }
}

void MakcuConnection::buttonListener()
{
    flushBuffers();
    sendCommand("km.buttons(1)");
    Sleep(100);

    auto popcount = [](uint8_t n) {
        int c = 0;
        while (n) { c += n & 1; n >>= 1; }
        return c;
    };

    while (listening_ && is_open_)
    {
        if (serial_.available() > 0)
        {
            std::string resp = serial_.read(1);
            if (resp.empty()) { Sleep(5); continue; }

            uint8_t value = static_cast<uint8_t>(resp[0]);
            // Validate: buttons byte must have exactly 1 bit set or be 0
            if (value > 31 || popcount(value) != 1) {
                Sleep(5);
                continue;
            }
            left_.store  (static_cast<bool>(value & (1 << 0)));
            right_.store (static_cast<bool>(value & (1 << 1)));
            middle_.store(static_cast<bool>(value & (1 << 2)));
            side1_.store (static_cast<bool>(value & (1 << 3)));
            side2_.store (static_cast<bool>(value & (1 << 4)));
        }
        Sleep(5);
    }

    if (is_open_ && serial_.isOpen())
        sendCommand("km.buttons(0)");
}

bool MakcuConnection::isLeftPressed()   const { return left_.load();   }
bool MakcuConnection::isRightPressed()  const { return right_.load();  }
bool MakcuConnection::isMiddlePressed() const { return middle_.load(); }
bool MakcuConnection::isSide1Pressed()  const { return side1_.load();  }
bool MakcuConnection::isSide2Pressed()  const { return side2_.load();  }
