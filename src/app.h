// Top-level application globals shared across modules.
#ifndef APP_H
#define APP_H

#include <mutex>
#include "config/config.h"
#include "detection/detector.h"
#include "input/mouse.h"
#include "input/makcuConnection.h"

class CaptureThread;
extern CaptureThread* globalCaptureThread;

extern std::mutex configMutex;
extern Config config;
extern Detector detector;
extern MouseThread* globalMouseThread;
extern MakcuConnection* makcuSerial;
extern std::atomic<bool> shooting;

#endif // APP_H
