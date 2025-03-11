// ai_colorbot — top-level application globals
// Defines extern declarations for shared singletons and state used across modules.
#ifndef APP_H
#define APP_H

#include "config/config.h"
#include "detection/detector.h"
#include "input/mouse.h"
#include "input/makcuConnection.h"

class CaptureThread;
extern CaptureThread* globalCaptureThread;

extern Config config;
extern Detector detector;
extern MouseThread* globalMouseThread;
extern MakcuConnection* makcuSerial;
extern std::atomic<bool> shooting;

#endif // APP_H
