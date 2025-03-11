// Utility helpers: file queries, model enumeration, monitor detection.
#ifndef OTHER_TOOLS_H
#define OTHER_TOOLS_H

#include <string>
#include <vector>
#include <d3d11.h>

bool        fileExists(const std::string& path);
std::string replaceExtension(const std::string& filename, const std::string& newExtension);
std::string intToString(int value);

std::vector<std::string> getEngineFiles();
std::vector<std::string> getModelFiles();
std::vector<std::string> getOnnxFiles();
std::vector<std::string> getAvailableModels();
std::vector<std::string>::difference_type getModelIndex(const std::vector<std::string>& engineModels);

std::string getEnvironmentVars();
std::string getTensorrtPath();

int      getActiveMonitors();
HMONITOR getMonitorHandleByIndex(int monitorIndex);

bool checkWin1903();
void welcomeMessage();

#endif // OTHER_TOOLS_H
