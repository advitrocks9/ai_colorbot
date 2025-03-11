#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <dxgi1_2.h>

#include <string>
#include <vector>
#include <filesystem>
#include <sstream>
#include <set>
#include <sys/stat.h>
#include <iostream>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <VersionHelpers.h>

#include "util/otherTools.h"
#include "app.h"

bool fileExists(const std::string& path)
{
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

std::string replaceExtension(const std::string& filename, const std::string& newExtension)
{
    size_t lastDot = filename.find_last_of('.');
    if (lastDot == std::string::npos)
        return filename + newExtension;
    return filename.substr(0, lastDot) + newExtension;
}

std::string intToString(int value)
{
    return std::to_string(value);
}

std::vector<std::string> getEngineFiles()
{
    std::vector<std::string> engineFiles;
    for (const auto& entry : std::filesystem::directory_iterator("models/"))
    {
        if (entry.is_regular_file() &&
            (entry.path().extension() == ".engine" || entry.path().extension() == ".onnx"))
        {
            engineFiles.push_back(entry.path().filename().string());
        }
    }
    return engineFiles;
}

std::vector<std::string> getModelFiles()
{
    return getEngineFiles();
}

std::vector<std::string> getOnnxFiles()
{
    std::vector<std::string> onnxFiles;
    for (const auto& entry : std::filesystem::directory_iterator("models/"))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx")
            onnxFiles.push_back(entry.path().filename().string());
    }
    return onnxFiles;
}

std::vector<std::string>::difference_type getModelIndex(const std::vector<std::string>& engineModels)
{
    auto it = std::find(engineModels.begin(), engineModels.end(), config.ai_model);
    return (it != engineModels.end()) ? std::distance(engineModels.begin(), it) : 0;
}

std::string getEnvironmentVars()
{
    char* envVarValue = nullptr;
    size_t len = 0;
    std::string pathEnv;

    errno_t err = _dupenv_s(&envVarValue, &len, "PATH");
    if (err == 0 && envVarValue != nullptr)
    {
        pathEnv = envVarValue;
        free(envVarValue);
    }
    return pathEnv;
}

std::string getTensorrtPath()
{
    std::string envPath = getEnvironmentVars();
    std::stringstream ss(envPath);
    std::string token;

    while (std::getline(ss, token, ';'))
    {
        std::string lower = token;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("tensorrt") != std::string::npos)
        {
            size_t pos = token.find_last_of("\\/");
            if (pos != std::string::npos)
            {
                std::string tail = token.substr(pos + 1);
                if (tail == "lib" || tail == "bin")
                    token = token.substr(0, pos);
            }
            return token;
        }
    }
    return "";
}

int getActiveMonitors()
{
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))))
        return -1;

    int monitorsWithCudaSupport = 0;
    IDXGIAdapter1* adapter = nullptr;

    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
    {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0)
        {
            CUdevice cuDevice;
            for (int dev = 0; dev < deviceCount; ++dev)
            {
                CUresult cuRes = cuDeviceGet(&cuDevice, dev);
                if (cuRes == CUDA_SUCCESS)
                {
                    CUuuid uuid;
                    cuRes = cuDeviceGetUuid(&uuid, cuDevice);
                    if (cuRes == CUDA_SUCCESS &&
                        memcmp(&uuid, &desc.AdapterLuid, sizeof(uuid)) == 0)
                    {
                        IDXGIOutput* output = nullptr;
                        for (UINT j = 0; adapter->EnumOutputs(j, &output) != DXGI_ERROR_NOT_FOUND; ++j)
                        {
                            monitorsWithCudaSupport++;
                            output->Release();
                        }
                    }
                }
            }
        }
        adapter->Release();
    }

    factory->Release();
    return monitorsWithCudaSupport;
}

HMONITOR getMonitorHandleByIndex(int monitorIndex)
{
    struct MonitorSearch
    {
        int targetIndex;
        int currentIndex;
        HMONITOR targetMonitor;
    };

    MonitorSearch search = { monitorIndex, 0, nullptr };

    EnumDisplayMonitors(nullptr, nullptr,
        [](HMONITOR hMonitor, HDC, LPRECT, LPARAM lParam) -> BOOL
        {
            MonitorSearch* s = reinterpret_cast<MonitorSearch*>(lParam);
            if (s->currentIndex == s->targetIndex)
            {
                s->targetMonitor = hMonitor;
                return FALSE;
            }
            s->currentIndex++;
            return TRUE;
        },
        reinterpret_cast<LPARAM>(&search));

    return search.targetMonitor;
}

std::vector<std::string> getAvailableModels()
{
    std::vector<std::string> available;
    std::vector<std::string> engineFiles = getEngineFiles();
    std::vector<std::string> onnxFiles   = getOnnxFiles();

    std::set<std::string> engineStems;
    for (const auto& f : engineFiles)
        engineStems.insert(std::filesystem::path(f).stem().string());

    for (const auto& f : engineFiles)
        available.push_back(f);

    for (const auto& f : onnxFiles)
    {
        if (engineStems.find(std::filesystem::path(f).stem().string()) == engineStems.end())
            available.push_back(f);
    }

    return available;
}

bool checkWin1903()
{
    return IsWindows10OrGreater() && IsWindowsVersionOrGreater(10, 0, 18362);
}

void welcomeMessage()
{
    std::cout << "\n\nai_colorbot started!\n" << std::endl;
}
