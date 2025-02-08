// DXGI Desktop Duplication capture — D3D11 device setup, frame acquisition,
// and CUDA interop for zero-copy GPU→GPU transfer.
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include "capture/duplicationApiCapture.h"
#include "config/config.h"
#include "util/otherTools.h"

// SafeRelease helper: releases a COM interface and nulls the pointer.
template <typename T>
inline void SafeRelease(T** ppInterface)
{
    if (*ppInterface)
    {
        (*ppInterface)->Release();
        *ppInterface = nullptr;
    }
}

struct FrameContext
{
    ID3D11Texture2D* texture = nullptr;
};

// DDAManager wraps the D3D11 device, DXGI output, and IDXGIOutputDuplication
// object needed to acquire desktop frames from a specific monitor.
class DDAManager
{
public:
    DDAManager() = default;
    ~DDAManager() { Release(); }

    HRESULT Initialize(
        int monitorIndex,
        int captureWidth,
        int captureHeight,
        int& outScreenWidth,
        int& outScreenHeight,
        ID3D11Device** outDevice,
        ID3D11DeviceContext** outContext)
    {
        HRESULT hr;
        IDXGIFactory1* factory = nullptr;
        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory));
        if (FAILED(hr)) return hr;

        IDXGIAdapter1* adapter = nullptr;
        IDXGIOutput*   output  = nullptr;
        int  currentMonitor    = 0;
        bool found             = false;

        for (UINT ai = 0; ; ++ai)
        {
            hr = factory->EnumAdapters1(ai, &adapter);
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            if (FAILED(hr)) continue;

            for (UINT oi = 0; ; ++oi)
            {
                hr = adapter->EnumOutputs(oi, &output);
                if (hr == DXGI_ERROR_NOT_FOUND) break;
                if (FAILED(hr)) continue;

                DXGI_OUTPUT_DESC desc;
                output->GetDesc(&desc);
                if (!desc.AttachedToDesktop)
                {
                    SafeRelease(&output);
                    continue;
                }

                if (currentMonitor == monitorIndex)
                {
                    found = true;
                    break;
                }

                ++currentMonitor;
                SafeRelease(&output);
            }

            if (found) break;
            SafeRelease(&adapter);
        }

        SafeRelease(&factory);
        if (!found) return DXGI_ERROR_NOT_FOUND;

        D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        hr = D3D11CreateDevice(
            adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            nullptr,
            flags,
            levels,
            ARRAYSIZE(levels),
            D3D11_SDK_VERSION,
            &m_device,
            nullptr,
            &m_context
        );
        SafeRelease(&adapter);
        if (FAILED(hr)) return hr;

        hr = output->QueryInterface(__uuidof(IDXGIOutput1),
                                    reinterpret_cast<void**>(&m_output1));
        SafeRelease(&output);
        if (FAILED(hr)) return hr;

        hr = m_output1->DuplicateOutput(m_device, &m_duplication);
        if (FAILED(hr)) return hr;

        DXGI_OUTPUT_DESC oDesc{};
        m_output1->GetDesc(&oDesc);
        outScreenWidth  = static_cast<int>(oDesc.DesktopCoordinates.right  -
                                           oDesc.DesktopCoordinates.left);
        outScreenHeight = static_cast<int>(oDesc.DesktopCoordinates.bottom -
                                           oDesc.DesktopCoordinates.top);

        if (outDevice)  *outDevice  = m_device;
        if (outContext) *outContext = m_context;
        return S_OK;
    }

    // Blocks for up to `timeout` ms waiting for the next desktop frame.
    HRESULT AcquireFrame(FrameContext& frameCtx, UINT timeout = 100)
    {
        if (!m_duplication) return E_FAIL;
        DXGI_OUTDUPL_FRAME_INFO info;
        IDXGIResource* resource = nullptr;

        HRESULT hr = m_duplication->AcquireNextFrame(timeout, &info, &resource);
        if (FAILED(hr)) return hr;

        hr = resource->QueryInterface(__uuidof(ID3D11Texture2D),
                                      reinterpret_cast<void**>(&frameCtx.texture));
        resource->Release();
        return hr;
    }

    void ReleaseFrame()
    {
        if (m_duplication)
            m_duplication->ReleaseFrame();
    }

    void Release()
    {
        if (m_duplication)
        {
            m_duplication->ReleaseFrame();
            m_duplication->Release();
            m_duplication = nullptr;
        }
        SafeRelease(&m_output1);
        SafeRelease(&m_context);
        SafeRelease(&m_device);
    }

private:
    ID3D11Device*           m_device      = nullptr;
    ID3D11DeviceContext*    m_context     = nullptr;
    IDXGIOutputDuplication* m_duplication = nullptr;
    IDXGIOutput1*           m_output1     = nullptr;
};

DuplicationAPIScreenCapture::DuplicationAPIScreenCapture(
    int monitorIndex,
    int desiredWidth,
    int desiredHeight)
    : m_ddaManager(std::make_unique<DDAManager>()),
      regionWidth(desiredWidth),
      regionHeight(desiredHeight)
{
    HRESULT hr = m_ddaManager->Initialize(
        monitorIndex,
        regionWidth,
        regionHeight,
        screenWidth,
        screenHeight,
        &d3dDevice,
        &d3dContext
    );
    if (FAILED(hr))
    {
        std::cerr << "[DDA] Initialize failed hr=0x" << std::hex << hr << std::endl;
        return;
    }
    createSharedTextureGPU();
    createStagingTextureCPU();
}

DuplicationAPIScreenCapture::~DuplicationAPIScreenCapture()
{
    if (m_ddaManager) m_ddaManager->Release();
    SafeRelease(&stagingTextureCPU);
    SafeRelease(&sharedTexture);
    if (cudaResource) {
        cudaGraphicsUnregisterResource(cudaResource);
        cudaResource = nullptr;
    }
    if (cudaStream) {
        cudaStreamDestroy(cudaStream);
        cudaStream = nullptr;
    }
}

cv::cuda::GpuMat DuplicationAPIScreenCapture::GetNextFrameGpu()
{
    FrameContext ctx;
    HRESULT hr = m_ddaManager->AcquireFrame(ctx, 100);
    if (FAILED(hr)) return cv::cuda::GpuMat();

    // Copy the centre-crop region into the shared D3D11 texture
    D3D11_BOX box{
        static_cast<UINT>((screenWidth  - regionWidth)  / 2),
        static_cast<UINT>((screenHeight - regionHeight) / 2),
        0u,
        static_cast<UINT>((screenWidth  + regionWidth)  / 2),
        static_cast<UINT>((screenHeight + regionHeight) / 2),
        1u
    };
    d3dContext->CopySubresourceRegion(sharedTexture, 0, 0, 0, 0, ctx.texture, 0, &box);
    m_ddaManager->ReleaseFrame();
    ctx.texture->Release();

    // Zero-copy: map the D3D11 texture as a CUDA array and memcpy on GPU
    cv::cuda::GpuMat gpu(regionHeight, regionWidth, CV_8UC4);
    cudaGraphicsMapResources(1, &cudaResource, cudaStream);
    cudaArray_t arr;
    cudaGraphicsSubResourceGetMappedArray(&arr, cudaResource, 0, 0);
    cudaMemcpy2DFromArrayAsync(
        gpu.data, gpu.step,
        arr, 0, 0,
        static_cast<size_t>(regionWidth) * 4,
        static_cast<size_t>(regionHeight),
        cudaMemcpyDeviceToDevice,
        cudaStream
    );
    cudaGraphicsUnmapResources(1, &cudaResource, cudaStream);
    cudaStreamSynchronize(cudaStream);

    return gpu;
}

cv::Mat DuplicationAPIScreenCapture::GetNextFrameCpu()
{
    FrameContext ctx;
    HRESULT hr = m_ddaManager->AcquireFrame(ctx, 100);
    if (FAILED(hr)) return cv::Mat();

    D3D11_BOX box{
        static_cast<UINT>((screenWidth  - regionWidth)  / 2),
        static_cast<UINT>((screenHeight - regionHeight) / 2),
        0u,
        static_cast<UINT>((screenWidth  + regionWidth)  / 2),
        static_cast<UINT>((screenHeight + regionHeight) / 2),
        1u
    };
    d3dContext->CopySubresourceRegion(stagingTextureCPU, 0, 0, 0, 0, ctx.texture, 0, &box);
    m_ddaManager->ReleaseFrame();
    ctx.texture->Release();

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (FAILED(d3dContext->Map(stagingTextureCPU, 0, D3D11_MAP_READ, 0, &mapped)))
        return cv::Mat();

    cv::Mat cpu(regionHeight, regionWidth, CV_8UC4);
    for (int y = 0; y < regionHeight; ++y)
    {
        memcpy(cpu.ptr(y),
               static_cast<BYTE*>(mapped.pData) + y * mapped.RowPitch,
               static_cast<size_t>(regionWidth) * 4);
    }
    d3dContext->Unmap(stagingTextureCPU, 0);

    return cpu;
}

bool DuplicationAPIScreenCapture::createSharedTextureGPU()
{
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width            = static_cast<UINT>(regionWidth);
    desc.Height           = static_cast<UINT>(regionHeight);
    desc.MipLevels        = 1;
    desc.ArraySize        = 1;
    desc.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage            = D3D11_USAGE_DEFAULT;
    desc.BindFlags        = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.MiscFlags        = D3D11_RESOURCE_MISC_SHARED;

    if (FAILED(d3dDevice->CreateTexture2D(&desc, nullptr, &sharedTexture)))
        return false;

    if (cudaGraphicsD3D11RegisterResource(&cudaResource, sharedTexture,
                                          cudaGraphicsRegisterFlagsNone) != cudaSuccess)
        return false;

    cudaStreamCreate(&cudaStream);
    return true;
}

bool DuplicationAPIScreenCapture::createStagingTextureCPU()
{
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width            = static_cast<UINT>(regionWidth);
    desc.Height           = static_cast<UINT>(regionHeight);
    desc.MipLevels        = 1;
    desc.ArraySize        = 1;
    desc.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage            = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;

    return SUCCEEDED(d3dDevice->CreateTexture2D(&desc, nullptr, &stagingTextureCPU));
}
