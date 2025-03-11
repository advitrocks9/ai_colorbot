// DXGI Desktop Duplication capture backend with D3D11/CUDA interop.
#ifndef DUPLICATION_API_CAPTURE_H
#define DUPLICATION_API_CAPTURE_H

#include <d3d11.h>
#include <dxgi1_2.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include "capture/capture.h"

class DDAManager;

class DuplicationAPIScreenCapture : public IScreenCapture
{
public:
    DuplicationAPIScreenCapture(int monitorIndex, int desiredWidth, int desiredHeight);
    ~DuplicationAPIScreenCapture();

    cv::cuda::GpuMat GetNextFrameGpu() override;
    cv::Mat          GetNextFrameCpu() override;

private:
    std::unique_ptr<DDAManager> m_ddaManager;

    ID3D11Device*           d3dDevice         = nullptr;
    ID3D11DeviceContext*    d3dContext         = nullptr;
    IDXGIOutputDuplication* m_duplication      = nullptr;
    IDXGIOutput1*           m_output1          = nullptr;

    // GPU-side shared texture for zero-copy CUDA interop
    ID3D11Texture2D*        sharedTexture      = nullptr;
    cudaGraphicsResource*   cudaResource       = nullptr;
    cudaStream_t            cudaStream         = nullptr;

    // CPU-readable staging texture for the non-CUDA path
    ID3D11Texture2D*        stagingTextureCPU  = nullptr;

    int screenWidth  = 0;
    int screenHeight = 0;
    int regionWidth  = 0;
    int regionHeight = 0;

    bool createSharedTextureGPU();
    bool createStagingTextureCPU();
};

#endif // DUPLICATION_API_CAPTURE_H
