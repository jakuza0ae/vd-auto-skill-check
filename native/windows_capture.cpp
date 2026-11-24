#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Microsoft::WRL::ComPtr;

constexpr UINT reference_width = 1920;
constexpr UINT reference_height = 1080;
constexpr UINT64 ns_per_second = 1'000'000'000;

struct Options {
    std::optional<UINT> device;
    std::optional<UINT> output;
    UINT fps = 120;
    std::array<UINT, 4> roi{800, 420, 320, 240};
    bool list_outputs = false;
    bool help = false;
};

void check(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        const char* hint = result == DXGI_ERROR_ACCESS_LOST
            ? " (desktop or display mode changed; start capture again)" : "";
        char message[256];
        std::snprintf(message, sizeof(message), "%s failed (0x%08lX)%s",
                      operation, static_cast<unsigned long>(result), hint);
        throw std::runtime_error(message);
    }
}

UINT parse_uint(std::string_view text) {
    UINT value = 0;
    const auto* end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::runtime_error("Expected a nonnegative integer");
    }
    return value;
}

std::array<UINT, 4> parse_roi(std::string_view text) {
    std::array<UINT, 4> roi{};
    for (size_t index = 0; index < roi.size(); ++index) {
        const size_t comma = text.find(',');
        if ((comma == std::string_view::npos) != (index == roi.size() - 1)) {
            throw std::runtime_error("ROI must be X,Y,W,H in 1920x1080 reference pixels");
        }
        roi[index] = parse_uint(text.substr(0, comma));
        if (comma != std::string_view::npos) {
            text.remove_prefix(comma + 1);
        }
    }
    if (roi[0] >= reference_width || roi[1] >= reference_height || !roi[2] || !roi[3] ||
        roi[2] > reference_width - roi[0] || roi[3] > reference_height - roi[1]) {
        throw std::runtime_error("ROI is outside the 1920x1080 reference image");
    }
    return roi;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option = argv[i];
        if (option == "--help") {
            options.help = true;
        } else if (option == "--list-outputs") {
            options.list_outputs = true;
        } else {
            if (++i == argc) {
                throw std::runtime_error("Missing option value; run with --help");
            }
            const std::string_view value = argv[i];
            if (option == "--device-index") {
                options.device = parse_uint(value);
            } else if (option == "--output-index") {
                options.output = parse_uint(value);
            } else if (option == "--fps") {
                options.fps = parse_uint(value);
            } else if (option == "--roi") {
                options.roi = parse_roi(value);
            } else {
                throw std::runtime_error("Unknown option; run with --help");
            }
        }
    }
    if (options.fps < 1 || options.fps > 240) {
        throw std::runtime_error("FPS must be in [1, 240]");
    }
    return options;
}

struct Output {
    ComPtr<IDXGIOutput> output;
    DXGI_OUTPUT_DESC desc{};
    bool primary = false;
};

struct Device {
    ComPtr<IDXGIAdapter1> adapter;
    std::vector<Output> outputs;
};

std::vector<Device> enumerate_devices(IDXGIFactory1* factory) {
    std::vector<Device> devices;
    for (UINT device_index = 0;; ++device_index) {
        Device device;
        const HRESULT result = factory->EnumAdapters1(device_index, device.adapter.GetAddressOf());
        if (result == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        check(result, "EnumAdapters1");
        for (UINT output_index = 0;; ++output_index) {
            Output output;
            const HRESULT status = device.adapter->EnumOutputs(output_index, output.output.GetAddressOf());
            if (status == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            check(status, "EnumOutputs");
            check(output.output->GetDesc(&output.desc), "Get output description");
            if (!output.desc.AttachedToDesktop || !output.desc.Monitor) {
                continue;
            }
            MONITORINFO info{};
            info.cbSize = sizeof(info);
            if (!GetMonitorInfoW(output.desc.Monitor, &info)) {
                throw std::runtime_error("Cannot read the monitor layout; start capture again");
            }
            output.primary = (info.dwFlags & MONITORINFOF_PRIMARY) != 0;
            device.outputs.push_back(std::move(output));
        }
        if (!device.outputs.empty()) {
            devices.push_back(std::move(device));
        }
    }
    if (devices.empty()) {
        throw std::runtime_error("DXGI could not find a monitor attached to an interactive desktop");
    }
    return devices;
}

size_t primary_device(const std::vector<Device>& devices) {
    for (size_t index = 0; index < devices.size(); ++index) {
        for (const auto& output : devices[index].outputs) {
            if (output.primary) {
                return index;
            }
        }
    }
    return 0;
}

size_t primary_output(const Device& device) {
    for (size_t index = 0; index < device.outputs.size(); ++index) {
        if (device.outputs[index].primary) {
            return index;
        }
    }
    return 0;
}

void list_outputs(const std::vector<Device>& devices) {
    for (size_t device_index = 0; device_index < devices.size(); ++device_index) {
        const auto& outputs = devices[device_index].outputs;
        for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
            const auto& output = outputs[output_index];
            const RECT rect = output.desc.DesktopCoordinates;
            std::wprintf(L"--device-index %zu --output-index %zu  %ls  %ldx%ld at (%ld,%ld)%ls\n",
                         device_index, output_index, output.desc.DeviceName,
                         rect.right - rect.left, rect.bottom - rect.top,
                         rect.left, rect.top, output.primary ? L"  primary" : L"");
        }
    }
}

struct FrameHeader {
    char magic[4] = {'V', 'D', 'D', '1'};
    UINT32 version = 1;
    UINT64 sequence = 0;
    UINT64 present_ns = 0;
    UINT64 copied_ns = 0;
    UINT64 monitor = 0;
    UINT32 source_width = 0;
    UINT32 source_height = 0;
    UINT32 roi_width = 0;
    UINT32 roi_height = 0;
    INT32 roi_x = 0;
    INT32 roi_y = 0;
    INT32 source_x = 0;
    INT32 source_y = 0;
};

static_assert(sizeof(FrameHeader) == 72, "Windows capture wire header changed");

UINT scale_coordinate(UINT value, UINT size, UINT reference_size) {
    return static_cast<UINT>((static_cast<UINT64>(value) * size + reference_size / 2) / reference_size);
}

FrameHeader frame_header(const Output& output, const Options& options) {
    const RECT rect = output.desc.DesktopCoordinates;
    if (rect.right <= rect.left || rect.bottom <= rect.top) {
        throw std::runtime_error("DXGI returned an empty monitor rectangle");
    }
    FrameHeader header;
    header.monitor = reinterpret_cast<uintptr_t>(output.desc.Monitor);
    header.source_width = rect.right - rect.left;
    header.source_height = rect.bottom - rect.top;
    header.source_x = rect.left;
    header.source_y = rect.top;
    header.roi_x = scale_coordinate(options.roi[0], header.source_width, reference_width);
    header.roi_y = scale_coordinate(options.roi[1], header.source_height, reference_height);
    header.roi_width = std::min(scale_coordinate(options.roi[2], header.source_width, reference_width),
                               header.source_width - header.roi_x);
    header.roi_height = std::min(scale_coordinate(options.roi[3], header.source_height, reference_height),
                                header.source_height - header.roi_y);
    if (!header.roi_width || !header.roi_height) {
        throw std::runtime_error("The scaled capture region is empty");
    }
    return header;
}

LONGLONG qpc_now() {
    LARGE_INTEGER count{};
    if (!QueryPerformanceCounter(&count)) {
        throw std::runtime_error("QueryPerformanceCounter failed");
    }
    return count.QuadPart;
}

UINT64 qpc_nanoseconds(LONGLONG ticks, LONGLONG frequency) {
    return static_cast<UINT64>(ticks / frequency) * ns_per_second +
           static_cast<UINT64>((ticks % frequency) * (1'000'000'000.0 / frequency));
}

class Pacer {
public:
    Pacer(UINT fps, LONGLONG frequency) : ticks_per_frame_(frequency / fps), frequency_(frequency) {
        timer_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                        TIMER_MODIFY_STATE | SYNCHRONIZE);
        if (!timer_) {
            timer_ = CreateWaitableTimerW(nullptr, FALSE, nullptr);
        }
        if (!timer_) {
            throw std::runtime_error("Cannot create the DXGI capture pacing timer");
        }
    }

    ~Pacer() { CloseHandle(timer_); }
    Pacer(const Pacer&) = delete;
    Pacer& operator=(const Pacer&) = delete;

    void published_frame(LONGLONG acquired) { next_frame_ = acquired + ticks_per_frame_; }

    void wait() const {
        const auto remaining = next_frame_ - qpc_now();
        if (remaining > 0) {
            LARGE_INTEGER delay{};
            delay.QuadPart = -((remaining * 10'000'000 + frequency_ - 1) / frequency_);
            if (!SetWaitableTimer(timer_, &delay, 0, nullptr, nullptr, FALSE) ||
                WaitForSingleObject(timer_, INFINITE) != WAIT_OBJECT_0) {
                throw std::runtime_error("DXGI capture pacing timer failed");
            }
        }
    }

private:
    HANDLE timer_ = nullptr;
    LONGLONG ticks_per_frame_;
    LONGLONG frequency_;
    LONGLONG next_frame_ = 0;
};

class AcquiredFrame {
public:
    explicit AcquiredFrame(IDXGIOutputDuplication* duplication) : duplication_(duplication) {}
    ~AcquiredFrame() { if (held_) duplication_->ReleaseFrame(); }
    AcquiredFrame(const AcquiredFrame&) = delete;
    AcquiredFrame& operator=(const AcquiredFrame&) = delete;

    HRESULT acquire_next(DXGI_OUTDUPL_FRAME_INFO* frame, IDXGIResource** resource) {
        // Retain ownership until the next acquire so no intermediate desktop copies are needed.
        if (held_) {
            held_ = false;
            check(duplication_->ReleaseFrame(), "Release desktop frame");
        }
        const HRESULT status = duplication_->AcquireNextFrame(50, frame, resource);
        held_ = SUCCEEDED(status);
        return status;
    }

private:
    IDXGIOutputDuplication* duplication_;
    bool held_ = false;
};

bool write_pipe(HANDLE pipe, const void* data, DWORD size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    while (size > 0) {
        DWORD written = 0;
        if (!WriteFile(pipe, bytes, size, &written, nullptr)) {
            const auto error = GetLastError();
            if (error == ERROR_BROKEN_PIPE || error == ERROR_NO_DATA) {
                return false;
            }
            check(HRESULT_FROM_WIN32(error), "Write capture pipe");
        }
        if (!written) {
            throw std::runtime_error("Capture pipe write made no progress");
        }
        bytes += written;
        size -= written;
    }
    return true;
}

bool parent_closed(HANDLE control) {
    if (PeekNamedPipe(control, nullptr, 0, nullptr, nullptr, nullptr)) {
        return false;
    }
    const auto error = GetLastError();
    if (error == ERROR_BROKEN_PIPE || error == ERROR_NO_DATA) {
        return true;
    }
    check(HRESULT_FROM_WIN32(error), "Read capture control pipe");
    return false;
}

void capture(const Options& options, const Device& adapter, const Output& output) {
    FrameHeader header = frame_header(output, options);
    const HANDLE pipe = GetStdHandle(STD_OUTPUT_HANDLE);
    const HANDLE control = GetStdHandle(STD_INPUT_HANDLE);
    const bool has_control_pipe = GetFileType(control) == FILE_TYPE_PIPE;
    if (GetFileType(pipe) != FILE_TYPE_PIPE) {
        throw std::runtime_error("Capture needs a stdout pipe opened by the Python solver; use --list-outputs to list monitors");
    }
    if (output.desc.Rotation > DXGI_MODE_ROTATION_IDENTITY) {
        throw std::runtime_error("Rotated monitors are unsupported by this capture worker");
    }
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    check(D3D11CreateDevice(adapter.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                            D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
                            device.GetAddressOf(), nullptr, context.GetAddressOf()), "Create D3D11 device");
    ComPtr<IDXGIOutput1> capture_output;
    check(output.output.As(&capture_output), "Get IDXGIOutput1");
    ComPtr<IDXGIOutputDuplication> duplication;
    check(capture_output->DuplicateOutput(device.Get(), duplication.GetAddressOf()), "Start DXGI Desktop Duplication");

    D3D11_TEXTURE2D_DESC surface{};
    surface.Width = header.roi_width;
    surface.Height = header.roi_height;
    surface.MipLevels = 1;
    surface.ArraySize = 1;
    surface.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    surface.SampleDesc.Count = 1;
    surface.Usage = D3D11_USAGE_STAGING;
    surface.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    check(device->CreateTexture2D(&surface, nullptr, staging.GetAddressOf()), "Create ROI staging texture");
    const D3D11_BOX region{static_cast<UINT>(header.roi_x), static_cast<UINT>(header.roi_y), 0,
                           header.roi_x + header.roi_width, header.roi_y + header.roi_height, 1};
    const size_t row_bytes = static_cast<size_t>(header.roi_width) * 4;
    std::vector<unsigned char> pixels(row_bytes * header.roi_height);

    LARGE_INTEGER frequency{};
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) {
        throw std::runtime_error("QueryPerformanceFrequency failed");
    }
    Pacer pace(options.fps, frequency.QuadPart);
    AcquiredFrame acquired_frame(duplication.Get());
    LONGLONG previous_present = 0;
    while (!has_control_pipe || !parent_closed(control)) {
        pace.wait();
        DXGI_OUTDUPL_FRAME_INFO frame{};
        ComPtr<IDXGIResource> resource;
        const HRESULT result = acquired_frame.acquire_next(&frame, resource.GetAddressOf());
        if (result == DXGI_ERROR_WAIT_TIMEOUT) {
            continue;
        }
        check(result, "Acquire desktop frame");
        const LONGLONG present = frame.LastPresentTime.QuadPart;
        // A mouse-only update carries no new desktop image. Never re-age the old pixels.
        if (present <= previous_present) {
            continue;
        }
        const auto acquired = qpc_now();
        if (frame.ProtectedContentMaskedOut) {
            throw std::runtime_error("DXGI masked protected content on this monitor");
        }
        ComPtr<ID3D11Texture2D> texture;
        check(resource.As(&texture), "Get desktop texture");
        D3D11_TEXTURE2D_DESC source{};
        texture->GetDesc(&source);
        if (source.Width != header.source_width || source.Height != header.source_height ||
            source.Format != surface.Format) {
            throw std::runtime_error("DXGI desktop geometry or pixel format changed; start capture again");
        }
        context->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, texture.Get(), 0, &region);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        check(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map ROI staging texture");
        const auto* data = static_cast<const unsigned char*>(mapped.pData);
        for (UINT y = 0; y < header.roi_height; ++y) {
            std::memcpy(pixels.data() + y * row_bytes, data + y * static_cast<size_t>(mapped.RowPitch), row_bytes);
        }
        context->Unmap(staging.Get(), 0);
        const auto copied = qpc_now();

        header.present_ns = qpc_nanoseconds(present, frequency.QuadPart);
        header.copied_ns = qpc_nanoseconds(copied, frequency.QuadPart);
        if (!write_pipe(pipe, &header, sizeof(header)) ||
            !write_pipe(pipe, pixels.data(), static_cast<DWORD>(pixels.size()))) {
            return;
        }
        ++header.sequence;
        previous_present = present;
        pace.published_frame(acquired);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.help) {
            std::puts("vd-windows-capture [--list-outputs] [--device-index N] [--output-index N] "
                      "[--fps N] [--roi X,Y,W,H]");
            return 0;
        }
        if (!SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
            check(HRESULT_FROM_WIN32(GetLastError()), "Enable per-monitor DPI awareness");
        }
        ComPtr<IDXGIFactory1> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())), "Create DXGI factory");
        const auto devices = enumerate_devices(factory.Get());
        if (options.list_outputs) {
            list_outputs(devices);
            return 0;
        }
        const size_t device_index = options.device ? *options.device : primary_device(devices);
        if (device_index >= devices.size()) {
            throw std::runtime_error("Unknown device index; run with --list-outputs");
        }
        const auto& device = devices[device_index];
        const size_t output_index = options.output ? *options.output : primary_output(device);
        if (output_index >= device.outputs.size()) {
            throw std::runtime_error("Unknown output index; run with --list-outputs");
        }
        capture(options, device, device.outputs[output_index]);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
