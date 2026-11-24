#include "vd_capture_layer.h"
#include "bright_spv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <thread>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <iterator>
#include <mutex>
#include <pwd.h>
#include <signal.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kDefaultCaptureIntervalNs = 200000000ULL; // 5 FPS
constexpr size_t kCaptureSlots = 3;

struct InstanceData {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr next_gipa = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkEnumeratePhysicalDeviceGroups vkEnumeratePhysicalDeviceGroups = nullptr;
    PFN_vkEnumeratePhysicalDeviceGroupsKHR vkEnumeratePhysicalDeviceGroupsKHR = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
};

struct CaptureResources {
    bool initialized = false;
    bool permanently_disabled = false;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    bool memory_coherent = false;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    uint64_t generation = 0;
    std::shared_ptr<std::atomic<bool>> lifetime;
    uint64_t source_present_id = 0;
    uint64_t source_ns = 0;
    uint32_t width = 0, height = 0, x = 0, y = 0, source_width = 0, source_height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

struct DeviceData {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    InstanceData *instance_data = nullptr;
    PFN_vkGetDeviceProcAddr next_gdpa = nullptr;

    PFN_vkDestroyDevice vkDestroyDevice = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
    PFN_vkGetDeviceQueue2 vkGetDeviceQueue2 = nullptr;
    PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR = nullptr;
    PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR = nullptr;
    PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR = nullptr;
    PFN_vkQueuePresentKHR vkQueuePresentKHR = nullptr;

    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
    PFN_vkResetCommandPool vkResetCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
    PFN_vkCreateBuffer vkCreateBuffer = nullptr;
    PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory = nullptr;
    PFN_vkFreeMemory vkFreeMemory = nullptr;
    PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
    PFN_vkMapMemory vkMapMemory = nullptr;
    PFN_vkUnmapMemory vkUnmapMemory = nullptr;
    PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges = nullptr;
    PFN_vkCreateFence vkCreateFence = nullptr;
    PFN_vkDestroyFence vkDestroyFence = nullptr;
    PFN_vkResetFences vkResetFences = nullptr;
    PFN_vkWaitForFences vkWaitForFences = nullptr;
    PFN_vkGetFenceStatus vkGetFenceStatus = nullptr;
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
    std::vector<VkSemaphore> retired_present_semaphores;
    PFN_vkCreateSemaphore vkCreateSemaphore = nullptr;
    PFN_vkDestroySemaphore vkDestroySemaphore = nullptr;
    PFN_vkQueueSubmit vkQueueSubmit = nullptr;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer = nullptr;

    // Brightness post-process (only resolved/used when VD_BRIGHT_ENABLE=1).
    PFN_vkCreateImage vkCreateImage = nullptr;
    PFN_vkDestroyImage vkDestroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements = nullptr;
    PFN_vkBindImageMemory vkBindImageMemory = nullptr;
    PFN_vkCreateImageView vkCreateImageView = nullptr;
    PFN_vkDestroyImageView vkDestroyImageView = nullptr;
    PFN_vkCreateSampler vkCreateSampler = nullptr;
    PFN_vkDestroySampler vkDestroySampler = nullptr;
    PFN_vkCreateRenderPass vkCreateRenderPass = nullptr;
    PFN_vkDestroyRenderPass vkDestroyRenderPass = nullptr;
    PFN_vkCreateFramebuffer vkCreateFramebuffer = nullptr;
    PFN_vkDestroyFramebuffer vkDestroyFramebuffer = nullptr;
    PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
    PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelines = nullptr;
    PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
    PFN_vkCmdCopyImage vkCmdCopyImage = nullptr;
    PFN_vkCmdBeginRenderPass vkCmdBeginRenderPass = nullptr;
    PFN_vkCmdEndRenderPass vkCmdEndRenderPass = nullptr;
    PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
    PFN_vkCmdSetViewport vkCmdSetViewport = nullptr;
    PFN_vkCmdSetScissor vkCmdSetScissor = nullptr;
    PFN_vkCmdDraw vkCmdDraw = nullptr;
};

struct QueueData {
    DeviceData *device_data = nullptr;
    uint32_t family_index = 0;
    std::mutex capture_mutex;
    std::array<CaptureResources, kCaptureSlots> captures;
    std::atomic<bool> stopping{false};
    std::thread worker;
    ~QueueData() { stopping.store(true); if (worker.joinable()) worker.join(); }
};

struct PresentTiming {
    uint64_t last_ns=0, first_ns=0, samples=0, sum_ns=0, max_ns=0;
    std::array<uint64_t,256> histogram{};
    void observe(uint64_t now_ns) {
        if (!first_ns) first_ns=now_ns;
        if (last_ns && now_ns>last_ns) {
            const uint64_t interval=now_ns-last_ns;
            ++samples;
            sum_ns+=interval;
            max_ns=std::max(max_ns,interval);
            ++histogram[std::min<uint64_t>(255,interval/1000000ULL)];
        }
        last_ns=now_ns;
    }
};

struct BrightFrame {
    VkImageView view = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore done = VK_NULL_HANDLE;
    bool submitted = false;
};

struct BrightResources {
    bool ready = false;
    bool failed = false;
    uint32_t family = 0;
    VkImage copy = VK_NULL_HANDLE;
    VkDeviceMemory copy_memory = VK_NULL_HANDLE;
    VkImageView copy_view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    std::vector<BrightFrame> frames;
};

struct SwapchainInfo {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    uint32_t image_count = 0;
    std::vector<VkImage> images;
    bool capture_supported = false;
    uint64_t last_capture_ns = 0;
    uint64_t generation = 0;
    std::shared_ptr<std::atomic<bool>> lifetime = std::make_shared<std::atomic<bool>>(true);
    std::vector<VkSemaphore> present_semaphores;
    PresentTiming timing;
    bool bright_supported = false;
    std::shared_ptr<BrightResources> bright;
};

std::mutex g_state_mutex;
std::mutex g_log_mutex;
std::mutex g_init_mutex;
std::mutex g_fps_mutex;
std::mutex g_shm_mutex;
std::mutex g_status_mutex;
std::unordered_map<void *, std::unique_ptr<InstanceData>> g_instances;
std::unordered_map<void *, InstanceData *> g_physical_device_to_instance;
std::unordered_map<void *, std::unique_ptr<DeviceData>> g_devices;
std::unordered_map<void *, std::unique_ptr<QueueData>> g_queues;
std::unordered_map<VkSwapchainKHR, SwapchainInfo> g_swapchains;
DeviceData *g_primary_device = nullptr;
InstanceData *g_primary_instance = nullptr;

std::string g_base_dir;
std::string g_log_path;
std::string g_status_path;
std::string g_shm_path;
int g_shm_fd = -1;
void *g_shm_mapping = nullptr;
size_t g_shm_mapping_size = 0;
VdLayerShmHeader *g_shm_header = nullptr;
uint8_t *g_shm_pixels = nullptr;
FILE *g_log_file = nullptr;

std::atomic<uint64_t> g_frame_counter{0};
std::atomic<uint64_t> g_capture_counter{0};
std::atomic<uint64_t> g_last_present_ns{0};
std::atomic<uint64_t> g_generation{0};
std::atomic<uint32_t> g_dropped{0};
void *g_control_mapping = nullptr;
uint64_t g_published_source_id = 0;
VkSwapchainKHR g_latest_present_swapchain = VK_NULL_HANDLE;
uint64_t g_last_fps_time_ns = 0;
uint64_t g_last_fps_frame_count = 0;
float g_current_fps = 0.0f;
bool g_is_disabled = false;
bool g_initialized = false;
bool g_verbose_log = false;
bool g_capture_enabled = false;
uint64_t g_capture_interval_ns = kDefaultCaptureIntervalNs;

// Brightness: armed by VD_BRIGHT_ENABLE=1 at launch; toggled live by the watcher thread.
struct BrightSettings {
    enum Mode { kAuto, kOn, kOff } mode = kAuto;
    float strength = 2.2f;
    float lift = 0.04f;
    float detail = 0.6f;
    std::vector<uint64_t> universes{6739698191ULL};  // Violence District
};
bool g_bright_armed = false;
std::atomic<bool> g_bright_active{false};
std::atomic<float> g_bright_strength{2.2f}, g_bright_lift{0.04f}, g_bright_detail{0.6f};
std::atomic<bool> g_bright_stop{false};
std::thread g_bright_thread;
time_t g_init_wall_time = 0;

uint64_t get_time_ns() {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

bool env_true(const char *name) {
    const char *value = getenv(name);
    if (!value) return false;
    return strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "yes") == 0 || strcmp(value, "on") == 0;
}

void log_line(bool debug_only, const char *format, va_list args) {
    if (g_is_disabled || (debug_only && !g_verbose_log)) return;
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    const double sec = static_cast<double>(get_time_ns()) / 1e9;

    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_file) {
        fprintf(g_log_file, "[VD_LAYER %.3f] %s\n", sec, buffer);
        fflush(g_log_file);
    }
    if (g_verbose_log) {
        fprintf(stderr, "[VD_LAYER %.3f] %s\n", sec, buffer);
        fflush(stderr);
    }
}

void log_msg(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_line(false, format, args);
    va_end(args);
}

void log_debug(const char *format, ...) {
    va_list args;
    va_start(args, format);
    log_line(true, format, args);
    va_end(args);
}

std::string get_process_name() {
    char comm[64] = {0};
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) {
        if (fgets(comm, sizeof(comm), f)) comm[strcspn(comm, "\r\n")] = '\0';
        fclose(f);
    }
    if (comm[0]) return std::string(comm);

    char path[256] = {0};
    const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len > 0) {
        path[len] = '\0';
        const char *slash = strrchr(path, '/');
        return slash ? std::string(slash + 1) : std::string(path);
    }
    return "unknown";
}

void shm_begin_write() {
    if (g_shm_header) __atomic_add_fetch(&g_shm_header->sequence, 1u, __ATOMIC_SEQ_CST);
}

void shm_end_write() {
    if (g_shm_header) __atomic_add_fetch(&g_shm_header->sequence, 1u, __ATOMIC_SEQ_CST);
}

void increment_dropped_frames() {
    g_dropped.fetch_add(1);
}

void stop_capture_workers() {
    std::vector<QueueData *> queues;
    { std::lock_guard<std::mutex> lock(g_state_mutex);
      for (auto &entry : g_queues) {
          entry.second->stopping.store(true);
          queues.push_back(entry.second.get());
      } }
    for (auto *q : queues) if (q->worker.joinable()) q->worker.join();
    g_bright_stop.store(true);
    if (g_bright_thread.joinable()) g_bright_thread.join();
}

void bright_watcher();

std::string layer_directory() {
    if (const char *custom_dir = getenv("VD_LAYER_DIR"); custom_dir && *custom_dir) {
        return custom_dir;
    }
    if (const char *xdg_data = getenv("XDG_DATA_HOME"); xdg_data && *xdg_data) {
        return std::string(xdg_data) + "/vulkan";
    }
    constexpr const char *sober_dir = "/.var/app/org.vinegarhq.Sober/data/vulkan";
    if (const char *home = getenv("HOME"); home && *home) {
        return std::string(home) + sober_dir;
    }

    const long size_hint = sysconf(_SC_GETPW_R_SIZE_MAX);
    std::vector<char> buffer(size_hint > 0 ? static_cast<size_t>(size_hint) : 1024);
    passwd entry{}, *result = nullptr;
    int status;
    while ((status = getpwuid_r(getuid(), &entry, buffer.data(), buffer.size(), &result)) == ERANGE) {
        buffer.resize(buffer.size() * 2);
    }
    if (status == 0 && result && entry.pw_dir && *entry.pw_dir) {
        return std::string(entry.pw_dir) + sober_dir;
    }
    return {};
}

void init_layer_runtime() {
    std::lock_guard<std::mutex> init_lock(g_init_mutex);
    if (g_initialized) return;
    g_initialized = true;
    std::atexit(stop_capture_workers);

    if (env_true("DISABLE_VD_LAYER")) {
        g_is_disabled = true;
        return;
    }

    g_verbose_log = env_true("VD_LAYER_DEBUG") || env_true("VD_LAYER_VERBOSE");
    g_capture_enabled = env_true("VD_CAPTURE_ENABLE");
    if (const char *interval_ms = getenv("VD_CAPTURE_INTERVAL_MS")) {
        char *end = nullptr;
        const unsigned long parsed = strtoul(interval_ms, &end, 10);
        if (end && *end == '\0') g_capture_interval_ns = static_cast<uint64_t>(parsed) * 1000000ULL;
    }

    g_base_dir = layer_directory();
    if (g_base_dir.empty()) {
        log_msg("Cannot resolve the home directory. Set VD_LAYER_DIR to enable capture.");
        g_capture_enabled = false;
        g_is_disabled = true;
        return;
    }

    mkdir(g_base_dir.c_str(), 0755);
    g_log_path = g_base_dir + "/vd_layer.log";
    g_status_path = g_base_dir + "/vd_layer_status.json";
    g_shm_path = g_base_dir + "/vd_layer_shm.dat";
    g_log_file = fopen(g_log_path.c_str(), "a");

    const pid_t pid = getpid();
    const std::string proc_name = get_process_name();
    log_msg("=== VD Vulkan Capture Layer initialized for PID %d (%s) ===", static_cast<int>(pid), proc_name.c_str());
    log_msg("Directory: %s (Verbose: %s, Capture: %s, Interval: %.1f ms)",
            g_base_dir.c_str(), g_verbose_log ? "ON" : "OFF", g_capture_enabled ? "ON" : "OFF",
            static_cast<double>(g_capture_interval_ns) / 1000000.0);

    g_shm_fd = open(g_shm_path.c_str(), O_RDWR | O_CREAT, 0666);
    if (g_shm_fd >= 0) {
        g_shm_mapping_size = sizeof(VdLayerShmHeader) + VD_CAPTURE_MAX_BYTES;
        if (ftruncate(g_shm_fd, static_cast<off_t>(g_shm_mapping_size)) == 0) {
            g_shm_mapping = mmap(nullptr, g_shm_mapping_size, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
            if (g_shm_mapping != MAP_FAILED) {
                g_shm_header = static_cast<VdLayerShmHeader *>(g_shm_mapping);
                g_shm_pixels = static_cast<uint8_t *>(g_shm_mapping) + sizeof(VdLayerShmHeader);

                bool stale = true;
                if (g_shm_header->magic == VD_LAYER_SHM_MAGIC && g_shm_header->pid != 0 &&
                    g_shm_header->pid != static_cast<uint32_t>(pid)) {
                    stale = kill(static_cast<pid_t>(g_shm_header->pid), 0) != 0;
                }

                if (stale || g_shm_header->magic != VD_LAYER_SHM_MAGIC ||
                    g_shm_header->version != VD_LAYER_SHM_VERSION ||
                    g_shm_header->struct_size != sizeof(VdLayerShmHeader)) {
                    memset(g_shm_mapping, 0, g_shm_mapping_size);
                }

                shm_begin_write();
                g_shm_header->magic = VD_LAYER_SHM_MAGIC;
                g_shm_header->version = VD_LAYER_SHM_VERSION;
                g_shm_header->struct_size = sizeof(VdLayerShmHeader);
                g_shm_header->status_flags = VD_STATUS_ACTIVE | (g_capture_enabled ? VD_STATUS_CAPTURE_ENABLED : 0u);
                g_shm_header->pid = static_cast<uint32_t>(pid);
                g_shm_header->init_timestamp_ns = get_time_ns();
                g_shm_header->last_present_ns = g_shm_header->init_timestamp_ns;
                g_shm_header->roi_data_offset = sizeof(VdLayerShmHeader);
                strncpy(g_shm_header->process_name, proc_name.c_str(), sizeof(g_shm_header->process_name) - 1);
                g_shm_header->process_name[sizeof(g_shm_header->process_name) - 1] = '\0';
                shm_end_write();
                msync(g_shm_mapping, sizeof(VdLayerShmHeader), MS_ASYNC);
                log_msg("Shared memory mapped at %s (header=%zu, pixels=%u)",
                        g_shm_path.c_str(), sizeof(VdLayerShmHeader), VD_CAPTURE_MAX_BYTES);
            } else {
                g_shm_mapping = nullptr;
                log_msg("mmap failed for %s", g_shm_path.c_str());
            }
        } else {
            log_msg("ftruncate failed for %s", g_shm_path.c_str());
        }
    } else {
        log_msg("open failed for shm file %s", g_shm_path.c_str());
    }

    const std::string control_path = g_base_dir + "/vd_layer_control.dat";
    const int control_fd = open(control_path.c_str(), O_RDWR | O_CREAT, 0600);
    if (control_fd >= 0) {
        struct stat st{};
        if (fstat(control_fd, &st) == 0 && (st.st_size >= 16 || ftruncate(control_fd, 16) == 0)) {
            void *mapping = mmap(nullptr, 16, PROT_READ, MAP_SHARED, control_fd, 0);
            if (mapping != MAP_FAILED) g_control_mapping = mapping;
        }
        close(control_fd);
    }
    g_last_fps_time_ns = get_time_ns();

    g_bright_armed = env_true("VD_BRIGHT_ENABLE");
    if (g_bright_armed) {
        g_init_wall_time = time(nullptr);
        log_msg("Brightness effect armed (config: %s/vd_bright.conf)", g_base_dir.c_str());
        g_bright_thread = std::thread(bright_watcher);
    }
}

bool consumer_active(uint64_t now_ns) {
    if (!g_control_mapping) return false;
    const auto *words = static_cast<const uint32_t *>(g_control_mapping);
    if (__atomic_load_n(words, __ATOMIC_ACQUIRE) != VD_LAYER_SHM_MAGIC ||
        __atomic_load_n(words + 1, __ATOMIC_ACQUIRE) != VD_LAYER_SHM_VERSION) return false;
    const auto *deadline = reinterpret_cast<const uint64_t *>(words + 2);
    const uint64_t expires = __atomic_load_n(deadline, __ATOMIC_ACQUIRE);
    return expires > now_ns && expires - now_ns <= 2000000000ULL;
}

void write_status_json(bool active = true) {
    if (g_is_disabled || g_status_path.empty()) return;
    std::lock_guard<std::mutex> status_lock(g_status_mutex);
    PresentTiming timing;
    uint64_t timing_generation=0;
    {
        std::lock_guard<std::mutex> state_lock(g_state_mutex);
        auto it=g_swapchains.find(g_latest_present_swapchain);
        if (it!=g_swapchains.end()) {
            timing=it->second.timing;
            timing_generation=it->second.generation;
        }
    }
    VdLayerShmHeader snapshot{};
    bool has_header=false;
    {
        std::lock_guard<std::mutex> shm_lock(g_shm_mutex);
        if (g_shm_header) { snapshot=*g_shm_header; has_header=true; }
    }
    const auto *header=has_header ? &snapshot : nullptr;
    const pid_t pid = getpid();
    const std::string tmp_path = g_status_path + "." + std::to_string(pid) + ".tmp";
    FILE *f = fopen(tmp_path.c_str(), "w");
    if (!f) return;

    float fps = 0.0f;
    {
        std::lock_guard<std::mutex> lock(g_fps_mutex);
        fps = g_current_fps;
    }

    const std::string proc_name = get_process_name();
    const uint32_t width = header ? header->swapchain_width : 0;
    const uint32_t height = header ? header->swapchain_height : 0;
    const uint32_t format = header ? header->swapchain_format : 0;
    const uint32_t mode = header ? header->swapchain_present_mode : 0;
    const uint32_t images = header ? header->swapchain_image_count : 0;
    const uint64_t captures = header ? header->roi_capture_count : 0;

    fprintf(f, "{\n");
    fprintf(f, "  \"layer_name\": \"%s\",\n", VD_LAYER_NAME);
    fprintf(f, "  \"version\": %d,\n", VD_LAYER_IMPL_VERSION);
    fprintf(f, "  \"active\": %s,\n", active ? "true" : "false");
    fprintf(f, "  \"capture_enabled\": %s,\n", g_capture_enabled ? "true" : "false");
    fprintf(f, "  \"pid\": %d,\n", static_cast<int>(pid));
    fprintf(f, "  \"process_name\": \"%s\",\n", proc_name.c_str());
    fprintf(f, "  \"frames_presented\": %llu,\n", static_cast<unsigned long long>(g_frame_counter.load()));
    fprintf(f, "  \"captures\": %llu,\n", static_cast<unsigned long long>(captures));
    fprintf(f, "  \"fps\": %.2f,\n", fps);
    fprintf(f, "  \"swapchain\": {\"width\": %u, \"height\": %u, \"format\": %u, \"present_mode\": %u, \"image_count\": %u},\n",
            width, height, format, mode, images);
    fprintf(f, "  \"status_time_ns\": %llu,\n", static_cast<unsigned long long>(get_time_ns()));
    fprintf(f, "  \"init_timestamp_ns\": %llu,\n", static_cast<unsigned long long>(header ? header->init_timestamp_ns : 0));
    fprintf(f, "  \"consumer_active\": %s,\n", consumer_active(get_time_ns()) ? "true" : "false");
    fprintf(f, "  \"present_timing\": {\"generation\": %llu, \"first_ns\": %llu, \"last_ns\": %llu, \"samples\": %llu, \"sum_ns\": %llu, \"max_ns\": %llu, \"bucket_width_ns\": 1000000, \"histogram\": [",
        static_cast<unsigned long long>(timing_generation), static_cast<unsigned long long>(timing.first_ns),
        static_cast<unsigned long long>(timing.last_ns), static_cast<unsigned long long>(timing.samples),
        static_cast<unsigned long long>(timing.sum_ns), static_cast<unsigned long long>(timing.max_ns));
    for (size_t i=0;i<timing.histogram.size();++i)
        fprintf(f, "%s%llu", i ? "," : "", static_cast<unsigned long long>(timing.histogram[i]));
    fprintf(f, "]},\n");
    if (header) {
        fprintf(f, "  \"roi\": {\"x\": %u, \"y\": %u, \"width\": %u, \"height\": %u, \"stride\": %u, \"size\": %u}\n",
                header->roi_x, header->roi_y, header->roi_width,
                header->roi_height, header->roi_stride, header->roi_data_size);
    } else {
        fprintf(f, "  \"roi\": null\n");
    }
    fprintf(f, "}\n");
    fclose(f);
    rename(tmp_path.c_str(), g_status_path.c_str());
}

VkLayerInstanceCreateInfo *get_instance_chain_info(const VkInstanceCreateInfo *create_info, VkLayerFunction func) {
    auto *chain = reinterpret_cast<VkLayerInstanceCreateInfo *>(const_cast<void *>(create_info->pNext));
    while (chain) {
        if (chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && chain->function == func) return chain;
        chain = reinterpret_cast<VkLayerInstanceCreateInfo *>(const_cast<void *>(chain->pNext));
    }
    return nullptr;
}

VkLayerDeviceCreateInfo *get_device_chain_info(const VkDeviceCreateInfo *create_info, VkLayerFunction func) {
    auto *chain = reinterpret_cast<VkLayerDeviceCreateInfo *>(const_cast<void *>(create_info->pNext));
    while (chain) {
        if (chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && chain->function == func) return chain;
        chain = reinterpret_cast<VkLayerDeviceCreateInfo *>(const_cast<void *>(chain->pNext));
    }
    return nullptr;
}

// UNORM only: sampling/writing an sRGB view would apply the curve in linear space.
bool format_supported_for_bright(VkFormat format) {
    return format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_R8G8B8A8_UNORM ||
        format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || format == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
}

bool format_supported_for_capture(VkFormat format) {
    return format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB ||
           format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB;
}

uint32_t find_host_memory_type(DeviceData *ddata, uint32_t type_bits, bool *coherent) {
    if (!ddata || !ddata->instance_data || !ddata->instance_data->vkGetPhysicalDeviceMemoryProperties) return UINT32_MAX;
    VkPhysicalDeviceMemoryProperties props{};
    ddata->instance_data->vkGetPhysicalDeviceMemoryProperties(ddata->physical_device, &props);

    uint32_t visible_fallback = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        const VkMemoryPropertyFlags flags = props.memoryTypes[i].propertyFlags;
        if (!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
            if (coherent) *coherent = true;
            return i;
        }
        if (visible_fallback == UINT32_MAX) visible_fallback = i;
    }
    if (visible_fallback != UINT32_MAX && coherent) *coherent = false;
    return visible_fallback;
}

void destroy_capture_resources(QueueData *qdata, CaptureResources &c) {
    if (!qdata || !qdata->device_data) return;
    DeviceData *d = qdata->device_data;
    if (c.mapped && d->vkUnmapMemory && c.staging_memory) d->vkUnmapMemory(d->device, c.staging_memory);
    c.mapped = nullptr;
    if (c.fence && d->vkDestroyFence) d->vkDestroyFence(d->device, c.fence, nullptr);
    if (c.staging_buffer && d->vkDestroyBuffer) d->vkDestroyBuffer(d->device, c.staging_buffer, nullptr);
    if (c.staging_memory && d->vkFreeMemory) d->vkFreeMemory(d->device, c.staging_memory, nullptr);
    if (c.command_pool && d->vkDestroyCommandPool) d->vkDestroyCommandPool(d->device, c.command_pool, nullptr);
    c = CaptureResources{};
}

bool init_capture_resources(QueueData *qdata, CaptureResources &c) {
    if (!qdata || !qdata->device_data) return false;
    DeviceData *d = qdata->device_data;
    if (c.initialized) return true;
    if (c.permanently_disabled) return false;

    const bool funcs_ok = d->vkCreateCommandPool && d->vkDestroyCommandPool && d->vkResetCommandPool &&
        d->vkAllocateCommandBuffers && d->vkCreateBuffer && d->vkDestroyBuffer &&
        d->vkGetBufferMemoryRequirements && d->vkAllocateMemory && d->vkFreeMemory &&
        d->vkBindBufferMemory && d->vkMapMemory && d->vkCreateFence && d->vkDestroyFence &&
        d->vkResetFences && d->vkGetFenceStatus && d->vkCreateSemaphore && d->vkDestroySemaphore &&
        d->vkQueueSubmit && d->vkBeginCommandBuffer && d->vkEndCommandBuffer &&
        d->vkCmdPipelineBarrier && d->vkCmdCopyImageToBuffer;
    if (!funcs_ok) {
        log_msg("Capture disabled: required Vulkan device functions are missing");
        c.permanently_disabled = true;
        return false;
    }

    auto fail = [&]() {
        log_msg("Capture resource initialization failed; capture disabled for queue family %u", qdata->family_index);
        destroy_capture_resources(qdata,c);
        c.permanently_disabled = true;
        increment_dropped_frames();
        return false;
    };

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = qdata->family_index;
    if (d->vkCreateCommandPool(d->device, &pool_info, nullptr, &c.command_pool) != VK_SUCCESS) return fail();

    VkCommandBufferAllocateInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_info.commandPool = c.command_pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;
    if (d->vkAllocateCommandBuffers(d->device, &cmd_info, &c.command_buffer) != VK_SUCCESS) return fail();

    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = VD_CAPTURE_MAX_BYTES;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d->vkCreateBuffer(d->device, &buffer_info, nullptr, &c.staging_buffer) != VK_SUCCESS) return fail();

    VkMemoryRequirements requirements{};
    d->vkGetBufferMemoryRequirements(d->device, c.staging_buffer, &requirements);
    c.memory_coherent = false;
    const uint32_t memory_type = find_host_memory_type(d, requirements.memoryTypeBits, &c.memory_coherent);
    if (memory_type == UINT32_MAX) return fail();

    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = requirements.size;
    alloc_info.memoryTypeIndex = memory_type;
    if (d->vkAllocateMemory(d->device, &alloc_info, nullptr, &c.staging_memory) != VK_SUCCESS) return fail();
    if (d->vkBindBufferMemory(d->device, c.staging_buffer, c.staging_memory, 0) != VK_SUCCESS) return fail();
    if (d->vkMapMemory(d->device, c.staging_memory, 0, VK_WHOLE_SIZE, 0, &c.mapped) != VK_SUCCESS) return fail();

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (d->vkCreateFence(d->device, &fence_info, nullptr, &c.fence) != VK_SUCCESS) return fail();

    c.initialized = true;
    log_msg("Capture resources ready on queue family %u (host coherent=%s)", qdata->family_index, c.memory_coherent ? "YES" : "NO");
    return true;
}

void capture_worker(QueueData *qdata);

void register_queue(DeviceData *ddata, VkQueue queue, uint32_t family_index) {
    if (!ddata || !queue) return;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto &slot = g_queues[reinterpret_cast<void *>(queue)];
        if (!slot) slot = std::make_unique<QueueData>();
        slot->device_data = ddata;
        slot->family_index = family_index;
        if (!slot->worker.joinable() && g_capture_enabled) {
            for (auto &c : slot->captures) init_capture_resources(slot.get(),c);
            slot->worker = std::thread(capture_worker,slot.get());
        }
    }
    log_debug("Registered queue %p to device %p (family %u)", reinterpret_cast<void *>(queue), reinterpret_cast<void *>(ddata->device), family_index);
}

void publish_roi(const CaptureResources &c, uint64_t completed_ns) {
    if (!g_shm_header || !g_shm_pixels || !c.mapped) return;
    std::lock_guard<std::mutex> lock(g_shm_mutex);
    if (c.source_present_id <= g_published_source_id || !c.lifetime || !c.lifetime->load()) return;
    const uint32_t size = c.width*c.height*VD_CAPTURE_BYTES_PER_PIXEL;
    shm_begin_write();
    memcpy(g_shm_pixels,c.mapped,size);
    if (!c.lifetime->load()) {
        g_shm_header->status_flags &= ~VD_STATUS_PIXELS_READY;
        shm_end_write();
        return;
    }
    g_shm_header->status_flags |= VD_STATUS_CAPTURE_ENABLED | VD_STATUS_PIXELS_READY;
    g_shm_header->roi_x=c.x;
    g_shm_header->roi_y=c.y;
    g_shm_header->roi_width=c.width;
    g_shm_header->roi_height=c.height;
    g_shm_header->roi_stride=c.width*VD_CAPTURE_BYTES_PER_PIXEL;
    g_shm_header->roi_data_offset=sizeof(VdLayerShmHeader);
    g_shm_header->roi_data_size=size;
    g_shm_header->roi_capture_count=g_capture_counter.fetch_add(1)+1;
    g_shm_header->roi_capture_timestamp_ns=c.source_ns;
    g_shm_header->source_present_id=c.source_present_id;
    g_shm_header->swapchain_generation=c.generation;
    g_shm_header->copy_completed_ns=completed_ns;
    g_shm_header->published_ns=get_time_ns();
    g_shm_header->swapchain_width=c.source_width;
    g_shm_header->swapchain_height=c.source_height;
    g_shm_header->swapchain_format=static_cast<uint32_t>(c.format);
    g_published_source_id=c.source_present_id;
    shm_end_write();
}

void capture_worker(QueueData *qdata) {
    DeviceData *d=qdata->device_data;
    uint64_t last_status=0;
    while (!qdata->stopping.load()) {
        bool pending_work=false;
        {
            std::lock_guard<std::mutex> lock(qdata->capture_mutex);
            for (auto &c : qdata->captures) {
                if (!c.pending) continue;
                VkResult ready=d->vkGetFenceStatus(d->device,c.fence);
                if (ready==VK_NOT_READY) { pending_work=true; continue; }
                if (ready==VK_SUCCESS) {
                    const uint64_t completed=get_time_ns();
                    bool valid=true;
                    if (!c.memory_coherent) {
                        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                        range.memory=c.staging_memory;
                        range.offset=0;
                        range.size=VK_WHOLE_SIZE;
                        valid=d->vkInvalidateMappedMemoryRanges &&
                            d->vkInvalidateMappedMemoryRanges(d->device,1,&range)==VK_SUCCESS;
                    }
                    if (valid) publish_roi(c,completed);
                    else increment_dropped_frames();
                } else {
                    c.permanently_disabled=true;
                    increment_dropped_frames();
                }
                c.pending=false;
            }
        }
        const uint64_t now=get_time_ns();
        if (now-last_status>=1000000000ULL) {
            {
                std::lock_guard<std::mutex> lock(g_shm_mutex);
                if (g_shm_header) {
                    shm_begin_write();
                    g_shm_header->frame_count=g_frame_counter.load();
                    g_shm_header->last_present_ns=g_last_present_ns.load();
                    g_shm_header->dropped_frames=g_dropped.load();
                    { std::lock_guard<std::mutex> fps_lock(g_fps_mutex);
                      g_shm_header->current_fps=g_current_fps; }
                    shm_end_write();
                }
            }
            write_status_json(true);
            last_status=now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(
            pending_work || consumer_active(now) ? 1 : 10));
    }
}

// On success the capture submit consumed the app's wait semaphores; *signal_out must be waited next.
bool capture_before_present(VkQueue queue, QueueData *qdata, const VkPresentInfoKHR *present_info, VkSemaphore *signal_out, uint64_t now_ns, uint64_t source_id) {
    if (!g_capture_enabled || !consumer_active(now_ns) || !qdata || !qdata->device_data || !present_info || !signal_out) return false;
    if (present_info->swapchainCount != 1 || !present_info->pSwapchains || !present_info->pImageIndices) return false;

    const VkSwapchainKHR swapchain = present_info->pSwapchains[0];
    const uint32_t image_index = present_info->pImageIndices[0];
    struct CaptureTarget {
        uint32_t width=0,height=0;
        VkFormat format=VK_FORMAT_UNDEFINED;
        VkImage image=VK_NULL_HANDLE;
        VkSemaphore semaphore=VK_NULL_HANDLE;
        uint64_t generation=0;
        std::shared_ptr<std::atomic<bool>> lifetime;
    } sc;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_swapchains.find(swapchain);
        if (it == g_swapchains.end() || !it->second.capture_supported) return false;
        if (g_capture_interval_ns > 0 && now_ns - it->second.last_capture_ns < g_capture_interval_ns) return false;
        if (image_index >= it->second.images.size() || image_index >= it->second.present_semaphores.size()) return false;
        it->second.last_capture_ns = now_ns;
        sc.width=it->second.width;
        sc.height=it->second.height;
        sc.format=it->second.format;
        sc.image=it->second.images[image_index];
        sc.semaphore=it->second.present_semaphores[image_index];
        sc.generation=it->second.generation;
        sc.lifetime=it->second.lifetime;
    }

    DeviceData *d = qdata->device_data;
    std::unique_lock<std::mutex> capture_lock(qdata->capture_mutex,std::try_to_lock);
    if (!capture_lock.owns_lock()) { increment_dropped_frames(); return false; }
    CaptureResources *available=nullptr;
    for (auto &slot : qdata->captures)
        if (slot.initialized && !slot.permanently_disabled && !slot.pending) { available=&slot; break; }
    if (!available) { increment_dropped_frames(); return false; }
    CaptureResources &c=*available;

    const uint32_t roi_width = std::min<uint32_t>(VD_CAPTURE_MAX_WIDTH, sc.width);
    const uint32_t roi_height = std::min<uint32_t>(VD_CAPTURE_MAX_HEIGHT, sc.height);
    const uint32_t roi_x = (sc.width - roi_width) / 2;
    const uint32_t roi_y = (sc.height - roi_height) / 2;

    if (d->vkResetCommandPool(d->device, c.command_pool, 0) != VK_SUCCESS) {
        increment_dropped_frames();
        return false;
    }

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (d->vkBeginCommandBuffer(c.command_buffer, &begin_info) != VK_SUCCESS) {
        increment_dropped_frames();
        return false;
    }

    VkImageMemoryBarrier to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to_transfer.srcAccessMask = 0;
    to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_transfer.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.image = sc.image;
    to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    to_transfer.subresourceRange.baseMipLevel = 0;
    to_transfer.subresourceRange.levelCount = 1;
    to_transfer.subresourceRange.baseArrayLayer = 0;
    to_transfer.subresourceRange.layerCount = 1;
    d->vkCmdPipelineBarrier(c.command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &to_transfer);

    VkBufferImageCopy copy{};
    copy.bufferOffset = 0;
    copy.bufferRowLength = 0;
    copy.bufferImageHeight = 0;
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.mipLevel = 0;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageOffset = {static_cast<int32_t>(roi_x), static_cast<int32_t>(roi_y), 0};
    copy.imageExtent = {roi_width, roi_height, 1};
    d->vkCmdCopyImageToBuffer(c.command_buffer, sc.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              c.staging_buffer, 1, &copy);

    VkBufferMemoryBarrier to_host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    to_host.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    to_host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    to_host.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    to_host.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    to_host.buffer=c.staging_buffer;
    to_host.offset=0;
    to_host.size=VK_WHOLE_SIZE;
    d->vkCmdPipelineBarrier(c.command_buffer,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,
        0,0,nullptr,1,&to_host,0,nullptr);

    VkImageMemoryBarrier to_present = to_transfer;
    to_present.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_present.dstAccessMask = 0;
    to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    d->vkCmdPipelineBarrier(c.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &to_present);

    if (d->vkEndCommandBuffer(c.command_buffer) != VK_SUCCESS) {
        increment_dropped_frames();
        return false;
    }

    std::vector<VkPipelineStageFlags> wait_stages(present_info->waitSemaphoreCount, VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkSemaphore capture_done = sc.semaphore;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = present_info->waitSemaphoreCount;
    submit.pWaitSemaphores = present_info->pWaitSemaphores;
    submit.pWaitDstStageMask = wait_stages.empty() ? nullptr : wait_stages.data();
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &c.command_buffer;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &capture_done;

    if (d->vkResetFences(d->device, 1, &c.fence) != VK_SUCCESS) {
        increment_dropped_frames();
        return false;
    }
    const VkResult submit_result = d->vkQueueSubmit(queue, 1, &submit, c.fence);
    if (submit_result != VK_SUCCESS) {
        increment_dropped_frames();
        destroy_capture_resources(qdata,c);
        c.permanently_disabled = true;
        return false;
    }

    c.pending=true;
    c.swapchain=swapchain;
    c.generation=sc.generation;
    c.lifetime=sc.lifetime;
    c.source_present_id=source_id;
    c.source_ns=now_ns;
    c.width=roi_width;
    c.height=roi_height;
    c.x=roi_x;
    c.y=roi_y;
    c.source_width=sc.width;
    c.source_height=sc.height;
    c.format=sc.format;
    *signal_out = capture_done;
    return true;
}

// ---------------------------------------------------------------------------
// Brightness post-process. Fail-open: any error disables it and frames pass untouched.

struct BrightPush { float texel[2]; float strength, lift, detail; };

uint32_t find_device_memory_type(DeviceData *d, uint32_t type_bits) {
    if (!d || !d->instance_data || !d->instance_data->vkGetPhysicalDeviceMemoryProperties) return UINT32_MAX;
    VkPhysicalDeviceMemoryProperties props{};
    d->instance_data->vkGetPhysicalDeviceMemoryProperties(d->physical_device, &props);
    uint32_t any = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        if (props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) return i;
        if (any == UINT32_MAX) any = i;
    }
    return any;
}

bool bright_functions_ok(const DeviceData *d) {
    return d->vkCreateImage && d->vkDestroyImage && d->vkGetImageMemoryRequirements && d->vkBindImageMemory &&
        d->vkCreateImageView && d->vkDestroyImageView && d->vkCreateSampler && d->vkDestroySampler &&
        d->vkCreateRenderPass && d->vkDestroyRenderPass && d->vkCreateFramebuffer && d->vkDestroyFramebuffer &&
        d->vkCreateShaderModule && d->vkDestroyShaderModule && d->vkCreateDescriptorSetLayout &&
        d->vkDestroyDescriptorSetLayout && d->vkCreatePipelineLayout && d->vkDestroyPipelineLayout &&
        d->vkCreateGraphicsPipelines && d->vkDestroyPipeline && d->vkCreateDescriptorPool &&
        d->vkDestroyDescriptorPool && d->vkAllocateDescriptorSets && d->vkUpdateDescriptorSets &&
        d->vkCmdCopyImage && d->vkCmdBeginRenderPass && d->vkCmdEndRenderPass && d->vkCmdBindPipeline &&
        d->vkCmdBindDescriptorSets && d->vkCmdPushConstants && d->vkCmdSetViewport && d->vkCmdSetScissor &&
        d->vkCmdDraw && d->vkCreateCommandPool && d->vkDestroyCommandPool && d->vkAllocateCommandBuffers &&
        d->vkAllocateMemory && d->vkFreeMemory && d->vkCreateFence && d->vkDestroyFence && d->vkResetFences &&
        d->vkWaitForFences && d->vkCreateSemaphore && d->vkDestroySemaphore && d->vkQueueSubmit &&
        d->vkBeginCommandBuffer && d->vkEndCommandBuffer && d->vkCmdPipelineBarrier;
}

// Caller guarantees the GPU no longer uses these resources.
void bright_destroy(DeviceData *d, BrightResources &b) {
    if (!d) return;
    const VkDevice dev = d->device;
    for (auto &f : b.frames) {
        if (f.framebuffer) d->vkDestroyFramebuffer(dev, f.framebuffer, nullptr);
        if (f.view) d->vkDestroyImageView(dev, f.view, nullptr);
        if (f.fence) d->vkDestroyFence(dev, f.fence, nullptr);
        if (f.done) d->vkDestroySemaphore(dev, f.done, nullptr);
    }
    if (b.pool) d->vkDestroyCommandPool(dev, b.pool, nullptr);
    if (b.pipeline) d->vkDestroyPipeline(dev, b.pipeline, nullptr);
    if (b.pipeline_layout) d->vkDestroyPipelineLayout(dev, b.pipeline_layout, nullptr);
    if (b.descriptor_pool) d->vkDestroyDescriptorPool(dev, b.descriptor_pool, nullptr);
    if (b.set_layout) d->vkDestroyDescriptorSetLayout(dev, b.set_layout, nullptr);
    if (b.render_pass) d->vkDestroyRenderPass(dev, b.render_pass, nullptr);
    if (b.sampler) d->vkDestroySampler(dev, b.sampler, nullptr);
    if (b.copy_view) d->vkDestroyImageView(dev, b.copy_view, nullptr);
    if (b.copy) d->vkDestroyImage(dev, b.copy, nullptr);
    if (b.copy_memory) d->vkFreeMemory(dev, b.copy_memory, nullptr);
    const bool failed = b.failed;
    b = BrightResources{};
    b.failed = failed;
}

// Wait for in-flight effect work on this swapchain, then free it.
void bright_retire(DeviceData *d, const std::shared_ptr<BrightResources> &b) {
    if (!d || !b) return;
    for (auto &f : b->frames)
        if (f.submitted && f.fence) d->vkWaitForFences(d->device, 1, &f.fence, VK_TRUE, UINT64_MAX);
    {
        // A queued present may still wait on `done`; free those with the capture semaphores.
        std::lock_guard<std::mutex> lock(g_state_mutex);
        for (auto &f : b->frames)
            if (f.done) d->retired_present_semaphores.push_back(std::exchange(f.done, VK_NULL_HANDLE));
    }
    if (bright_functions_ok(d)) bright_destroy(d, *b);
}

VkImageView make_view(DeviceData *d, VkImage image, VkFormat format) {
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    return d->vkCreateImageView(d->device, &info, nullptr, &view) == VK_SUCCESS ? view : VK_NULL_HANDLE;
}

bool bright_init(DeviceData *d, BrightResources &b, uint32_t family, const std::vector<VkImage> &images,
                 VkFormat format, uint32_t width, uint32_t height) {
    if (b.ready) return true;
    if (b.failed) return false;
    auto fail = [&](const char *what) {
        log_msg("Brightness disabled for swapchain: %s", what);
        bright_destroy(d, b);
        b.failed = true;
        return false;
    };
    if (!bright_functions_ok(d)) return fail("missing device functions");
    if (!d->instance_data || !d->instance_data->vkGetPhysicalDeviceQueueFamilyProperties) return fail("no queue family query");
    {
        uint32_t count = 0;
        d->instance_data->vkGetPhysicalDeviceQueueFamilyProperties(d->physical_device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        d->instance_data->vkGetPhysicalDeviceQueueFamilyProperties(d->physical_device, &count, families.data());
        if (family >= count || !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) return fail("present queue has no graphics");
    }
    const VkDevice dev = d->device;
    b.family = family;

    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent = {width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d->vkCreateImage(dev, &image_info, nullptr, &b.copy) != VK_SUCCESS) return fail("copy image");
    VkMemoryRequirements req{};
    d->vkGetImageMemoryRequirements(dev, b.copy, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_device_memory_type(d, req.memoryTypeBits);
    if (alloc.memoryTypeIndex == UINT32_MAX ||
        d->vkAllocateMemory(dev, &alloc, nullptr, &b.copy_memory) != VK_SUCCESS ||
        d->vkBindImageMemory(dev, b.copy, b.copy_memory, 0) != VK_SUCCESS) return fail("copy memory");
    if (!(b.copy_view = make_view(d, b.copy, format))) return fail("copy view");

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_LINEAR;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (d->vkCreateSampler(dev, &sampler, nullptr, &b.sampler) != VK_SUCCESS) return fail("sampler");

    VkAttachmentDescription attachment{};
    attachment.format = format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;  // every pixel is rewritten
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].srcAccessMask = 0;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = 0;
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount = 1;
    rp.pAttachments = &attachment;
    rp.subpassCount = 1;
    rp.pSubpasses = &subpass;
    rp.dependencyCount = 2;
    rp.pDependencies = deps;
    if (d->vkCreateRenderPass(dev, &rp, nullptr, &b.render_pass) != VK_SUCCESS) return fail("render pass");

    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl.bindingCount = 1;
    dsl.pBindings = &binding;
    if (d->vkCreateDescriptorSetLayout(dev, &dsl, nullptr, &b.set_layout) != VK_SUCCESS) return fail("set layout");
    VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(BrightPush)};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &b.set_layout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &push;
    if (d->vkCreatePipelineLayout(dev, &pl, nullptr, &b.pipeline_layout) != VK_SUCCESS) return fail("pipeline layout");

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets = 1;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &pool_size;
    if (d->vkCreateDescriptorPool(dev, &dp, nullptr, &b.descriptor_pool) != VK_SUCCESS) return fail("descriptor pool");
    VkDescriptorSetAllocateInfo ds{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ds.descriptorPool = b.descriptor_pool;
    ds.descriptorSetCount = 1;
    ds.pSetLayouts = &b.set_layout;
    if (d->vkAllocateDescriptorSets(dev, &ds, &b.set) != VK_SUCCESS) return fail("descriptor set");
    VkDescriptorImageInfo image_desc{b.sampler, b.copy_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = b.set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image_desc;
    d->vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);

    VkShaderModule modules[2]{};
    const uint32_t *code[2] = {kBrightVertSpv, kBrightFragSpv};
    const size_t sizes[2] = {sizeof(kBrightVertSpv), sizeof(kBrightFragSpv)};
    for (int i = 0; i < 2; ++i) {
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = sizes[i];
        sm.pCode = code[i];
        if (d->vkCreateShaderModule(dev, &sm, nullptr, &modules[i]) != VK_SUCCESS) {
            if (modules[0]) d->vkDestroyShaderModule(dev, modules[0], nullptr);
            return fail("shader module");
        }
    }
    VkPipelineShaderStageCreateInfo stages[2]{};
    for (int i = 0; i < 2; ++i) {
        stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[i].stage = i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[i].module = modules[i];
        stages[i].pName = "main";
    }
    VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    const VkDynamicState dynamic_states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vertex;
    gp.pInputAssemblyState = &assembly;
    gp.pViewportState = &viewport;
    gp.pRasterizationState = &raster;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &blend;
    gp.pDynamicState = &dynamic;
    gp.layout = b.pipeline_layout;
    gp.renderPass = b.render_pass;
    const VkResult pipeline_result = d->vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &b.pipeline);
    d->vkDestroyShaderModule(dev, modules[0], nullptr);
    d->vkDestroyShaderModule(dev, modules[1], nullptr);
    if (pipeline_result != VK_SUCCESS) return fail("pipeline");

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = family;
    if (d->vkCreateCommandPool(dev, &pool_info, nullptr, &b.pool) != VK_SUCCESS) return fail("command pool");

    b.frames.resize(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        BrightFrame &f = b.frames[i];
        if (!(f.view = make_view(d, images[i], format))) return fail("swapchain view");
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = b.render_pass;
        fb.attachmentCount = 1;
        fb.pAttachments = &f.view;
        fb.width = width;
        fb.height = height;
        fb.layers = 1;
        if (d->vkCreateFramebuffer(dev, &fb, nullptr, &f.framebuffer) != VK_SUCCESS) return fail("framebuffer");
        VkCommandBufferAllocateInfo cmd{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmd.commandPool = b.pool;
        cmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd.commandBufferCount = 1;
        if (d->vkAllocateCommandBuffers(dev, &cmd, &f.cmd) != VK_SUCCESS) return fail("command buffer");
        VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (d->vkCreateFence(dev, &fence, nullptr, &f.fence) != VK_SUCCESS) return fail("fence");
        VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (d->vkCreateSemaphore(dev, &sem, nullptr, &f.done) != VK_SUCCESS) return fail("semaphore");
    }
    b.ready = true;
    log_msg("Brightness resources ready: %ux%u format=%u images=%zu", width, height, format, images.size());
    return true;
}

// Records copy(swapchain -> copy image) + fullscreen draw(copy -> swapchain). If `wait` is set it
// replaces the app's wait semaphores (capture already consumed them).
bool bright_before_present(VkQueue queue, QueueData *qdata, const VkPresentInfoKHR *present_info,
                           VkSemaphore wait, VkSemaphore *signal_out) {
    if (!qdata || !qdata->device_data || !present_info || present_info->swapchainCount != 1 ||
        !present_info->pSwapchains || !present_info->pImageIndices) return false;
    DeviceData *d = qdata->device_data;
    const uint32_t index = present_info->pImageIndices[0];
    std::shared_ptr<BrightResources> b;
    std::vector<VkImage> images;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_swapchains.find(present_info->pSwapchains[0]);
        if (it == g_swapchains.end() || !it->second.bright_supported || index >= it->second.images.size()) return false;
        if (!it->second.bright) it->second.bright = std::make_shared<BrightResources>();
        b = it->second.bright;
        if (!b->ready) images = it->second.images;
        format = it->second.format;
        width = it->second.width;
        height = it->second.height;
    }
    if (b->failed) return false;
    if (!b->ready && !bright_init(d, *b, qdata->family_index, images, format, width, height)) return false;
    if (b->family != qdata->family_index || index >= b->frames.size()) return false;
    BrightFrame &f = b->frames[index];
    if (f.submitted && d->vkWaitForFences(d->device, 1, &f.fence, VK_TRUE, 50000000ULL) != VK_SUCCESS) return false;
    VkImage image = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_swapchains.find(present_info->pSwapchains[0]);
        if (it == g_swapchains.end()) return false;
        image = it->second.images[index];
    }

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (d->vkBeginCommandBuffer(f.cmd, &begin) != VK_SUCCESS) return false;

    VkImageMemoryBarrier to_copy[2]{};
    for (auto &barrier : to_copy) {
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    to_copy[0].image = image;  // presentable -> copy source
    to_copy[0].oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    to_copy[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_copy[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_copy[1].image = b->copy;  // previous frame's sampling finished (queue order) -> copy target
    to_copy[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_copy[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_copy[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    d->vkCmdPipelineBarrier(f.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 2, to_copy);
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = region.srcSubresource;
    region.extent = {width, height, 1};
    d->vkCmdCopyImage(f.cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      b->copy, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageMemoryBarrier to_sample = to_copy[1];
    to_sample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_sample.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    d->vkCmdPipelineBarrier(f.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &to_sample);

    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = b->render_pass;
    rp.framebuffer = f.framebuffer;
    rp.renderArea.extent = {width, height};
    d->vkCmdBeginRenderPass(f.cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    d->vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, b->pipeline);
    d->vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, b->pipeline_layout, 0, 1, &b->set, 0, nullptr);
    BrightPush push{{1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height)},
                    g_bright_strength.load(), g_bright_lift.load(), g_bright_detail.load()};
    d->vkCmdPushConstants(f.cmd, b->pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {width, height}};
    d->vkCmdSetViewport(f.cmd, 0, 1, &viewport);
    d->vkCmdSetScissor(f.cmd, 0, 1, &scissor);
    d->vkCmdDraw(f.cmd, 3, 1, 0, 0);
    d->vkCmdEndRenderPass(f.cmd);  // final layout PRESENT_SRC_KHR
    if (d->vkEndCommandBuffer(f.cmd) != VK_SUCCESS) return false;

    const uint32_t wait_count = wait ? 1u : present_info->waitSemaphoreCount;
    const VkSemaphore *waits = wait ? &wait : present_info->pWaitSemaphores;
    std::vector<VkPipelineStageFlags> stages(wait_count, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = wait_count;
    submit.pWaitSemaphores = waits;
    submit.pWaitDstStageMask = stages.empty() ? nullptr : stages.data();
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &f.cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &f.done;
    if (d->vkResetFences(d->device, 1, &f.fence) != VK_SUCCESS) return false;
    f.submitted = false;
    if (d->vkQueueSubmit(queue, 1, &submit, f.fence) != VK_SUCCESS) {
        log_msg("Brightness submit failed; effect disabled for this swapchain");
        b->failed = true;
        return false;
    }
    f.submitted = true;
    *signal_out = f.done;
    return true;
}

// --- config + Sober log watcher (runs on its own thread, never on the present path) ---

std::string trim(const std::string &s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

BrightSettings parse_bright_conf(const std::string &text) {
    BrightSettings out;
    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(start, end - start);
        start = end + 1;
        if (const size_t hash = line.find('#'); hash != std::string::npos) line.resize(hash);
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq)), value = trim(line.substr(eq + 1));
        char *e = nullptr;
        if (key == "mode") {
            out.mode = value == "on" ? BrightSettings::kOn : value == "off" ? BrightSettings::kOff : BrightSettings::kAuto;
        } else if (key == "strength" || key == "lift" || key == "detail") {
            const float v = strtof(value.c_str(), &e);
            if (e == value.c_str() || !(v == v)) continue;
            if (key == "strength") out.strength = std::min(6.0f, std::max(1.0f, v));
            else if (key == "lift") out.lift = std::min(0.3f, std::max(0.0f, v));
            else out.detail = std::min(3.0f, std::max(0.0f, v));
        } else if (key == "universes") {
            out.universes.clear();
            const char *p = value.c_str();
            while (*p) {
                const unsigned long long id = strtoull(p, &e, 10);
                if (e == p) { ++p; continue; }
                out.universes.push_back(id);
                p = e;
            }
        }
    }
    return out;
}

// Returns the new "inside a target game" state after one Sober log line.
bool bright_log_line(const std::string &line, const std::vector<uint64_t> &universes, bool inside) {
    if (line.find("[FLog::GameJoinLoadTime]") != std::string::npos) {
        const size_t at = line.find("universeid:");
        if (at == std::string::npos) return inside;
        const uint64_t id = strtoull(line.c_str() + at + 11, nullptr, 10);
        return std::find(universes.begin(), universes.end(), id) != universes.end();
    }
    if (line.find("[FLog::SingleSurfaceApp] setStage: (stage:LuaApp)") != std::string::npos) return false;
    return inside;
}

const char kDefaultBrightConf[] =
    "# VD brightness post-process. Re-read every 0.5 s while Sober runs.\n"
    "# mode: auto (only in the universes below), on (every game), off\n"
    "mode = auto\n"
    "# strength 1..6 lifts shadows (1 = off), lift 0..0.3 raises black, detail 0..3 local contrast\n"
    "strength = 2.2\n"
    "lift = 0.04\n"
    "detail = 0.6\n"
    "# Roblox universe ids (Violence District = 6739698191)\n"
    "universes = 6739698191\n";

void bright_watcher() {
    const std::string conf_path = g_base_dir + "/vd_bright.conf";
    std::string log_link;
    if (const char *xdg = getenv("XDG_DATA_HOME"); xdg && *xdg) log_link = std::string(xdg) + "/sober/sober_logs/latest.log";
    if (FILE *f = fopen(conf_path.c_str(), "r")) fclose(f);
    else if (FILE *w = fopen(conf_path.c_str(), "w")) { fputs(kDefaultBrightConf, w); fclose(w); }

    std::string target, partial;
    long offset = 0;
    bool inside = false, last_active = false;
    BrightSettings s = parse_bright_conf(kDefaultBrightConf);
    while (!g_bright_stop.load()) {
        std::string text;
        if (FILE *f = fopen(conf_path.c_str(), "r")) {
            char buf[2048];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
            fclose(f);
        }
        if (!text.empty()) s = parse_bright_conf(text);
        g_bright_strength.store(s.strength);
        g_bright_lift.store(s.lift);
        g_bright_detail.store(s.detail);

        if (!log_link.empty()) {
            char resolved[4096];
            const ssize_t len = readlink(log_link.c_str(), resolved, sizeof(resolved) - 1);
            struct stat st{};
            if (len > 0) {
                resolved[len] = '\0';
                std::string now_target = resolved[0] == '/' ? resolved
                    : log_link.substr(0, log_link.rfind('/') + 1) + resolved;
                // Ignore the previous session's log until Sober repoints latest.log.
                if (now_target != target && stat(now_target.c_str(), &st) == 0 && st.st_mtime >= g_init_wall_time) {
                    target = now_target;
                    offset = 0;
                    partial.clear();
                    inside = false;
                }
            }
            if (!target.empty() && stat(target.c_str(), &st) == 0 && st.st_size < offset) {
                offset = 0;  // truncated in place
                partial.clear();
            }
            if (!target.empty()) {
                if (FILE *f = fopen(target.c_str(), "r")) {
                    if (fseek(f, offset, SEEK_SET) == 0) {
                        char buf[65536];
                        size_t n;
                        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
                            offset += static_cast<long>(n);
                            partial.append(buf, n);
                            if (partial.size() > (1u << 20)) partial.clear();  // no newline in 1 MiB: not a log line
                            size_t nl;
                            while ((nl = partial.find('\n')) != std::string::npos) {
                                inside = bright_log_line(partial.substr(0, nl), s.universes, inside);
                                partial.erase(0, nl + 1);
                            }
                        }
                    }
                    fclose(f);
                }
            }
        }
        const bool active = s.mode == BrightSettings::kOn || (s.mode == BrightSettings::kAuto && inside);
        if (active != last_active) {
            log_msg("Brightness %s (mode=%d strength=%.2f lift=%.2f detail=%.2f)", active ? "ON" : "OFF",
                    static_cast<int>(s.mode), s.strength, s.lift, s.detail);
            last_active = active;
        }
        g_bright_active.store(active);
        for (int i = 0; i < 10 && !g_bright_stop.load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

} // namespace

VKAPI_ATTR VkResult VKAPI_CALL vd_vkCreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                                                     const VkAllocationCallbacks *pAllocator,
                                                     VkInstance *pInstance) {
    init_layer_runtime();
    log_msg("vd_vkCreateInstance called");
    VkLayerInstanceCreateInfo *chain = get_instance_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain || !chain->u.pLayerInfo) {
        log_msg("ERROR: No instance link info found in pCreateInfo");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance next_create = reinterpret_cast<PFN_vkCreateInstance>(next_gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    VkResult result = next_create(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS) return result;

    auto data = std::make_unique<InstanceData>();
    data->instance = *pInstance;
    data->next_gipa = next_gipa;
    data->vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(next_gipa(*pInstance, "vkDestroyInstance"));
    data->vkCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(next_gipa(*pInstance, "vkCreateDevice"));
    data->vkEnumeratePhysicalDevices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(next_gipa(*pInstance, "vkEnumeratePhysicalDevices"));
    data->vkEnumeratePhysicalDeviceGroups = reinterpret_cast<PFN_vkEnumeratePhysicalDeviceGroups>(next_gipa(*pInstance, "vkEnumeratePhysicalDeviceGroups"));
    data->vkEnumeratePhysicalDeviceGroupsKHR = reinterpret_cast<PFN_vkEnumeratePhysicalDeviceGroupsKHR>(next_gipa(*pInstance, "vkEnumeratePhysicalDeviceGroupsKHR"));
    data->vkEnumerateDeviceExtensionProperties = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(next_gipa(*pInstance, "vkEnumerateDeviceExtensionProperties"));
    data->vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(next_gipa(*pInstance, "vkGetPhysicalDeviceMemoryProperties"));
    data->vkGetPhysicalDeviceQueueFamilyProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(next_gipa(*pInstance, "vkGetPhysicalDeviceQueueFamilyProperties"));
    data->vkGetPhysicalDeviceSurfaceCapabilitiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(next_gipa(*pInstance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        InstanceData *ptr = data.get();
        g_instances[reinterpret_cast<void *>(*pInstance)] = std::move(data);
        g_primary_instance = ptr;
    }
    log_msg("Instance %p created successfully", reinterpret_cast<void *>(*pInstance));
    write_status_json(true);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vd_vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    log_msg("vd_vkDestroyInstance called for %p", reinterpret_cast<void *>(instance));
    PFN_vkDestroyInstance next_destroy = nullptr;
    bool no_instances = false;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find(reinterpret_cast<void *>(instance));
        if (it != g_instances.end()) {
            next_destroy = it->second->vkDestroyInstance;
            InstanceData *ptr = it->second.get();
            for (auto pit = g_physical_device_to_instance.begin(); pit != g_physical_device_to_instance.end();) {
                pit = pit->second == ptr ? g_physical_device_to_instance.erase(pit) : std::next(pit);
            }
            g_instances.erase(it);
            if (g_primary_instance == ptr) g_primary_instance = g_instances.empty() ? nullptr : g_instances.begin()->second.get();
        }
        no_instances = g_instances.empty();
    }
    if (no_instances && g_shm_header) {
        { std::lock_guard<std::mutex> lock(g_shm_mutex);
          shm_begin_write(); g_shm_header->status_flags = 0; shm_end_write(); }
        write_status_json(false);
    }
    if (next_destroy) next_destroy(instance, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumeratePhysicalDevices(VkInstance instance, uint32_t *count, VkPhysicalDevice *devices) {
    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find(reinterpret_cast<void *>(instance));
        idata = it != g_instances.end() ? it->second.get() : g_primary_instance;
    }
    if (!idata || !idata->vkEnumeratePhysicalDevices) return VK_ERROR_INITIALIZATION_FAILED;
    const VkResult result = idata->vkEnumeratePhysicalDevices(instance, count, devices);
    if ((result == VK_SUCCESS || result == VK_INCOMPLETE) && devices && count) {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        for (uint32_t i = 0; i < *count; ++i) if (devices[i]) g_physical_device_to_instance[reinterpret_cast<void *>(devices[i])] = idata;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumeratePhysicalDeviceGroups(VkInstance instance, uint32_t *count, VkPhysicalDeviceGroupProperties *groups) {
    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find(reinterpret_cast<void *>(instance));
        idata = it != g_instances.end() ? it->second.get() : g_primary_instance;
    }
    PFN_vkEnumeratePhysicalDeviceGroups fn = idata ? idata->vkEnumeratePhysicalDeviceGroups : nullptr;
    if (!fn && idata) fn = reinterpret_cast<PFN_vkEnumeratePhysicalDeviceGroups>(idata->vkEnumeratePhysicalDeviceGroupsKHR);
    if (!fn) return VK_ERROR_INITIALIZATION_FAILED;
    const VkResult result = fn(instance, count, groups);
    if ((result == VK_SUCCESS || result == VK_INCOMPLETE) && groups && count) {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        for (uint32_t g = 0; g < *count; ++g) {
            for (uint32_t i = 0; i < groups[g].physicalDeviceCount; ++i) {
                if (groups[g].physicalDevices[i]) g_physical_device_to_instance[reinterpret_cast<void *>(groups[g].physicalDevices[i])] = idata;
            }
        }
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkCreateDevice(VkPhysicalDevice physicalDevice,
                                                   const VkDeviceCreateInfo *pCreateInfo,
                                                   const VkAllocationCallbacks *pAllocator,
                                                   VkDevice *pDevice) {
    log_msg("vd_vkCreateDevice called for physicalDevice %p", reinterpret_cast<void *>(physicalDevice));
    VkLayerDeviceCreateInfo *chain = get_device_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);
    if (!chain || !chain->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr next_gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr next_gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;

    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_physical_device_to_instance.find(reinterpret_cast<void *>(physicalDevice));
        idata = it != g_physical_device_to_instance.end() ? it->second : g_primary_instance;
    }
    const VkInstance instance = idata ? idata->instance : VK_NULL_HANDLE;
    PFN_vkCreateDevice next_create = next_gipa ? reinterpret_cast<PFN_vkCreateDevice>(next_gipa(instance, "vkCreateDevice")) : nullptr;
    if (!next_create && idata) next_create = idata->vkCreateDevice;
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;

    VkResult result = next_create(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (result != VK_SUCCESS) return result;

    auto data = std::make_unique<DeviceData>();
    data->physical_device = physicalDevice;
    data->device = *pDevice;
    data->instance_data = idata;
    data->next_gdpa = next_gdpa;
#define VD_RESOLVE_DEVICE(name) data->name = reinterpret_cast<PFN_##name>(next_gdpa(*pDevice, #name))
    VD_RESOLVE_DEVICE(vkDestroyDevice);
    VD_RESOLVE_DEVICE(vkGetDeviceQueue);
    VD_RESOLVE_DEVICE(vkGetDeviceQueue2);
    VD_RESOLVE_DEVICE(vkCreateSwapchainKHR);
    VD_RESOLVE_DEVICE(vkDestroySwapchainKHR);
    VD_RESOLVE_DEVICE(vkGetSwapchainImagesKHR);
    VD_RESOLVE_DEVICE(vkQueuePresentKHR);
    VD_RESOLVE_DEVICE(vkCreateCommandPool);
    VD_RESOLVE_DEVICE(vkDestroyCommandPool);
    VD_RESOLVE_DEVICE(vkResetCommandPool);
    VD_RESOLVE_DEVICE(vkAllocateCommandBuffers);
    VD_RESOLVE_DEVICE(vkCreateBuffer);
    VD_RESOLVE_DEVICE(vkDestroyBuffer);
    VD_RESOLVE_DEVICE(vkGetBufferMemoryRequirements);
    VD_RESOLVE_DEVICE(vkAllocateMemory);
    VD_RESOLVE_DEVICE(vkFreeMemory);
    VD_RESOLVE_DEVICE(vkBindBufferMemory);
    VD_RESOLVE_DEVICE(vkMapMemory);
    VD_RESOLVE_DEVICE(vkUnmapMemory);
    VD_RESOLVE_DEVICE(vkInvalidateMappedMemoryRanges);
    VD_RESOLVE_DEVICE(vkCreateFence);
    VD_RESOLVE_DEVICE(vkDestroyFence);
    VD_RESOLVE_DEVICE(vkResetFences);
    VD_RESOLVE_DEVICE(vkWaitForFences);
    VD_RESOLVE_DEVICE(vkGetFenceStatus);
    VD_RESOLVE_DEVICE(vkDeviceWaitIdle);
    VD_RESOLVE_DEVICE(vkCreateSemaphore);
    VD_RESOLVE_DEVICE(vkDestroySemaphore);
    VD_RESOLVE_DEVICE(vkQueueSubmit);
    VD_RESOLVE_DEVICE(vkBeginCommandBuffer);
    VD_RESOLVE_DEVICE(vkEndCommandBuffer);
    VD_RESOLVE_DEVICE(vkCmdPipelineBarrier);
    VD_RESOLVE_DEVICE(vkCmdCopyImageToBuffer);
    VD_RESOLVE_DEVICE(vkCreateImage);
    VD_RESOLVE_DEVICE(vkDestroyImage);
    VD_RESOLVE_DEVICE(vkGetImageMemoryRequirements);
    VD_RESOLVE_DEVICE(vkBindImageMemory);
    VD_RESOLVE_DEVICE(vkCreateImageView);
    VD_RESOLVE_DEVICE(vkDestroyImageView);
    VD_RESOLVE_DEVICE(vkCreateSampler);
    VD_RESOLVE_DEVICE(vkDestroySampler);
    VD_RESOLVE_DEVICE(vkCreateRenderPass);
    VD_RESOLVE_DEVICE(vkDestroyRenderPass);
    VD_RESOLVE_DEVICE(vkCreateFramebuffer);
    VD_RESOLVE_DEVICE(vkDestroyFramebuffer);
    VD_RESOLVE_DEVICE(vkCreateShaderModule);
    VD_RESOLVE_DEVICE(vkDestroyShaderModule);
    VD_RESOLVE_DEVICE(vkCreateDescriptorSetLayout);
    VD_RESOLVE_DEVICE(vkDestroyDescriptorSetLayout);
    VD_RESOLVE_DEVICE(vkCreatePipelineLayout);
    VD_RESOLVE_DEVICE(vkDestroyPipelineLayout);
    VD_RESOLVE_DEVICE(vkCreateGraphicsPipelines);
    VD_RESOLVE_DEVICE(vkDestroyPipeline);
    VD_RESOLVE_DEVICE(vkCreateDescriptorPool);
    VD_RESOLVE_DEVICE(vkDestroyDescriptorPool);
    VD_RESOLVE_DEVICE(vkAllocateDescriptorSets);
    VD_RESOLVE_DEVICE(vkUpdateDescriptorSets);
    VD_RESOLVE_DEVICE(vkCmdCopyImage);
    VD_RESOLVE_DEVICE(vkCmdBeginRenderPass);
    VD_RESOLVE_DEVICE(vkCmdEndRenderPass);
    VD_RESOLVE_DEVICE(vkCmdBindPipeline);
    VD_RESOLVE_DEVICE(vkCmdBindDescriptorSets);
    VD_RESOLVE_DEVICE(vkCmdPushConstants);
    VD_RESOLVE_DEVICE(vkCmdSetViewport);
    VD_RESOLVE_DEVICE(vkCmdSetScissor);
    VD_RESOLVE_DEVICE(vkCmdDraw);
#undef VD_RESOLVE_DEVICE

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        DeviceData *ptr = data.get();
        g_devices[reinterpret_cast<void *>(*pDevice)] = std::move(data);
        g_primary_device = ptr;
    }
    log_msg("Device %p created successfully (Swapchain support: %s)", reinterpret_cast<void *>(*pDevice), g_primary_device->vkCreateSwapchainKHR ? "YES" : "NO");
    write_status_json(true);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vd_vkDestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    log_msg("vd_vkDestroyDevice called for %p", reinterpret_cast<void *>(device));
    PFN_vkDestroyDevice next_destroy = nullptr;
    DeviceData *target = nullptr;
    std::vector<std::unique_ptr<QueueData>> queues_to_destroy;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        if (it != g_devices.end()) {
            target = it->second.get();
            next_destroy = target->vkDestroyDevice;
            for (auto qit = g_queues.begin(); qit != g_queues.end();) {
                if (qit->second->device_data == target) {
                    queues_to_destroy.push_back(std::move(qit->second));
                    qit = g_queues.erase(qit);
                } else {
                    ++qit;
                }
            }
        }
    }
    for (auto &q : queues_to_destroy) {
        q->stopping.store(true);
        if (q->worker.joinable()) q->worker.join();
    }
    if (target && target->vkDeviceWaitIdle) target->vkDeviceWaitIdle(device);
    for (auto &q : queues_to_destroy) {
        for (auto &c : q->captures) destroy_capture_resources(q.get(),c);
    }
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        if (it != g_devices.end()) {
            DeviceData *ptr = it->second.get();
            for (auto &sc : g_swapchains) {
                if (sc.second.device==device && sc.second.bright && bright_functions_ok(ptr))
                    bright_destroy(ptr, *sc.second.bright);
                if (sc.second.device==device)
                    ptr->retired_present_semaphores.insert(ptr->retired_present_semaphores.end(),
                        sc.second.present_semaphores.begin(),sc.second.present_semaphores.end());
            }
            for (VkSemaphore sem : ptr->retired_present_semaphores)
                if (ptr->vkDestroySemaphore) ptr->vkDestroySemaphore(device,sem,nullptr);
            for (auto sc=g_swapchains.begin();sc!=g_swapchains.end();)
                sc=sc->second.device==device ? g_swapchains.erase(sc) : std::next(sc);
            g_devices.erase(it);
            if (g_primary_device == ptr) g_primary_device = g_devices.empty() ? nullptr : g_devices.begin()->second.get();
        }
    }
    if (next_destroy) next_destroy(device, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL vd_vkGetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue *queue) {
    DeviceData *d = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        d = it != g_devices.end() ? it->second.get() : g_primary_device;
    }
    if (d && d->vkGetDeviceQueue) {
        d->vkGetDeviceQueue(device, family, index, queue);
        if (queue && *queue) register_queue(d, *queue, family);
    }
}

VKAPI_ATTR void VKAPI_CALL vd_vkGetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2 *info, VkQueue *queue) {
    DeviceData *d = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        d = it != g_devices.end() ? it->second.get() : g_primary_device;
    }
    if (d && d->vkGetDeviceQueue2) {
        d->vkGetDeviceQueue2(device, info, queue);
        if (queue && *queue && info) register_queue(d, *queue, info->queueFamilyIndex);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkCreateSwapchainKHR(VkDevice device,
                                                        const VkSwapchainCreateInfoKHR *pCreateInfo,
                                                        const VkAllocationCallbacks *pAllocator,
                                                        VkSwapchainKHR *pSwapchain) {
    log_msg("vd_vkCreateSwapchainKHR called: %ux%u format=%u mode=%u minImages=%u usage=0x%x",
            pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height, pCreateInfo->imageFormat,
            pCreateInfo->presentMode, pCreateInfo->minImageCount, pCreateInfo->imageUsage);

    DeviceData *d = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        d = it != g_devices.end() ? it->second.get() : g_primary_device;
    }
    if (!d || !d->vkCreateSwapchainKHR) return VK_ERROR_INITIALIZATION_FAILED;

    VkSwapchainCreateInfoKHR local = *pCreateInfo;
    const VkSwapchainCreateInfoKHR *create_info = pCreateInfo;
    bool capture_supported = false;
    bool bright_supported = false;
    bool added_transfer_usage = false;
    const bool want_capture = g_capture_enabled && format_supported_for_capture(pCreateInfo->imageFormat);
    const bool want_bright = g_bright_armed && format_supported_for_bright(pCreateInfo->imageFormat);
    if ((want_capture || want_bright) && d->instance_data &&
        d->instance_data->vkGetPhysicalDeviceSurfaceCapabilitiesKHR) {
        VkSurfaceCapabilitiesKHR caps{};
        if (d->instance_data->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d->physical_device, pCreateInfo->surface, &caps) == VK_SUCCESS &&
            (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
            local.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            capture_supported = want_capture;
            bright_supported = want_bright && (caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            if (bright_supported) local.imageUsage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            create_info = &local;
            added_transfer_usage = local.imageUsage != pCreateInfo->imageUsage;
        } else {
            log_msg("Capture/brightness disabled for swapchain: surface does not support TRANSFER_SRC");
        }
    } else if (g_capture_enabled && !want_capture) {
        log_msg("Capture disabled for swapchain: unsupported 4-byte format %u", pCreateInfo->imageFormat);
    }

    VkResult result = d->vkCreateSwapchainKHR(device, create_info, pAllocator, pSwapchain);
    if (result != VK_SUCCESS && added_transfer_usage) {
        log_msg("Swapchain creation with extra usage failed (%d); retrying untouched create info", static_cast<int>(result));
        capture_supported = false;
        bright_supported = false;
        result = d->vkCreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
    }
    if (result != VK_SUCCESS) return result;

    SwapchainInfo sc;
    sc.swapchain = *pSwapchain;
    sc.device = device;
    sc.surface = pCreateInfo->surface;
    sc.width = pCreateInfo->imageExtent.width;
    sc.height = pCreateInfo->imageExtent.height;
    sc.format = pCreateInfo->imageFormat;
    sc.present_mode = pCreateInfo->presentMode;
    sc.capture_supported = capture_supported;
    sc.bright_supported = bright_supported;

    if (d->vkGetSwapchainImagesKHR) {
        uint32_t count = 0;
        if (d->vkGetSwapchainImagesKHR(device, *pSwapchain, &count, nullptr) == VK_SUCCESS && count > 0) {
            sc.images.resize(count);
            if (d->vkGetSwapchainImagesKHR(device, *pSwapchain, &count, sc.images.data()) == VK_SUCCESS) {
                sc.images.resize(count);
                sc.image_count = count;
            } else {
                sc.images.clear();
            }
        }
    }
    sc.generation=g_generation.fetch_add(1)+1;
    if (sc.images.empty()) sc.capture_supported = sc.bright_supported = false;
    if (sc.capture_supported) {
        VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (uint32_t i=0;i<sc.image_count;++i) {
            VkSemaphore sem=VK_NULL_HANDLE;
            if (d->vkCreateSemaphore(device,&sem_info,nullptr,&sem)!=VK_SUCCESS) {
                sc.capture_supported=false;
                break;
            }
            sc.present_semaphores.push_back(sem);
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_swapchains[*pSwapchain] = sc;
    }
    if (g_shm_header) {
        std::lock_guard<std::mutex> shm_lock(g_shm_mutex);
        shm_begin_write();
        g_shm_header->status_flags &= ~VD_STATUS_PIXELS_READY;
        g_shm_header->status_flags |= VD_STATUS_SWAPCHAIN_READY;
        if (sc.capture_supported && g_capture_enabled) g_shm_header->status_flags |= VD_STATUS_CAPTURE_ENABLED;
        g_shm_header->swapchain_width = sc.width;
        g_shm_header->swapchain_height = sc.height;
        g_shm_header->swapchain_format = static_cast<uint32_t>(sc.format);
        g_shm_header->swapchain_present_mode = static_cast<uint32_t>(sc.present_mode);
        g_shm_header->swapchain_image_count = sc.image_count;
        shm_end_write();
    }
    log_msg("Swapchain %p created: %ux%u format=%u images=%u capture=%s bright=%s",
            reinterpret_cast<void *>(*pSwapchain), sc.width, sc.height, sc.format, sc.image_count,
            sc.capture_supported ? "YES" : "NO", sc.bright_supported ? "YES" : "NO");
    write_status_json(true);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vd_vkDestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks *pAllocator) {
    log_msg("vd_vkDestroySwapchainKHR called for %p", reinterpret_cast<void *>(swapchain));
    PFN_vkDestroySwapchainKHR next_destroy = nullptr;
    bool any = false;
    uint64_t retired_generation=0;
    std::shared_ptr<BrightResources> bright;
    DeviceData *bright_device = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto sc=g_swapchains.find(swapchain);
        auto bright_dev=g_devices.find(reinterpret_cast<void *>(device));
        if (sc!=g_swapchains.end() && bright_dev!=g_devices.end()) {
            bright = std::move(sc->second.bright);
            bright_device = bright_dev->second.get();
        }
        auto dev=g_devices.find(reinterpret_cast<void *>(device));
        if (sc!=g_swapchains.end() && dev!=g_devices.end())
            dev->second->retired_present_semaphores.insert(dev->second->retired_present_semaphores.end(),
                sc->second.present_semaphores.begin(),sc->second.present_semaphores.end());
        if (sc!=g_swapchains.end()) {
            sc->second.lifetime->store(false);
            retired_generation=sc->second.generation;
        }
        g_swapchains.erase(swapchain);
        any = !g_swapchains.empty();
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        DeviceData *d = it != g_devices.end() ? it->second.get() : g_primary_device;
        if (d) next_destroy = d->vkDestroySwapchainKHR;
    }
    std::vector<QueueData *> queues;
    { std::lock_guard<std::mutex> lock(g_state_mutex);
      for (auto &entry : g_queues)
          if (entry.second->device_data->device==device) queues.push_back(entry.second.get()); }
    for (auto *q : queues) {
        std::lock_guard<std::mutex> lock(q->capture_mutex);
        for (auto &c : q->captures) {
            if (c.pending && c.swapchain==swapchain) {
                q->device_data->vkWaitForFences(device,1,&c.fence,VK_TRUE,UINT64_MAX);
                c.pending=false;
            }
        }
    }
    bright_retire(bright_device, bright);
    if (g_shm_header) {
        std::lock_guard<std::mutex> shm_lock(g_shm_mutex);
        shm_begin_write();
        if (!any) g_shm_header->status_flags &= ~VD_STATUS_SWAPCHAIN_READY;
        if (!any || g_shm_header->swapchain_generation==retired_generation)
            g_shm_header->status_flags &= ~VD_STATUS_PIXELS_READY;
        shm_end_write();
    }
    write_status_json(true);
    if (next_destroy) next_destroy(device, swapchain, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pPresentInfo) {
    const uint64_t now_ns = get_time_ns();
    QueueData *qdata = nullptr;
    DeviceData *d = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        if (pPresentInfo && pPresentInfo->swapchainCount==1 && pPresentInfo->pSwapchains) {
            auto sc=g_swapchains.find(pPresentInfo->pSwapchains[0]);
            if (sc!=g_swapchains.end()) {
                sc->second.timing.observe(now_ns);
                g_latest_present_swapchain=sc->first;
            }
        }
        auto qit = g_queues.find(reinterpret_cast<void *>(queue));
        if (qit != g_queues.end()) {
            qdata = qit->second.get();
            d = qdata->device_data;
        } else {
            d = g_primary_device;
        }
    }
    if (!d || !d->vkQueuePresentKHR) return VK_ERROR_INITIALIZATION_FAILED;

    const uint64_t frame = g_frame_counter.fetch_add(1) + 1;
    g_last_present_ns.store(now_ns);

    {
        std::lock_guard<std::mutex> lock(g_fps_mutex);
        if (frame % 60 == 0 || now_ns - g_last_fps_time_ns >= 1000000000ULL) {
            const uint64_t elapsed = now_ns - g_last_fps_time_ns;
            if (elapsed) {
                g_current_fps = static_cast<float>(static_cast<double>(frame - g_last_fps_frame_count) * 1e9 / static_cast<double>(elapsed));
            }
            g_last_fps_time_ns = now_ns;
            g_last_fps_frame_count = frame;
        }
    }

    VkSemaphore wait = VK_NULL_HANDLE;
    if (qdata) capture_before_present(queue, qdata, pPresentInfo, &wait, now_ns, frame);
    // Effect runs after the capture copy, so the solver always reads the original pixels.
    if (g_bright_armed && qdata && g_bright_active.load(std::memory_order_relaxed)) {
        VkSemaphore brightened = VK_NULL_HANDLE;
        if (bright_before_present(queue, qdata, pPresentInfo, wait, &brightened)) wait = brightened;
    }
    if (wait == VK_NULL_HANDLE) return d->vkQueuePresentKHR(queue, pPresentInfo);
    VkPresentInfoKHR chained_present = *pPresentInfo;
    chained_present.waitSemaphoreCount = 1;
    chained_present.pWaitSemaphores = &wait;
    return d->vkQueuePresentKHR(queue, &chained_present);
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateInstanceLayerProperties(uint32_t *count, VkLayerProperties *props) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (!props) { *count = 1; return VK_SUCCESS; }
    if (*count < 1) return VK_INCOMPLETE;
    *count = 1;
    memset(&props[0], 0, sizeof(props[0]));
    strncpy(props[0].layerName, VD_LAYER_NAME, VK_MAX_EXTENSION_NAME_SIZE - 1);
    props[0].specVersion = VD_LAYER_SPEC_VERSION;
    props[0].implementationVersion = VD_LAYER_IMPL_VERSION;
    strncpy(props[0].description, VD_LAYER_DESCRIPTION, VK_MAX_DESCRIPTION_SIZE - 1);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateInstanceExtensionProperties(const char *layer, uint32_t *count, VkExtensionProperties *) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (layer && strcmp(layer, VD_LAYER_NAME) == 0) { *count = 0; return VK_SUCCESS; }
    return VK_ERROR_LAYER_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateDeviceLayerProperties(VkPhysicalDevice, uint32_t *count, VkLayerProperties *props) {
    return vd_vkEnumerateInstanceLayerProperties(count, props);
}

VKAPI_ATTR VkResult VKAPI_CALL vd_vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice, const char *layer, uint32_t *count, VkExtensionProperties *props) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (layer && strcmp(layer, VD_LAYER_NAME) == 0) { *count = 0; return VK_SUCCESS; }
    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_physical_device_to_instance.find(reinterpret_cast<void *>(physicalDevice));
        idata = it != g_physical_device_to_instance.end() ? it->second : g_primary_instance;
    }
    return idata && idata->vkEnumerateDeviceExtensionProperties
        ? idata->vkEnumerateDeviceExtensionProperties(physicalDevice, layer, count, props)
        : VK_SUCCESS;
}

extern "C" {

VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    if (!name) return nullptr;
    log_debug("vkGetInstanceProcAddr: instance=%p name=%s", reinterpret_cast<void *>(instance), name);
    if (strcmp(name, "vkGetInstanceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vkGetInstanceProcAddr);
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceProcAddr);
    if (strcmp(name, "vkNegotiateLoaderLayerInterfaceVersion") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vkNegotiateLoaderLayerInterfaceVersion);
    if (strcmp(name, "vkCreateInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkCreateInstance);
    if (strcmp(name, "vkEnumerateInstanceLayerProperties") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumerateInstanceLayerProperties);
    if (strcmp(name, "vkEnumerateInstanceExtensionProperties") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumerateInstanceExtensionProperties);
    if (!instance) return nullptr;

    if (strcmp(name, "vkDestroyInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkDestroyInstance);
    if (strcmp(name, "vkEnumeratePhysicalDevices") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumeratePhysicalDevices);
    if (strcmp(name, "vkEnumeratePhysicalDeviceGroups") == 0 || strcmp(name, "vkEnumeratePhysicalDeviceGroupsKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumeratePhysicalDeviceGroups);
    if (strcmp(name, "vkCreateDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkCreateDevice);
    if (strcmp(name, "vkEnumerateDeviceLayerProperties") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumerateDeviceLayerProperties);
    if (strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkEnumerateDeviceExtensionProperties);
    if (strcmp(name, "vkDestroyDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkDestroyDevice);
    if (strcmp(name, "vkGetDeviceQueue") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkGetDeviceQueue);
    if (strcmp(name, "vkGetDeviceQueue2") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkGetDeviceQueue2);
    if (strcmp(name, "vkCreateSwapchainKHR") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkCreateSwapchainKHR);
    if (strcmp(name, "vkDestroySwapchainKHR") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkDestroySwapchainKHR);
    if (strcmp(name, "vkQueuePresentKHR") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkQueuePresentKHR);

    InstanceData *idata = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_instances.find(reinterpret_cast<void *>(instance));
        idata = it != g_instances.end() ? it->second.get() : g_primary_instance;
    }
    return idata && idata->next_gipa ? idata->next_gipa(instance, name) : nullptr;
}

VD_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    if (!name) return nullptr;
    if (strcmp(name, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceProcAddr);
    if (strcmp(name, "vkDestroyDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkDestroyDevice);
    if (strcmp(name, "vkGetDeviceQueue") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkGetDeviceQueue);
    if (strcmp(name, "vkGetDeviceQueue2") == 0) return reinterpret_cast<PFN_vkVoidFunction>(vd_vkGetDeviceQueue2);

    DeviceData *d = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_devices.find(reinterpret_cast<void *>(device));
        d = it != g_devices.end() ? it->second.get() : g_primary_device;
    }
    if (strcmp(name, "vkCreateSwapchainKHR") == 0) return d && d->vkCreateSwapchainKHR ? reinterpret_cast<PFN_vkVoidFunction>(vd_vkCreateSwapchainKHR) : nullptr;
    if (strcmp(name, "vkDestroySwapchainKHR") == 0) return d && d->vkDestroySwapchainKHR ? reinterpret_cast<PFN_vkVoidFunction>(vd_vkDestroySwapchainKHR) : nullptr;
    if (strcmp(name, "vkQueuePresentKHR") == 0) return d && d->vkQueuePresentKHR ? reinterpret_cast<PFN_vkVoidFunction>(vd_vkQueuePresentKHR) : nullptr;
    return d && d->next_gdpa ? d->next_gdpa(device, name) : nullptr;
}

VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *version) {
    log_msg("vkNegotiateLoaderLayerInterfaceVersion called, loader version: %u", version ? version->loaderLayerInterfaceVersion : 0);
    if (!version || version->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT || version->loaderLayerInterfaceVersion < 1)
        return VK_ERROR_INITIALIZATION_FAILED;
    version->loaderLayerInterfaceVersion = std::min<uint32_t>(version->loaderLayerInterfaceVersion, 2);
    version->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    version->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
    version->pfnGetPhysicalDeviceProcAddr = nullptr;
    log_msg("Negotiation successful (interface version %u, pfnGetPhysicalDeviceProcAddr = nullptr)", version->loaderLayerInterfaceVersion);
    return VK_SUCCESS;
}

VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *count, VkLayerProperties *props) {
    return vd_vkEnumerateInstanceLayerProperties(count, props);
}
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *layer, uint32_t *count, VkExtensionProperties *props) {
    return vd_vkEnumerateInstanceExtensionProperties(layer, count, props);
}
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physical, uint32_t *count, VkLayerProperties *props) {
    return vd_vkEnumerateDeviceLayerProperties(physical, count, props);
}
VD_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physical, const char *layer, uint32_t *count, VkExtensionProperties *props) {
    return vd_vkEnumerateDeviceExtensionProperties(physical, layer, count, props);
}

} // extern "C"
