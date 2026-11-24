#include "../native/vulkan_layer/vd_capture_layer.cpp"
#include <cassert>

namespace {
int submits=0,presents=0,waits=0,destroyed_semaphores=0;
bool drained=false;
std::atomic<bool> fence_observed{false};
VkResult submit_result=VK_SUCCESS,ready_result=VK_NOT_READY;
VkSemaphore original=reinterpret_cast<VkSemaphore>(44);
VkSemaphore last_wait=VK_NULL_HANDLE;
VkResult VKAPI_CALL reset_pool(VkDevice,VkCommandPool,VkCommandPoolResetFlags) { return VK_SUCCESS; }
VkResult VKAPI_CALL begin_cmd(VkCommandBuffer,const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
VkResult VKAPI_CALL end_cmd(VkCommandBuffer) { return VK_SUCCESS; }
VkResult VKAPI_CALL reset_fences(VkDevice,uint32_t,const VkFence*) { return VK_SUCCESS; }
VkResult VKAPI_CALL fence_status(VkDevice,VkFence) { fence_observed.store(true); return ready_result; }
VkResult VKAPI_CALL wait_fences(VkDevice,uint32_t,const VkFence*,VkBool32,uint64_t) { ++waits; return VK_SUCCESS; }
VkResult VKAPI_CALL submit(VkQueue,uint32_t,const VkSubmitInfo *info,VkFence) {
    ++submits;
    assert(info->waitSemaphoreCount==1 && info->pWaitSemaphores[0]==original);
    assert(info->signalSemaphoreCount==1);
    return submit_result;
}
VkResult VKAPI_CALL present(VkQueue,const VkPresentInfoKHR *info) {
    ++presents;
    assert(info->waitSemaphoreCount==1);
    last_wait=info->pWaitSemaphores[0];
    return VK_SUCCESS;
}
void VKAPI_CALL barrier(VkCommandBuffer,VkPipelineStageFlags,VkPipelineStageFlags,VkDependencyFlags,
    uint32_t,const VkMemoryBarrier*,uint32_t,const VkBufferMemoryBarrier*,uint32_t,const VkImageMemoryBarrier*) {}
void VKAPI_CALL copy(VkCommandBuffer,VkImage,VkImageLayout,VkBuffer,uint32_t,const VkBufferImageCopy*) {}
VkResult VKAPI_CALL idle(VkDevice) { drained=true; return VK_SUCCESS; }
void VKAPI_CALL destroy_sem(VkDevice,VkSemaphore,const VkAllocationCallbacks*) { assert(drained); ++destroyed_semaphores; }
void VKAPI_CALL destroy_swapchain(VkDevice,VkSwapchainKHR,const VkAllocationCallbacks*) {}
void VKAPI_CALL destroy_device(VkDevice,const VkAllocationCallbacks*) { assert(drained); }

// --- brightness mocks: every create returns a fresh fake handle ---
uintptr_t next_handle=5000;
template<class T> VkResult make(T *out) { *out=reinterpret_cast<T>(++next_handle); return VK_SUCCESS; }
struct BrightSubmit { std::vector<VkSemaphore> waits; VkSemaphore signal; };
std::vector<BrightSubmit> bright_submits;
VkResult bright_submit_result=VK_SUCCESS;
VkResult VKAPI_CALL submit2(VkQueue,uint32_t,const VkSubmitInfo *info,VkFence) {
    bright_submits.push_back({{info->pWaitSemaphores,info->pWaitSemaphores+info->waitSemaphoreCount},info->pSignalSemaphores[0]});
    // Capture submits signal present_semaphores (200..); the effect signals its own semaphore.
    const bool effect=reinterpret_cast<uintptr_t>(info->pSignalSemaphores[0])>5000;
    return effect ? bright_submit_result : VK_SUCCESS;
}
}

namespace {
void install_bright_mocks(DeviceData *d) {
    d->vkCreateImage=[](VkDevice,const VkImageCreateInfo*,const VkAllocationCallbacks*,VkImage *o){ return make(o); };
    d->vkDestroyImage=[](VkDevice,VkImage,const VkAllocationCallbacks*){};
    d->vkGetImageMemoryRequirements=[](VkDevice,VkImage,VkMemoryRequirements *r){ *r={}; r->size=64; r->memoryTypeBits=1; };
    d->vkBindImageMemory=[](VkDevice,VkImage,VkDeviceMemory,VkDeviceSize){ return VK_SUCCESS; };
    d->vkAllocateMemory=[](VkDevice,const VkMemoryAllocateInfo*,const VkAllocationCallbacks*,VkDeviceMemory *o){ return make(o); };
    d->vkFreeMemory=[](VkDevice,VkDeviceMemory,const VkAllocationCallbacks*){};
    d->vkCreateImageView=[](VkDevice,const VkImageViewCreateInfo*,const VkAllocationCallbacks*,VkImageView *o){ return make(o); };
    d->vkDestroyImageView=[](VkDevice,VkImageView,const VkAllocationCallbacks*){};
    d->vkCreateSampler=[](VkDevice,const VkSamplerCreateInfo*,const VkAllocationCallbacks*,VkSampler *o){ return make(o); };
    d->vkDestroySampler=[](VkDevice,VkSampler,const VkAllocationCallbacks*){};
    d->vkCreateRenderPass=[](VkDevice,const VkRenderPassCreateInfo*,const VkAllocationCallbacks*,VkRenderPass *o){ return make(o); };
    d->vkDestroyRenderPass=[](VkDevice,VkRenderPass,const VkAllocationCallbacks*){};
    d->vkCreateFramebuffer=[](VkDevice,const VkFramebufferCreateInfo*,const VkAllocationCallbacks*,VkFramebuffer *o){ return make(o); };
    d->vkDestroyFramebuffer=[](VkDevice,VkFramebuffer,const VkAllocationCallbacks*){};
    d->vkCreateShaderModule=[](VkDevice,const VkShaderModuleCreateInfo*,const VkAllocationCallbacks*,VkShaderModule *o){ return make(o); };
    d->vkDestroyShaderModule=[](VkDevice,VkShaderModule,const VkAllocationCallbacks*){};
    d->vkCreateDescriptorSetLayout=[](VkDevice,const VkDescriptorSetLayoutCreateInfo*,const VkAllocationCallbacks*,VkDescriptorSetLayout *o){ return make(o); };
    d->vkDestroyDescriptorSetLayout=[](VkDevice,VkDescriptorSetLayout,const VkAllocationCallbacks*){};
    d->vkCreatePipelineLayout=[](VkDevice,const VkPipelineLayoutCreateInfo*,const VkAllocationCallbacks*,VkPipelineLayout *o){ return make(o); };
    d->vkDestroyPipelineLayout=[](VkDevice,VkPipelineLayout,const VkAllocationCallbacks*){};
    d->vkCreateGraphicsPipelines=[](VkDevice,VkPipelineCache,uint32_t,const VkGraphicsPipelineCreateInfo*,const VkAllocationCallbacks*,VkPipeline *o){ return make(o); };
    d->vkDestroyPipeline=[](VkDevice,VkPipeline,const VkAllocationCallbacks*){};
    d->vkCreateDescriptorPool=[](VkDevice,const VkDescriptorPoolCreateInfo*,const VkAllocationCallbacks*,VkDescriptorPool *o){ return make(o); };
    d->vkDestroyDescriptorPool=[](VkDevice,VkDescriptorPool,const VkAllocationCallbacks*){};
    d->vkAllocateDescriptorSets=[](VkDevice,const VkDescriptorSetAllocateInfo*,VkDescriptorSet *o){ return make(o); };
    d->vkUpdateDescriptorSets=[](VkDevice,uint32_t,const VkWriteDescriptorSet*,uint32_t,const VkCopyDescriptorSet*){};
    d->vkCreateCommandPool=[](VkDevice,const VkCommandPoolCreateInfo*,const VkAllocationCallbacks*,VkCommandPool *o){ return make(o); };
    d->vkDestroyCommandPool=[](VkDevice,VkCommandPool,const VkAllocationCallbacks*){};
    d->vkAllocateCommandBuffers=[](VkDevice,const VkCommandBufferAllocateInfo*,VkCommandBuffer *o){ return make(o); };
    d->vkCreateFence=[](VkDevice,const VkFenceCreateInfo*,const VkAllocationCallbacks*,VkFence *o){ return make(o); };
    d->vkDestroyFence=[](VkDevice,VkFence,const VkAllocationCallbacks*){};
    d->vkCreateSemaphore=[](VkDevice,const VkSemaphoreCreateInfo*,const VkAllocationCallbacks*,VkSemaphore *o){ return make(o); };
    d->vkCmdCopyImage=[](VkCommandBuffer,VkImage,VkImageLayout,VkImage,VkImageLayout,uint32_t,const VkImageCopy*){};
    d->vkCmdBeginRenderPass=[](VkCommandBuffer,const VkRenderPassBeginInfo*,VkSubpassContents){};
    d->vkCmdEndRenderPass=[](VkCommandBuffer){};
    d->vkCmdBindPipeline=[](VkCommandBuffer,VkPipelineBindPoint,VkPipeline){};
    d->vkCmdBindDescriptorSets=[](VkCommandBuffer,VkPipelineBindPoint,VkPipelineLayout,uint32_t,uint32_t,const VkDescriptorSet*,uint32_t,const uint32_t*){};
    d->vkCmdPushConstants=[](VkCommandBuffer,VkPipelineLayout,VkShaderStageFlags,uint32_t,uint32_t,const void*){};
    d->vkCmdSetViewport=[](VkCommandBuffer,uint32_t,uint32_t,const VkViewport*){};
    d->vkCmdSetScissor=[](VkCommandBuffer,uint32_t,uint32_t,const VkRect2D*){};
    d->vkCmdDraw=[](VkCommandBuffer,uint32_t,uint32_t,uint32_t,uint32_t){};
    d->vkQueueSubmit=submit2;
}
}

int main() {
    PresentTiming timing;
    for (uint64_t at : {1000000ULL,2000000ULL,18000000ULL,318000000ULL}) timing.observe(at);
    assert(timing.samples==3 && timing.sum_ns==317000000ULL && timing.max_ns==300000000ULL);
    assert(timing.histogram[1]==1 && timing.histogram[16]==1 && timing.histogram[255]==1);
    alignas(8) uint64_t control[2]={static_cast<uint64_t>(VD_LAYER_SHM_MAGIC)|(uint64_t(VD_LAYER_SHM_VERSION)<<32),get_time_ns()+2000000000ULL};
    g_control_mapping=control;
    g_capture_enabled=true;
    g_capture_interval_ns=0;
    VdLayerShmHeader header{};
    std::array<uint8_t,VD_CAPTURE_MAX_BYTES> pixels{};
    g_shm_header=&header;
    g_shm_pixels=pixels.data();
    auto device=std::make_unique<DeviceData>();
    device->device=reinterpret_cast<VkDevice>(1);
    device->vkResetCommandPool=reset_pool;
    device->vkBeginCommandBuffer=begin_cmd;
    device->vkEndCommandBuffer=end_cmd;
    device->vkResetFences=reset_fences;
    device->vkGetFenceStatus=fence_status;
    device->vkWaitForFences=wait_fences;
    device->vkQueueSubmit=submit;
    device->vkQueuePresentKHR=present;
    device->vkCmdPipelineBarrier=barrier;
    device->vkCmdCopyImageToBuffer=copy;
    device->vkDeviceWaitIdle=idle;
    device->vkDestroySemaphore=destroy_sem;
    device->vkDestroySwapchainKHR=destroy_swapchain;
    device->vkDestroyDevice=destroy_device;
    DeviceData *d=device.get();
    g_primary_device=d;
    g_devices[reinterpret_cast<void*>(d->device)]=std::move(device);
    auto queue=std::make_unique<QueueData>();
    queue->device_data=d;
    QueueData *q=queue.get();
    VkQueue queue_handle=reinterpret_cast<VkQueue>(2);
    std::array<uint8_t,VD_CAPTURE_MAX_BYTES> mapped{};
    mapped.fill(73);
    for (auto &c:q->captures) {
        c.initialized=true;
        c.memory_coherent=true;
        c.mapped=mapped.data();
    }
    g_queues[reinterpret_cast<void*>(queue_handle)]=std::move(queue);
    SwapchainInfo sc;
    sc.device=d->device;
    sc.swapchain=reinterpret_cast<VkSwapchainKHR>(3);
    sc.capture_supported=true;
    sc.generation=1;
    sc.width=320;
    sc.height=240;
    sc.format=VK_FORMAT_R8G8B8A8_UNORM;
    for (int i=0;i<3;++i) {
        sc.images.push_back(reinterpret_cast<VkImage>(100+i));
        sc.present_semaphores.push_back(reinterpret_cast<VkSemaphore>(200+i));
    }
    g_swapchains[sc.swapchain]=sc;
    uint32_t image=0;
    VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    info.swapchainCount=1;
    info.pSwapchains=&sc.swapchain;
    info.pImageIndices=&image;
    info.waitSemaphoreCount=1;
    info.pWaitSemaphores=&original;
    for (image=0;image<3;++image) {
        assert(vd_vkQueuePresentKHR(queue_handle,&info)==VK_SUCCESS);
        assert(last_wait==sc.present_semaphores[image]);
    }
    assert(submits==3 && presents==3 && waits==0);
    image=0;
    assert(vd_vkQueuePresentKHR(queue_handle,&info)==VK_SUCCESS);
    assert(submits==3 && last_wait==original && waits==0);

    ready_result=VK_SUCCESS;
    std::unique_lock<std::mutex> publication_block(g_shm_mutex);
    std::thread worker(capture_worker,q);
    while (!fence_observed.load()) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    assert(g_state_mutex.try_lock());
    g_state_mutex.unlock();
    // A blocked pixel publication must never block the game's presentation.
    assert(vd_vkQueuePresentKHR(queue_handle,&info)==VK_SUCCESS);
    assert(last_wait==original && submits==3);
    publication_block.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    q->stopping.store(true);
    worker.join();
    assert(header.source_present_id==3 && header.swapchain_generation==1);
    assert(header.roi_capture_timestamp_ns<=header.copy_completed_ns);
    assert(header.copy_completed_ns<=header.published_ns);
    assert(pixels[0]==73);
    assert(waits==0);
    for (auto &c:q->captures) assert(!c.pending);

    // A free staging slot does not determine the presentation semaphore.
    assert(vd_vkQueuePresentKHR(queue_handle,&info)==VK_SUCCESS);
    assert(last_wait==sc.present_semaphores[0] && submits==4);
    control[1]=0;
    assert(vd_vkQueuePresentKHR(queue_handle,&info)==VK_SUCCESS);
    assert(last_wait==original && submits==4);
    control[1]=get_time_ns()+2000000000ULL;
    submit_result=VK_ERROR_DEVICE_LOST;
    assert(vd_vkQueuePresentKHR(queue_handle,&info)==VK_SUCCESS);
    assert(last_wait==original && submits==5);
    submit_result=VK_SUCCESS;
    ready_result=VK_ERROR_DEVICE_LOST;
    q->stopping.store(false);
    std::thread lost_worker(capture_worker,q);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    q->stopping.store(true);
    lost_worker.join();
    assert(!q->captures[0].pending && q->captures[0].permanently_disabled);
    assert(header.source_present_id==3);

    // Old semaphores survive recreation; pending copies drain at teardown only.
    q->captures[2].pending=true;
    q->captures[2].swapchain=sc.swapchain;
    vd_vkDestroySwapchainKHR(d->device,sc.swapchain,nullptr);
    assert(waits==1 && destroyed_semaphores==0);
    assert(g_swapchains.empty());
    assert(!sc.lifetime->load());
    CaptureResources retired=q->captures[2];
    retired.source_present_id=999;
    publish_roi(retired,get_time_ns());
    assert(header.source_present_id==3 && pixels[0]==73);
    SwapchainInfo replacement;
    assert(replacement.lifetime->load() && replacement.lifetime!=sc.lifetime);
    retired.lifetime=replacement.lifetime;
    retired.generation=2;
    publish_roi(retired,get_time_ns());
    assert(header.source_present_id==999 && header.swapchain_generation==2);
    assert(d->retired_present_semaphores.size()==3);
    VkDevice handle=d->device;
    vd_vkDestroyDevice(handle,nullptr);
    assert(drained && destroyed_semaphores==3 && g_queues.empty());

    // --- brightness effect: config/log parsing ---
    BrightSettings cfg=parse_bright_conf("mode = on # x\nstrength=9\nlift=-1\ndetail=nan\nuniverses=1, 22 x333\n");
    assert(cfg.mode==BrightSettings::kOn && cfg.strength==6.0f && cfg.lift==0.0f && cfg.detail==0.6f);
    assert((cfg.universes==std::vector<uint64_t>{1,22,333}));
    assert(parse_bright_conf("mode=garbage").mode==BrightSettings::kAuto);
    const std::vector<uint64_t> vd{6739698191ULL};
    assert(bright_log_line("x [FLog::GameJoinLoadTime] Report game_join_loadtime: placeid:93978595733734, universeid:6739698191, ugcgame",vd,false));
    assert(!bright_log_line("x [FLog::GameJoinLoadTime] Report game_join_loadtime: placeid:1, universeid:10774406718, ugc",vd,true));
    assert(!bright_log_line("x [FLog::SingleSurfaceApp] setStage: (stage:LuaApp)",vd,true));
    assert(bright_log_line("x [FLog::Network] unrelated",vd,true));

    // --- brightness effect: present chain app -> capture -> effect -> present ---
    g_bright_armed=true;
    g_bright_active.store(true);
    drained=false;
    auto dev2=std::make_unique<DeviceData>();
    dev2->device=reinterpret_cast<VkDevice>(11);
    dev2->vkResetCommandPool=reset_pool;
    dev2->vkBeginCommandBuffer=begin_cmd;
    dev2->vkEndCommandBuffer=end_cmd;
    dev2->vkResetFences=reset_fences;
    dev2->vkGetFenceStatus=fence_status;
    dev2->vkWaitForFences=wait_fences;
    dev2->vkQueuePresentKHR=present;
    dev2->vkCmdPipelineBarrier=barrier;
    dev2->vkCmdCopyImageToBuffer=copy;
    dev2->vkDeviceWaitIdle=idle;
    dev2->vkDestroySemaphore=destroy_sem;
    dev2->vkDestroySwapchainKHR=destroy_swapchain;
    dev2->vkDestroyDevice=destroy_device;
    install_bright_mocks(dev2.get());
    InstanceData instance2{};
    instance2.vkGetPhysicalDeviceMemoryProperties=[](VkPhysicalDevice,VkPhysicalDeviceMemoryProperties *m){
        *m={}; m->memoryTypeCount=1; m->memoryTypes[0].propertyFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT; };
    instance2.vkGetPhysicalDeviceQueueFamilyProperties=[](VkPhysicalDevice,uint32_t *n,VkQueueFamilyProperties *f){
        *n=1; if (f) { *f={}; f->queueFlags=VK_QUEUE_GRAPHICS_BIT; } };
    dev2->instance_data=&instance2;
    DeviceData *d2=dev2.get();
    g_primary_device=d2;
    g_devices[reinterpret_cast<void*>(d2->device)]=std::move(dev2);
    auto queue2=std::make_unique<QueueData>();
    queue2->device_data=d2;
    QueueData *q2=queue2.get();
    for (auto &c:q2->captures) { c.initialized=true; c.memory_coherent=true; c.mapped=mapped.data(); }
    VkQueue queue2_handle=reinterpret_cast<VkQueue>(12);
    g_queues[reinterpret_cast<void*>(queue2_handle)]=std::move(queue2);
    SwapchainInfo sc2=sc;
    sc2.device=d2->device;
    sc2.swapchain=reinterpret_cast<VkSwapchainKHR>(13);
    sc2.bright_supported=true;
    sc2.lifetime=std::make_shared<std::atomic<bool>>(true);
    g_swapchains[sc2.swapchain]=sc2;
    info.pSwapchains=&sc2.swapchain;
    image=0;
    // Capture copies the original frame first, the effect waits on the capture and present on the effect.
    assert(vd_vkQueuePresentKHR(queue2_handle,&info)==VK_SUCCESS);
    assert(bright_submits.size()==2);
    assert(bright_submits[0].waits==std::vector<VkSemaphore>{original} && bright_submits[0].signal==sc2.present_semaphores[0]);
    assert(bright_submits[1].waits==std::vector<VkSemaphore>{sc2.present_semaphores[0]});
    assert(last_wait==bright_submits[1].signal && last_wait!=sc2.present_semaphores[0]);
    // Effect off (outside VD): no extra submit, capture unchanged.
    g_bright_active.store(false);
    image=1;
    assert(vd_vkQueuePresentKHR(queue2_handle,&info)==VK_SUCCESS);
    assert(bright_submits.size()==3 && last_wait==sc2.present_semaphores[1]);
    // No capture consumer: the effect waits on the app's own semaphores.
    g_bright_active.store(true);
    control[1]=0;
    image=2;
    assert(vd_vkQueuePresentKHR(queue2_handle,&info)==VK_SUCCESS);
    assert(bright_submits.size()==4 && bright_submits[3].waits==std::vector<VkSemaphore>{original});
    assert(last_wait==bright_submits[3].signal);
    // Effect submit failure fails open: present waits on the capture and the effect stays off.
    control[1]=get_time_ns()+2000000000ULL;
    bright_submit_result=VK_ERROR_DEVICE_LOST;
    q2->captures[0].pending=false;
    image=0;
    const int fence_waits_before=waits;
    assert(vd_vkQueuePresentKHR(queue2_handle,&info)==VK_SUCCESS);
    assert(waits==fence_waits_before+1);  // frame 0 effect was in flight
    assert(bright_submits.size()==6 && last_wait==sc2.present_semaphores[0]);
    q2->captures[0].pending=false;
    assert(vd_vkQueuePresentKHR(queue2_handle,&info)==VK_SUCCESS);
    assert(bright_submits.size()==7 && last_wait==sc2.present_semaphores[0]);
    // Teardown: effect semaphores outlive the swapchain like the capture ones.
    const size_t retired_before=d2->retired_present_semaphores.size();
    vd_vkDestroySwapchainKHR(d2->device,sc2.swapchain,nullptr);
    assert(d2->retired_present_semaphores.size()==retired_before+3+3);
    const int destroyed_before=destroyed_semaphores;
    vd_vkDestroyDevice(d2->device,nullptr);
    assert(destroyed_semaphores==destroyed_before+6 && g_queues.empty());
    g_bright_armed=false;

    g_shm_header=nullptr;
    g_shm_pixels=nullptr;
    puts("async capture contract: PASS (saturation, semaphore routing, lease expiry, timestamps, device loss, teardown, brightness chain)");
}
