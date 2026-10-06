#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHTHREADARGS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHTHREADARGS_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Turns the arguments of a guest indirect dispatch sized in threads into a
// VkDispatchIndirectCommand on the GPU. A one-thread compute pass rounds the thread counts up to
// workgroups, so the CPU never reads arguments a shader may have written: reading them back made
// the GPU thread submit and wait for the GPU once per such dispatch.
class DispatchThreadArgs {
public:
	// Guest arguments: three thread counts. Output: three workgroup counts.
	static constexpr uint64_t ArgumentsSize = 3 * sizeof(uint32_t);
	static constexpr uint64_t OutputSize    = 3 * sizeof(uint32_t);

	explicit DispatchThreadArgs(GraphicContext& graphics);
	~DispatchThreadArgs();
	KYTY_CLASS_NO_COPY(DispatchThreadArgs);

	// Records the conversion and the barriers that order it after earlier GPU writes of the
	// arguments and before the indirect dispatch. `group_threads` is the dispatched shader's
	// workgroup size per axis, each at least 1. Must be recorded outside a rendering instance.
	void Record(vk::CommandBuffer command, const Buffer& arguments, uint64_t arguments_offset,
	            const Buffer& output, uint64_t output_offset,
	            const std::array<uint32_t, 3>& group_threads);

private:
	GraphicContext&         m_graphics;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_pipeline          = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHTHREADARGS_H_
