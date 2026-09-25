#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DCCCLEARHELPER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DCCCLEARHELPER_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <span>

namespace Libs::Graphics {

class CommandScheduler;
class StreamBuffer;
struct GraphicContext;

// Applies DCC fast clears whose metadata the GPU wrote without reading that metadata back: a scan
// pass decides per slice whether the metadata holds a clear code, and an indirect fill pass writes
// the matching color only where it does.
class DccClearHelper final {
public:
	// The clear codes the scan recognizes, in the order of Request::colors.
	static constexpr std::array<uint8_t, 5> Codes {0x00, 0x20, 0x40, 0x80, 0xc0};

	struct Request {
		vk::Buffer metadata        = nullptr;
		uint64_t   metadata_offset = 0; // first slice; each following slice is slice_size later
		uint64_t   slice_size      = 0;
		uint32_t   width           = 0;
		uint32_t   height          = 0;
		// Only codes whose bit is set can match; colors are what the clear writes for each code.
		uint32_t                            valid_mask = 0;
		std::array<std::array<float, 4>, 5> colors {};
		// Overwrite a matched slice with the "not cleared" code, as the CPU path does.
		bool consume = true;
	};

	DccClearHelper(GraphicContext& graphics, CommandScheduler& scheduler, StreamBuffer& stream);
	~DccClearHelper();
	KYTY_CLASS_NO_COPY(DccClearHelper);

	// Records the scan and fill for one slice per view. Every view is a single-layer 2D storage
	// view in GENERAL layout; the caller transitions the image and commits the write.
	void Record(const Request& request, std::span<const vk::ImageView> views);

private:
	struct Push {
		std::array<std::array<float, 4>, 5> colors;
		uint32_t                            valid_mask;
		uint32_t                            meta_base;
		uint32_t                            slice_dwords;
		uint32_t                            consume;
		uint32_t                            width;
		uint32_t                            height;
		uint32_t                            record;
	};

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	StreamBuffer&           m_stream;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_scan              = nullptr;
	vk::Pipeline            m_fill              = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DCCCLEARHELPER_H_
