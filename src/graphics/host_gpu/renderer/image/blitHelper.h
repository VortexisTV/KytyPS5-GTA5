#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <compare>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
class Image;
struct GraphicContext;
struct ImageViewInfo;

class BlitHelper final {
public:
	inline static constexpr auto ColorToMsDepthLayout =
	    vk::ImageLayout::eDepthStencilAttachmentOptimal;

	BlitHelper(GraphicContext& graphics, CommandScheduler& scheduler);
	~BlitHelper();
	KYTY_CLASS_NO_COPY(BlitHelper);

	void ReinterpretColorAsMsDepth(Image& source, Image& destination);
	void CopyDepthStencilToColor(Image& source, Image& destination, ImageViewInfo source_view,
	                             ImageViewInfo destination_view, vk::Rect2D scissor,
	                             uint32_t sample, bool stencil);

private:
	struct PipelineKey {
		uint32_t   samples = 1;
		vk::Format format  = vk::Format::eUndefined;
		// Zero is the existing color-to-depth blit; 1..4 select depth/stencil copy shaders.
		uint32_t copy_shader = 0;

		auto operator<=>(const PipelineKey&) const = default;
	};

	struct Pipeline {
		PipelineKey  key;
		vk::Pipeline handle = nullptr;
	};

	[[nodiscard]] vk::Pipeline GetPipeline(PipelineKey key);

	GraphicContext&                 m_graphics;
	CommandScheduler&               m_scheduler;
	vk::DescriptorSetLayout         m_descriptor_layout = nullptr;
	vk::PipelineLayout              m_pipeline_layout   = nullptr;
	vk::ShaderModule                m_vertex_shader     = nullptr;
	vk::ShaderModule                m_fragment_shader   = nullptr;
	std::array<vk::ShaderModule, 4> m_copy_shaders {};
	std::vector<Pipeline>           m_pipelines;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_
