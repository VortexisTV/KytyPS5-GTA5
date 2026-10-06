#include "graphics/host_gpu/renderer/image/blitHelper.h"

#include "common/assert.h"
#include "gpu_blit_shaders/gpu_blit_color_to_ms_depth_spv.h"
#include "gpu_blit_shaders/gpu_blit_depth_to_color_0_spv.h"
#include "gpu_blit_shaders/gpu_blit_depth_to_color_1_spv.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/gpu_blit_stencil_to_color_0_spv.h"
#include "gpu_blit_shaders/gpu_blit_stencil_to_color_1_spv.h"
#include "graphics/host_gpu/gpuCrashDiagnostics.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/renderTarget.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

BlitHelper::BlitHelper(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler) {
	vk::DescriptorSetLayoutBinding texture_binding {};
	texture_binding.binding         = 0;
	texture_binding.descriptorType  = vk::DescriptorType::eSampledImage;
	texture_binding.descriptorCount = 1;
	texture_binding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

	vk::DescriptorSetLayoutCreateInfo descriptor_info {};
	descriptor_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor_info.bindingCount = 1;
	descriptor_info.pBindings    = &texture_binding;
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr,
	                                                                 &m_descriptor_layout),
	                     "create BlitHelper descriptor layout");

	vk::PipelineLayoutCreateInfo layout_info {};
	const vk::PushConstantRange  copy_control {vk::ShaderStageFlagBits::eFragment, 0,
	                                           sizeof(uint32_t)};
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &copy_control;
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &m_descriptor_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_pipeline_layout),
	    "create BlitHelper pipeline layout");

	m_vertex_shader   = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, m_graphics.device);
	m_fragment_shader = CompileSPV(GPU_BLIT_COLOR_TO_MS_DEPTH_SPV, m_graphics.device);
	m_copy_shaders    = {CompileSPV(GPU_BLIT_DEPTH_TO_COLOR_0_SPV, m_graphics.device),
	                     CompileSPV(GPU_BLIT_DEPTH_TO_COLOR_1_SPV, m_graphics.device),
	                     CompileSPV(GPU_BLIT_STENCIL_TO_COLOR_0_SPV, m_graphics.device),
	                     CompileSPV(GPU_BLIT_STENCIL_TO_COLOR_1_SPV, m_graphics.device)};
}

BlitHelper::~BlitHelper() {
	for (const auto shader: m_copy_shaders) {
		m_graphics.device.destroyShaderModule(shader, nullptr);
	}
	for (const auto& pipeline: m_pipelines) {
		m_graphics.device.destroyPipeline(pipeline.handle, nullptr);
	}
	if (m_fragment_shader != nullptr) {
		m_graphics.device.destroyShaderModule(m_fragment_shader, nullptr);
	}
	if (m_vertex_shader != nullptr) {
		m_graphics.device.destroyShaderModule(m_vertex_shader, nullptr);
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

vk::Pipeline BlitHelper::GetPipeline(PipelineKey key) {
	const auto cached = std::ranges::find(m_pipelines, key, &Pipeline::key);
	if (cached != m_pipelines.end()) {
		return cached->handle;
	}

	const auto samples = vulkan_sample_count(key.samples);
	EXIT_IF(samples == vk::SampleCountFlagBits {} || key.format == vk::Format::eUndefined);

	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
	stages[0].module = m_vertex_shader;
	stages[0].pName  = "main";
	stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
	stages[1].module =
	    key.copy_shader == 0 ? m_fragment_shader : m_copy_shaders.at(key.copy_shader - 1);
	stages[1].pName = "main";

	vk::PipelineVertexInputStateCreateInfo   vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
	vk::PipelineViewportStateCreateInfo viewport {};
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;
	vk::PipelineRasterizationStateCreateInfo rasterization {};
	rasterization.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisample {};
	multisample.rasterizationSamples = samples;
	vk::PipelineDepthStencilStateCreateInfo depth {};
	depth.depthTestEnable  = key.copy_shader == 0;
	depth.depthWriteEnable = key.copy_shader == 0;
	depth.depthCompareOp   = vk::CompareOp::eAlways;
	vk::PipelineColorBlendStateCreateInfo color_blend {};
	vk::PipelineColorBlendAttachmentState color_attachment {};
	color_attachment.colorWriteMask = vk::ColorComponentFlagBits::eR;
	if (key.copy_shader != 0) {
		color_blend.attachmentCount = 1;
		color_blend.pAttachments    = &color_attachment;
	}
	const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic {};
	dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic.pDynamicStates    = dynamic_states.data();

	vk::PipelineRenderingCreateInfo rendering {};
	if (key.copy_shader == 0) {
		rendering.depthAttachmentFormat = key.format;
	} else {
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = &key.format;
	}

	vk::GraphicsPipelineCreateInfo create {};
	create.pNext               = &rendering;
	create.stageCount          = static_cast<uint32_t>(stages.size());
	create.pStages             = stages.data();
	create.pVertexInputState   = &vertex_input;
	create.pInputAssemblyState = &input_assembly;
	create.pViewportState      = &viewport;
	create.pRasterizationState = &rasterization;
	create.pMultisampleState   = &multisample;
	create.pDepthStencilState  = &depth;
	create.pColorBlendState    = &color_blend;
	create.pDynamicState       = &dynamic;
	create.layout              = m_pipeline_layout;

	vk::Pipeline pipeline = nullptr;
	RequireVulkanSuccess(
	    m_graphics.device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipeline),
	    "create image blit pipeline");
	m_pipelines.push_back({key, pipeline});
	return pipeline;
}

void BlitHelper::ReinterpretColorAsMsDepth(Image& source, Image& destination) {
	const auto& source_info      = source.info;
	const auto& destination_info = destination.info;
	EXIT_IF(DepthAspectTransferFormat(source_info.pixel_format) != vk::Format::eUndefined ||
	        DepthAspectTransferFormat(destination_info.pixel_format) == vk::Format::eUndefined ||
	        source_info.samples != 1 || destination_info.samples <= 1 ||
	        destination_info.samples > 4 || source.backing.image_type != vk::ImageType::e2D ||
	        destination.backing.image_type != vk::ImageType::e2D ||
	        source_info.extent.width != destination_info.extent.width ||
	        source_info.extent.height != destination_info.extent.height ||
	        source_info.extent.depth != 1 || destination_info.extent.depth != 1 ||
	        source.backing.image == nullptr || destination.backing.image == nullptr);
	m_scheduler.EndRendering();

	ImageViewInfo source_view_info {};
	source_view_info.format = source_info.pixel_format;
	source_view_info.type   = vk::ImageViewType::e2D;
	source_view_info.aspect = vk::ImageAspectFlagBits::eColor;
	source_view_info.usage  = vk::ImageUsageFlagBits::eSampled;
	const auto source_view  = source.FindView(source_view_info);

	ImageViewInfo destination_view_info {};
	destination_view_info.format = destination_info.pixel_format;
	destination_view_info.type   = vk::ImageViewType::e2D;
	destination_view_info.aspect = vk::ImageAspectFlagBits::eDepth;
	destination_view_info.usage  = vk::ImageUsageFlagBits::eDepthStencilAttachment;
	const auto destination_view  = destination.FindView(destination_view_info);

	auto& command_buffer = m_scheduler.Current();
	auto  command        = command_buffer.Handle();
	GpuZones::Mark(command, DrainStats::Zone::Blit);
	source.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {},
	               command);
	destination.Transit(ColorToMsDepthLayout, vk::AccessFlagBits2::eDepthStencilAttachmentWrite, {},
	                    command);

	vk::RenderingAttachmentInfo depth_attachment {};
	depth_attachment.imageView               = destination_view;
	depth_attachment.imageLayout             = ColorToMsDepthLayout;
	depth_attachment.loadOp                  = vk::AttachmentLoadOp::eClear;
	depth_attachment.storeOp                 = vk::AttachmentStoreOp::eStore;
	depth_attachment.clearValue.depthStencil = {0.0f, 0};

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent = {destination_info.extent.width, destination_info.extent.height};
	rendering.layerCount        = 1;
	rendering.pDepthAttachment  = &depth_attachment;
	command.beginRendering(&rendering);

	vk::DescriptorImageInfo descriptor_image {};
	descriptor_image.imageView   = source_view;
	descriptor_image.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::WriteDescriptorSet descriptor_write {};
	descriptor_write.dstBinding      = 0;
	descriptor_write.descriptorCount = 1;
	descriptor_write.descriptorType  = vk::DescriptorType::eSampledImage;
	descriptor_write.pImageInfo      = &descriptor_image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_pipeline_layout, 0, 1,
	                             &descriptor_write);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics,
	                     GetPipeline({destination_info.samples, destination_info.pixel_format}));

	const vk::Viewport viewport {0.0f,
	                             0.0f,
	                             static_cast<float>(destination_info.extent.width),
	                             static_cast<float>(destination_info.extent.height),
	                             0.0f,
	                             1.0f};
	const vk::Rect2D   scissor {{0, 0},
	                            {destination_info.extent.width, destination_info.extent.height}};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	MarkGpuCheckpoint(m_graphics, command, GpuCheckpointKind::Blit, 0, {},
	                  destination_info.extent.width, destination_info.extent.height);
	command.draw(3, 1, 0, 0);
	command.endRendering();
	// The blit replaced the pipeline and dynamic state the game's draws recorded.
	command_buffer.InvalidateGraphicsState();
}

void BlitHelper::CopyDepthStencilToColor(Image& source, Image& destination,
                                         ImageViewInfo source_view, ImageViewInfo destination_view,
                                         vk::Rect2D scissor, uint32_t sample, bool stencil) {
	EXIT_IF(source.backing.image == destination.backing.image ||
	        source.backing.image_type != vk::ImageType::e2D ||
	        destination.backing.image_type != vk::ImageType::e2D ||
	        destination.backing.samples != 1 || sample >= source.backing.samples ||
	        source_view.layer_count != destination_view.layer_count);
	if (scissor.extent.width == 0 || scissor.extent.height == 0) {
		return;
	}
	m_scheduler.EndRendering();
	auto&      command_buffer = m_scheduler.Current();
	const auto command        = command_buffer.Handle();
	GpuZones::Mark(command, DrainStats::Zone::Blit);
	const ImageSubresourceRange source_range {source_view.base_level, 1, source_view.base_layer,
	                                          source_view.layer_count};
	const ImageSubresourceRange destination_range {
	    destination_view.base_level, 1, destination_view.base_layer, destination_view.layer_count};
	source.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
	               source_range, command);
	destination.Transit(vk::ImageLayout::eColorAttachmentOptimal,
	                    vk::AccessFlagBits2::eColorAttachmentRead |
	                        vk::AccessFlagBits2::eColorAttachmentWrite,
	                    destination_range, command);
	source_view.type = destination_view.type = vk::ImageViewType::e2D;
	source_view.layer_count = destination_view.layer_count = 1;
	source_view.aspect =
	    stencil ? vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlagBits::eDepth;
	source_view.usage      = vk::ImageUsageFlagBits::eSampled;
	destination_view.usage = vk::ImageUsageFlagBits::eColorAttachment;
	const vk::Viewport viewport {
	    0.0f,
	    0.0f,
	    static_cast<float>(
	        std::max(destination.backing.extent.width >> destination_view.base_level, 1u)),
	    static_cast<float>(
	        std::max(destination.backing.extent.height >> destination_view.base_level, 1u)),
	    0.0f,
	    1.0f};
	const uint32_t shader   = 1u + (stencil ? 2u : 0u) + (source.backing.samples > 1 ? 1u : 0u);
	const auto     pipeline = GetPipeline({1, destination_view.format, shader});
	for (uint32_t layer = 0; layer < source_range.layer_count; ++layer) {
		source_view.base_layer      = source_range.base_layer + layer;
		destination_view.base_layer = destination_range.base_layer + layer;
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = destination.FindView(destination_view);
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		vk::RenderingInfo rendering {};
		rendering.renderArea           = scissor;
		rendering.layerCount           = 1;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		command.beginRendering(rendering);
		const vk::DescriptorImageInfo image_info {nullptr, source.FindView(source_view),
		                                          vk::ImageLayout::eShaderReadOnlyOptimal};
		vk::WriteDescriptorSet        write {};
		write.dstBinding      = 0;
		write.descriptorCount = 1;
		write.descriptorType  = vk::DescriptorType::eSampledImage;
		write.pImageInfo      = &image_info;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_pipeline_layout, 0, write);
		command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0,
		                      sizeof(sample), &sample);
		command.setViewport(0, viewport);
		command.setScissor(0, scissor);
		MarkGpuCheckpoint(m_graphics, command, GpuCheckpointKind::Blit, 0, {}, scissor.extent.width,
		                  scissor.extent.height);
		command.draw(3, 1, 0, 0);
		command.endRendering();
	}
	command_buffer.InvalidateGraphicsState();
}

} // namespace Libs::Graphics
