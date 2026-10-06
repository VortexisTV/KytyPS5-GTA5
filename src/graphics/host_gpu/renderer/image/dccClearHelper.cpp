#include "graphics/host_gpu/renderer/image/dccClearHelper.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_blit_shaders/gpu_dcc_clear_fill_spv.h"
#include "gpu_blit_shaders/gpu_dcc_clear_scan_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>

namespace Libs::Graphics {

namespace {

// One VkDispatchIndirectCommand padded to a uvec4, then the color: the shaders' Record.
constexpr uint64_t RecordSize = 32;

vk::MemoryBarrier MemoryBarrier(vk::AccessFlags source, vk::AccessFlags destination) {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = source;
	barrier.dstAccessMask = destination;
	return barrier;
}

} // namespace

DccClearHelper::DccClearHelper(GraphicContext& graphics, CommandScheduler& scheduler,
                               StreamBuffer& stream)
    : m_graphics(graphics), m_scheduler(scheduler), m_stream(stream) {
	static_assert(sizeof(Push) == 108);
	const std::array bindings {
	    vk::DescriptorSetLayoutBinding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                    vk::ShaderStageFlagBits::eCompute, nullptr},
	    vk::DescriptorSetLayoutBinding {1, vk::DescriptorType::eStorageBuffer, 1,
	                                    vk::ShaderStageFlagBits::eCompute, nullptr},
	    vk::DescriptorSetLayoutBinding {2, vk::DescriptorType::eStorageImage, 1,
	                                    vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo descriptor_info {};
	descriptor_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor_info.bindingCount = static_cast<uint32_t>(bindings.size());
	descriptor_info.pBindings    = bindings.data();
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr,
	                                                                 &m_descriptor_layout),
	                     "create DCC clear descriptor layout");

	const vk::PushConstantRange  push_range {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &m_descriptor_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_pipeline_layout),
	    "create DCC clear pipeline layout");

	const auto create = [&](std::span<const uint32_t> code, const char* operation) {
		const auto module = CompileSPV(code, m_graphics.device);
		vk::ComputePipelineCreateInfo info {};
		info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		info.stage.module = module;
		info.stage.pName  = "main";
		info.layout       = m_pipeline_layout;
		vk::Pipeline pipeline = nullptr;
		const auto   result =
		    m_graphics.device.createComputePipelines(nullptr, 1, &info, nullptr, &pipeline);
		m_graphics.device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, operation);
		return pipeline;
	};
	m_scan = create(GPU_DCC_CLEAR_SCAN_SPV, "create DCC clear scan pipeline");
	m_fill = create(GPU_DCC_CLEAR_FILL_SPV, "create DCC clear fill pipeline");
}

DccClearHelper::~DccClearHelper() {
	for (auto pipeline: {m_scan, m_fill}) {
		if (pipeline != nullptr) {
			m_graphics.device.destroyPipeline(pipeline, nullptr);
		}
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

void DccClearHelper::Record(const Request& request, std::span<const vk::ImageView> views) {
	const auto count = static_cast<uint32_t>(views.size());
	EXIT_IF(count == 0 || request.metadata == nullptr || request.slice_size == 0 ||
	        request.slice_size % sizeof(uint32_t) != 0 || request.width == 0 ||
	        request.height == 0);
	const uint64_t alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 16);
	// Each call gets its own records, so no later call can overwrite them before they are read.
	const auto [mapped, records_offset] = m_stream.Map(count * RecordSize, alignment);
	EXIT_IF(mapped == nullptr);
	m_stream.Commit();

	const auto meta_begin = Common::AlignDown(request.metadata_offset, alignment);
	const auto meta_skip  = request.metadata_offset - meta_begin;
	const vk::DescriptorBufferInfo metadata_info {request.metadata, meta_begin,
	                                              meta_skip + request.slice_size * count};
	const vk::DescriptorBufferInfo records_info {m_stream.Handle(), records_offset,
	                                             count * RecordSize};

	Push push {};
	push.colors       = request.colors;
	push.valid_mask   = request.valid_mask;
	push.slice_dwords = static_cast<uint32_t>(request.slice_size / sizeof(uint32_t));
	push.consume      = request.consume ? 1u : 0u;
	push.width        = request.width;
	push.height       = request.height;

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	// Earlier GPU work wrote the metadata, and may still be reading what the scan consumes.
	const auto before = MemoryBarrier(vk::AccessFlagBits::eMemoryWrite,
	                                  vk::AccessFlagBits::eShaderRead |
	                                      vk::AccessFlagBits::eShaderWrite);
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0, nullptr,
	                        0, nullptr);

	std::array<vk::WriteDescriptorSet, 2> scan_writes {};
	scan_writes[0].dstBinding      = 0;
	scan_writes[0].descriptorCount = 1;
	scan_writes[0].descriptorType  = vk::DescriptorType::eStorageBuffer;
	scan_writes[0].pBufferInfo     = &metadata_info;
	scan_writes[1].dstBinding      = 1;
	scan_writes[1].descriptorCount = 1;
	scan_writes[1].descriptorType  = vk::DescriptorType::eStorageBuffer;
	scan_writes[1].pBufferInfo     = &records_info;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_scan);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0,
	                             static_cast<uint32_t>(scan_writes.size()), scan_writes.data());
	for (uint32_t slice = 0; slice < count; slice++) {
		push.meta_base = static_cast<uint32_t>(meta_skip / sizeof(uint32_t)) +
		                 slice * push.slice_dwords;
		push.record    = slice;
		command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
		                      sizeof(push), &push);
		command.dispatch(1, 1, 1);
	}

	const auto scanned = MemoryBarrier(vk::AccessFlagBits::eShaderWrite,
	                                   vk::AccessFlagBits::eIndirectCommandRead |
	                                       vk::AccessFlagBits::eShaderRead);
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eDrawIndirect |
	                            vk::PipelineStageFlagBits::eComputeShader,
	                        {}, 1, &scanned, 0, nullptr, 0, nullptr);

	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fill);
	for (uint32_t slice = 0; slice < count; slice++) {
		const vk::DescriptorImageInfo target {nullptr, views[slice], vk::ImageLayout::eGeneral};
		std::array<vk::WriteDescriptorSet, 2> fill_writes {};
		fill_writes[0].dstBinding      = 1;
		fill_writes[0].descriptorCount = 1;
		fill_writes[0].descriptorType  = vk::DescriptorType::eStorageBuffer;
		fill_writes[0].pBufferInfo     = &records_info;
		fill_writes[1].dstBinding      = 2;
		fill_writes[1].descriptorCount = 1;
		fill_writes[1].descriptorType  = vk::DescriptorType::eStorageImage;
		fill_writes[1].pImageInfo      = &target;
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0,
		                             static_cast<uint32_t>(fill_writes.size()),
		                             fill_writes.data());
		push.record = slice;
		command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
		                      sizeof(push), &push);
		command.dispatchIndirect(m_stream.Handle(), records_offset + slice * RecordSize);
	}

	// The consumed metadata and the filled texels are ordinary GPU writes to everything after.
	const auto filled = MemoryBarrier(vk::AccessFlagBits::eShaderWrite,
	                                  vk::AccessFlagBits::eMemoryRead |
	                                      vk::AccessFlagBits::eMemoryWrite);
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eAllCommands, {}, 1, &filled, 0, nullptr, 0,
	                        nullptr);
}

} // namespace Libs::Graphics
