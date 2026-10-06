#include "graphics/host_gpu/renderer/dispatchThreadArgs.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_tiler_shaders/dispatch_thread_args_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>

namespace Libs::Graphics {

namespace {

// Mirrors the push-constant block of dispatch_thread_args.comp.
struct ShaderParams {
	uint32_t arguments_word;
	uint32_t output_word;
	uint32_t group_threads_x;
	uint32_t group_threads_y;
	uint32_t group_threads_z;
};

} // namespace

DispatchThreadArgs::DispatchThreadArgs(GraphicContext& graphics): m_graphics(graphics) {
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	    "create dispatch thread arguments descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(ShaderParams)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create dispatch thread arguments pipeline layout");

	const auto module = CompileSPV(DISPATCH_THREAD_ARGS_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create dispatch thread arguments pipeline");
	SetVulkanObjectNameF(graphics.device, m_pipeline, "Kyty.DispatchThreadArgs");
}

DispatchThreadArgs::~DispatchThreadArgs() {
	if (m_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

void DispatchThreadArgs::Record(vk::CommandBuffer command, const Buffer& arguments,
                                uint64_t arguments_offset, const Buffer& output,
                                uint64_t output_offset,
                                const std::array<uint32_t, 3>& group_threads) {
	const auto alignment = m_graphics.StorageMinAlignment();
	EXIT_IF(arguments_offset % sizeof(uint32_t) != 0 || output_offset % sizeof(uint32_t) != 0 ||
	        group_threads[0] == 0 || group_threads[1] == 0 || group_threads[2] == 0);
	const auto arguments_binding = Common::AlignDown(arguments_offset, alignment);
	const auto output_binding    = Common::AlignDown(output_offset, alignment);
	const auto arguments_size    = arguments_offset - arguments_binding + ArgumentsSize;
	const auto output_size       = output_offset - output_binding + OutputSize;
	EXIT_IF(arguments_binding > arguments.Size() ||
	        arguments_size > arguments.Size() - arguments_binding ||
	        output_binding > output.Size() || output_size > output.Size() - output_binding);

	// A dispatch earlier in this queue may have written the arguments, and an earlier dispatch
	// may still read a slot this ring is reusing.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	const vk::DescriptorBufferInfo infos[] {
	    {arguments.Handle(), arguments_binding, arguments_size},
	    {output.Handle(), output_binding, output_size},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const ShaderParams shader_params {
	    .arguments_word =
	        static_cast<uint32_t>((arguments_offset - arguments_binding) / sizeof(uint32_t)),
	    .output_word = static_cast<uint32_t>((output_offset - output_binding) / sizeof(uint32_t)),
	    .group_threads_x = group_threads[0],
	    .group_threads_y = group_threads[1],
	    .group_threads_z = group_threads[2],
	};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(shader_params), &shader_params);
	command.dispatch(1, 1, 1);

	vk::MemoryBarrier2 after {};
	after.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	after.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	after.dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect;
	after.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead;
	dependency.pMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
}

} // namespace Libs::Graphics
