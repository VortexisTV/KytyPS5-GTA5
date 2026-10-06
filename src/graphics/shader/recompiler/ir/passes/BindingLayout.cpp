#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

[[noreturn]] void BindingFail(const char* message) {
	EXIT("shader binding layout failed: %s", message);
	std::abort();
}

std::vector<uint32_t> CollectUserData(const Program& program) {
	std::array<bool, NumScalarRegs> registers {};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::GetUserData || !inst.HasUses()) {
				continue;
			}
			if (inst.Arg(0).GetType() != Type::ScalarReg) {
				BindingFail("typed shader contains an invalid user-data register");
			}
			const auto index = RegIndex(inst.Arg(0).ScalarRegister());
			if (index >= NumScalarRegs) {
				BindingFail("typed shader contains an invalid user-data register");
			}
			registers[index] = true;
		}
	}
	std::vector<uint32_t> result;
	for (uint32_t index = 0; index < registers.size(); index++) {
		if (registers[index]) {
			result.push_back(index);
		}
	}
	return result;
}

void AddBinding(BindingLayout& layout, DescriptorBindingKind kind,
                std::vector<uint32_t> resources = {}) {
	layout.descriptors.push_back({kind, std::move(resources)});
}

bool UsesGds(const Program& program) {
	bool uses_gds = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (SharedAccessOf(inst.GetOpcode()) == SharedAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				BindingFail("typed shader contains invalid shared-memory metadata");
			}
			const auto kind = program.memory_info[index].kind;
			if (kind != ResourceKind::Lds && kind != ResourceKind::Gds) {
				BindingFail("typed shader contains invalid shared-memory metadata");
			}
			uses_gds |= kind == ResourceKind::Gds;
		}
	}
	return uses_gds;
}

} // namespace

bool UsesFlattenedSrt(const Program& program) {
	return std::ranges::any_of(program.blocks, [](const Block* block) {
		return std::ranges::any_of(*block, [](const Inst& inst) {
			return inst.GetOpcode() == ValueOpcode::ReadConst;
		});
	}) || std::ranges::any_of(program.info.images, [](const ImageResource& image) {
		return image.indirect_search_iterations != 0u;
	});
}

void AllocateBindings(Program& program, uint32_t push_data_start_dword) {
	if (!program.shader_info_complete || program.binding_layout_complete) {
		EXIT("shader binding layout failed: %s", !program.shader_info_complete
		                                             ? "shader info is not ready"
		                                             : "binding layout already allocated");
	}
	// Dispatch thread counts occupy the first push-data dwords, ahead of shader data.
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::DispatchThreadCount) {
				push_data_start_dword =
				    std::max(push_data_start_dword, PushData::DispatchThreadDwordCount);
			}
		}
	}
	BindingLayout next;
	next.user_data_registers = CollectUserData(program);
	next.memory_offset_dword = static_cast<uint32_t>(next.user_data_registers.size());
	next.memory_offset_count = static_cast<uint32_t>(program.info.buffers.size());
	next.push_data_start_dword =
	    PushData::StartFor(push_data_start_dword, next.ShaderDataDwords());

	if (!program.info.buffers.empty()) {
		std::vector<uint32_t> resources(program.info.buffers.size());
		for (uint32_t i = 0; i < resources.size(); i++) {
			resources[i] = i;
		}
		AddBinding(next, DescriptorBindingKind::Buffers, std::move(resources));
	}

	std::array<std::vector<uint32_t>, ImageBindingCount> image_groups;
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto kind = DescriptorBindingForImage(program.info.images[i]);
		if (!kind.has_value()) {
			EXIT("shader binding layout failed: image %u has an invalid binding class", i);
		}
		const auto group = ImageBindingIndex(*kind);
		if (group >= image_groups.size()) {
			EXIT("shader binding layout failed: image %u has an unmapped binding class", i);
		}
		auto&      resources = image_groups[group];
		const auto dynamic   = program.info.images[i].mip_mode == ImageMipMode::DynamicStorage;
		const auto count     = dynamic ? program.info.images[i].mip_count : 1u;
		if (count == 0u || (!dynamic && program.info.images[i].mip_count != 1u)) {
			EXIT("shader binding layout failed: image %u has invalid specialized mip count %u", i,
			     program.info.images[i].mip_count);
		}
		resources.insert(resources.end(), count, i);
	}
	for (uint32_t i = 0; i < image_groups.size(); i++) {
		if (!image_groups[i].empty()) {
			AddBinding(next, static_cast<DescriptorBindingKind>(FirstImageBinding + i),
			           std::move(image_groups[i]));
		}
	}

	if (!program.info.samplers.empty()) {
		std::vector<uint32_t> resources(program.info.samplers.size());
		for (uint32_t i = 0; i < resources.size(); i++) {
			resources[i] = i;
		}
		AddBinding(next, DescriptorBindingKind::Samplers, std::move(resources));
	}
	if (UsesGds(program)) {
		AddBinding(next, DescriptorBindingKind::Gds);
	}
	if (program.info.uses_dma) {
		AddBinding(next, DescriptorBindingKind::BdaPagetable);
		AddBinding(next, DescriptorBindingKind::FaultBuffer);
	}
	if (UsesFlattenedSrt(program)) {
		AddBinding(next, DescriptorBindingKind::FlattenedSrt);
	}

	if (next.ShaderDataDwords() != 0 && !next.UsesPushData()) {
		AddBinding(next, DescriptorBindingKind::ShaderData);
	}
	if (UsesLoopWatchdog(program)) {
		AddBinding(next, DescriptorBindingKind::LoopWatchdog);
	}

	program.bindings                = std::move(next);
	program.binding_layout_complete = true;
}

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind) {
	for (const auto& binding: layout.descriptors) {
		if (binding.kind == kind) {
			return &binding;
		}
	}
	return nullptr;
}

const BlockInfo* LoopLatchHeader(const Program& program, size_t index, bool& conditional,
                                 bool& back_edge_on_true) {
	if (program.dispatcher_fallback || index >= program.block_info.size()) {
		return nullptr;
	}
	const auto& latch = program.block_info[index];
	const auto& term  = latch.terminator;
	// A latch that heads a loop keeps its OpLoopMerge in the block the back edge targets, so the
	// watchdog cannot split it.
	if (term.loop_header) {
		return nullptr;
	}
	conditional = term.kind == CFG::TerminatorKind::ConditionalBranch;
	if (conditional ? latch.condition.IsEmpty() || term.true_block == term.false_block
	                : term.kind != CFG::TerminatorKind::Branch) {
		return nullptr;
	}
	for (const bool on_true: {true, false}) {
		if (!on_true && !conditional) {
			break;
		}
		const auto target = on_true ? term.true_block : term.false_block;
		const auto header = std::ranges::find(program.block_info, target, &BlockInfo::id);
		if (header != program.block_info.end() && header->terminator.loop_header &&
		    header->terminator.continue_block == latch.id) {
			back_edge_on_true = on_true;
			return &*header;
		}
	}
	return nullptr;
}

const BlockInfo* ConditionalLatchHeader(const Program& program, size_t index,
                                        bool& back_edge_on_true) {
	bool       conditional = false;
	const auto header      = LoopLatchHeader(program, index, conditional, back_edge_on_true);
	return conditional ? header : nullptr;
}

const BlockInfo* DispatcherBackEdgeTarget(const Program& program, size_t index) {
	if (!program.dispatcher_fallback || index >= program.block_info.size()) {
		return nullptr;
	}
	const auto&      block  = program.block_info[index];
	const auto&      term   = block.terminator;
	const BlockInfo* result = nullptr;
	const auto       visit  = [&](uint32_t target) {
		const auto found = std::ranges::find(program.block_info, target, &BlockInfo::id);
		if (found != program.block_info.end() && found->start_pc <= block.start_pc &&
		    (result == nullptr || found->start_pc < result->start_pc)) {
			result = &*found;
		}
	};
	switch (term.kind) {
		case CFG::TerminatorKind::Branch: visit(term.true_block); break;
		case CFG::TerminatorKind::ConditionalBranch:
			visit(term.true_block);
			visit(term.false_block);
			break;
		case CFG::TerminatorKind::IndirectBranch:
			for (const auto target: term.indirect_targets) {
				visit(target);
			}
			break;
		default: break;
	}
	return result;
}

bool UsesLoopWatchdog(const Program& program) {
	if (!program.loop_watchdog) {
		return false;
	}
	for (size_t index = 0; index < program.block_info.size(); index++) {
		bool conditional = false;
		bool on_true     = false;
		if (LoopLatchHeader(program, index, conditional, on_true) != nullptr ||
		    DispatcherBackEdgeTarget(program, index) != nullptr) {
			return true;
		}
	}
	return false;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
