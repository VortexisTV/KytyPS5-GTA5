#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/assert.h"
#include "common/perfStats.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <array>
#include <fmt/format.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

std::string Diagnostic(const ResourcePlan& program, uint32_t pc, const std::string& message) {
	return fmt::format("shader SRT: hash=0x{:016x} stage={} pc=0x{:08x} {}", program.shader_hash,
	                   StageName(program.stage), pc, message);
}

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto kind = values.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMin32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type == RuntimeValueType::Integer &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type == RuntimeValueType::Integer && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};

class PlanBuilder {
public:
	explicit PlanBuilder(Program& program): m_program(program) {}

	void Run() {
		m_program.srt_reads.clear();
		m_program.dynamic_reads.clear();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) {
					const auto flags = inst.Flags<MemoryFlags>();
					if (flags.index < m_program.memory_info.size()) {
						const auto kind       = m_program.memory_info[flags.index].kind;
						const bool crosswired = (op == ValueOpcode::LoadAddressU32 &&
						                         kind == ResourceKind::ScalarBuffer) ||
						                        (op == ValueOpcode::ReadConstBuffer &&
						                         kind == ResourceKind::ScalarAddress);
						if (crosswired) {
							Fail(flags.pc,
							     fmt::format("{} has incompatible scalar memory metadata",
							                 ValueOpcodeName(op)));
						}
					}
				}
				if (IsDescriptorHandle(inst.GetOpcode())) {
					for (size_t index = 0; index < inst.NumArgs(); index++) {
						Collect(inst.Arg(index), 0);
					}
				}
			}
		}
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 && IsRawRead(m_program, inst) &&
				    inst.Arg(1).Resolve().IsImmediate() &&
				    ValidateRuntimeValue(m_program, Value(&inst))) {
					Collect(Value(&inst), inst.Flags<MemoryFlags>().pc);
				}
			}
		}
		PatchReads();
	}

private:
	struct Patch {
		Inst*    inst = nullptr;
		uint32_t slot = 0;
		bool     keep = false;
	};

	[[noreturn]] void Fail(uint32_t pc, const std::string& message) const {
		const auto diagnostic = Diagnostic(m_program, pc, message);
		EXIT("shader SRT planning failed: %s", diagnostic.c_str());
		std::abort();
	}

	void Collect(Value value, uint32_t use_pc) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return;
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			Fail(use_pc, "invalid typed planning value");
		}
		const auto cycle = std::ranges::find(m_visiting, inst);
		if (cycle != m_visiting.end()) {
			const auto contains_phi = std::any_of(cycle, m_visiting.end(), [](const Inst* value) {
				return value->GetOpcode() == ValueOpcode::Phi;
			});
			if (contains_phi) {
				return;
			}
			Fail(use_pc, fmt::format("cyclic typed planning value {} without a phi",
			                         ValueOpcodeName(inst->GetOpcode())));
		}
		if (std::ranges::find(m_visited, inst) != m_visited.end()) {
			return;
		}
		m_visiting.push_back(inst);
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			Collect(inst->Arg(index), use_pc);
		}
		m_visiting.pop_back();
		m_visited.push_back(inst);
		if (!IsRawRead(m_program, *inst)) {
			return;
		}
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) {
			if (std::ranges::find(m_program.dynamic_reads, value) ==
			    m_program.dynamic_reads.end()) {
				m_program.dynamic_reads.push_back(value);
			}
			return;
		}
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); slot++) {
			if (EquivalentValue(m_program, value, m_program.srt_reads[slot].value)) {
				m_patches.push_back({inst, slot, false});
				return;
			}
		}
		const auto slot = static_cast<uint32_t>(m_program.srt_reads.size());
		m_program.srt_reads.push_back({value, slot});
		m_patches.push_back({inst, slot, true});
	}

	void PatchReads() {
		for (const auto& patch: m_patches) {
			auto* block = patch.inst->Parent();
			auto& list  = block->Instructions();
			auto  where =
			    std::ranges::find_if(list, [&](const Inst& inst) { return &inst == patch.inst; });
			const auto resource =
			    Value(&*block->PrependNewInst(where, ValueOpcode::GetSrtResource));
			const auto flat = Value(&*block->PrependNewInst(where, ValueOpcode::ReadConst,
			                                                {resource, Value(patch.slot)}));
			const auto uses = patch.inst->Uses();
			for (const auto& use: uses) {
				use.user->SetArg(use.operand, flat);
			}
			for (auto& info: m_program.block_info) {
				if (info.condition.Resolve() == Value(patch.inst)) {
					info.condition = flat;
				}
				if (info.indirect_target.Resolve() == Value(patch.inst)) {
					info.indirect_target = flat;
				}
			}
			if (patch.keep) {
				const auto memory = patch.inst->Flags<MemoryFlags>().index;
				if (memory < m_program.memory_info.size()) {
					m_program.memory_info[memory].planning_only = true;
				}
				block->AppendNewInst(ValueOpcode::ReferenceU32, {Value(patch.inst)});
			}
		}
	}

	Program&           m_program;
	std::vector<Inst*> m_visiting;
	std::vector<Inst*> m_visited;
	std::vector<Patch> m_patches;
};

} // namespace

// The value graph of a plan is fixed, yet the interpreter re-resolves it through IR pointers on
// every refresh. This is the same graph flattened once into nodes addressed by index. Evaluation
// stays lazy and memoized exactly like the interpreter's - a select still evaluates only the
// branch it takes, an inactive source is still never read - so reads happen where they always
// did. A walker has one of two node spaces: a self-contained one (the clean walker), and a
// delegating one whose select predicates and clean flat slots belong to its clean partner.
struct CompiledSrt {
	enum class Op : uint8_t {
		Fail,
		Const,
		UserData,
		ShaderBase,
		Alias,
		ExtractU64,
		CarryLow,
		CarryHigh,
		ConstructU64,
		RawRead,
		RawReadBuffer,
		IAdd32,
		IAdd64,
		ISub32,
		ISub64,
		IMul32,
		IMul64,
		UMin32,
		ConvertF32U32,
		ConvertU32F32,
		FPMul32,
		FPTrunc32,
		FPIsNan32,
		FPOrdLessThanEqual32,
		FPOrdGreaterThanEqual32,
		BitwiseAnd32,
		BitwiseAnd64,
		BitwiseOr32,
		BitwiseXor32,
		BitwiseNot32,
		ShiftLeftLogical32,
		ShiftLeftLogical64,
		ShiftRightLogical32,
		ShiftRightLogical64,
		ShiftRightArithmetic32,
		ShiftRightArithmetic64,
		BitFieldUExtract,
		BitFieldSExtract,
		BitFieldInsert,
		Select,
		IEqual32,
		INotEqual32,
		ULessThan32,
		UGreaterThan32,
		SGreaterThanEqual32,
		LogicalAnd,
		LogicalOr,
		LogicalXor,
		LogicalNot,
	};

	static constexpr uint32_t Invalid = UINT32_MAX;

	struct Node {
		Op                      op         = Op::Fail;
		bool                    self_space = true;
		std::array<uint32_t, 5> args {};
		uint64_t                imm = 0;
	};

	// Entry points per space; Invalid where the interpreter has to evaluate the value.
	struct Roots {
		std::vector<std::array<uint32_t, 8>> sources;
		std::vector<uint32_t>                reads;
		std::vector<uint32_t>                conditions;
		std::array<uint32_t, 4>              fill {Invalid, Invalid, Invalid, Invalid};
	};

	std::vector<Node>    nodes;
	std::array<Roots, 2> roots; // self-contained, delegating
};

namespace {

// Mirrors SrtWalker::EvaluateInst case by case. Anything whose interpreted outcome it cannot
// reproduce exactly compiles to Invalid, which leaves that value to the interpreter.
class SrtCompiler {
public:
	using Op                          = CompiledSrt::Op;
	static constexpr uint32_t Invalid = CompiledSrt::Invalid;

	SrtCompiler(const ResourcePlan& program, CompiledSrt& out): m_program(program), m_out(out) {}

	bool Run() {
		if (!m_program.srt_plan_complete) {
			return false;
		}
		bool any = false;
		for (const bool delegating: {false, true}) {
			auto& roots = m_out.roots[delegating ? 1 : 0];
			roots.sources.resize(m_program.descriptor_sources.size());
			for (size_t index = 0; index < m_program.descriptor_sources.size(); index++) {
				const auto& source = m_program.descriptor_sources[index];
				auto&       slots  = roots.sources[index];
				slots.fill(Invalid);
				for (uint32_t dword = 0; dword < source.dword_count && dword < slots.size();
				     dword++) {
					slots[dword] = Compile(source.dwords[dword], delegating);
					any |= slots[dword] != Invalid;
				}
			}
			roots.reads.resize(m_program.srt_reads.size());
			for (size_t index = 0; index < m_program.srt_reads.size(); index++) {
				roots.reads[index] = Compile(m_program.srt_reads[index].value, delegating);
			}
			roots.conditions.resize(m_program.control_flow.size(), Invalid);
			for (size_t index = 0; index < m_program.control_flow.size(); index++) {
				const auto& condition = m_program.control_flow[index].condition;
				if (!condition.IsEmpty()) {
					roots.conditions[index] = Compile(condition, delegating);
				}
			}
			for (uint32_t index = 0; index < m_program.uniform_fill.fill.words &&
			                         index < roots.fill.size();
			     index++) {
				roots.fill[index] = Compile(m_program.uniform_fill.values[index], delegating);
			}
		}
		return any;
	}

private:
	uint32_t Emit(Op op, bool delegating, std::initializer_list<uint32_t> args = {},
	              uint64_t imm = 0) {
		for (const auto arg: args) {
			if (arg == Invalid) {
				return Invalid;
			}
		}
		CompiledSrt::Node node {.op = op, .self_space = !delegating, .imm = imm};
		std::copy(args.begin(), args.end(), node.args.begin());
		m_out.nodes.push_back(node);
		return static_cast<uint32_t>(m_out.nodes.size() - 1u);
	}

	uint32_t Compile(Value value, bool delegating) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			switch (value.GetType()) {
				case Type::U1: return Emit(Op::Const, delegating, {}, value.U1());
				case Type::U8: return Emit(Op::Const, delegating, {}, value.U8());
				case Type::U16: return Emit(Op::Const, delegating, {}, value.U16());
				case Type::U32: return Emit(Op::Const, delegating, {}, value.U32());
				case Type::U64: return Emit(Op::Const, delegating, {}, value.U64());
				case Type::F32:
					return Emit(Op::Const, delegating, {},
					            std::bit_cast<uint32_t>(value.F32Value()));
				default: return Emit(Op::Fail, delegating);
			}
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return Emit(Op::Fail, delegating);
		}
		auto& done = m_done[delegating ? 1 : 0];
		if (const auto found = done.find(inst); found != done.end()) {
			return found->second;
		}
		// The interpreter fails a cycle only when it walks into it; leave that to it.
		auto& visiting = m_visiting[delegating ? 1 : 0];
		if (!visiting.insert(inst).second) {
			return Invalid;
		}
		const auto node = CompileInst(*inst, delegating);
		visiting.erase(inst);
		done.emplace(inst, node);
		return node;
	}

	uint32_t Unary(const Inst& inst, Op op, bool delegating) {
		return Emit(op, delegating, {Compile(inst.Arg(0), delegating)});
	}

	uint32_t Binary(const Inst& inst, Op op, bool delegating) {
		const auto a = Compile(inst.Arg(0), delegating);
		const auto b = a == Invalid ? Invalid : Compile(inst.Arg(1), delegating);
		return Emit(op, delegating, {a, b});
	}

	uint32_t Ternary(const Inst& inst, Op op, bool delegating) {
		const auto a = Compile(inst.Arg(0), delegating);
		const auto b = a == Invalid ? Invalid : Compile(inst.Arg(1), delegating);
		const auto c = b == Invalid ? Invalid : Compile(inst.Arg(2), delegating);
		return Emit(op, delegating, {a, b, c});
	}

	uint32_t CompileInst(const Inst& inst, bool delegating) {
		switch (inst.GetOpcode()) {
			case ValueOpcode::GetUserData: {
				if (inst.NumArgs() != 1 || inst.Arg(0).GetType() != Type::ScalarReg) {
					return Invalid;
				}
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base) {
					return Emit(Op::Fail, delegating);
				}
				return Emit(Op::UserData, delegating, {}, reg - m_program.user_data_base);
			}
			case ValueOpcode::GetShaderBase: return Emit(Op::ShaderBase, delegating);
			case ValueOpcode::Phi: {
				const auto invariant =
				    ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
				if (invariant.IsEmpty()) {
					return Emit(Op::Fail, delegating);
				}
				return Emit(Op::Alias, delegating, {Compile(invariant, delegating)});
			}
			// Its nested walkers evaluate under an EXEC mask this form does not model.
			case ValueOpcode::ReadFirstLane: return Invalid;
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32: return Unary(inst, Op::Alias, delegating);
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: {
				const auto index = inst.Arg(1).Resolve();
				if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
					return Emit(Op::Fail, delegating);
				}
				const auto component = index.U32();
				if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
					return Emit(Op::ExtractU64, delegating, {Compile(inst.Arg(0), delegating)},
					            component);
				}
				const auto* source = inst.Arg(0).ResolveInstruction();
				if (source == nullptr) {
					return Emit(Op::Fail, delegating);
				}
				if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
					if (source->NumArgs() < 2) {
						return Invalid;
					}
					return Emit(Op::Alias, delegating, {Compile(source->Arg(component), delegating)});
				}
				if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
					if (source->NumArgs() < 2) {
						return Invalid;
					}
					const auto a = Compile(source->Arg(0), delegating);
					const auto b = a == Invalid ? Invalid : Compile(source->Arg(1), delegating);
					return Emit(component == 0u ? Op::CarryLow : Op::CarryHigh, delegating, {a, b});
				}
				return Emit(Op::Fail, delegating);
			}
			case ValueOpcode::CompositeConstructU64: return Binary(inst, Op::ConstructU64, delegating);
			case ValueOpcode::ReadConst: {
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
				    slot.U32() >= m_program.srt_reads.size()) {
					return Emit(Op::Fail, delegating);
				}
				// A delegating walker hands clean slots to its clean partner.
				const bool clean = delegating && slot.U32() < m_program.clean_flat_slots.size() &&
				                   m_program.clean_flat_slots[slot.U32()] != 0u;
				return Emit(Op::Alias, delegating,
				            {Compile(m_program.srt_reads[slot.U32()].value, delegating && !clean)});
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer: {
				if (!IsRawRead(m_program, inst)) {
					return Emit(Op::Fail, delegating);
				}
				const auto  flags  = inst.Flags<MemoryFlags>();
				const auto& mem    = m_program.memory_info[flags.index];
				const auto* handle = inst.Arg(0).ResolveInstruction();
				if (handle == nullptr) {
					return Emit(Op::Fail, delegating);
				}
				if (handle->NumArgs() < 2 || inst.NumArgs() < 2) {
					return Invalid;
				}
				const auto low    = Compile(handle->Arg(0), delegating);
				const auto high   = low == Invalid ? Invalid : Compile(handle->Arg(1), delegating);
				const auto offset = high == Invalid ? Invalid : Compile(inst.Arg(1), delegating);
				if (inst.GetOpcode() == ValueOpcode::LoadAddressU32) {
					return Emit(Op::RawRead, delegating, {low, high, offset}, mem.offset);
				}
				if (handle->NumArgs() != 4u || static_cast<int32_t>(mem.offset) < 0) {
					return Emit(Op::Fail, delegating);
				}
				const auto records = offset == Invalid ? Invalid : Compile(handle->Arg(2), delegating);
				const auto word3 = records == Invalid ? Invalid : Compile(handle->Arg(3), delegating);
				return Emit(Op::RawReadBuffer, delegating, {low, high, offset, records, word3},
				            mem.offset);
			}
			case ValueOpcode::IAdd32: return Binary(inst, Op::IAdd32, delegating);
			case ValueOpcode::IAdd64: return Binary(inst, Op::IAdd64, delegating);
			case ValueOpcode::ISub32: return Binary(inst, Op::ISub32, delegating);
			case ValueOpcode::ISub64: return Binary(inst, Op::ISub64, delegating);
			case ValueOpcode::IMul32: return Binary(inst, Op::IMul32, delegating);
			case ValueOpcode::IMul64: return Binary(inst, Op::IMul64, delegating);
			case ValueOpcode::UMin32: return Binary(inst, Op::UMin32, delegating);
			case ValueOpcode::ConvertF32U32: return Unary(inst, Op::ConvertF32U32, delegating);
			case ValueOpcode::ConvertU32F32: return Unary(inst, Op::ConvertU32F32, delegating);
			case ValueOpcode::FPMul32: return Binary(inst, Op::FPMul32, delegating);
			case ValueOpcode::FPTrunc32: return Unary(inst, Op::FPTrunc32, delegating);
			case ValueOpcode::FPIsNan32: return Unary(inst, Op::FPIsNan32, delegating);
			case ValueOpcode::FPOrdLessThanEqual32:
				return Binary(inst, Op::FPOrdLessThanEqual32, delegating);
			case ValueOpcode::FPOrdGreaterThanEqual32:
				return Binary(inst, Op::FPOrdGreaterThanEqual32, delegating);
			case ValueOpcode::BitwiseAnd32: return Binary(inst, Op::BitwiseAnd32, delegating);
			case ValueOpcode::BitwiseAnd64: return Binary(inst, Op::BitwiseAnd64, delegating);
			case ValueOpcode::BitwiseOr32: return Binary(inst, Op::BitwiseOr32, delegating);
			case ValueOpcode::BitwiseXor32: return Binary(inst, Op::BitwiseXor32, delegating);
			case ValueOpcode::BitwiseNot32: return Unary(inst, Op::BitwiseNot32, delegating);
			case ValueOpcode::ShiftLeftLogical32:
				return Binary(inst, Op::ShiftLeftLogical32, delegating);
			case ValueOpcode::ShiftLeftLogical64:
				return Binary(inst, Op::ShiftLeftLogical64, delegating);
			case ValueOpcode::ShiftRightLogical32:
				return Binary(inst, Op::ShiftRightLogical32, delegating);
			case ValueOpcode::ShiftRightLogical64:
				return Binary(inst, Op::ShiftRightLogical64, delegating);
			case ValueOpcode::ShiftRightArithmetic32:
				return Binary(inst, Op::ShiftRightArithmetic32, delegating);
			case ValueOpcode::ShiftRightArithmetic64:
				return Binary(inst, Op::ShiftRightArithmetic64, delegating);
			case ValueOpcode::BitFieldUExtract: return Ternary(inst, Op::BitFieldUExtract, delegating);
			case ValueOpcode::BitFieldSExtract: return Ternary(inst, Op::BitFieldSExtract, delegating);
			case ValueOpcode::BitFieldInsert: {
				if (inst.NumArgs() < 4) {
					return Invalid;
				}
				const auto a = Compile(inst.Arg(0), delegating);
				const auto b = a == Invalid ? Invalid : Compile(inst.Arg(1), delegating);
				const auto c = b == Invalid ? Invalid : Compile(inst.Arg(2), delegating);
				const auto d = c == Invalid ? Invalid : Compile(inst.Arg(3), delegating);
				return Emit(Op::BitFieldInsert, delegating, {a, b, c, d});
			}
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32: {
				// A delegating walker asks its clean partner for the predicate.
				const auto predicate = Compile(inst.Arg(0), false);
				const auto a = predicate == Invalid ? Invalid : Compile(inst.Arg(1), delegating);
				const auto b = a == Invalid ? Invalid : Compile(inst.Arg(2), delegating);
				return Emit(Op::Select, delegating, {predicate, a, b});
			}
			case ValueOpcode::IEqual32: return Binary(inst, Op::IEqual32, delegating);
			case ValueOpcode::INotEqual32: return Binary(inst, Op::INotEqual32, delegating);
			case ValueOpcode::ULessThan32: return Binary(inst, Op::ULessThan32, delegating);
			case ValueOpcode::UGreaterThan32: return Binary(inst, Op::UGreaterThan32, delegating);
			case ValueOpcode::SGreaterThanEqual32:
				return Binary(inst, Op::SGreaterThanEqual32, delegating);
			case ValueOpcode::LogicalAnd: return Binary(inst, Op::LogicalAnd, delegating);
			case ValueOpcode::LogicalOr: return Binary(inst, Op::LogicalOr, delegating);
			case ValueOpcode::LogicalXor: return Binary(inst, Op::LogicalXor, delegating);
			case ValueOpcode::LogicalNot: return Unary(inst, Op::LogicalNot, delegating);
			default: return Emit(Op::Fail, delegating);
		}
	}

	const ResourcePlan&                          m_program;
	CompiledSrt&                                 m_out;
	std::array<std::unordered_map<const Inst*, uint32_t>, 2> m_done;
	std::array<std::unordered_set<const Inst*>, 2>           m_visiting;
};

} // namespace

bool SrtWalker::CompileSrt(const ResourcePlan& program) {
	if (!program.compiled_srt_attempted) {
		program.compiled_srt_attempted = true;
		auto compiled                  = std::make_shared<CompiledSrt>();
		if (SrtCompiler(program, *compiled).Run()) {
			program.compiled_srt        = std::move(compiled);
			program.compiled_srt_checks = 16;
		}
	}
	return program.compiled_srt != nullptr;
}

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
                     Value active_mask)
    : m_program(program), m_runtime(runtime), m_clean_flat_slots(clean_flat_slots),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_context(AcquireContext(program)) {
	// Only the two walker shapes a resource refresh builds have a compiled form.
	if (!m_active_mask.IsEmpty()) {
		return;
	}
	if (clean_evaluator == nullptr && clean_flat_slots.empty()) {
		m_space = Space::Self;
	} else if (clean_evaluator != nullptr && clean_evaluator->m_space == Space::Self &&
	           clean_flat_slots.data() == program.clean_flat_slots.data() &&
	           clean_flat_slots.size() == program.clean_flat_slots.size()) {
		m_space = Space::Delegating;
	} else {
		return;
	}
	if (!CompileSrt(program)) {
		m_space = Space::None;
		return;
	}
	m_compiled = program.compiled_srt.get();
	if (m_context.compiled.size() < m_compiled->nodes.size()) {
		m_context.compiled.resize(m_compiled->nodes.size());
	}
}

void SrtWalker::Interpret() {
	m_compiled = nullptr;
	m_space    = Space::None;
}

const void* SrtWalker::CompiledRoots() const {
	if (m_compiled == nullptr ||
	    (m_space == Space::Delegating && m_clean_evaluator->m_compiled != m_compiled)) {
		return nullptr;
	}
	return &m_compiled->roots[m_space == Space::Self ? 0 : 1];
}

// A walker with compiled roots that still has to interpret this value.
void SrtWalker::CountInterpreted(const void* roots) {
	if (roots != nullptr) {
		PerfStats::Add(PerfStats::CounterId::SrtInterpreted);
	}
}

bool SrtWalker::EvaluateNode(uint32_t index, uint64_t& result) {
	if (m_space == Space::Delegating && m_compiled->nodes[index].self_space) {
		return m_clean_evaluator->EvaluateNode(index, result);
	}
	auto& memo = m_context.compiled[index];
	if (memo.generation == m_context.generation) {
		result = memo.value;
		return true;
	}
	if (memo.generation == (m_context.generation | 1u)) {
		return false;
	}
	uint64_t   out       = 0;
	const bool evaluated = RunNode(index, out);
	auto&      entry     = m_context.compiled[index];
	entry.value          = out;
	entry.generation     = evaluated ? m_context.generation : (m_context.generation | 1u);
	result               = out;
	return evaluated;
}

bool SrtWalker::RunNode(uint32_t index, uint64_t& result) {
	using Op          = CompiledSrt::Op;
	const auto& node  = m_compiled->nodes[index];
	uint64_t    a     = 0;
	uint64_t    b     = 0;
	uint64_t    c     = 0;
	const auto  arg   = [&](uint32_t which, uint64_t& value) {
		return EvaluateNode(node.args[which], value);
	};
	const auto binary  = [&]() { return arg(0, a) && arg(1, b); };
	const auto ternary = [&]() { return arg(0, a) && arg(1, b) && arg(2, c); };
	switch (node.op) {
		case Op::Fail: return false;
		case Op::Const: result = node.imm; return true;
		case Op::UserData:
			if (node.imm >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[node.imm];
			return true;
		case Op::ShaderBase: result = m_runtime.shader_base; return true;
		case Op::Alias: return arg(0, result);
		case Op::ExtractU64:
			if (!arg(0, a)) {
				return false;
			}
			result = static_cast<uint32_t>(a >> (node.imm * 32u));
			return true;
		case Op::CarryLow:
		case Op::CarryHigh: {
			if (!binary()) {
				return false;
			}
			const auto sum =
			    static_cast<uint64_t>(static_cast<uint32_t>(a)) + static_cast<uint32_t>(b);
			result = node.op == Op::CarryLow ? static_cast<uint32_t>(sum)
			                                 : static_cast<uint32_t>(sum >> 32u);
			return true;
		}
		case Op::ConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) | (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case Op::RawRead:
		case Op::RawReadBuffer: {
			uint64_t offset = 0;
			if (!arg(0, a) || !arg(1, b) || !arg(2, offset)) {
				return false;
			}
			const auto base      = ((b << 32u) | static_cast<uint32_t>(a)) & AddressMask;
			const auto immediate = static_cast<int64_t>(static_cast<int32_t>(node.imm));
			uint64_t   address   = 0;
			if (node.op == Op::RawReadBuffer) {
				uint64_t records = 0;
				uint64_t word3   = 0;
				if (!arg(3, records) || !arg(4, word3)) {
					return false;
				}
				const auto byte_offset =
				    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
				const auto aligned = byte_offset & ~uint64_t {3};
				const auto stride  = (static_cast<uint32_t>(b) >> 16u) & 0x3fffu;
				const auto size    = stride == 0u
				                         ? static_cast<uint64_t>(static_cast<uint32_t>(records))
				                         : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
				if (aligned > size || size - aligned < sizeof(uint32_t)) {
					return false;
				}
				address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
			} else {
				const auto relative = (immediate & ~int64_t {3}) +
				                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
				if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
					return false;
				}
			}
			uint32_t word = 0;
			if (m_runtime.read_memory != nullptr) {
				if (!m_runtime.read_memory(m_runtime.userdata, address, {&word, 1})) {
					return false;
				}
			} else {
				std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
			}
			result = word;
			return true;
		}
		case Op::IAdd32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a + b);
			return true;
		case Op::IAdd64:
			if (!binary()) return false;
			result = a + b;
			return true;
		case Op::ISub32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a - b);
			return true;
		case Op::ISub64:
			if (!binary()) return false;
			result = a - b;
			return true;
		case Op::IMul32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a * b);
			return true;
		case Op::IMul64:
			if (!binary()) return false;
			result = a * b;
			return true;
		case Op::UMin32:
			if (!binary()) return false;
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			return true;
		case Op::ConvertF32U32:
			if (!arg(0, a)) return false;
			result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case Op::ConvertU32F32: {
			if (!arg(0, a)) return false;
			const auto value = Float32(a);
			if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX) {
				return false;
			}
			result = static_cast<uint32_t>(value);
			return true;
		}
		case Op::FPMul32:
			if (!binary()) return false;
			result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
			return true;
		case Op::FPTrunc32:
			if (!arg(0, a)) return false;
			result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
			return true;
		case Op::FPIsNan32:
			if (!arg(0, a)) return false;
			result = std::isnan(Float32(a));
			return true;
		case Op::FPOrdLessThanEqual32:
			if (!binary()) return false;
			result = Float32(a) <= Float32(b);
			return true;
		case Op::FPOrdGreaterThanEqual32:
			if (!binary()) return false;
			result = Float32(a) >= Float32(b);
			return true;
		case Op::BitwiseAnd32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a & b);
			return true;
		case Op::BitwiseAnd64:
			if (!binary()) return false;
			result = a & b;
			return true;
		case Op::BitwiseOr32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a | b);
			return true;
		case Op::BitwiseXor32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a ^ b);
			return true;
		case Op::BitwiseNot32:
			if (!arg(0, a)) return false;
			result = ~static_cast<uint32_t>(a);
			return true;
		case Op::ShiftLeftLogical32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) << (b & 31u);
			return true;
		case Op::ShiftLeftLogical64:
			if (!binary()) return false;
			result = a << (b & 63u);
			return true;
		case Op::ShiftRightLogical32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) >> (b & 31u);
			return true;
		case Op::ShiftRightLogical64:
			if (!binary()) return false;
			result = a >> (b & 63u);
			return true;
		case Op::ShiftRightArithmetic32:
			if (!binary()) return false;
			result =
			    static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
			return true;
		case Op::ShiftRightArithmetic64:
			if (!binary()) return false;
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			return true;
		case Op::BitFieldUExtract: {
			if (!ternary()) return false;
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			const auto mask = width == 32u  ? UINT32_MAX
			                  : width == 0u ? 0u
			                                : (uint32_t {1} << width) - 1u;
			result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			return true;
		}
		case Op::BitFieldSExtract: {
			if (!ternary()) return false;
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = 0;
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
			auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
			if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
				bits |= ~mask;
			}
			result = bits;
			return true;
		}
		case Op::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !arg(3, d)) return false;
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case Op::Select:
			if (!arg(0, a)) return false;
			return arg(a != 0u ? 1u : 2u, result);
		case Op::IEqual32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
			return true;
		case Op::INotEqual32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
			return true;
		case Op::ULessThan32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
			return true;
		case Op::UGreaterThan32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
			return true;
		case Op::SGreaterThanEqual32:
			if (!binary()) return false;
			result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
			         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			return true;
		case Op::LogicalAnd:
			if (!binary()) return false;
			result = (a != 0u) && (b != 0u);
			return true;
		case Op::LogicalOr:
			if (!binary()) return false;
			result = (a != 0u) || (b != 0u);
			return true;
		case Op::LogicalXor:
			if (!binary()) return false;
			result = (a != 0u) != (b != 0u);
			return true;
		case Op::LogicalNot:
			if (!arg(0, a)) return false;
			result = a == 0u;
			return true;
	}
	return false;
}

bool SrtWalker::EvaluateRead(uint32_t index, uint32_t& result) {
	const auto* roots = static_cast<const CompiledSrt::Roots*>(CompiledRoots());
	if (roots == nullptr || roots->reads[index] == CompiledSrt::Invalid) {
		CountInterpreted(roots);
		return Evaluate(m_program.srt_reads[index].value, result);
	}
	uint64_t wide = 0;
	if (!EvaluateNode(roots->reads[index], wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

bool SrtWalker::EvaluateCondition(uint32_t block, uint32_t& result) {
	const auto* roots = static_cast<const CompiledSrt::Roots*>(CompiledRoots());
	if (roots == nullptr || roots->conditions[block] == CompiledSrt::Invalid) {
		CountInterpreted(roots);
		return Evaluate(m_program.control_flow[block].condition, result);
	}
	uint64_t wide = 0;
	if (!EvaluateNode(roots->conditions[block], wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

bool SrtWalker::EvaluateUniformFill(uint32_t index, uint32_t& result) {
	const auto* roots = static_cast<const CompiledSrt::Roots*>(CompiledRoots());
	if (roots == nullptr || index >= roots->fill.size() ||
	    roots->fill[index] == CompiledSrt::Invalid) {
		CountInterpreted(roots);
		return Evaluate(m_program.uniform_fill.values[index], result);
	}
	uint64_t wide = 0;
	if (!EvaluateNode(roots->fill[index], wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

SrtWalker::~SrtWalker() { --m_program.evaluation_depth; }

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext(const ResourcePlan& program) {
	if (program.evaluation_depth == program.evaluation_contexts.size()) {
		program.evaluation_contexts.emplace_back();
	}
	auto& context = program.evaluation_contexts[program.evaluation_depth++];
	context.generation += 2;
	return context;
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
			default: return false;
		}
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// The low generation bit marks an instruction that is still being evaluated.
	if (m_context.values[index].generation == (m_context.generation | 1u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = 0;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		return EvaluateWide(source->Arg(component), result);
	}
	if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
		uint64_t lhs = 0;
		uint64_t rhs = 0;
		if (!Arg(*source, 0, lhs) || !Arg(*source, 1, rhs)) {
			return false;
		}
		const auto sum =
		    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
		result =
		    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
		return true;
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<MemoryFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   address   = 0;
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
		const auto aligned = byte_offset & ~uint64_t {3};
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (aligned > size || size - aligned < sizeof(uint32_t)) {
			return false;
		}
		address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	if (m_runtime.read_memory != nullptr) {
		if (!m_runtime.read_memory(m_runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
}

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
	const auto ternary = [&]() {
		return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			return true;
		}
		case ValueOpcode::GetShaderBase: result = m_runtime.shader_base; return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			const auto clean_runtime = CleanRuntime(m_runtime);
			SrtWalker  clean_active(m_program, clean_runtime, {}, nullptr, inst.Arg(1));
			SrtWalker  active(m_program, m_runtime, m_clean_flat_slots, &clean_active,
			                  inst.Arg(1));
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::ReadConst: {
			const auto slot = inst.Arg(1).Resolve();
			if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return false;
			}
			if (slot.U32() < m_clean_flat_slots.size() &&
			    m_clean_flat_slots[slot.U32()] != 0u && m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateWide(m_program.srt_reads[slot.U32()].value,
				                                       result);
			}
			return EvaluateWide(m_program.srt_reads[slot.U32()].value, result);
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
			if (IsRawRead(m_program, inst)) {
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::IAdd32:
			if (binary()) {
				result = static_cast<uint32_t>(a + b);
				return true;
			}
			return false;
		case ValueOpcode::IAdd64:
			if (binary()) {
				result = a + b;
				return true;
			}
			return false;
		case ValueOpcode::ISub32:
			if (binary()) {
				result = static_cast<uint32_t>(a - b);
				return true;
			}
			return false;
		case ValueOpcode::ISub64:
			if (binary()) {
				result = a - b;
				return true;
			}
			return false;
		case ValueOpcode::IMul32:
			if (binary()) {
				result = static_cast<uint32_t>(a * b);
				return true;
			}
			return false;
		case ValueOpcode::IMul64:
			if (binary()) {
				result = a * b;
				return true;
			}
			return false;
		case ValueOpcode::UMin32:
			if (binary()) {
				result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::ConvertF32U32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			}
			return false;
		case ValueOpcode::ConvertU32F32:
			if (Arg(inst, 0, a)) {
				const auto value = Float32(a);
				if (!std::isfinite(value) || value < 0.0f ||
				    static_cast<double>(value) > UINT32_MAX) {
					return false;
				}
				result = static_cast<uint32_t>(value);
				return true;
			}
			return false;
		case ValueOpcode::FPMul32:
			if (binary()) {
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			}
			return false;
		case ValueOpcode::FPTrunc32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			}
			return false;
		case ValueOpcode::FPIsNan32:
			if (Arg(inst, 0, a)) {
				result = std::isnan(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPOrdLessThanEqual32:
			if (binary()) {
				result = Float32(a) <= Float32(b);
				return true;
			}
			return false;
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (binary()) {
				result = Float32(a) >= Float32(b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd32:
			if (binary()) {
				result = static_cast<uint32_t>(a & b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd64:
			if (binary()) {
				result = a & b;
				return true;
			}
			return false;
		case ValueOpcode::BitwiseOr32:
			if (binary()) {
				result = static_cast<uint32_t>(a | b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseXor32:
			if (binary()) {
				result = static_cast<uint32_t>(a ^ b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseNot32:
			if (Arg(inst, 0, a)) {
				result = ~static_cast<uint32_t>(a);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical64:
			if (binary()) {
				result = a << (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical64:
			if (binary()) {
				result = a >> (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic32:
			if (binary()) {
				result = static_cast<uint32_t>(
				    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic64:
			if (binary()) {
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			}
			return false;
		case ValueOpcode::BitFieldUExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u  ? UINT32_MAX
				                  : width == 0u ? 0u
				                                : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldSExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !Arg(inst, 3, d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return Arg(inst, a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::IEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::INotEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::UGreaterThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::SGreaterThanEqual32:
			if (binary()) {
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd:
			if (binary()) {
				result = (a != 0u) && (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::LogicalOr:
			if (binary()) {
				result = (a != 0u) || (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::LogicalXor:
			if (binary()) {
				result = (a != 0u) != (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::LogicalNot:
			if (Arg(inst, 0, a)) {
				result = a == 0u;
				return true;
			}
			return false;
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU8:
		case ValueOpcode::UndefU16:
		case ValueOpcode::UndefU32:
		case ValueOpcode::UndefU64: return false;
		default: break;
	}
	return false;
}
bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	const auto* roots = static_cast<const CompiledSrt::Roots*>(CompiledRoots());
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		const auto node = roots != nullptr && index < roots->sources[source].size()
		                      ? roots->sources[source][index]
		                      : CompiledSrt::Invalid;
		if (node == CompiledSrt::Invalid) {
			CountInterpreted(roots);
			if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
				return false;
			}
			continue;
		}
		uint64_t wide = 0;
		if (!EvaluateNode(node, wide)) {
			return false;
		}
		result.dwords[index] = static_cast<uint32_t>(wide);
	}
	return true;
}

std::span<const uint8_t> SrtWalker::FindActiveSources() {
	if (m_program.control_flow.empty()) {
		return {};
	}
	auto& active = m_program.active_sources;
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow) {
		for (const auto source: block.sources) {
			active.at(source) = 0u;
		}
	}
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index)) {
			continue;
		}
		visited[index] = 1u;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) {
			active[source] = 1u;
		}
		uint32_t condition = 0;
		if (!block.condition.IsEmpty() && m_runtime.read_specialization_memory != nullptr &&
		    EvaluateCondition(index, condition)) {
			pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	return active;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) {
		return false;
	}
	flat.resize(m_program.srt_reads.size());
	for (uint32_t index = 0; index < m_program.srt_reads.size(); index++) {
		const auto& read  = m_program.srt_reads[index];
		const bool  clean = read.flat_offset < m_clean_flat_slots.size() &&
		                   m_clean_flat_slots[read.flat_offset] != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr)) {
			return false;
		}
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		if (read.flat_offset >= flat.size() ||
		    !evaluator.EvaluateRead(index, flat[read.flat_offset])) {
			return false;
		}
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

void BuildSrtPlan(Program& program) {
	if (program.resource_tracking_complete) {
		EXIT("shader SRT planning failed: cannot rebuild SRT after resource tracking");
	}
	program.srt_plan_complete = false;
	PlanBuilder(program).Run();
	program.srt_plan_complete = true;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
