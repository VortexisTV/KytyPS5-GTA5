#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"

#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {
namespace {

struct TessellationAddress {
	enum class Kind { Unknown, Affine, PackedControlPoint };
	Kind     kind        = Kind::Unknown;
	uint32_t coefficient = 0;
	uint32_t constant    = 0;

	bool operator==(const TessellationAddress&) const = default;
};

bool IsLocalWrite(Decoder::Opcode opcode) {
	using Decoder::Opcode;
	return opcode == Opcode::DS_WRITE_B32 || opcode == Opcode::DS_WRITE_B64 ||
	       opcode == Opcode::DS_WRITE_B96 || opcode == Opcode::DS_WRITE_B128 ||
	       opcode == Opcode::DS_WRITE2_B32 || opcode == Opcode::DS_WRITE2_B64;
}

bool IsLocalRead(Decoder::Opcode opcode) {
	using Decoder::Opcode;
	return opcode == Opcode::DS_READ_B32 || opcode == Opcode::DS_READ_B64 ||
	       opcode == Opcode::DS_READ_B96 || opcode == Opcode::DS_READ_B128 ||
	       opcode == Opcode::DS_READ2_B32 || opcode == Opcode::DS_READ2_B64;
}

struct TessellationInterface {
	uint32_t stride      = 0;
	uint32_t patch_begin = 0;
	uint32_t patch_end   = 0;
};

TessellationInterface ReflectTessellationInterface(const Decoder::Program& program, bool local,
                                                   uint32_t control_points, uint32_t input_stride) {
	using namespace Decoder;
	using Address = TessellationAddress;
	using Kind    = Address::Kind;
	std::array<Address, IR::NumVectorRegs> registers {};
	registers[local ? 3u : 1u] =
	    local ? Address {Kind::Affine, 1u, 0u} : Address {Kind::PackedControlPoint};
	const auto constant = [](uint32_t value) { return Address {Kind::Affine, 0u, value}; };
	const auto read     = [&](const Operand& operand) -> Address {
		if (operand.absolute || operand.negate || operand.sdwa_sext || operand.op_sel ||
		    operand.op_sel_hi || operand.negate_hi || operand.dpp)
			return {};
		Address value;
		if (operand.kind == OperandKind::IntegerInlineConstant ||
		    operand.kind == OperandKind::LiteralConstant) {
			value = constant(operand.value);
		} else if (operand.kind == OperandKind::Vgpr) {
			value = registers.at(operand.reg);
		}
		// The native packed HS ID's low byte is the patch ordinal. The logical
		// interface addresses one patch, as in the frontend's v1 initialization.
		if (value.kind == Kind::PackedControlPoint && operand.sdwa_sel == 0u) return constant(0u);
		return operand.sdwa_sel == 6u ? value : Address {};
	};
	const auto affine = [](uint64_t coefficient, uint64_t offset) {
		return coefficient <= UINT32_MAX && offset <= UINT32_MAX
		           ? Address {Kind::Affine, static_cast<uint32_t>(coefficient),
		                      static_cast<uint32_t>(offset)}
		           : Address {};
	};
	const auto add = [&](Address lhs, Address rhs) {
		return lhs.kind == Kind::Affine && rhs.kind == Kind::Affine
		           ? affine(uint64_t {lhs.coefficient} + rhs.coefficient,
		                    uint64_t {lhs.constant} + rhs.constant)
		           : Address {};
	};
	const auto multiply = [&](Address lhs, Address rhs) {
		if ((lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant == 0u) ||
		    (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant == 0u))
			return constant(0u);
		if (lhs.kind != Kind::Affine || rhs.kind != Kind::Affine) return Address {};
		if (rhs.coefficient != 0u) std::swap(lhs, rhs);
		return rhs.coefficient == 0u ? affine(uint64_t {lhs.coefficient} * rhs.constant,
		                                      uint64_t {lhs.constant} * rhs.constant)
		                             : Address {};
	};
	const auto low24 = [&](Address value) {
		if (value.kind == Kind::Affine && value.coefficient == 0u)
			return constant(value.constant & 0xffffffu);
		return value.kind == Kind::Affine &&
		               uint64_t {value.coefficient} * (control_points - 1u) + value.constant <=
		                   0xffffffu
		           ? value
		           : Address {};
	};
	// A forward branch carries the register state to its target, where it merges with the
	// fallthrough path: a register the two paths define differently is unknown after the join.
	// Loops would need a fixed point instead.
	std::map<uint32_t, decltype(registers)> joins;
	const auto merge = [](decltype(registers)& into, const decltype(registers)& from) {
		for (size_t index = 0; index < into.size(); index++) {
			if (into[index] != from[index]) into[index] = {};
		}
	};
	struct ConstantStore {
		uint32_t address, offset, dwords;
	};
	std::vector<ConstantStore> constant_stores;
	bool                       reachable = true;
	uint32_t                   stride    = 0;
	for (const auto& inst: program.instructions) {
		if (const auto join = joins.find(inst.pc); join != joins.end()) {
			if (reachable) {
				merge(registers, join->second);
			} else {
				registers = join->second;
			}
			reachable = true;
			joins.erase(join);
		}
		if (!reachable) continue;
		if (IsDirectBranch(inst.opcode)) {
			EXIT_NOT_IMPLEMENTED(inst.branch_target <= inst.pc);
			if (const auto [join, inserted] = joins.try_emplace(inst.branch_target, registers);
			    !inserted) {
				merge(join->second, registers);
			}
			reachable = inst.opcode != Opcode::S_BRANCH;
			continue;
		}
		const bool local_store = IsLocalWrite(inst.opcode);
		const bool buffer_store = inst.opcode == Opcode::BUFFER_STORE_DWORD ||
		                          inst.opcode == Opcode::BUFFER_STORE_DWORDX2 ||
		                          inst.opcode == Opcode::BUFFER_STORE_DWORDX3 ||
		                          inst.opcode == Opcode::BUFFER_STORE_DWORDX4;
		const bool control_store =
		    !local && buffer_store && inst.src2.kind == OperandKind::Sgpr && inst.src2.reg == 2u;
		const bool control_read = !local && IsLocalRead(inst.opcode);
		if ((local && local_store) || control_store || control_read) {
			const auto address = read(inst.src0);
			if (address.kind != Kind::Affine) {
				EXIT("%s tessellation address is not affine at pc 0x%08x\n", local ? "LS" : "HS",
				     inst.pc);
			}
			if (address.coefficient != 0u) {
				const auto expected = control_read ? input_stride : stride;
				EXIT_NOT_IMPLEMENTED((address.coefficient & 3u) != 0u ||
				                     (expected != 0u && expected != address.coefficient));
				if (!control_read) stride = address.coefficient;
			} else if (control_store) {
				constant_stores.push_back({address.constant, inst.offset, inst.data_dwords});
			}
		}
		// Store vdata is a source, despite occupying the decoder's dst field.
		if (local_store || buffer_store || inst.dst.kind != OperandKind::Vgpr) continue;
		const auto lhs   = read(inst.src0);
		const auto rhs   = read(inst.src1);
		const auto third = read(inst.src2);
		Address    value;
		switch (inst.opcode) {
			case Opcode::V_BFE_U32:
				if (lhs.kind == Kind::PackedControlPoint && rhs.kind == Kind::Affine &&
				    rhs.coefficient == 0u && third.kind == Kind::Affine &&
				    third.coefficient == 0u) {
					if (rhs.constant == 8u && third.constant == 5u) value = {Kind::Affine, 1u, 0u};
					if (rhs.constant == 0u && third.constant == 8u) value = constant(0u);
				}
				break;
			case Opcode::V_MUL_U32_U24: value = multiply(low24(lhs), low24(rhs)); break;
			case Opcode::V_MAD_U32_U24: value = add(multiply(low24(lhs), low24(rhs)), third); break;
			case Opcode::V_ADD_NC_U32: value = add(lhs, rhs); break;
			case Opcode::V_LSHL_ADD_U32:
				if (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant < 32u)
					value = add(multiply(lhs, constant(1u << rhs.constant)), third);
				break;
			case Opcode::V_LSHLREV_B32:
				if (lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant < 32u)
					value = multiply(rhs, constant(1u << lhs.constant));
				break;
			case Opcode::V_SUB_NC_U32:
				if (lhs.kind == Kind::Affine && rhs.kind == Kind::Affine &&
				    lhs.coefficient >= rhs.coefficient && lhs.constant >= rhs.constant)
					value = affine(lhs.coefficient - rhs.coefficient, lhs.constant - rhs.constant);
				break;
			default: break;
		}
		if (inst.dst.sdwa_sel != 6u || inst.dst.op_sel || inst.dst.omod || inst.dst.clamp ||
		    inst.dst.dpp)
			value = {};
		for (uint32_t index = 0;
		     index < std::max(inst.data_dwords, 1u) && inst.dst.reg + index < registers.size(); index++) {
			registers[inst.dst.reg + index] = {};
		}
		registers.at(inst.dst.reg) = value;
	}
	// Targets past the end are stage exits. One inside the program that no instruction starts
	// at would have dropped its path's definitions.
	EXIT_NOT_IMPLEMENTED(!joins.empty() && joins.begin()->first <= program.instructions.back().pc);
	EXIT_NOT_IMPLEMENTED(stride == 0u);
	// As in the lowering, a constant store past every control point's record is a patch constant.
	TessellationInterface result {.stride = stride};
	for (const auto& store: constant_stores) {
		if (store.address < stride * control_points) continue;
		const auto begin   = store.address + store.offset;
		result.patch_begin = result.patch_end == 0u ? begin : std::min(result.patch_begin, begin);
		result.patch_end   = std::max(result.patch_end, begin + 4u * store.dwords);
	}
	return result;
}

const IR::Inst* TessellationBufferBase(const IR::Inst& inst) {
	if (IR::BufferAccessOf(inst.GetOpcode()) == IR::BufferAccess::None || inst.NumArgs() <= 3u)
		return nullptr;
	const auto* base = inst.Arg(3).Resolve().TryInstruction();
	return base != nullptr && base->GetOpcode() == IR::ValueOpcode::TessellationBase ? base
	                                                                                 : nullptr;
}

IR::Value ActiveAddress(IR::Block& block, IR::Block::iterator before, IR::Value value,
                        IR::Value predicate, std::unordered_map<IR::Inst*, IR::Value>& resolved) {
	using namespace IR;
	value        = value.Resolve();
	auto* source = value.TryInstruction();
	if (source == nullptr) return value;
	if (const auto found = resolved.find(source); found != resolved.end()) return found->second;
	if (source->GetOpcode() == ValueOpcode::SelectU32) {
		return source->Arg(0).Resolve() == predicate.Resolve()
		           ? ActiveAddress(block, before, source->Arg(1), predicate, resolved)
		           : value;
	}
	// Integer arithmetic is lane-wise, so a Select may sit under any of these, as when a
	// predicated V_ADD_NC_U32 feeds the multiply that scales a patch address.
	const auto opcode = source->GetOpcode();
	if (opcode != ValueOpcode::IAdd32 && opcode != ValueOpcode::ISub32 &&
	    opcode != ValueOpcode::IMul32 && opcode != ValueOpcode::ShiftLeftLogical32 &&
	    opcode != ValueOpcode::BitFieldUExtract)
		return value;
	std::array<Value, 3> args;
	bool                 changed = false;
	for (size_t index = 0; index < source->NumArgs(); index++) {
		args[index] = ActiveAddress(block, before, source->Arg(index), predicate, resolved);
		changed     = changed || args[index] != source->Arg(index).Resolve();
	}
	if (changed) {
		const auto flags = source->Flags<uint64_t>();
		auto copy = opcode == ValueOpcode::BitFieldUExtract
		                ? block.PrependNewInst(before, opcode, {args[0], args[1], args[2]}, flags)
		                : block.PrependNewInst(before, opcode, {args[0], args[1]}, flags);
		value = Value(&*copy);
	}
	resolved.emplace(source, value);
	return value;
}

} // namespace

void AnalyzeTessellationPrograms(std::span<const uint32_t> local, std::span<const uint32_t> control,
                                 ShaderTessellationInputInfo& info) {
	const auto       local_program = Decoder::DecodeFrontProgram(local);
	Decoder::Program control_program;
	Decoder::DecodeProgram(control, control_program);
	info.ls_stride =
	    ReflectTessellationInterface(local_program, true, info.input_control_points, 0u).stride;
	const auto hs = ReflectTessellationInterface(control_program, false,
	                                             info.output_control_points, info.ls_stride);
	info.hs_stride   = hs.stride;
	info.patch_begin = hs.patch_begin;
	info.patch_end   = hs.patch_end;
	LOGF("Tessellation interface: input_cp=%u output_cp=%u ls_stride=%u hs_stride=%u "
	     "patch=[0x%x, 0x%x)\n",
	     info.input_control_points, info.output_control_points, info.ls_stride, info.hs_stride,
	     info.patch_begin, info.patch_end);
}

void LowerTessellationMemory(IR::Program& program, const CompileOptions& options) {
	using namespace IR;
	if (options.stage != ShaderType::Local && options.stage != ShaderType::TessellationControl &&
	    options.stage != ShaderType::TessellationEvaluation) {
		return;
	}
	const auto& tess = options.input_info.vertex->tess;
	// Ring addresses can reuse data VGPRs. Their inactive values are irrelevant to
	// a store guarded by the same EXEC predicate, but must remain intact elsewhere.
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			if (TessellationBufferBase(*it) == nullptr) continue;
			std::unordered_map<Inst*, Value> resolved;
			it->SetArg(
			    2, ActiveAddress(*block, it, it->Arg(2), it->Arg(it->NumArgs() - 1u), resolved));
		}
	}
	ConstantPropagationPass(program.blocks);
	uint32_t reads = 0, writes = 0, factors = 0;
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto&      inst   = *it;
			const auto shared = SharedAccessOf(inst.GetOpcode());
			const auto buffer = BufferAccessOf(inst.GetOpcode());
			if (shared == SharedAccess::None && buffer == BufferAccess::None) {
				continue;
			}
			const auto&           memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			bool                  write  = false;
			uint32_t              components = 0;
			TessellationAttribute kind;
			Value                 address, predicate;
			if (shared != SharedAccess::None && memory.kind == ResourceKind::Lds) {
				write = shared == SharedAccess::Write;
				EXIT_NOT_IMPLEMENTED(memory.data_bits != 32u ||
				                     (options.stage == ShaderType::Local
				                          ? !write
				                          : options.stage != ShaderType::TessellationControl ||
				                                shared != SharedAccess::Read));
				kind       = write ? TessellationAttribute::LocalOutput
				                   : TessellationAttribute::ControlInput;
				components = SharedComponentCount(inst.GetOpcode());
				address    = inst.Arg(0);
				predicate  = inst.Arg(inst.NumArgs() - 1u);
			} else if (buffer != BufferAccess::None && memory.kind == ResourceKind::Buffer) {
				const auto* base = TessellationBufferBase(inst);
				if (base == nullptr) {
					continue;
				}
				EXIT_NOT_IMPLEMENTED(memory.data_bits != 32u || memory.formatted || memory.idxen ||
				                     !memory.offen || buffer == BufferAccess::Atomic);
				write      = buffer == BufferAccess::Write;
				components = BufferComponentCount(inst.GetOpcode());
				address    = inst.Arg(2).Resolve();
				predicate  = inst.Arg(inst.NumArgs() - 1u);
				if (base->Arg(0).U32() == 1u) {
					EXIT_NOT_IMPLEMENTED(!write ||
					                     options.stage != ShaderType::TessellationControl);
					kind = TessellationAttribute::Factor;
					factors += components;
				} else {
					kind = write ? TessellationAttribute::ControlOutput
					             : TessellationAttribute::EvaluationInput;
					EXIT_NOT_IMPLEMENTED(write
					                         ? options.stage != ShaderType::TessellationControl
					                         : options.stage != ShaderType::TessellationEvaluation);
					if (address.IsImmediate() &&
					    address.U32() >= tess.hs_stride * tess.output_control_points) {
						kind = write ? TessellationAttribute::PatchOutput
						             : TessellationAttribute::PatchInput;
					}
				}
			} else {
				continue;
			}
			const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
				return Value(&*block->PrependNewInst(it, opcode, args));
			};
			std::array<Value, 4> values;
			for (uint32_t component = 0; component < components; component++) {
				const auto offset = memory.offset + 4u * component;
				const auto byte_address =
				    offset == 0u ? address : emit(ValueOpcode::IAdd32, {address, Value(offset)});
				if (write) {
					Value data = shared != SharedAccess::None ? inst.Arg(component + 1u)
					             : components == 1u
					                 ? inst.Arg(4)
					                 : emit(components == 2u   ? ValueOpcode::CompositeExtractU32x2
					                        : components == 3u ? ValueOpcode::CompositeExtractU32x3
					                                           : ValueOpcode::CompositeExtractU32x4,
					                        {inst.Arg(4), Value(component)});
					emit(ValueOpcode::SetTessellationAttribute,
					     {Value(static_cast<uint32_t>(kind)), byte_address, data, predicate});
					writes++;
				} else {
					values[component] =
					    emit(ValueOpcode::GetTessellationAttribute,
					         {Value(static_cast<uint32_t>(kind)), byte_address, predicate});
					reads++;
				}
			}
			if (!write) {
				Value replacement = values[0];
				if (components == 2u)
					replacement =
					    emit(ValueOpcode::CompositeConstructU32x2, {values[0], values[1]});
				if (components == 3u)
					replacement = emit(ValueOpcode::CompositeConstructU32x3,
					                   {values[0], values[1], values[2]});
				if (components == 4u)
					replacement = emit(ValueOpcode::CompositeConstructU32x4,
					                   {values[0], values[1], values[2], values[3]});
				inst.ReplaceUsesWith(replacement);
			}
			inst.Invalidate();
		}
	}
	ConstantPropagationPass(program.blocks);
	RemoveIdentities(program.blocks);
	EliminateDeadCode(program.blocks);
	LOGF("%s tessellation lowering: reads=%u writes=%u factors=%u\n",
	     options.stage == ShaderType::Local                 ? "LS"
	     : options.stage == ShaderType::TessellationControl ? "HS"
	                                                       : "TES",
	     reads, writes, factors);
}

} // namespace Libs::Graphics::ShaderRecompiler
