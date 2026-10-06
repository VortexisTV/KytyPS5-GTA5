#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"

#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

void EmitKillIfBoolFalse(EmitterState& state, uint32_t active) {
	const auto kill_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	const auto inactive    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalNot, TypeBool(state), inactive, active);
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, inactive, kill_label, merge_label);
	EmitLabel(state, kill_label);
	state.builder.AddFunction(spv::OpKill);
	EmitLabel(state, merge_label);
}

void EmitKillIfPixelValidMaskInactive(EmitterState& state) {
	if (state.pixel_valid_mask_variable == 0) {
		return;
	}

	const auto mask_value = state.builder.AllocateId();
	const auto active     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), mask_value,
	                          state.pixel_valid_mask_variable);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active, mask_value,
	                          ConstantU32(state, 0));
	EmitKillIfBoolFalse(state, active);
}

uint32_t SpillPointerType(ValueEmitContext& ctx, IR::Type type) {
	const auto value_type = TypeId(ctx.state, type);
	return value_type == 0 ? 0 : TypePointer(ctx.state, spv::StorageClassFunction, value_type);
}

struct DeferredPhiPatch {
	DeferredPhi     phi;
	const IR::Inst* instruction = nullptr;
	uint32_t        half        = 0;
};

// A latch that is its loop's continue target and leaves the loop on one branch takes its back
// edge through an empty continue block of its own. Left in the continue construct, the loop body
// and its subgroup operations miscompile on NVIDIA (GeForce driver 616.56): lanes that already
// left the loop stay in readfirstlane's ballot, so a GTA V skinning waterfall never ends.
struct SplitLatch {
	uint32_t         continue_label = 0;
	const IR::Block* header         = nullptr;
};

struct StructuredFunctionState {
	std::unordered_map<const IR::Block*, uint32_t>   block_exit_labels;
	std::unordered_map<const IR::Block*, SplitLatch> split_latches;
	std::vector<DeferredPhiPatch>                    deferred_phis;
};

struct DispatcherFunctionState {
	std::array<std::unordered_map<const IR::Inst*, uint32_t>, 2> spills;
	uint32_t                                                     header_label       = 0;
	uint32_t                                                     select_label       = 0;
	uint32_t                                                     after_switch_label = 0;
	uint32_t                                                     continue_label     = 0;
	uint32_t                                                     merge_label        = 0;
};

void StoreDispatcherPhiEdge(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                            const IR::Block* from, const IR::Block* to) {
	if (to == nullptr) {
		return;
	}
	for (const auto& phi: *to) {
		if (phi.GetOpcode() != IR::ValueOpcode::Phi) {
			break;
		}
		for (size_t index = 0; index < phi.NumArgs(); index++) {
			if (phi.PhiBlock(index) == from) {
				ctx.state.builder.AddFunction(spv::OpStore, dispatcher.spills[ctx.half].at(&phi),
				                              ctx.Def(phi.Arg(index)));
				break;
			}
		}
	}
}

const IR::Block* TargetBlock(const IR::Program& program, uint32_t id) {
	const auto found = std::ranges::find_if(
	    program.block_info, [&](const IR::BlockInfo& info) { return info.id == id; });
	if (found == program.block_info.end()) {
		return nullptr;
	}
	return program.blocks[static_cast<size_t>(found - program.block_info.begin())];
}

void EmitReturn(ValueEmitContext& ctx) {
	EmitKillIfPixelValidMaskInactive(ctx.state);
	ctx.state.builder.AddFunction(spv::OpReturn);
}

uint32_t LoopWatchdogElement(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.loop_watchdog_variable, ConstantU32(state, 0),
	                          ConstantU32(state, index));
	return pointer;
}

// Whether the invocation has used up its loop iterations.
uint32_t EmitLoopWatchdogExhausted(EmitterState& state) {
	const auto count = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), count, state.loop_watchdog_counter);
	const auto exhausted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), exhausted, count,
	                          ConstantU32(state, IR::LoopWatchdog::IterationLimit));
	return exhausted;
}

// The branch condition of a block that can leave a watched loop, taking the exit once the
// invocation's iterations are used up. The exit edge exists already, so no phi gains a predecessor.
uint32_t EmitLoopWatchdogExit(ValueEmitContext& ctx, const IR::BlockInfo& info, bool exit_on_true) {
	auto&      state     = ctx.state;
	const auto condition = ctx.Def(info.condition);
	const auto exhausted = EmitLoopWatchdogExhausted(state);
	const auto result    = state.builder.AllocateId();
	if (exit_on_true) {
		state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), result, condition, exhausted);
	} else {
		const auto within = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalNot, TypeBool(state), within, exhausted);
		state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), result, condition, within);
	}
	return result;
}

// The guest pc a watched loop is recorded under. A header the structurizer made up holds no guest
// code, so the loop is named after the first block of its body that does.
uint32_t LoopWatchdogPc(const IR::Program& program, const IR::BlockInfo& header) {
	const auto* block = &header;
	for (size_t step = 0; step < program.block_info.size(); step++) {
		if (block->start_pc != block->end_pc || block->terminator.kind != CFG::TerminatorKind::Branch) {
			break;
		}
		const auto next =
		    std::ranges::find(program.block_info, block->terminator.true_block, &IR::BlockInfo::id);
		if (next == program.block_info.end()) {
			break;
		}
		block = &*next;
	}
	return block->start_pc;
}

// Counts one back-edge check of a structured loop latch. An invocation that reaches the limit or,
// when loops are timed, outlasts its tick budget is cut and records the cut, the first one the
// shader and loop header; every AbortCheckInterval checks an invocation also uses up its
// iterations once any cut has happened. A conditional latch then leaves through its exit edge, and
// its branch condition is returned; an unconditional one leaves at the loop's next exit branch
// (EmitLoopWatchdogExit), and 0 is returned.
uint32_t EmitLoopWatchdog(ValueEmitContext& ctx, const IR::BlockInfo& latch,
                          const IR::BlockInfo& header, bool conditional, bool back_edge_on_true) {
	using Watchdog       = IR::LoopWatchdog;
	auto&      state     = ctx.state;
	const auto condition = conditional ? ctx.Def(latch.condition) : 0u;
	const auto loop_pc   = LoopWatchdogPc(state.program, header);
	const auto previous  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), previous, state.loop_watchdog_counter);
	const auto count = EmitAddU32(state, previous, ConstantU32(state, 1));
	state.builder.AddFunction(spv::OpStore, state.loop_watchdog_counter, count);
	// Only the back edge that reaches the limit is a cut. An invocation made to leave its loops
	// after another's cut has its count set to the limit, and its next back edge, in an outer
	// loop, passes it without being counted again.
	auto tripped = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), tripped, count,
	                          ConstantU32(state, Watchdog::IterationLimit));
	// A latch about to leave its loop anyway is not cut.
	const auto continuing = [&](uint32_t cut) {
		if (!conditional) {
			return cut;
		}
		auto takes_back_edge = condition;
		if (!back_edge_on_true) {
			takes_back_edge = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLogicalNot, TypeBool(state), takes_back_edge,
			                          condition);
		}
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), result, takes_back_edge, cut);
		return result;
	};
	tripped          = continuing(tripped);
	const auto phase = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), phase, count,
	                          ConstantU32(state, Watchdog::AbortCheckInterval - 1u));
	const auto periodic = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), periodic, phase,
	                          ConstantU32(state, 0));
	const auto rare = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), rare, tripped, periodic);
	EmitIfCondition(state, rare, [&]() {
		const auto scope   = ConstantU32(state, spv::ScopeDevice);
		const auto relaxed = ConstantU32(state, spv::MemorySemanticsMaskNone);
		auto       cut_now = tripped;
		uint32_t   by_time = 0;
		const auto hash    = state.program.shader_hash;
		const auto slot    = Watchdog::Slot(hash);
		if (state.loop_watchdog_start != 0) {
			// Only the low clock word is kept: it wraps after seconds, well past any budget.
			const auto clock = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpReadClockKHR, TypeU32Vector(state, 2), clock, scope);
			const auto now = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), now, clock, 0);
			const auto first_check = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpIEqual, TypeBool(state), first_check, count,
			                          ConstantU32(state, Watchdog::AbortCheckInterval));
			const auto saved = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeU32(state), saved,
			                          state.loop_watchdog_start);
			const auto start = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, TypeU32(state), start, first_check, now,
			                          saved);
			state.builder.AddFunction(spv::OpStore, state.loop_watchdog_start, start);
			const auto elapsed = EmitBinaryU32(state, spv::OpISub, now, start);
			const auto budget  = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeU32(state), budget,
			                          LoopWatchdogElement(state, Watchdog::TickBudget));
			auto expired = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), expired, elapsed,
			                          budget);
			if (state.program.stage == ShaderType::Compute) {
				// The dispatch's clock starts at its first invocation's first abort check.
				const auto slot_start = LoopWatchdogElement(state, slot + Watchdog::SlotStart);
				const auto now_is_zero = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpIEqual, TypeBool(state), now_is_zero, now,
				                          ConstantU32(state, 0));
				const auto stamp = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpSelect, TypeU32(state), stamp, now_is_zero,
				                          ConstantU32(state, 1), now);
				const auto current = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), current, slot_start,
				                          scope, relaxed);
				const auto unset = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpIEqual, TypeBool(state), unset, current,
				                          ConstantU32(state, 0));
				const auto dispatch_start = EmitValueOrDefaultIfCondition(
				    state, unset, TypeU32(state), current, [&]() {
					    const auto original = state.builder.AllocateId();
					    state.builder.AddFunction(spv::OpAtomicCompareExchange, TypeU32(state),
					                              original, slot_start, scope, relaxed, relaxed,
					                              stamp, ConstantU32(state, 0));
					    const auto won = state.builder.AllocateId();
					    state.builder.AddFunction(spv::OpIEqual, TypeBool(state), won, original,
					                              ConstantU32(state, 0));
					    const auto chosen = state.builder.AllocateId();
					    state.builder.AddFunction(spv::OpSelect, TypeU32(state), chosen, won, stamp,
					                              original);
					    return chosen;
				    });
				// Another invocation's stamp may be a little ahead of this one's clock: a difference
				// that wrapped is not an overrun.
				const auto dispatch_elapsed = EmitBinaryU32(state, spv::OpISub, now, dispatch_start);
				const auto forward          = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpULessThan, TypeBool(state), forward,
				                          dispatch_elapsed, ConstantU32(state, 0x80000000u));
				const auto dispatch_budget = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpLoad, TypeU32(state), dispatch_budget,
				                          LoopWatchdogElement(state, Watchdog::DispatchTickBudget));
				const auto enabled = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), enabled,
				                          dispatch_budget, ConstantU32(state, 0));
				const auto over = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), over,
				                          dispatch_elapsed, dispatch_budget);
				auto dispatch_expired = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), dispatch_expired,
				                          enabled, over);
				const auto counted = dispatch_expired;
				dispatch_expired   = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), dispatch_expired,
				                          counted, forward);
				const auto per_invocation = expired;
				expired                   = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), expired, per_invocation,
				                          dispatch_expired);
			}
			// An invocation already made to leave its loops is not timed out again.
			const auto below_limit = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpULessThan, TypeBool(state), below_limit, count,
			                          ConstantU32(state, Watchdog::IterationLimit));
			const auto timed = expired;
			expired          = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), expired, timed,
			                          below_limit);
			const auto limit_reached = cut_now;
			const auto expired_now   = continuing(expired);
			cut_now                  = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), cut_now, limit_reached,
			                          expired_now);
			const auto under_limit = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLogicalNot, TypeBool(state), under_limit,
			                          limit_reached);
			by_time = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), by_time, under_limit,
			                          expired_now);
		}
		EmitIfCondition(state, cut_now, [&]() {
			const auto trips = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), trips,
			                          LoopWatchdogElement(state, Watchdog::TripCount), scope,
			                          relaxed, ConstantU32(state, 1));
			const auto slot_trips = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), slot_trips,
			                          LoopWatchdogElement(state, slot + Watchdog::SlotTrips), scope,
			                          relaxed, ConstantU32(state, 1));
			const auto most = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAtomicUMax, TypeU32(state), most,
			                          LoopWatchdogElement(state, Watchdog::CutIterations), scope,
			                          relaxed, count);
			if (by_time != 0) {
				EmitIfCondition(state, by_time, [&]() {
					const auto time_cuts = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), time_cuts,
					                          LoopWatchdogElement(state, Watchdog::TimeCuts), scope,
					                          relaxed, ConstantU32(state, 1));
					const auto slot_time_cuts = state.builder.AllocateId();
					state.builder.AddFunction(
					    spv::OpAtomicIAdd, TypeU32(state), slot_time_cuts,
					    LoopWatchdogElement(state, slot + Watchdog::SlotTimeCuts), scope, relaxed,
					    ConstantU32(state, 1));
				});
			}
			const std::array<std::pair<uint32_t, uint32_t>, 3> slot_record {{
			    {slot + Watchdog::SlotHashLow, static_cast<uint32_t>(hash)},
			    {slot + Watchdog::SlotHashHigh, static_cast<uint32_t>(hash >> 32u)},
			    {slot + Watchdog::SlotLoopPc, loop_pc},
			}};
			for (const auto& [index, value]: slot_record) {
				state.builder.AddFunction(spv::OpStore, LoopWatchdogElement(state, index),
				                          ConstantU32(state, value));
			}
			const auto claimed = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAtomicCompareExchange, TypeU32(state), claimed,
			                          LoopWatchdogElement(state, Watchdog::Claimed), scope, relaxed,
			                          relaxed, ConstantU32(state, 1), ConstantU32(state, 0));
			const auto first = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpIEqual, TypeBool(state), first, claimed,
			                          ConstantU32(state, 0));
			EmitIfCondition(state, first, [&]() {
				const std::array<std::pair<uint32_t, uint32_t>, 3> record {{
				    {Watchdog::HashLow, static_cast<uint32_t>(hash)},
				    {Watchdog::HashHigh, static_cast<uint32_t>(hash >> 32u)},
				    {Watchdog::LoopPc, loop_pc},
				}};
				for (const auto& [index, value]: record) {
					state.builder.AddFunction(spv::OpStore, LoopWatchdogElement(state, index),
					                          ConstantU32(state, value));
				}
			});
		});
		// The invocation's own cut, if any, counts: its iterations are used up too. Only cuts of this
		// shader do: another shader's runaway must not cut short a loop that would have finished.
		const auto trips = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), trips,
		                          LoopWatchdogElement(state, slot + Watchdog::SlotTrips), scope,
		                          relaxed);
		const auto cut = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), cut, trips,
		                          ConstantU32(state, 0));
		const auto remaining = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), remaining, cut,
		                          ConstantU32(state, Watchdog::IterationLimit), count);
		state.builder.AddFunction(spv::OpStore, state.loop_watchdog_counter, remaining);
	});
	if (!conditional) {
		return 0;
	}
	const auto exhausted = EmitLoopWatchdogExhausted(state);
	const auto result    = state.builder.AllocateId();
	if (back_edge_on_true) {
		const auto within = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalNot, TypeBool(state), within, exhausted);
		state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), result, condition, within);
	} else {
		state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), result, condition, exhausted);
	}
	return result;
}

void EmitStructuredTerminator(ValueEmitContext& ctx, const IR::Block* block,
                              const IR::BlockInfo& info, uint32_t precomputed_condition = 0,
                              const StructuredFunctionState* structured = nullptr) {
	const auto& program    = ctx.state.program;
	const auto& term       = info.terminator;
	const auto  split_latch = [&](const IR::Block* latch) -> const SplitLatch* {
		if (structured == nullptr) {
			return nullptr;
		}
		const auto found = structured->split_latches.find(latch);
		return found != structured->split_latches.end() ? &found->second : nullptr;
	};
	const auto emit_merge = [&]() {
		if (term.loop_header) {
			const auto* merge = TargetBlock(program, term.merge_block);
			const auto* cont  = TargetBlock(program, term.continue_block);
			if (merge != nullptr && cont != nullptr) {
				const auto* split = split_latch(cont);
				ctx.state.builder.AddFunction(spv::OpLoopMerge, ctx.Label(merge),
				                              split != nullptr ? split->continue_label
				                                               : ctx.Label(cont),
				                              spv::LoopControlMaskNone);
			}
		} else if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
		           term.merge_block != UINT32_MAX) {
			if (const auto* merge = TargetBlock(program, term.merge_block); merge != nullptr) {
				ctx.state.builder.AddFunction(spv::OpSelectionMerge, ctx.Label(merge),
				                              spv::SelectionControlMaskNone);
			}
		}
	};

	switch (term.kind) {
		case CFG::TerminatorKind::Branch: {
			const auto* target = TargetBlock(program, term.true_block);
			if (target == nullptr) {
				EmitReturn(ctx);
				return;
			}
			emit_merge();
			ctx.state.builder.AddFunction(spv::OpBranch, ctx.Label(target));
			return;
		}
		case CFG::TerminatorKind::ConditionalBranch: {
			const auto* true_block  = TargetBlock(program, term.true_block);
			const auto* false_block = TargetBlock(program, term.false_block);
			if (true_block == nullptr || false_block == nullptr || info.condition.IsEmpty()) {
				EmitReturn(ctx);
				return;
			}
			const auto condition =
			    precomputed_condition != 0 ? precomputed_condition : ctx.Def(info.condition);
			emit_merge();
			const auto* split = split_latch(block);
			const auto  label = [&](const IR::Block* target) {
				return split != nullptr && target == split->header ? split->continue_label
				                                                   : ctx.Label(target);
			};
			ctx.state.builder.AddFunction(spv::OpBranchConditional, condition, label(true_block),
			                              label(false_block));
			if (split != nullptr) {
				EmitLabel(ctx.state, split->continue_label);
				ctx.state.builder.AddFunction(spv::OpBranch, ctx.Label(split->header));
			}
			return;
		}
		default: EmitReturn(ctx); return;
	}
}

void EmitDispatcherTarget(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                          const IR::Block* from, uint32_t target) {
	const auto* block = TargetBlock(ctx.state.program, target);
	if (block != nullptr) {
		StoreDispatcherPhiEdge(ctx, dispatcher, from, block);
		if (ctx.other_half != nullptr) {
			StoreDispatcherPhiEdge(*ctx.other_half, dispatcher, from, block);
		}
	}
}

uint32_t EmitDispatcherNextPc(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                              const IR::Block* block, const IR::BlockInfo& info) {
	const auto& term = info.terminator;
	switch (term.kind) {
		case CFG::TerminatorKind::Branch:
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			return ConstantU32(ctx.state, term.true_block);
		case CFG::TerminatorKind::ConditionalBranch: {
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			EmitDispatcherTarget(ctx, dispatcher, block, term.false_block);
			const auto selected = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(
			    spv::OpSelect, TypeU32(ctx.state), selected, ctx.Def(info.condition),
			    ConstantU32(ctx.state, term.true_block), ConstantU32(ctx.state, term.false_block));
			return selected;
		}
		case CFG::TerminatorKind::IndirectBranch: {
			for (const auto target: term.indirect_targets) {
				EmitDispatcherTarget(ctx, dispatcher, block, target);
			}
			uint32_t selected = ConstantU32(ctx.state, UINT32_MAX);
			if (!info.indirect_target.IsEmpty()) {
				const auto  selector = ctx.Def(info.indirect_target);
				const auto& values   = term.indirect_selector_code != UINT32_MAX
				                           ? term.indirect_selector_values
				                           : term.indirect_target_pcs;
				const auto& targets  = term.indirect_selector_code != UINT32_MAX
				                           ? term.indirect_selector_targets
				                           : term.indirect_targets;
				for (size_t index = 0; index < std::min(values.size(), targets.size()); index++) {
					const auto match = ctx.state.builder.AllocateId();
					const auto next  = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpIEqual, TypeBool(ctx.state), match,
					                              selector, ConstantU32(ctx.state, values[index]));
					ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), next, match,
					                              ConstantU32(ctx.state, targets[index]), selected);
					selected = next;
				}
			}
			return selected;
		}
		default: return ConstantU32(ctx.state, UINT32_MAX);
	}
}

template <typename T>
decltype(auto) Arg(ValueEmitContext& ctx, const IR::Inst& inst, size_t index) {
	if constexpr (std::is_same_v<T, const IR::Inst&>) {
		return inst;
	} else if constexpr (std::is_same_v<T, IR::Value>) {
		return inst.Arg(index);
	} else if constexpr (std::is_same_v<T, IR::ScalarReg>) {
		return inst.Arg(index).ScalarRegister();
	} else {
		static_assert(std::is_same_v<T, uint32_t>);
		return ctx.Def(inst.Arg(index));
	}
}

template <typename Context, typename Return, typename... Args>
void Invoke(Return (*emit)(Context&, Args...), ValueEmitContext& ctx, const IR::Inst& inst) {
	// A full instruction keeps metadata and predicated/lane operand loads lazy.
	static_assert(std::is_same_v<Context, ValueEmitContext> ||
	              std::is_same_v<Context, EmitterState>);
	auto& context = [&]() -> Context& {
		if constexpr (std::is_same_v<Context, EmitterState>)
			return ctx.state;
		else
			return ctx;
	}();
	constexpr bool has_inst = (std::is_same_v<Args, const IR::Inst&> || ...);
	[&]<size_t... I>(std::index_sequence<I...>) {
		static_assert(((!std::is_same_v<Args, const IR::Inst&> || I == 0) && ...));
		const auto call = [&] {
			return emit(context, Arg<Args>(ctx, inst, I - (has_inst && I != 0))...);
		};
		if constexpr (std::is_void_v<Return>) {
			call();
		} else {
			static_assert(std::is_same_v<Return, uint32_t>);
			ctx.Define(inst, call());
		}
	}(std::index_sequence_for<Args...> {});
}

// A link select(index == constant, value, rest) of an indexed select chain. V_MOVRELS reads
// become one: a chain over every register above the source, on a uniform index (M0).
struct IndexedSelectLink {
	IR::Value index;
	uint32_t  constant = 0;
	IR::Value value;
	IR::Value rest;
};

bool IndexedSelectLinkOf(const IR::Inst& inst, IndexedSelectLink& link) {
	if (inst.GetOpcode() != IR::ValueOpcode::SelectU32 || inst.NumArgs() != 3) {
		return false;
	}
	const auto* compare = inst.Arg(0).Resolve().TryInstruction();
	if (compare == nullptr || compare->GetOpcode() != IR::ValueOpcode::IEqual32) {
		return false;
	}
	auto index    = compare->Arg(0).Resolve();
	auto constant = compare->Arg(1).Resolve();
	if (index.IsImmediate()) {
		std::swap(index, constant);
	}
	if (index.IsImmediate() || !constant.IsImmediate() || constant.GetType() != IR::Type::U32) {
		return false;
	}
	link = {index, constant.U32(), inst.Arg(1), inst.Arg(2)};
	return true;
}

// The link below `inst` on the same index, when only `inst` uses it and it is in the same block.
const IR::Inst* NextIndexedSelect(const IR::Inst& inst, const IndexedSelectLink& link,
                                  IndexedSelectLink& next) {
	const auto* rest = link.rest.Resolve().TryInstruction();
	if (rest == nullptr || rest->Parent() != inst.Parent() || rest->UseCount() != 1 ||
	    !IndexedSelectLinkOf(*rest, next) || !(next.index == link.index)) {
		return nullptr;
	}
	return rest;
}

// Shorter chains stay selects.
constexpr size_t MinIndexedSelects = 16;

// KYTY_DEBUG_INDEXED_SWITCH=0 keeps every chain as selects (A/B runs).
bool IndexedSwitchEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DEBUG_INDEXED_SWITCH");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void FindIndexedSelects(EmitterState& state) {
	state.indexed_selects_found = true;
	if (!IndexedSwitchEnabled()) {
		return;
	}
	std::unordered_set<const IR::Inst*> inner;
	IndexedSelectLink                   link;
	IndexedSelectLink                   next;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (IndexedSelectLinkOf(inst, link)) {
				if (const auto* rest = NextIndexedSelect(inst, link, next)) {
					inner.insert(rest);
				}
			}
		}
	}
	std::vector<const IR::Inst*> members;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (inner.contains(&inst) || !IndexedSelectLinkOf(inst, link)) {
				continue;
			}
			members.clear();
			for (const auto* current = &inst;;) {
				const auto* rest = NextIndexedSelect(*current, link, next);
				if (rest == nullptr) {
					break;
				}
				members.push_back(rest);
				current = rest;
				link    = next;
			}
			if (members.size() + 1u >= MinIndexedSelects) {
				state.indexed_select_heads.insert(&inst);
				state.indexed_select_members.insert(members.begin(), members.end());
			}
		}
	}
}

// Emits a long indexed select chain at its outermost select as an OpSwitch on the index, which
// the driver lowers to a few scalar branches on a uniform index, into short select chains.
// Returns false for other instructions.
bool EmitIndexedSelect(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (!state.indexed_selects_found) {
		FindIndexedSelects(state);
	}
	if (state.indexed_select_members.contains(&inst)) {
		return true;
	}
	if (!state.indexed_select_heads.contains(&inst)) {
		return false;
	}
	IndexedSelectLink link;
	IndexedSelectLinkOf(inst, link);
	const auto selector = ctx.Def(link.index);
	// Outer links take precedence over inner ones with the same constant, as in the chain.
	std::vector<std::pair<uint32_t, uint32_t>> cases; // Constant, value.
	std::unordered_set<uint32_t>               constants;
	for (const auto* current = &inst;;) {
		if (constants.insert(link.constant).second) {
			cases.emplace_back(link.constant, ctx.Def(link.value));
		}
		IndexedSelectLink next;
		const auto*       rest = NextIndexedSelect(*current, link, next);
		if (rest == nullptr) {
			break;
		}
		current = rest;
		link    = next;
	}
	const auto fallback = ctx.Def(link.rest);
	// A switch on index / 16 into short select chains: the driver runs one chain of at most 16
	// selects, and the code stays about as large as the whole chain. One case per index value
	// compiled twice as slowly (a block per register), for little more speed.
	constexpr uint32_t BucketBits = 4;
	std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> buckets;
	for (const auto& [constant, value]: cases) {
		if (value != fallback) {
			buckets[constant >> BucketBits].emplace_back(constant, value);
		}
	}
	const auto bucket_selector =
	    EmitBinaryU32(state, spv::OpShiftRightLogical, selector, ConstantU32(state, BucketBits));
	const auto            default_label = state.builder.AllocateId();
	const auto            merge_label   = state.builder.AllocateId();
	std::vector<uint32_t> labels;
	std::vector<uint32_t> switch_words {spv::OpSwitch, bucket_selector, default_label};
	for (const auto& [bucket, entries]: buckets) {
		labels.push_back(state.builder.AllocateId());
		switch_words.push_back(bucket);
		switch_words.push_back(labels.back());
	}
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(switch_words);
	std::vector<uint32_t> phi_words {spv::OpPhi, TypeU32(state), state.builder.AllocateId(),
	                                 fallback, default_label};
	EmitLabel(state, default_label);
	state.builder.AddFunction(spv::OpBranch, merge_label);
	size_t label_index = 0;
	for (const auto& [bucket, entries]: buckets) {
		const auto label = labels[label_index++];
		EmitLabel(state, label);
		// The constants are distinct, so the order of the selects does not matter.
		auto selected = fallback;
		for (const auto& [constant, value]: entries) {
			const auto match = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpIEqual, TypeBool(state), match, selector,
			                          ConstantU32(state, constant));
			const auto next = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, TypeU32(state), next, match, value, selected);
			selected = next;
		}
		state.builder.AddFunction(spv::OpBranch, merge_label);
		phi_words.push_back(selected);
		phi_words.push_back(label);
	}
	EmitLabel(state, merge_label);
	state.builder.AddFunction(phi_words);
	ctx.Define(inst, phi_words[2]);
	return true;
}

void EmitDirectInstruction(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::SelectU32 && EmitIndexedSelect(ctx, inst)) {
		return;
	}
	switch (inst.GetOpcode()) {
#define VALUE_OPCODE(name, ...)                                                                    \
	case IR::ValueOpcode::name: return Invoke(Emit##name, ctx, inst);
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.inc"
#undef VALUE_OPCODE
		default: ctx.Fail(inst, "has no direct SPIR-V emitter");
	}
}

void EmitStructuredInstruction(ValueEmitContext& ctx, StructuredFunctionState& structured,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0 || inst.NumArgs() == 0) {
			ctx.Fail(inst, "has no native SPIR-V representation");
		}
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			const auto* predecessor = inst.PhiBlock(index);
			if (predecessor == nullptr || !ctx.state.labels.contains(predecessor)) {
				ctx.Fail(inst, "has a predecessor outside the structured function");
			}
		}
		structured.deferred_phis.push_back(
		    {ctx.state.builder.AddDeferredPhi(type, ctx.Result(inst), inst.NumArgs()), &inst,
		     ctx.half});
		return;
	}
	EmitDirectInstruction(ctx, inst);
}

void EmitDispatcherInstruction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0) {
			ctx.Fail(inst, "cannot be loaded by the dispatcher");
		}
		ctx.state.builder.AddFunction(spv::OpLoad, type, ctx.Result(inst),
		                              dispatcher.spills[ctx.half].at(&inst));
		return;
	}
	EmitDirectInstruction(ctx, inst);
	if (const auto found = dispatcher.spills[ctx.half].find(&inst);
	    found != dispatcher.spills[ctx.half].end()) {
		ctx.state.builder.AddFunction(spv::OpStore, found->second,
		                              ctx.Def(IR::Value(const_cast<IR::Inst*>(&inst))));
	}
}

template <typename EmitInstruction>
void EmitBlock(ValueEmitContext& ctx, const IR::Block* block, EmitInstruction&& emit_instruction) {
	ctx.state.current_block = block;
	EmitLabel(ctx.state, ctx.Label(block));
	bool emitted_non_phi = false;
	for (const auto& inst: *block) {
		if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
			if (emitted_non_phi) {
				ctx.Fail(inst, "appears after a non-Phi instruction");
			}
		} else {
			emitted_non_phi = true;
		}
		for (uint32_t half = 0; half < ctx.state.lane_count; half++) {
			auto& lane          = half == 0 ? ctx : *ctx.other_half;
			ctx.state.lane_half = half;
			if (half == 0 || (inst.GetOpcode() != IR::ValueOpcode::Barrier &&
			                  inst.GetOpcode() != IR::ValueOpcode::MeshAllocate)) {
				emit_instruction(lane, inst);
			}
		}
		ctx.state.lane_half = 0;
	}
}

void PatchStructuredPhis(ValueEmitContext& ctx, StructuredFunctionState& structured) {
	for (const auto& deferred: structured.deferred_phis) {
		auto& lane = deferred.half == 0 ? ctx : *ctx.other_half;
		for (size_t index = 0; index < deferred.instruction->NumArgs(); index++) {
			const auto* predecessor = deferred.instruction->PhiBlock(index);
			const auto  found       = structured.block_exit_labels.find(predecessor);
			if (found == structured.block_exit_labels.end()) {
				ctx.Fail(*deferred.instruction, "has a predecessor that was not emitted");
			}
			auto label = found->second;
			// A split latch reaches its header through its own continue block.
			if (const auto split = structured.split_latches.find(predecessor);
			    split != structured.split_latches.end() &&
			    split->second.header == deferred.instruction->Parent()) {
				label = split->second.continue_label;
			}
			ctx.state.builder.PatchDeferredPhi(
			    deferred.phi, index, lane.Def(deferred.instruction->Arg(index)), label);
		}
	}
}

void EmitStructuredFunction(ValueEmitContext& ctx) {
	const auto& program = ctx.state.program;
	StructuredFunctionState      structured;
	std::unordered_set<uint32_t> watched_merges;
	for (size_t index = 0; index < program.blocks.size(); index++) {
		bool conditional = false;
		bool on_true     = false;
		const auto* header = IR::LoopLatchHeader(program, index, conditional, on_true);
		if (header == nullptr) {
			continue;
		}
		watched_merges.insert(header->terminator.merge_block);
		if (conditional) {
			structured.split_latches.emplace(
			    program.blocks[index],
			    SplitLatch {ctx.state.builder.AllocateId(), TargetBlock(program, header->id)});
		}
	}
	ctx.state.builder.AddFunction(spv::OpBranch, ctx.Label(program.blocks.front()));
	for (size_t index = 0; index < program.blocks.size(); index++) {
		const auto* block = program.blocks[index];
		EmitBlock(ctx, block, [&](ValueEmitContext& lane, const IR::Inst& inst) {
			EmitStructuredInstruction(lane, structured, inst);
		});
		const auto& info        = program.block_info[index];
		const auto& term        = info.terminator;
		uint32_t    condition   = 0;
		bool        conditional = false;
		bool        on_true     = false;
		if (ctx.state.loop_watchdog_counter != 0) {
			if (const auto* header = IR::LoopLatchHeader(program, index, conditional, on_true)) {
				condition = EmitLoopWatchdog(ctx, info, *header, conditional, on_true);
			} else if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
			           !info.condition.IsEmpty() &&
			           watched_merges.contains(term.true_block) !=
			               watched_merges.contains(term.false_block)) {
				condition =
				    EmitLoopWatchdogExit(ctx, info, watched_merges.contains(term.true_block));
			}
		}
		// The watchdog may have split the block; its branch leaves from the current label.
		structured.block_exit_labels.emplace(block, ctx.state.current_label);
		EmitStructuredTerminator(ctx, block, program.block_info[index], condition, &structured);
	}
	PatchStructuredPhis(ctx, structured);
}

void EmitDispatcherFunction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher) {
	auto&       state = ctx.state;
	const auto* entry = state.program.blocks.front();
	state.builder.AddFunction(spv::OpBranch, ctx.Label(entry));
	EmitBlock(ctx, entry, [&](ValueEmitContext& lane, const IR::Inst& inst) {
		EmitDispatcherInstruction(lane, dispatcher, inst);
	});
	const auto initial_pc =
	    EmitDispatcherNextPc(ctx, dispatcher, entry, state.program.block_info.front());
	const auto initial_parent = state.current_label;
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);

	EmitLabel(state, dispatcher.header_label);
	const auto pc      = state.builder.AllocateId();
	const auto next_pc = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), pc, initial_pc, initial_parent, next_pc,
	                          dispatcher.continue_label);
	const auto done = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), done, pc,
	                          ConstantU32(ctx.state, UINT32_MAX));
	state.builder.AddFunction(spv::OpLoopMerge, dispatcher.merge_label, dispatcher.continue_label,
	                          spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, done, dispatcher.merge_label,
	                          dispatcher.select_label);

	EmitLabel(state, dispatcher.select_label);
	state.builder.AddFunction(spv::OpSelectionMerge, dispatcher.after_switch_label,
	                          spv::SelectionControlMaskNone);
	std::vector<uint32_t> words {spv::OpSwitch, pc, dispatcher.after_switch_label};
	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		words.push_back(state.program.block_info[index].id);
		words.push_back(ctx.Label(state.program.blocks[index]));
	}
	state.builder.AddFunction(words);
	std::vector<uint32_t> next_pc_words {spv::OpPhi, TypeU32(state), next_pc,
	                                     ConstantU32(state, UINT32_MAX), dispatcher.select_label};

	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		EmitBlock(ctx, state.program.blocks[index],
		          [&](ValueEmitContext& lane, const IR::Inst& inst) {
			          EmitDispatcherInstruction(lane, dispatcher, inst);
		          });
		auto selected = EmitDispatcherNextPc(ctx, dispatcher, state.program.blocks[index],
		                                     state.program.block_info[index]);
		// A dispatcher has no loop exit to leave through, so a cut invocation ends.
		if (state.loop_watchdog_counter != 0) {
			if (const auto* target = IR::DispatcherBackEdgeTarget(state.program, index)) {
				EmitLoopWatchdog(ctx, state.program.block_info[index], *target, false, false);
				const auto exhausted = EmitLoopWatchdogExhausted(state);
				const auto next      = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpSelect, TypeU32(state), next, exhausted,
				                          ConstantU32(state, UINT32_MAX), selected);
				selected = next;
			}
		}
		next_pc_words.push_back(selected);
		next_pc_words.push_back(state.current_label);
		state.builder.AddFunction(spv::OpBranch, dispatcher.after_switch_label);
	}
	EmitLabel(state, dispatcher.after_switch_label);
	state.builder.AddFunction(next_pc_words);
	state.builder.AddFunction(spv::OpBranch, dispatcher.continue_label);
	EmitLabel(state, dispatcher.continue_label);
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);
	EmitLabel(state, dispatcher.merge_label);
	EmitReturn(ctx);
}

} // namespace

uint32_t TypeId(EmitterState& state, IR::Type type) {
	switch (type) {
		case IR::Type::U1: return TypeBool(state);
		case IR::Type::U8:
		case IR::Type::U16:
		case IR::Type::U32:
		case IR::Type::F16: return TypeU32(state);
		case IR::Type::U64: return TypeU64(state);
		case IR::Type::U32x2: return TypeU32Pair(state);
		case IR::Type::F32: return TypeF32(state);
		case IR::Type::F64: return TypeF64(state);
		case IR::Type::U32x3: return TypeU32Vector(state, 3);
		case IR::Type::U32x4: return TypeU32Vector(state, 4);
		case IR::Type::F32x2: return TypeF32Vector(state, 2);
		default: return 0;
	}
}

uint32_t ValueEmitContext::Def(IR::Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case IR::Type::U1: return ConstantBool(state, value.U1());
			case IR::Type::U8: return ConstantU32(state, value.U8());
			case IR::Type::U16: return ConstantU32(state, value.U16());
			case IR::Type::U32: return ConstantU32(state, value.U32());
			case IR::Type::U64: return ConstantU64(state, value.U64());
			case IR::Type::F16: return ConstantU32(state, value.F16Bits());
			case IR::Type::F32:
				return ConstantF32(state, std::bit_cast<uint32_t>(value.F32Value()));
			default: break;
		}
	}
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr) {
		Fail("direct SPIR-V emitter received a non-value argument");
	}
	if (dispatcher_spills != nullptr && state.current_block != nullptr &&
	    inst->Parent() != state.current_block) {
		if (const auto found = dispatcher_spills->find(inst); found != dispatcher_spills->end()) {
			if (const auto loaded = dispatcher_block_loads.find(inst);
			    loaded != dispatcher_block_loads.end() &&
			    loaded->second.first == state.current_label) {
				return loaded->second.second;
			}
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeId(state, inst->GetType()), id,
			                          found->second);
			dispatcher_block_loads.insert_or_assign(inst, std::pair {state.current_label, id});
			return id;
		}
	}
	return Result(*inst);
}

uint32_t ValueEmitContext::Arg(const IR::Inst& inst, size_t index) {
	return Def(inst.Arg(index));
}

uint32_t ValueEmitContext::HalfArg(const IR::Inst& inst, size_t index, uint32_t lane_half) {
	return lane_half == half ? Arg(inst, index) : other_half->Arg(inst, index);
}

uint32_t ValueEmitContext::Ballot(IR::Value predicate) {
	const auto ballot_type = TypeU32Vector(state, 4);
	const auto scope       = ConstantU32(state, spv::ScopeSubgroup);
	const auto live        = [&](uint32_t value) {
		if (state.helper_invocation_variable == 0) {
			return value;
		}
		const auto helper = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeBool(state), helper,
		                          state.helper_invocation_variable);
		const auto not_helper = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalNot, TypeBool(state), not_helper, helper);
		const auto masked = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), masked, value, not_helper);
		return masked;
	};
	const auto low = state.builder.AllocateId();
	state.builder.AddFunction(
	    spv::OpGroupNonUniformBallot, ballot_type, low, scope,
	    live(other_half == nullptr || half == 0 ? Def(predicate) : other_half->Def(predicate)));
	if (other_half == nullptr) {
		return low;
	}
	const auto high      = state.builder.AllocateId();
	const auto low_word  = state.builder.AllocateId();
	const auto high_word = state.builder.AllocateId();
	const auto ballot    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, high, scope,
	                          live(half == 1 ? Def(predicate) : other_half->Def(predicate)));
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low_word, low, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high_word, high, 0);
	state.builder.AddFunction(spv::OpCompositeConstruct, ballot_type, ballot, low_word, high_word,
	                          ConstantU32(state, 0), ConstantU32(state, 0));
	return ballot;
}

uint32_t ValueEmitContext::FirstLane(uint32_t ballot) {
	if (other_half == nullptr && WaveHalvesInHostSubgroup(state)) {
		const auto first = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, TypeU32(state), first, GlslStd450(state),
		                          GLSLstd450FindILsb, EmitOwnWaveHalfWord(state, ballot));
		return EmitAddU32(state, first, EmitOwnWaveHalfBase(state));
	}
	if (other_half == nullptr) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallotFindLSB, TypeU32(state), result,
		                          ConstantU32(state, spv::ScopeSubgroup), ballot);
		return result;
	}
	const auto low        = state.builder.AllocateId();
	const auto high       = state.builder.AllocateId();
	const auto low_first  = state.builder.AllocateId();
	const auto high_first = state.builder.AllocateId();
	const auto low_active = state.builder.AllocateId();
	const auto result     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), low_first, GlslStd450(state),
	                          GLSLstd450FindILsb, low);
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), high_first, GlslStd450(state),
	                          GLSLstd450FindILsb, high);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), low_active, low,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), result, low_active, low_first,
	                          EmitAddU32(state, high_first, ConstantU32(state, 32)));
	return result;
}

uint32_t ValueEmitContext::Shuffle(const IR::Inst& inst, size_t index, uint32_t lane) {
	const auto type  = TypeId(state, inst.Arg(index).GetType());
	const auto scope = ConstantU32(state, spv::ScopeSubgroup);
	const auto low   = state.builder.AllocateId();
	if (other_half == nullptr) {
		state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope, Arg(inst, index),
		                          lane);
		return low;
	}
	const auto physical_lane =
	    EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31));
	const auto high    = state.builder.AllocateId();
	const auto in_high = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope,
	                          HalfArg(inst, index, 0), physical_lane);
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, high, scope,
	                          HalfArg(inst, index, 1), physical_lane);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), in_high,
	                          EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 32)),
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, type, value, in_high, high, low);
	return value;
}

uint32_t ValueEmitContext::Result(const IR::Inst& inst) {
	if (const auto found = definitions.find(&inst); found != definitions.end()) {
		return found->second;
	}
	const auto id = state.builder.AllocateId();
	definitions.emplace(&inst, id);
	return id;
}

uint32_t ValueEmitContext::Define(const IR::Inst& inst, uint32_t value) {
	if (const auto found = definitions.find(&inst); found != definitions.end()) {
		if (found->second != value) {
			state.builder.AddFunction(spv::OpCopyObject, TypeId(state, inst.GetType()),
			                          found->second, value);
		}
		return found->second;
	}
	definitions.emplace(&inst, value);
	return value;
}

uint32_t ValueEmitContext::ResourceIndex(IR::Value value, IR::ValueOpcode opcode) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != opcode) {
		Fail("typed resource handle has the wrong producer");
	}
	return inst->Flags<uint32_t>();
}

const IR::Inst* ValueEmitContext::ImageAddress(IR::Value value) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != IR::ValueOpcode::MakeImageAddress) {
		Fail("typed image address was not constructed by MakeImageAddress");
	}
	return inst;
}

const IR::MemoryInfo& ValueEmitContext::Memory(const IR::Inst& inst) const {
	return state.program.memory_info.at(inst.Flags<IR::MemoryFlags>().index);
}

const IR::ExportInfo& ValueEmitContext::Export(const IR::Inst& inst) const {
	return state.program.export_info.at(inst.Flags<IR::ExportFlags>().index);
}

uint32_t ValueEmitContext::Label(const IR::Block* block) const {
	return state.labels.at(block);
}

[[noreturn]] void ValueEmitContext::Fail(const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage), reason);
	std::abort();
}

[[noreturn]] void ValueEmitContext::Fail(const IR::Inst& inst, const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u opcode=%s reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage),
	     IR::ValueOpcodeName(inst.GetOpcode()), reason);
	std::abort();
}

void EmitProgram(EmitterState& state) {
	const auto&      program = state.program;
	ValueEmitContext ctx(state);
	ValueEmitContext high(state);
	if (state.lane_count == 2) {
		ctx.other_half  = &high;
		high.other_half = &ctx;
		high.half       = 1;
	}
	std::optional<DispatcherFunctionState> dispatcher;
	if (state.program.stage == ShaderType::Pixel && state.requirements.pixel_valid_mask) {
		state.pixel_valid_mask_variable = state.builder.AllocateId();
		state.builder.AddName(state.pixel_valid_mask_variable, "pixel_valid_mask_active");
	}
	for (const auto* block: program.blocks) {
		const auto label = state.builder.AllocateId();
		state.labels.emplace(block, label);
	}
	if (state.program.dispatcher_fallback) {
		auto& dispatch = dispatcher.emplace();
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() != IR::ValueOpcode::Phi) {
					continue;
				}
				if (SpillPointerType(ctx, inst.GetType()) == 0) {
					ctx.Fail(inst, "cannot be stored by the dispatcher");
					break;
				}
				dispatch.spills[0].emplace(&inst, state.builder.AllocateId());
			}
		}
		const auto mark_cross_block = [&](IR::Value value, const IR::Block* consumer) {
			value                  = value.Resolve();
			const auto* definition = value.TryInstruction();
			if (definition == nullptr || definition->Parent() == consumer ||
			    definition->Parent() == program.blocks.front()) {
				return;
			}
			if (SpillPointerType(ctx, definition->GetType()) == 0) {
				ctx.Fail(*definition, "cannot be stored by the dispatcher");
				return;
			}
			if (!dispatch.spills[0].contains(definition)) {
				dispatch.spills[0].emplace(definition, state.builder.AllocateId());
			}
		};
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				for (size_t index = 0; index < inst.NumArgs(); index++) {
					const auto* consumer =
					    inst.GetOpcode() == IR::ValueOpcode::Phi ? inst.PhiBlock(index) : block;
					mark_cross_block(inst.Arg(index), consumer);
				}
			}
		}
		for (size_t index = 0; index < program.blocks.size(); index++) {
			mark_cross_block(program.block_info[index].condition, program.blocks[index]);
			mark_cross_block(program.block_info[index].indirect_target, program.blocks[index]);
		}
		dispatch.header_label       = state.builder.AllocateId();
		dispatch.select_label       = state.builder.AllocateId();
		dispatch.after_switch_label = state.builder.AllocateId();
		dispatch.continue_label     = state.builder.AllocateId();
		dispatch.merge_label        = state.builder.AllocateId();
		ctx.dispatcher_spills       = &dispatch.spills[0];
		if (state.lane_count == 2) {
			for (const auto& [inst, id]: dispatch.spills[0]) {
				dispatch.spills[1].emplace(inst, state.builder.AllocateId());
			}
			high.dispatcher_spills = &dispatch.spills[1];
		}
	}
	DefineGetBdaPointer(state);
	for (const auto* block: program.blocks) {
		if (std::ranges::any_of(*block, [](const IR::Inst& inst) {
			    return inst.GetOpcode() == IR::ValueOpcode::SwizzleU32 ||
			           inst.GetOpcode() == IR::ValueOpcode::SharedAtomicFMin32 ||
			           inst.GetOpcode() == IR::ValueOpcode::SharedAtomicFMax32;
		    })) {
			ctx.scratch_u32_variable = state.builder.AllocateId();
			if (state.lane_count == 2) {
				high.scratch_u32_variable = state.builder.AllocateId();
			}
			break;
		}
	}
	if (state.loop_watchdog_variable != 0) {
		state.loop_watchdog_counter = state.builder.AllocateId();
		if (state.program.loop_watchdog_clock) {
			state.loop_watchdog_start = state.builder.AllocateId();
		}
	}
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state),
	                          state.mesh_guest_func != 0 ? state.mesh_guest_func : state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.entry_label);
	if (state.requirements.function_lds) {
		state.builder.AddFunction(
		    spv::OpVariable,
		    TypeU32ArrayPointer(state, spv::StorageClassFunction, LdsDwordCount(state)),
		    state.lds_variable, spv::StorageClassFunction);
	}
	if (state.requirements.function_scratch) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.builder.AddFunction(
			    spv::OpVariable,
			    TypeU32ArrayPointer(state, spv::StorageClassFunction, state.program.scratch_dwords),
			    state.scratch_variable[half], spv::StorageClassFunction);
		}
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.pixel_valid_mask_variable, spv::StorageClassFunction);
	}
	if (state.loop_watchdog_counter != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.loop_watchdog_counter, spv::StorageClassFunction,
		                          ConstantU32(state, 0));
	}
	if (state.loop_watchdog_start != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.loop_watchdog_start, spv::StorageClassFunction,
		                          ConstantU32(state, 0));
	}
	for (uint32_t half = 0; half < state.lane_count; half++) {
		auto& lane = half == 0 ? ctx : high;
		if (state.program.dispatcher_fallback) {
			for (const auto* block: program.blocks) {
				for (const auto& inst: *block) {
					if (const auto found = dispatcher->spills[half].find(&inst);
					    found != dispatcher->spills[half].end()) {
						state.builder.AddFunction(spv::OpVariable,
						                          SpillPointerType(lane, inst.GetType()),
						                          found->second, spv::StorageClassFunction);
					}
				}
			}
		}
		if (lane.scratch_u32_variable != 0) {
			state.builder.AddFunction(spv::OpVariable,
			                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
			                          lane.scratch_u32_variable, spv::StorageClassFunction);
		}
	}
	if (state.gds_variable != 0) {
		state.gds_length = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), state.gds_length,
		                          state.gds_variable, 0);
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable,
		                          ConstantU32(state, 1));
	}
	EmitMemoryOffsets(state);
	if (program.blocks.empty()) {
		EmitReturn(ctx);
	} else if (state.program.dispatcher_fallback) {
		EmitDispatcherFunction(ctx, *dispatcher);
	} else {
		EmitStructuredFunction(ctx);
	}
	state.builder.AddFunction(spv::OpFunctionEnd);
	if (state.program.stage == ShaderType::Mesh) {
		EmitMeshEntryPoint(state);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
