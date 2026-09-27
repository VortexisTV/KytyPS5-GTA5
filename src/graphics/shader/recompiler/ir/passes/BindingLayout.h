#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

void AllocateBindings(Program& program, uint32_t push_data_start_dword = 0);

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind);

// The loop header a structured loop's latch (its continue target, a block apart from the header)
// takes the back edge to, or null for any other block. conditional tells whether the latch's
// branch can also leave the loop, and back_edge_on_true which of its targets is the back edge.
// The loop watchdog counts iterations in every such latch.
const BlockInfo* LoopLatchHeader(const Program& program, size_t index, bool& conditional,
                                 bool& back_edge_on_true);
// LoopLatchHeader for a conditional latch only. The SPIR-V emitter gives such a latch its own
// continue block.
const BlockInfo* ConditionalLatchHeader(const Program& program, size_t index,
                                        bool& back_edge_on_true);
// In a dispatcher-fallback program, the earliest block a block can branch back to (one starting at
// or before it, which any guest loop needs), or null. The loop watchdog counts each run of such a
// block as a loop iteration.
const BlockInfo* DispatcherBackEdgeTarget(const Program& program, size_t index);
bool             UsesLoopWatchdog(const Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_ */
