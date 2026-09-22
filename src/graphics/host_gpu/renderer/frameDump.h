#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEDUMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEDUMP_H_

#include <cstdint>
#include <string>

namespace Libs::Graphics {

class RenderContext;

// Debug capture: when "kyty_dump.trigger" exists in the working directory, writes a preview and
// channel statistics of every live render target to _FrameDump/NN, then deletes the trigger.
void FrameDumpOnFlip(RenderContext& context, uint64_t surface_address);

// Watching one target: when "kyty_watch.trigger" holds a guest address, every draw and dispatch
// that writes that image over the next frame appends what it left in it to _FrameDump/watch.txt.
// That is how a stray value is traced back to the operation that wrote it.
[[nodiscard]] uint64_t FrameDumpWatchAddress();

// Records a copy of the watched image, so it must not be called inside a render pass. `kind` and
// the shader addresses only label the line; `details` is appended to it, for what was bound.
void FrameDumpWatchOperation(RenderContext& context, uint64_t address, const char* kind,
                             uint64_t first_shader, uint64_t second_shader,
                             const std::string& details = {});

// The same for an image the operation read, so that an output already fed by a spoiled input can
// be told apart from one an operation spoiled itself.
void FrameDumpWatchInput(RenderContext& context, uint64_t address, const char* role);

// Writes a guest program beside the watch log, once per address, so it can be disassembled.
void FrameDumpWatchShader(uint64_t address);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMEDUMP_H_
