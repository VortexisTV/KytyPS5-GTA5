#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUCRASHDIAGNOSTICS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUCRASHDIAGNOSTICS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

enum class GpuCheckpointKind : uint32_t {
	Dispatch,
	DispatchIndirect,
	// Recorded after a dispatch's completion barrier: reaching it means the dispatch finished.
	DispatchDone,
	Draw,
	DrawMesh,
	// Work the emulator records on its own.
	FaultProcess,
	Tiler,
	DccClear,
	Blit,
};

// Guest memory a dispatch reads, captured at record time so a report can show what the game wrote.
struct GpuCheckpointBuffer {
	uint64_t address = 0;
	uint64_t size    = 0;
};

// The inputs of a dispatch, as the shader sees them when it is recorded.
struct GpuCheckpointDispatch {
	std::span<const uint32_t>            user_data;
	// Values the SRT walker read on the CPU for the shader: SRT loads and constant-offset reads.
	std::span<const uint32_t>            flattened_srt;
	std::span<const GpuCheckpointBuffer> buffers;
	// Thread counts of a dispatch sized in threads; zero for one sized in workgroups.
	std::array<uint32_t, 3> dispatch_threads {};
};

// Records the shaders of the draw or dispatch recorded next and marks the command stream with a
// VK_NV_device_diagnostic_checkpoints checkpoint, so a lost device can name the work it was running.
// Does nothing when the device lacks the extension.
void MarkGpuCheckpoint(const GraphicContext& graphics, vk::CommandBuffer buffer,
                       GpuCheckpointKind kind, uint64_t submit_id,
                       std::span<const uint64_t> shader_hashes, uint32_t arg0 = 0,
                       uint32_t arg1 = 0, uint32_t arg2 = 0,
                       const GpuCheckpointDispatch* dispatch = nullptr);

// Writes what the device reports about a failed call to the console and log: the
// VK_EXT_device_fault description and addresses, and the draws and dispatches the queue's last
// checkpoints mark. The code of each shader it names is dumped as DumpDiagnosticShader does. Only
// the first report is written; later failures follow from the same loss.
void ReportGpuFailure(const GraphicContext& graphics, const char* what, vk::Result result);

// Keeps a copy of a shader's guest code so a later report can dump it after the guest has moved on.
void RegisterDiagnosticShader(uint64_t hash, std::span<const uint32_t> code);

// Writes a registered shader's code and its decoded RDNA2 listing to
// <shader log folder>/diagnostics/<hash>_<reason>.bin and .rdna2, once per shader and reason.
void DumpDiagnosticShader(uint64_t hash, const char* reason);

// Names the shader the calling thread translates while it lives. A fatal error on that thread,
// such as an instruction the recompiler does not support, dumps the shader as
// DumpDiagnosticShader(hash, "fatal") does.
class DiagnosticShaderScope {
public:
	explicit DiagnosticShaderScope(uint64_t hash);
	~DiagnosticShaderScope();
	DiagnosticShaderScope(const DiagnosticShaderScope&)            = delete;
	DiagnosticShaderScope& operator=(const DiagnosticShaderScope&) = delete;

private:
	uint64_t m_previous = 0;
};

class Buffer;

// Sets the host-visible buffer shaders report to when the recompiler's loop watchdog cuts a guest
// loop short (laid out as ShaderRecompiler::IR::LoopWatchdog); null forgets it.
void RegisterLoopWatchdog(Buffer* buffer);

// Logs loops the watchdog cut short since the last call and dumps each shader it cut, then lets
// the next cut record its own shader. Call on the GPU thread.
void ReportLoopWatchdog();

// Notes a dispatch of a shader that stores through DMA; the first one dumps the shader's code as
// DumpDiagnosticShader(hash, "dma_writer") does. TakeDmaWriters returns the shaders noted since its
// last call, the ones a report of dropped DMA stores names.
void                                NoteDmaWriter(uint64_t hash);
[[nodiscard]] std::vector<uint64_t> TakeDmaWriters();

// Logs DMA stores the BDA page table dropped, by the guest pages no cached buffer covered when a
// shader stored to them, and the DMA-storing shaders dispatched meanwhile.
void ReportDroppedDmaStores(std::span<const uint64_t> pages, std::span<const uint64_t> writers,
                            uint64_t page_size);

// Remembers the latest dispatch of a shader that reaches guest memory through the BDA page table:
// its user data and the guest ranges it binds or caches ahead. When the loop watchdog first cuts
// the shader, those ranges are dumped next to its code, as the GPU and guest memory hold them.
void RecordDmaDispatch(uint64_t hash, std::span<const uint32_t> user_data,
                       std::vector<std::pair<uint64_t, uint64_t>> ranges);
// Adds the dispatch's compute input info, as raw bytes, and its group counts, so a dump
// can be replayed.
void RecordDmaDispatchShape(uint64_t hash, std::span<const uint8_t> input,
                            std::array<uint32_t, 3> groups);

// Remembers the guest ranges a compute dispatch writes through its bound buffers, and whether it
// also stores through DMA. When a runaway shader's memory is dumped, the recent dispatches that
// wrote it are listed in <hash>_writers.txt and their code is dumped.
void RecordDispatchWrites(uint64_t hash, std::span<const std::pair<uint64_t, uint64_t>> ranges,
                          bool dma_writer, std::array<uint32_t, 3> groups);

// Reads guest bytes for a dump: `gpu` gets them from the buffer that caches them (left empty when
// none does) and `cpu` from guest memory; `note` describes where they came from. Set by the buffer
// cache and called on the GPU thread; an empty reader forgets it.
using DiagnosticMemoryReader =
    std::function<void(uint64_t address, uint64_t size, std::vector<uint8_t>& gpu,
                       std::vector<uint8_t>& cpu, std::string& note)>;
void RegisterDiagnosticMemoryReader(DiagnosticMemoryReader reader);

// The guest range of the cached buffer that holds `address`, or {0, 0} when none does. Set by the
// buffer cache beside the reader, so a dump can take a whole structure the GPU built in a buffer,
// such as a ray-tracing BVH pool whose trees a shader reaches through pointers.
using DiagnosticBufferExtent = std::function<std::pair<uint64_t, uint64_t>(uint64_t address)>;
void RegisterDiagnosticBufferExtent(DiagnosticBufferExtent extent);

// Guest shaders the loop watchdog was recorded cutting in this many separate reports.
constexpr uint32_t RunawayShaderReports = 3;

// The shaders the loop watchdog keeps cutting (see RunawayShaderReports). The generation changes
// whenever one is added, so a caller can keep its own copy and refresh it cheaply.
[[nodiscard]] uint32_t              RunawayShaderGeneration();
[[nodiscard]] std::vector<uint64_t> RunawayShaders();

// A shader the watchdog cuts only for time is slow, not stuck: instead of listing it as a runaway,
// its direct dispatches run in this many bands of workgroup rows, each its own submission, so none
// nears the Windows GPU timeout. Starts at 1 and doubles after each report that cut it, up to
// MaxDispatchSplit; only then do its cuts count towards RunawayShaderReports.
constexpr uint32_t                  MaxDispatchSplit = 64;
[[nodiscard]] uint32_t              DispatchSplit(uint64_t hash);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUCRASHDIAGNOSTICS_H_
