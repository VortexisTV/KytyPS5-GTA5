#ifndef KYTY_COMMON_PERFSTATS_H_
#define KYTY_COMMON_PERFSTATS_H_

#include "common/common.h"

#include <array>
#include <cstdint>
#include <string>

// Per-interval performance statistics for finding the part of the emulator a slow scene loads.
// Collection is off unless the environment variable KYTY_PERF_STATS=1 is set.
// When on, a summary is printed after every second of guest frames and each interval is 
// appended as one row to _PerfStats.csv.

namespace PerfStats {

// Wall-clock time with an occurrence count and the longest single occurrence. Spans nest: the draw
// phases are part of Draw, the shader phases are part of DrawShaders, and Draw, GpuWait and
// FlipWait are part of GpuThreadBusy.
enum class SpanId : uint8_t {
	GpuThreadBusy,      // emulated GPU thread executing submissions or queued commands
	GpuThreadIdle,      // emulated GPU thread waiting for work
	GpuThreadBlocked,   // every queued submission suspended on a wait packet
	GpuThreadCommands,  // commands other threads ran on the emulated GPU thread
	GameWaitGpuIdle,    // guest thread blocked at a suspend point until the GPU thread drains
	GameWaitGpuCommand, // guest thread blocked on a synchronous GPU-thread command
	Draw,
	DrawTargets,        // render-target resolution and attachment acquisition
	DrawShaders,        // shader program lookup, including resource materialization
	ShaderPrepare,      // vertex and pixel shader header and resource parsing
	ShaderKey,          // static-state key and program table lookup; per stage, compute included
	ShaderMaterialize,  // resource snapshot and specialization from guest memory; per stage
	ShaderSrtWalk,      // part of ShaderMaterialize: SRT walk resolving descriptors and constants
	ShaderIndirect,     // part of ShaderMaterialize: indirect (bindless) image table probing
	ShaderSpecialize,   // part of ShaderMaterialize: specialization from resolved descriptors
	ShaderPermutation,  // specialization hash and compiled permutation search; per stage
	DrawBindings,       // textures, samplers, storage buffers and DMA preparation
	BindResolve,        // part of DrawBindings: textures, samplers and user data; compute included
	BindFindBuffers,    // part of DrawBindings: storage buffer cache lookups
	BindDmaSources,     // part of DrawBindings: caching the pages DMA address bases name
	BindRebindBuffers,  // part of DrawBindings: storage buffer descriptors and uploads
	BindRebindImages,   // part of DrawBindings: image view lookups after resolution
	DrawVertexIndex,    // vertex and index buffer acquisition
	DrawPipeline,       // graphics pipeline lookup or creation
	DrawRecord,         // descriptor commit, dynamic state and draw recording
	Dispatch,
	BdaPrepare,         // synchronizing cached buffers for a DMA draw or dispatch (or skipping it)
	HotPageHash,        // hashing hot pages for CPU changes, in BDA passes and buffer uploads
	BufferDownload,     // GPU-to-guest buffer readback, including the GPU drain
	QueueSubmit,
	GpuWait,            // emulated GPU thread blocked on host GPU completion
	GpuWaitReadback,    // part of GpuWait: reading GPU-written memory back into guest memory
	GpuWaitFaults,      // part of GpuWait: collecting the shader fault buffer
	GpuWaitStream,      // part of GpuWait: room in a stream or download ring buffer
	GpuWaitPredicate,   // part of GpuWait: the GPU writes a predicate is resolved from
	FlipWait,           // emulated GPU thread blocked until a flip is presented
	Present,            // presentation, including FIFO vsync blocking
	PageFault,          // guest access faults on GPU-tracked pages
	VirtualMapEdit,     // changing the guest virtual memory map, which holds its lock
	VirtualMapQuery,    // reading the guest virtual memory map, which holds its lock
	VirtualMapWait,     // a range check waiting for the virtual memory map lock
	GarbageCollect,
	ShaderCompileSync,  // translation and compilation on the emulated GPU thread
	ShaderCompileAsync, // translation and compilation on worker threads
	PipelineCreateSync,
	PipelineCreateAsync,
	PipelineCacheSave,
	AudioPushGap,       // time between accepted pushes on one AudioOut2 context
	AudioQueueWait,     // blocking audio output waiting for room in the host device queue
	Count
};

enum class CounterId : uint8_t {
	DrawsSkippedShader,
	DrawsSkippedPipeline,
	DrawsSkippedStage,        // unsupported shader stage mask or geometry-engine registers
	DrawsSkippedTessellation, // of those, the ones with a tessellation (LS or HS) shader
	DrawsSkippedVertexShader, // no vertex shader address
	DrawsSkippedState,        // no usable render targets or an unsupported topology
	DrawsSkippedEmpty,        // zero vertices, indices or instances
	DrawsIndirect,            // draws whose counts came from indirect arguments in guest memory
	DrawsIndexClamped,        // indirect draws whose index count INDEX_BUFFER_SIZE cut short
	PredicateSkips,           // predication evaluations that turned packet skipping on
	PredicateStale,           // predicates read without waiting for the GPU writes producing them
	PacketsSkippedPredicated, // command packets skipped because predication was on
	BdaBuffersVisited,
	BdaPassesSkipped, // BDA preparations with nothing new to upload since the last complete pass
	HotPagesHashed,   // hot pages hashed to find whether the CPU changed them
	HotPagesChanged,  // hashed hot pages whose contents had changed
	HotPagesCooled,   // hot pages that stopped changing and left the hot set
	BufferUploads,
	BufferUploadBytes,
	StreamUploads, // small CPU-modified bindings copied into the stream buffer
	StreamUploadBytes,
	BufferCreates,
	BufferDownloadBytes,
	ReadbacksShadow,     // buffer readbacks answered from a host-visible shadow copy
	ReadbackNoOwner,     // readbacks drained the GPU: no single small owning buffer
	ReadbackNotHot,      // readbacks drained the GPU: the owner had no shadow recorded yet
	ReadbackShadowStale, // readbacks drained the GPU: its shadow was missing or too old
	ReadbackRecentWrite, // readbacks drained the GPU: the GPU wrote the range after the shadow
	ShadowsRecorded,     // shadow copies of readback-hot buffers taken at a flush
	BuffersEvicted,
	ImageCreates,
	ImageUploads,
	ImageUploadBytes,
	ImagesEvicted,
	WriteFaults,
	ReadFaults,
	ShadersCompiled,
	PipelinesCreated,
	ShaderGuestReads,   // guest words materialization read through the GPU-clean check
	ShaderRangeChecks,  // guest buffer ranges materialization validated
	IndirectProbes,     // indirect image table entries probed during materialization
	SrtInterpreted,     // SRT evaluations that fell back from compiled code to the interpreter
	ShaderInputRepeats, // materializations whose program and user data were seen recently
	AudioPushesSync,
	AudioPushesAsync,
	AudioPushesNotReady, // asynchronous pushes refused because the queue was full
	AudioUnderruns,      // host device queue found empty after playback started
	Count
};

// Most recent value; gauges are not reset between intervals.
enum class GaugeId : uint8_t {
	GpuMemoryMb,
	BufferGcTriggerMb,
	TextureGcTriggerMb,
	CachedBuffers,
	VirtualRanges, // entries in the guest virtual memory map
	Count
};

inline constexpr size_t SpanCount    = static_cast<size_t>(SpanId::Count);
inline constexpr size_t CounterCount = static_cast<size_t>(CounterId::Count);
inline constexpr size_t GaugeCount   = static_cast<size_t>(GaugeId::Count);

namespace Detail {

// Written during initialization, before emulation threads start.
inline bool g_enabled = false;

void RecordSpan(SpanId id, uint64_t ticks) noexcept;
void AddCounter(CounterId id, uint64_t value) noexcept;
void SetGauge(GaugeId id, uint64_t value) noexcept;

} // namespace Detail

[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_enabled;
}

// Common::Timer::QueryPerformanceCounter ticks.
[[nodiscard]] uint64_t Now() noexcept;
[[nodiscard]] uint64_t TicksPerSecond() noexcept;

inline void Add(CounterId id, uint64_t value = 1) noexcept {
	if (Enabled()) {
		Detail::AddCounter(id, value);
	}
}

inline void Set(GaugeId id, uint64_t value) noexcept {
	if (Enabled()) {
		Detail::SetGauge(id, value);
	}
}

// Records a duration the caller measured itself, such as the time between two events.
inline void Record(SpanId id, uint64_t ticks) noexcept {
	if (Enabled()) {
		Detail::RecordSpan(id, ticks);
	}
}

// Records the time from construction to Stop() or destruction, whichever comes first. A false
// condition makes the span inert.
class Span final {
public:
	explicit Span(SpanId id, bool condition = true) noexcept
	    : m_id(id), m_active(condition && Enabled()), m_start(m_active ? Now() : 0) {}
	~Span() { Stop(); }

	void Stop() noexcept {
		if (m_active) {
			m_active = false;
			Detail::RecordSpan(m_id, Now() - m_start);
		}
	}

	KYTY_CLASS_NO_COPY(Span);

private:
	SpanId   m_id;
	bool     m_active;
	uint64_t m_start;
};

struct SpanTotal {
	uint64_t ticks     = 0;
	uint64_t count     = 0;
	uint64_t max_ticks = 0;
};

struct Snapshot {
	double                             elapsed_seconds = 0.0; // since collection started
	uint64_t                           interval_ticks  = 0;
	uint64_t                           frames          = 0;
	uint64_t                           max_frame_ticks = 0;
	uint64_t                           cpu_ns          = 0; // process CPU time in the interval
	std::array<SpanTotal, SpanCount>   spans {};
	std::array<uint64_t, CounterCount> counters {};
	std::array<uint64_t, GaugeCount>   gauges {};

	[[nodiscard]] const SpanTotal& Get(SpanId id) const { return spans[static_cast<size_t>(id)]; }
	[[nodiscard]] uint64_t Get(CounterId id) const { return counters[static_cast<size_t>(id)]; }
	[[nodiscard]] uint64_t Get(GaugeId id) const { return gauges[static_cast<size_t>(id)]; }
};

// Returns the spans and counters accumulated since the previous call and resets them. Gauges keep
// their values; the frame and interval fields are left for the caller.
[[nodiscard]] Snapshot CollectAndReset() noexcept;

[[nodiscard]] std::string FormatSummary(const Snapshot& snapshot, uint64_t ticks_per_second);
[[nodiscard]] std::string FormatCsvHeader();
[[nodiscard]] std::string FormatCsvRow(const Snapshot& snapshot, uint64_t ticks_per_second);

// Present thread, after a guest flip is shown. Closes an interval once a second has passed.
void OnGuestFrame() noexcept;

void Initialize();
void Shutdown();
void EmergencyShutdown();

struct Lifecycle {
	static constexpr const char* name               = "PerfStats";
	static constexpr auto        initialize         = PerfStats::Initialize;
	static constexpr auto        shutdown           = PerfStats::Shutdown;
	static constexpr auto        emergency_shutdown = PerfStats::EmergencyShutdown;
};

} // namespace PerfStats

#endif /* KYTY_COMMON_PERFSTATS_H_ */
