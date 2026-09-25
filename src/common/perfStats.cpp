#include "common/perfStats.h"
#include "common/emulatorConfig.h"

#include "common/file.h"
#include "common/timer.h"

#include <algorithm>
#include <atomic>
#include <fmt/format.h>
#include <iterator>
#include <mutex>
#include <string_view>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#else
#include <cstdlib>
#include <sys/resource.h>
#include <sys/time.h>
#endif

namespace PerfStats {

namespace {

constexpr const char* CsvFileName   = "_PerfStats.csv";
constexpr const char* NotesFileName = "_PerfNotes.txt";

constexpr std::array<std::string_view, SpanCount> SpanNames = {
    "gpu_thread_busy",       "gpu_thread_idle",      "gpu_thread_blocked",
    "gpu_thread_commands",   "game_wait_gpu_idle",   "game_wait_gpu_command",
    "game_wait_gpu_flip",
    "draw",                  "draw_targets",         "draw_shaders",
    "shader_prepare",        "shader_key",           "shader_materialize",
    "shader_srt_walk",       "shader_indirect_images", "shader_specialize",
    "shader_permutation",
    "draw_bindings",         "bind_resolve",         "bind_find_buffers",
    "bind_dma_sources",      "bind_rebind_buffers",  "bind_rebind_images",
    "draw_vertex_index",     "draw_pipeline",
    "draw_record",           "dispatch",             "bda_prepare",
    "hot_page_hash",
    "buffer_download",       "buffer_upload",        "readback_guest_wait",
    "queue_submit",          "gpu_wait",
    "gpu_wait_readback",     "gpu_wait_faults",      "gpu_wait_stream",
    "gpu_wait_predicate",
    "flip_wait",             "present",              "page_fault",
    "virtual_map_edit",      "virtual_map_query",    "virtual_map_wait",
    "garbage_collect",       "shader_compile_sync",  "shader_compile_async",
    "pipeline_create_sync",  "pipeline_create_async", "pipeline_cache_save",
    "audio_push_gap",        "audio_queue_wait",
};

constexpr std::array<std::string_view, CounterCount> CounterNames = {
    "draws_skipped_shader", "draws_skipped_pipeline", "draws_skipped_stage",
    "draws_skipped_tessellation", "draws_skipped_vertex_shader", "draws_skipped_state",
    "draws_skipped_empty",
    "draws_indirect", "draws_index_clamped", "predicate_skips", "predicate_stale",
    "packets_skipped_predicated", "bda_buffers_visited",
    "bda_passes_skipped",
    "hot_pages_hashed",     "hot_pages_changed",      "hot_pages_cooled",
    "buffer_uploads",       "buffer_upload_bytes",    "stream_uploads",
    "stream_upload_bytes",  "buffer_creates",         "buffer_download_bytes",
    "readbacks_shadow",     "readback_no_owner",      "readback_not_hot",
    "readback_shadow_stale", "readback_recent_write", "shadows_recorded",
    "eager_shadow_flushes", "readbacks_async",        "readbacks_retried",
    "indirect_dispatches_gpu", "writes_beside_gpu",
    "dcc_gpu_scans",        "dcc_scans_skipped",      "dcc_cpu_readbacks",
    "buffers_evicted",      "image_creates",          "image_uploads",
    "image_upload_bytes",   "images_evicted",         "write_faults",
    "read_faults",          "shaders_compiled",       "pipelines_created",
    "shader_guest_reads",   "shader_range_checks",    "indirect_image_probes",
    "srt_interpreted",      "shader_input_repeats",
    "audio_pushes_sync",    "audio_pushes_async",     "audio_pushes_not_ready",
    "audio_underruns",
};

constexpr std::array<std::string_view, GaugeCount> GaugeNames = {
    "gpu_memory_mb",
    "buffer_gc_trigger_mb",
    "texture_gc_trigger_mb",
    "cached_buffers",
    "virtual_ranges",
};

template <size_t N>
constexpr bool AllNamed(const std::array<std::string_view, N>& names) {
	return std::ranges::none_of(names, [](std::string_view name) { return name.empty(); });
}

// A name missing from a table leaves an empty entry at its end.
static_assert(AllNamed(SpanNames) && AllNamed(CounterNames) && AllNamed(GaugeNames));

struct AtomicSpan {
	std::atomic<uint64_t> ticks {0};
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> max_ticks {0};
};

std::array<AtomicSpan, SpanCount>               g_spans;
std::array<std::atomic<uint64_t>, CounterCount> g_counters {};
std::array<std::atomic<uint64_t>, GaugeCount>   g_gauges {};

// Interval bookkeeping, touched by initialization, the present thread and shutdown.
struct Report {
	std::mutex   mutex;
	uint64_t     ticks_per_second = 0;
	uint64_t     start            = 0;
	uint64_t     interval_start   = 0;
	uint64_t     last_frame       = 0;
	uint64_t     max_frame        = 0;
	uint64_t     frames           = 0;
	uint64_t     cpu_start_ns     = 0;
	Common::File csv;
	bool         csv_open = false;
	Common::File notes;
	bool         notes_open = false;
};

Report& GetReport() {
	static Report report;
	return report;
}

uint64_t ProcessCpuNs() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	FILETIME creation {};
	FILETIME exited {};
	FILETIME kernel {};
	FILETIME user {};
	if (GetProcessTimes(GetCurrentProcess(), &creation, &exited, &kernel, &user) == 0) {
		return 0;
	}
	const auto to_ns = [](const FILETIME& time) {
		return ((static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime) * 100u;
	};
	return to_ns(kernel) + to_ns(user);
#else
	rusage usage {};
	if (getrusage(RUSAGE_SELF, &usage) != 0) {
		return 0;
	}
	const auto to_ns = [](const timeval& time) {
		return static_cast<uint64_t>(time.tv_sec) * 1000000000u +
		       static_cast<uint64_t>(time.tv_usec) * 1000u;
	};
	return to_ns(usage.ru_utime) + to_ns(usage.ru_stime);
#endif
}

bool EnvironmentRequestsStats() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	char       buffer[8] {};
	const auto length = GetEnvironmentVariableA("KYTY_PERF_STATS", buffer, sizeof(buffer));
	if (length == 0 || length >= sizeof(buffer)) {
		return false;
	}
	const std::string_view value {buffer, length};
#else
	const char* variable = std::getenv("KYTY_PERF_STATS");
	if (variable == nullptr) {
		return false;
	}
	const std::string_view value {variable};
#endif
	return value == "1" || value == "true";
}

double Milliseconds(uint64_t ticks, uint64_t ticks_per_second) {
	return ticks_per_second == 0
	           ? 0.0
	           : static_cast<double>(ticks) * 1000.0 / static_cast<double>(ticks_per_second);
}

double Megabytes(uint64_t bytes) {
	return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double IntervalSeconds(const Snapshot& snapshot, uint64_t ticks_per_second) {
	return Milliseconds(snapshot.interval_ticks, ticks_per_second) / 1000.0;
}

double FramesPerSecond(const Snapshot& snapshot, uint64_t ticks_per_second) {
	const auto seconds = IntervalSeconds(snapshot, ticks_per_second);
	return seconds > 0.0 ? static_cast<double>(snapshot.frames) / seconds : 0.0;
}

double AverageFrameMs(const Snapshot& snapshot, uint64_t ticks_per_second) {
	return snapshot.frames == 0 ? 0.0
	                            : Milliseconds(snapshot.interval_ticks, ticks_per_second) /
	                                  static_cast<double>(snapshot.frames);
}

double CpuCores(const Snapshot& snapshot, uint64_t ticks_per_second) {
	const auto seconds = IntervalSeconds(snapshot, ticks_per_second);
	return seconds > 0.0 ? static_cast<double>(snapshot.cpu_ns) / 1e9 / seconds : 0.0;
}

void CloseCsv(Report& report) {
	if (report.csv_open) {
		report.csv.Flush();
		report.csv.Close();
		report.csv_open = false;
	}
	if (report.notes_open) {
		report.notes.Flush();
		report.notes.Close();
		report.notes_open = false;
	}
}

} // namespace

namespace Detail {

void RecordSpan(SpanId id, uint64_t ticks) noexcept {
	auto& span = g_spans[static_cast<size_t>(id)];
	span.ticks.fetch_add(ticks, std::memory_order_relaxed);
	span.count.fetch_add(1, std::memory_order_relaxed);
	auto longest = span.max_ticks.load(std::memory_order_relaxed);
	while (ticks > longest &&
	       !span.max_ticks.compare_exchange_weak(longest, ticks, std::memory_order_relaxed)) {
	}
}

void AddCounter(CounterId id, uint64_t value) noexcept {
	g_counters[static_cast<size_t>(id)].fetch_add(value, std::memory_order_relaxed);
}

void SetGauge(GaugeId id, uint64_t value) noexcept {
	g_gauges[static_cast<size_t>(id)].store(value, std::memory_order_relaxed);
}

} // namespace Detail

uint64_t Now() noexcept {
	return Common::Timer::QueryPerformanceCounter();
}

uint64_t TicksPerSecond() noexcept {
	static const uint64_t ticks_per_second = Common::Timer::QueryPerformanceFrequency();
	return ticks_per_second;
}

Snapshot CollectAndReset() noexcept {
	Snapshot snapshot;
	for (size_t i = 0; i < SpanCount; i++) {
		snapshot.spans[i] = {
		    .ticks     = g_spans[i].ticks.exchange(0, std::memory_order_relaxed),
		    .count     = g_spans[i].count.exchange(0, std::memory_order_relaxed),
		    .max_ticks = g_spans[i].max_ticks.exchange(0, std::memory_order_relaxed),
		};
	}
	for (size_t i = 0; i < CounterCount; i++) {
		snapshot.counters[i] = g_counters[i].exchange(0, std::memory_order_relaxed);
	}
	for (size_t i = 0; i < GaugeCount; i++) {
		snapshot.gauges[i] = g_gauges[i].load(std::memory_order_relaxed);
	}
	return snapshot;
}

std::string FormatSummary(const Snapshot& snapshot, uint64_t ticks_per_second) {
	const auto frames = static_cast<double>(std::max<uint64_t>(snapshot.frames, 1));
	const auto ms     = [&](SpanId id) {
		return Milliseconds(snapshot.Get(id).ticks, ticks_per_second) / frames;
	};
	const auto count = [&](SpanId id) {
		return static_cast<double>(snapshot.Get(id).count) / frames;
	};
	const auto per_frame = [&](CounterId id) {
		return static_cast<double>(snapshot.Get(id)) / frames;
	};
	const auto mb = [&](CounterId id) { return Megabytes(snapshot.Get(id)) / frames; };
	// A skipped preparation is timed like a pass but visits no buffers.
	const auto bda_requests = snapshot.Get(SpanId::BdaPrepare).count;
	const auto bda_passes =
	    bda_requests - std::min(snapshot.Get(CounterId::BdaPassesSkipped), bda_requests);
	const auto buffers_per_pass =
	    bda_passes == 0 ? 0.0
	                    : static_cast<double>(snapshot.Get(CounterId::BdaBuffersVisited)) /
	                          static_cast<double>(bda_passes);

	std::string out;
	auto        it = std::back_inserter(out);
	fmt::format_to(it,
	               "[perf] {:.0f}s: {} frames in {:.2f}s ({:.1f} fps), frame avg {:.1f} ms, max {:.1f} "
	               "ms, CPU {:.2f} cores, GPU memory {} MB (GC above {}/{} MB), {} cached buffers\n",
	               snapshot.elapsed_seconds, snapshot.frames,
	               IntervalSeconds(snapshot, ticks_per_second),
	               FramesPerSecond(snapshot, ticks_per_second),
	               AverageFrameMs(snapshot, ticks_per_second),
	               Milliseconds(snapshot.max_frame_ticks, ticks_per_second),
	               CpuCores(snapshot, ticks_per_second), snapshot.Get(GaugeId::GpuMemoryMb),
	               snapshot.Get(GaugeId::BufferGcTriggerMb),
	               snapshot.Get(GaugeId::TextureGcTriggerMb), snapshot.Get(GaugeId::CachedBuffers));
	fmt::format_to(it,
	               "[perf] per frame: GPU thread busy {:.1f} ms, idle {:.1f} ms, blocked {:.1f} ms | "
	               "game waiting on GPU thread {:.1f} ms (flip {:.1f}), on GPU commands {:.1f} ms "
	               "({:.1f}) | flip wait {:.1f} ms | present {:.1f} ms\n",
	               ms(SpanId::GpuThreadBusy), ms(SpanId::GpuThreadIdle),
	               ms(SpanId::GpuThreadBlocked), ms(SpanId::GameWaitGpuIdle),
	               ms(SpanId::GameWaitGpuFlip), ms(SpanId::GameWaitGpuCommand),
	               count(SpanId::GameWaitGpuCommand), ms(SpanId::FlipWait), ms(SpanId::Present));
	fmt::format_to(it,
	               "[perf] per frame: {:.0f} draws {:.1f} ms (targets {:.1f}, shaders {:.1f}, "
	               "bindings {:.1f}, vertex/index {:.1f}, pipeline {:.1f}, record {:.1f}) | "
	               "{:.0f} dispatches {:.1f} ms\n",
	               count(SpanId::Draw), ms(SpanId::Draw), ms(SpanId::DrawTargets),
	               ms(SpanId::DrawShaders), ms(SpanId::DrawBindings), ms(SpanId::DrawVertexIndex),
	               ms(SpanId::DrawPipeline), ms(SpanId::DrawRecord), count(SpanId::Dispatch),
	               ms(SpanId::Dispatch));
	fmt::format_to(
	    it,
	    "[perf] per frame: dropped draws: {:.1f} waiting for shaders, {:.1f} for "
	    "pipelines, {:.1f} unsupported stages ({:.1f} tessellated), {:.1f} without a vertex "
	    "shader, {:.1f} "
	    "without state, {:.1f} empty | {:.1f} indirect draws ({:.1f} index-clamped) | "
	    "predication: {:.1f} skips on ({:.1f} unsynchronized), {:.1f} packets "
	    "skipped\n",
	    per_frame(CounterId::DrawsSkippedShader), per_frame(CounterId::DrawsSkippedPipeline),
	    per_frame(CounterId::DrawsSkippedStage), per_frame(CounterId::DrawsSkippedTessellation),
	    per_frame(CounterId::DrawsSkippedVertexShader), per_frame(CounterId::DrawsSkippedState),
	    per_frame(CounterId::DrawsSkippedEmpty), per_frame(CounterId::DrawsIndirect),
	    per_frame(CounterId::DrawsIndexClamped), per_frame(CounterId::PredicateSkips),
	    per_frame(CounterId::PredicateStale), per_frame(CounterId::PacketsSkippedPredicated));
	fmt::format_to(it,
	               "[perf] per frame: shader lookup {:.1f} ms: prepare {:.1f} ms, key {:.1f} ms ({:.0f} "
	               "lookups), materialize {:.1f} ms ({:.0f}), permutation {:.1f} ms\n",
	               ms(SpanId::DrawShaders), ms(SpanId::ShaderPrepare), ms(SpanId::ShaderKey),
	               count(SpanId::ShaderKey), ms(SpanId::ShaderMaterialize),
	               count(SpanId::ShaderMaterialize), ms(SpanId::ShaderPermutation));
	const auto materialize_other =
	    std::max(0.0, ms(SpanId::ShaderMaterialize) - ms(SpanId::ShaderSrtWalk) -
	                      ms(SpanId::ShaderIndirect) - ms(SpanId::ShaderSpecialize));
	fmt::format_to(it,
	               "[perf] per frame: materialize: SRT walk {:.1f} ms ({:.0f} interpreted), "
	               "indirect images {:.1f} ms ({:.0f} tables, {:.0f} probes), specialize {:.1f} "
	               "ms, other {:.1f} ms | {:.0f} guest reads, {:.0f} range checks, {:.0f} "
	               "repeated inputs\n",
	               ms(SpanId::ShaderSrtWalk), per_frame(CounterId::SrtInterpreted),
	               ms(SpanId::ShaderIndirect), count(SpanId::ShaderIndirect),
	               per_frame(CounterId::IndirectProbes), ms(SpanId::ShaderSpecialize),
	               materialize_other, per_frame(CounterId::ShaderGuestReads),
	               per_frame(CounterId::ShaderRangeChecks),
	               per_frame(CounterId::ShaderInputRepeats));
	fmt::format_to(it,
	               "[perf] per frame: bindings {:.1f} ms: resolve {:.1f} ms, find buffers {:.1f} "
	               "ms, DMA sources {:.1f} ms, BDA {:.1f} ms, rebind buffers {:.1f} ms, rebind "
	               "images {:.1f} ms\n",
	               ms(SpanId::DrawBindings), ms(SpanId::BindResolve), ms(SpanId::BindFindBuffers),
	               ms(SpanId::BindDmaSources), ms(SpanId::BdaPrepare),
	               ms(SpanId::BindRebindBuffers), ms(SpanId::BindRebindImages));
	fmt::format_to(
	    it,
	    "[perf] per frame: readbacks: {:.1f} from a shadow, {:.1f} drained the GPU "
	    "({:.1f} unowned, {:.1f} not hot yet, {:.1f} stale shadow, {:.1f} rewritten) | "
	    "{:.1f} shadows recorded ({:.1f} eager flushes) | {:.1f} waited off the GPU thread, "
	    "{:.1f} ms ({:.1f} retried) | avoided: {:.1f} indirect dispatches read on the GPU, "
	    "{:.1f} labels written beside GPU data\n",
	    per_frame(CounterId::ReadbacksShadow),
	    per_frame(CounterId::ReadbackNoOwner) + per_frame(CounterId::ReadbackNotHot) +
	        per_frame(CounterId::ReadbackShadowStale) + per_frame(CounterId::ReadbackRecentWrite),
	    per_frame(CounterId::ReadbackNoOwner), per_frame(CounterId::ReadbackNotHot),
	    per_frame(CounterId::ReadbackShadowStale), per_frame(CounterId::ReadbackRecentWrite),
	    per_frame(CounterId::ShadowsRecorded), per_frame(CounterId::EagerShadowFlushes),
	    per_frame(CounterId::ReadbacksAsync), ms(SpanId::ReadbackGuestWait),
	    per_frame(CounterId::ReadbacksRetried), per_frame(CounterId::IndirectDispatchesGpu),
	    per_frame(CounterId::WritesBesideGpu));
	fmt::format_to(it,
	               "[perf] per frame: GPU waits {:.1f} ms: readback {:.1f} ms ({:.1f}), faults "
	               "{:.1f} ms ({:.1f}), stream buffer {:.1f} ms ({:.1f}), predicates {:.1f} ms "
	               "({:.1f})\n",
	               ms(SpanId::GpuWait), ms(SpanId::GpuWaitReadback), count(SpanId::GpuWaitReadback),
	               ms(SpanId::GpuWaitFaults), count(SpanId::GpuWaitFaults),
	               ms(SpanId::GpuWaitStream), count(SpanId::GpuWaitStream),
	               ms(SpanId::GpuWaitPredicate), count(SpanId::GpuWaitPredicate));
	fmt::format_to(it,
	               "[perf] per frame: BDA {:.1f} passes {:.1f} ms ({:.1f} skipped, {:.0f} buffers per "
	               "full pass) | {:.1f} submits {:.1f} ms | {:.1f} GPU waits {:.1f} ms (longest {:.1f} "
	               "ms) | {:.1f} page faults {:.1f} ms ({:.1f} write) | GC {:.1f} ms\n",
	               count(SpanId::BdaPrepare), ms(SpanId::BdaPrepare),
	               per_frame(CounterId::BdaPassesSkipped), buffers_per_pass,
	               count(SpanId::QueueSubmit),
	               ms(SpanId::QueueSubmit), count(SpanId::GpuWait), ms(SpanId::GpuWait),
	               Milliseconds(snapshot.Get(SpanId::GpuWait).max_ticks, ticks_per_second),
	               count(SpanId::PageFault), ms(SpanId::PageFault),
	               per_frame(CounterId::WriteFaults), ms(SpanId::GarbageCollect));
	fmt::format_to(it,
	               "[perf] per frame: hot pages: {:.0f} hashed in {:.1f} ms, {:.0f} changed, "
	               "{:.0f} cooled down\n",
	               per_frame(CounterId::HotPagesHashed), ms(SpanId::HotPageHash),
	               per_frame(CounterId::HotPagesChanged), per_frame(CounterId::HotPagesCooled));
	fmt::format_to(it,
	               "[perf] per frame: virtual memory map: {:.1f} edits holding its lock {:.1f} ms "
	               "(longest {:.2f} ms), {:.1f} queries {:.1f} ms | range checks waited {:.1f} "
	               "times, {:.1f} ms (longest {:.2f} ms) | {} ranges\n",
	               count(SpanId::VirtualMapEdit), ms(SpanId::VirtualMapEdit),
	               Milliseconds(snapshot.Get(SpanId::VirtualMapEdit).max_ticks, ticks_per_second),
	               count(SpanId::VirtualMapQuery), ms(SpanId::VirtualMapQuery),
	               count(SpanId::VirtualMapWait), ms(SpanId::VirtualMapWait),
	               Milliseconds(snapshot.Get(SpanId::VirtualMapWait).max_ticks, ticks_per_second),
	               snapshot.Get(GaugeId::VirtualRanges));
	fmt::format_to(it,
	               "[perf] per frame: audio {:.1f} sync and {:.1f} async pushes ({:.1f} refused), "
	               "{:.2f} underruns | longest push gap {:.1f} ms | waiting for device room {:.1f} "
	               "ms\n",
	               per_frame(CounterId::AudioPushesSync), per_frame(CounterId::AudioPushesAsync),
	               per_frame(CounterId::AudioPushesNotReady), per_frame(CounterId::AudioUnderruns),
	               Milliseconds(snapshot.Get(SpanId::AudioPushGap).max_ticks, ticks_per_second),
	               ms(SpanId::AudioQueueWait));
	fmt::format_to(it,
	               "[perf] per frame: uploads {:.1f} buffer ({:.2f} MB), {:.1f} stream ({:.2f} MB), "
	               "{:.1f} image ({:.2f} MB) | created {:.1f} buffers, {:.1f} images | evicted "
	               "{:.1f} buffers, {:.1f} images | {:.1f} downloads ({:.2f} MB) {:.1f} ms | "
	               "compiled {:.1f} shaders ({:.1f} ms blocking), {:.1f} pipelines ({:.1f} ms "
	               "blocking)\n",
	               per_frame(CounterId::BufferUploads), mb(CounterId::BufferUploadBytes),
	               per_frame(CounterId::StreamUploads), mb(CounterId::StreamUploadBytes),
	               per_frame(CounterId::ImageUploads), mb(CounterId::ImageUploadBytes),
	               per_frame(CounterId::BufferCreates), per_frame(CounterId::ImageCreates),
	               per_frame(CounterId::BuffersEvicted), per_frame(CounterId::ImagesEvicted),
	               count(SpanId::BufferDownload), mb(CounterId::BufferDownloadBytes),
	               ms(SpanId::BufferDownload), per_frame(CounterId::ShadersCompiled),
	               ms(SpanId::ShaderCompileSync), per_frame(CounterId::PipelinesCreated),
	               ms(SpanId::PipelineCreateSync));
	return out;
}

std::string FormatCsvHeader() {
	std::string out = "elapsed_s,interval_s,frames,fps,frame_avg_ms,frame_max_ms,cpu_cores";
	auto        it  = std::back_inserter(out);
	for (const auto name: SpanNames) {
		fmt::format_to(it, ",{0}_ms,{0}_n,{0}_max_ms", name);
	}
	for (const auto name: CounterNames) {
		fmt::format_to(it, ",{}", name);
	}
	for (const auto name: GaugeNames) {
		fmt::format_to(it, ",{}", name);
	}
	out += '\n';
	return out;
}

std::string FormatCsvRow(const Snapshot& snapshot, uint64_t ticks_per_second) {
	std::string out;
	auto        it = std::back_inserter(out);
	fmt::format_to(it, "{:.3f},{:.3f},{},{:.3f},{:.3f},{:.3f},{:.3f}", snapshot.elapsed_seconds,
	               IntervalSeconds(snapshot, ticks_per_second), snapshot.frames,
	               FramesPerSecond(snapshot, ticks_per_second),
	               AverageFrameMs(snapshot, ticks_per_second),
	               Milliseconds(snapshot.max_frame_ticks, ticks_per_second),
	               CpuCores(snapshot, ticks_per_second));
	for (const auto& span: snapshot.spans) {
		fmt::format_to(it, ",{:.3f},{},{:.3f}", Milliseconds(span.ticks, ticks_per_second),
		               span.count, Milliseconds(span.max_ticks, ticks_per_second));
	}
	for (const auto value: snapshot.counters) {
		fmt::format_to(it, ",{}", value);
	}
	for (const auto value: snapshot.gauges) {
		fmt::format_to(it, ",{}", value);
	}
	out += '\n';
	return out;
}

void OnGuestFrame() noexcept {
	if (!Enabled()) {
		return;
	}
	const auto  now    = Now();
	auto&       report = GetReport();
	std::string summary;
	{
		std::lock_guard lock(report.mutex);
		if (report.last_frame != 0) {
			report.max_frame = std::max(report.max_frame, now - report.last_frame);
		}
		report.last_frame = now;
		report.frames++;
		if (now - report.interval_start < report.ticks_per_second) {
			return;
		}
		auto snapshot            = CollectAndReset();
		snapshot.elapsed_seconds = static_cast<double>(now - report.start) /
		                           static_cast<double>(report.ticks_per_second);
		snapshot.interval_ticks  = now - report.interval_start;
		snapshot.frames          = report.frames;
		snapshot.max_frame_ticks = report.max_frame;
		const auto cpu_ns        = ProcessCpuNs();
		snapshot.cpu_ns          = cpu_ns - std::min(cpu_ns, report.cpu_start_ns);
		report.cpu_start_ns      = cpu_ns;
		report.interval_start    = now;
		report.frames            = 0;
		report.max_frame         = 0;

		summary = FormatSummary(snapshot, report.ticks_per_second);
		if (report.csv_open) {
			const auto row = FormatCsvRow(snapshot, report.ticks_per_second);
			report.csv.Write(row.data(), static_cast<uint32_t>(row.size()));
			report.csv.Flush();
		}
	}
	std::fwrite(summary.data(), 1, summary.size(), stdout);
	std::fflush(stdout);
}

void Note(std::string_view line) {
	if (!Enabled()) {
		return;
	}
	auto&           report = GetReport();
	std::lock_guard lock(report.mutex);
	if (!report.notes_open) {
		return;
	}
	const auto text = fmt::format("{:.1f}s {}\n",
	                              static_cast<double>(Now() - report.start) /
	                                  static_cast<double>(report.ticks_per_second),
	                              line);
	report.notes.Write(text.data(), static_cast<uint32_t>(text.size()));
	report.notes.Flush();
}

void Initialize() {
	if (!Config::PerfStatsEnabled() && !EnvironmentRequestsStats()) {
		return;
	}
	auto&           report = GetReport();
	std::lock_guard lock(report.mutex);
	report.ticks_per_second = TicksPerSecond();
	report.start            = Now();
	report.interval_start   = report.start;
	report.cpu_start_ns     = ProcessCpuNs();
	report.csv_open         = report.csv.Create(CsvFileName);
	if (report.csv_open) {
		const auto header = FormatCsvHeader();
		report.csv.Write(header.data(), static_cast<uint32_t>(header.size()));
		report.csv.Flush();
	}
	report.notes_open = report.notes.Create(NotesFileName);
	std::printf("Performance statistics enabled: a summary follows every second of guest frames%s\n",
	            report.csv_open ? "; intervals are written to _PerfStats.csv"
	                            : "; _PerfStats.csv could not be created");
	std::fflush(stdout);
	Detail::g_enabled = true;
}

void Shutdown() {
	auto&           report = GetReport();
	std::lock_guard lock(report.mutex);
	CloseCsv(report);
}

void EmergencyShutdown() {
	// A crashing thread may hold the report lock; losing the last row beats a hang.
	auto&            report = GetReport();
	std::unique_lock lock(report.mutex, std::try_to_lock);
	if (lock.owns_lock()) {
		CloseCsv(report);
	}
}

} // namespace PerfStats
