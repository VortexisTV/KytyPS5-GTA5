#include "common/perfStats.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using PerfStats::CounterId;
using PerfStats::GaugeId;
using PerfStats::SpanId;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "PerfStatsTests: failed: %s\n", message);
		std::abort();
	}
}

size_t Columns(const std::string& line) {
	size_t columns = 1;
	for (const auto c: line) {
		columns += c == ',' ? 1u : 0u;
	}
	return columns;
}

void TestDisabledHooksRecordNothing() {
	PerfStats::Detail::g_enabled = false;
	PerfStats::Add(CounterId::BufferUploads, 5);
	PerfStats::Set(GaugeId::CachedBuffers, 7);
	{
		PerfStats::Span span(SpanId::Draw);
	}
	const auto snapshot = PerfStats::CollectAndReset();
	Check(snapshot.Get(CounterId::BufferUploads) == 0, "a disabled counter was recorded");
	Check(snapshot.Get(GaugeId::CachedBuffers) == 0, "a disabled gauge was recorded");
	Check(snapshot.Get(SpanId::Draw).count == 0, "a disabled span was recorded");
}

void TestIntervalsAccumulateAndReset() {
	PerfStats::Detail::g_enabled = true;
	PerfStats::Add(CounterId::BufferUploads);
	PerfStats::Add(CounterId::BufferUploadBytes, 4096);
	PerfStats::Add(CounterId::BufferUploadBytes, 4096);
	PerfStats::Set(GaugeId::CachedBuffers, 3);
	PerfStats::Set(GaugeId::CachedBuffers, 9);
	PerfStats::Detail::RecordSpan(SpanId::Draw, 10);
	PerfStats::Detail::RecordSpan(SpanId::Draw, 30);
	{
		PerfStats::Span span(SpanId::BdaPrepare);
		span.Stop();
		span.Stop();
	}
	{
		PerfStats::Span span(SpanId::GpuWait, false);
	}

	const auto first = PerfStats::CollectAndReset();
	Check(first.Get(CounterId::BufferUploads) == 1, "a counter increment was lost");
	Check(first.Get(CounterId::BufferUploadBytes) == 8192, "counter values were not summed");
	Check(first.Get(GaugeId::CachedBuffers) == 9, "a gauge did not keep its latest value");
	const auto& draw = first.Get(SpanId::Draw);
	Check(draw.ticks == 40 && draw.count == 2 && draw.max_ticks == 30, "span totals are wrong");
	Check(first.Get(SpanId::BdaPrepare).count == 1, "stopping a span twice recorded it twice");
	Check(first.Get(SpanId::GpuWait).count == 0, "a span with a false condition was recorded");

	const auto second = PerfStats::CollectAndReset();
	Check(second.Get(CounterId::BufferUploads) == 0 &&
	          second.Get(CounterId::BufferUploadBytes) == 0,
	      "counters were not reset between intervals");
	Check(second.Get(SpanId::Draw).count == 0 && second.Get(SpanId::Draw).max_ticks == 0,
	      "spans were not reset between intervals");
	Check(second.Get(GaugeId::CachedBuffers) == 9, "a gauge was reset between intervals");
	PerfStats::Detail::g_enabled = false;
}

void TestCsvRowsMatchHeader() {
	const auto          header = PerfStats::FormatCsvHeader();
	PerfStats::Snapshot snapshot;
	snapshot.frames         = 3;
	snapshot.interval_ticks = 1000;
	const auto row          = PerfStats::FormatCsvRow(snapshot, 1000);
	Check(!header.empty() && header.back() == '\n' && !row.empty() && row.back() == '\n',
	      "CSV lines are not newline terminated");
	Check(Columns(header) == Columns(row), "CSV row and header column counts differ");
	Check(header.find(",gpu_wait_predicate_ms,gpu_wait_predicate_n,") != std::string::npos,
	      "the CSV header stops before the last span");
	Check(header.find(",draw_ms,draw_n,draw_max_ms,") != std::string::npos,
	      "span columns are missing from the CSV header");
	Check(header.find(",packets_skipped_predicated,bda_buffers_visited,") != std::string::npos,
	      "counter columns are missing from the CSV header");
	Check(header.find(",cached_buffers,virtual_ranges\n") != std::string::npos,
	      "gauge columns are missing from the CSV header");
	Check(row.rfind("0.000,1.000,3,3.000,333.333,", 0) == 0, "CSV frame columns are wrong");
}

void TestSummaryReportsPerFrameValues() {
	PerfStats::Snapshot snapshot;
	snapshot.frames                                        = 4;
	snapshot.interval_ticks                                = 1000;
	snapshot.spans[static_cast<size_t>(SpanId::Draw)]       = {1000, 400, 10};
	snapshot.spans[static_cast<size_t>(SpanId::BdaPrepare)] = {600, 40, 20};
	snapshot.counters[static_cast<size_t>(CounterId::BdaBuffersVisited)] = 8000;
	snapshot.counters[static_cast<size_t>(CounterId::BdaPassesSkipped)]  = 20;
	snapshot.spans[static_cast<size_t>(SpanId::DrawShaders)]       = {400, 400, 5};
	snapshot.spans[static_cast<size_t>(SpanId::ShaderPrepare)]     = {80, 400, 1};
	snapshot.spans[static_cast<size_t>(SpanId::ShaderKey)]         = {40, 800, 1};
	snapshot.spans[static_cast<size_t>(SpanId::ShaderMaterialize)] = {200, 760, 2};
	snapshot.spans[static_cast<size_t>(SpanId::ShaderPermutation)] = {60, 760, 1};
	snapshot.spans[static_cast<size_t>(SpanId::AudioPushGap)]      = {4000, 800, 45};
	snapshot.spans[static_cast<size_t>(SpanId::AudioQueueWait)]    = {40, 800, 1};
	snapshot.counters[static_cast<size_t>(CounterId::AudioPushesSync)] = 800;
	snapshot.counters[static_cast<size_t>(CounterId::AudioUnderruns)]  = 2;

	snapshot.spans[static_cast<size_t>(SpanId::ShaderSrtWalk)]            = {40, 760, 1};
	snapshot.spans[static_cast<size_t>(SpanId::ShaderIndirect)]           = {20, 40, 1};
	snapshot.spans[static_cast<size_t>(SpanId::ShaderSpecialize)]         = {60, 760, 1};
	snapshot.counters[static_cast<size_t>(CounterId::SrtInterpreted)]     = 8;
	snapshot.counters[static_cast<size_t>(CounterId::IndirectProbes)]     = 400;
	snapshot.counters[static_cast<size_t>(CounterId::ShaderGuestReads)]   = 4000;
	snapshot.counters[static_cast<size_t>(CounterId::ShaderRangeChecks)]  = 800;
	snapshot.counters[static_cast<size_t>(CounterId::ShaderInputRepeats)] = 600;

	snapshot.spans[static_cast<size_t>(SpanId::VirtualMapEdit)]  = {80, 40, 12};
	snapshot.spans[static_cast<size_t>(SpanId::VirtualMapQuery)] = {8, 400, 1};
	snapshot.spans[static_cast<size_t>(SpanId::VirtualMapWait)]  = {60, 20, 9};
	snapshot.gauges[static_cast<size_t>(GaugeId::VirtualRanges)] = 12345;

	snapshot.counters[static_cast<size_t>(CounterId::DrawsSkippedShader)]       = 8;
	snapshot.counters[static_cast<size_t>(CounterId::DrawsSkippedPipeline)]     = 4;
	snapshot.counters[static_cast<size_t>(CounterId::DrawsSkippedStage)]        = 40;
	snapshot.counters[static_cast<size_t>(CounterId::DrawsSkippedTessellation)] = 36;
	snapshot.counters[static_cast<size_t>(CounterId::DrawsSkippedVertexShader)] = 12;
	snapshot.counters[static_cast<size_t>(CounterId::DrawsSkippedState)]        = 20;
	snapshot.counters[static_cast<size_t>(CounterId::DrawsSkippedEmpty)]        = 80;
	snapshot.counters[static_cast<size_t>(CounterId::DrawsIndirect)]            = 200;
	snapshot.counters[static_cast<size_t>(CounterId::DrawsIndexClamped)]        = 16;
	snapshot.counters[static_cast<size_t>(CounterId::PredicateSkips)]           = 24;
	snapshot.counters[static_cast<size_t>(CounterId::PredicateStale)]           = 4;
	snapshot.counters[static_cast<size_t>(CounterId::PacketsSkippedPredicated)] = 1200;

	snapshot.spans[static_cast<size_t>(SpanId::DrawBindings)]      = {320, 400, 4};
	snapshot.spans[static_cast<size_t>(SpanId::BindResolve)]       = {80, 800, 1};
	snapshot.spans[static_cast<size_t>(SpanId::BindFindBuffers)]   = {40, 800, 1};
	snapshot.spans[static_cast<size_t>(SpanId::BindDmaSources)]    = {20, 80, 1};
	snapshot.spans[static_cast<size_t>(SpanId::BindRebindBuffers)] = {60, 800, 1};
	snapshot.spans[static_cast<size_t>(SpanId::BindRebindImages)]  = {100, 800, 1};

	snapshot.counters[static_cast<size_t>(CounterId::ReadbacksShadow)]     = 24;
	snapshot.counters[static_cast<size_t>(CounterId::ReadbackNoOwner)]     = 4;
	snapshot.counters[static_cast<size_t>(CounterId::ReadbackNotHot)]      = 8;
	snapshot.counters[static_cast<size_t>(CounterId::ReadbackShadowStale)] = 12;
	snapshot.counters[static_cast<size_t>(CounterId::ReadbackRecentWrite)] = 16;
	snapshot.counters[static_cast<size_t>(CounterId::ShadowsRecorded)]     = 40;

	snapshot.spans[static_cast<size_t>(SpanId::GpuWait)]          = {400, 40, 30};
	snapshot.spans[static_cast<size_t>(SpanId::GpuWaitReadback)]  = {160, 8, 30};
	snapshot.spans[static_cast<size_t>(SpanId::GpuWaitFaults)]    = {120, 8, 20};
	snapshot.spans[static_cast<size_t>(SpanId::GpuWaitStream)]    = {80, 4, 20};
	snapshot.spans[static_cast<size_t>(SpanId::GpuWaitPredicate)] = {40, 20, 10};

	snapshot.spans[static_cast<size_t>(SpanId::HotPageHash)]           = {24, 80, 2};
	snapshot.counters[static_cast<size_t>(CounterId::HotPagesHashed)]  = 80000;
	snapshot.counters[static_cast<size_t>(CounterId::HotPagesChanged)] = 4000;
	snapshot.counters[static_cast<size_t>(CounterId::HotPagesCooled)]  = 60000;
	const auto summary = PerfStats::FormatSummary(snapshot, 1000);
	Check(summary.find("4 frames in 1.00s (4.0 fps)") != std::string::npos,
	      "the summary does not report the frame rate");
	Check(summary.find("100 draws 250.0 ms") != std::string::npos,
	      "the summary does not report per-frame draw cost");
	Check(summary.find("BDA 10.0 passes 150.0 ms (5.0 skipped, 400 buffers per full pass)") !=
	          std::string::npos,
	      "the summary does not report per-frame BDA cost");
	Check(summary.find("shader lookup 100.0 ms: prepare 20.0 ms, key 10.0 ms (200 lookups), "
	                   "materialize 50.0 ms (190), permutation 15.0 ms") != std::string::npos,
	      "the summary does not break down shader lookup");
	Check(summary.find("materialize: SRT walk 10.0 ms (2 interpreted), indirect images 5.0 ms (10 "
	                   "tables, 100 probes), specialize 15.0 ms, other 20.0 ms | 1000 guest reads, "
	                   "200 range checks, 150 repeated inputs") != std::string::npos,
	      "the summary does not break down materialization");
	Check(summary.find("virtual memory map: 10.0 edits holding its lock 20.0 ms (longest 12.00 "
	                   "ms), 100.0 queries 2.0 ms | range checks waited 5.0 times, 15.0 ms "
	                   "(longest 9.00 ms) | 12345 ranges") != std::string::npos,
	      "the summary does not report the virtual memory map lock");
	Check(summary.find("bindings 80.0 ms: resolve 20.0 ms, find buffers 10.0 ms, DMA sources "
	                   "5.0 ms, BDA 150.0 ms, rebind buffers 15.0 ms, rebind images 25.0 ms") !=
	          std::string::npos,
	      "the summary does not break down binding preparation");
	Check(summary.find("readbacks: 6.0 from a shadow, 10.0 drained the GPU (1.0 unowned, 2.0 "
	                   "not hot yet, 3.0 stale shadow, 4.0 rewritten) | 10.0 shadows recorded") !=
	          std::string::npos,
	      "the summary does not report readbacks");
	Check(summary.find("GPU waits 100.0 ms: readback 40.0 ms (2.0), faults 30.0 ms (2.0), stream "
	                   "buffer 20.0 ms (1.0), predicates 10.0 ms (5.0)") != std::string::npos,
	      "the summary does not break down GPU waits");
	Check(summary.find("dropped draws: 2.0 waiting for shaders, 1.0 for pipelines, 10.0 "
	                   "unsupported stages (9.0 tessellated), 3.0 without a vertex shader, 5.0 "
	                   "without state, 20.0 "
	                   "empty | 50.0 indirect draws (4.0 index-clamped) | predication: 6.0 skips "
	                   "on (1.0 unsynchronized), 300.0 packets skipped") != std::string::npos,
	      "the summary does not report dropped draws");
	Check(summary.find("hot pages: 20000 hashed in 6.0 ms, 1000 changed, 15000 cooled down") !=
	          std::string::npos,
	      "the summary does not report hot-page hashing");
	Check(summary.find("audio 200.0 sync and 0.0 async pushes (0.0 refused), 0.50 underruns | "
	                   "longest push gap 45.0 ms | waiting for device room 10.0 ms") !=
	          std::string::npos,
	      "the summary does not report audio pacing");
}

} // namespace

int main() {
	TestDisabledHooksRecordNothing();
	TestIntervalsAccumulateAndReset();
	TestCsvRowsMatchHeader();
	TestSummaryReportsPerFrameValues();
	std::printf("PerfStatsTests: ok\n");
	return 0;
}
