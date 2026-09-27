#include "graphics/host_gpu/gpuCrashDiagnostics.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {
namespace {

constexpr uint32_t MaxRecordedUserData = 16;
constexpr uint32_t MaxRecordedSrt      = 32;
constexpr uint32_t MaxRecordedBuffers  = 3;
constexpr uint32_t RecordedBufferBytes = 64;

struct RecordedBuffer {
	uint64_t address = 0;
	uint64_t size    = 0;
	// Whether guest memory held the latest bytes: false when the GPU had written the range since.
	bool                                      clean = false;
	bool                                      read  = false;
	std::array<uint32_t, RecordedBufferBytes / 4> dwords {};
};

struct CheckpointRecord {
	uint64_t                sequence     = 0;
	uint64_t                submit_id    = 0;
	GpuCheckpointKind       kind         = GpuCheckpointKind::Dispatch;
	uint32_t                shader_count = 0;
	std::array<uint64_t, 4> shaders {};
	std::array<uint32_t, 3> args {};
	uint32_t                user_data_count = 0;
	std::array<uint32_t, MaxRecordedUserData> user_data {};
	uint32_t                                  srt_count = 0;
	std::array<uint32_t, MaxRecordedSrt>      srt {};
	std::array<uint32_t, 3>                   dispatch_threads {};
	uint32_t                                  buffer_count = 0;
	std::array<RecordedBuffer, MaxRecordedBuffers> buffers {};
};

// Covers the draws and dispatches of several frames in flight; older records are overwritten.
constexpr uint64_t RecordCount = uint64_t {1} << 15u;
// How many unfinished draws and dispatches a report lists, oldest first.
constexpr uint64_t MaxListedInFlight = 32;

std::atomic<uint64_t> g_sequence {0};

CheckpointRecord* Records() {
	static const std::unique_ptr<CheckpointRecord[]> records(new CheckpointRecord[RecordCount]);
	return records.get();
}

struct ShaderRegistry {
	std::mutex                                           mutex;
	std::unordered_map<uint64_t, std::vector<uint32_t>> code;
	std::set<std::pair<uint64_t, std::string>>           dumped;
};

ShaderRegistry& Shaders() {
	static ShaderRegistry registry;
	return registry;
}

thread_local uint64_t g_translating_shader = 0;

void DumpTranslatingShader() {
	if (g_translating_shader != 0) {
		DumpDiagnosticShader(g_translating_shader, "fatal");
	}
}

std::atomic<Buffer*> g_loop_watchdog {nullptr};

std::mutex                             g_runaway_mutex;
std::unordered_map<uint64_t, uint32_t> g_runaway_reports;
std::vector<uint64_t>                  g_runaway_shaders;
std::atomic<uint32_t>                  g_runaway_generation {0};

struct LoopWatchdogCut {
	uint64_t hash  = 0;
	uint32_t pc    = 0;
	uint32_t trips = 0;
};

struct LoopWatchdogRecord {
	uint32_t                     trips   = 0;
	bool                         claimed = false;
	uint64_t                     hash    = 0;
	uint32_t                     pc      = 0;
	// Each shader slot that counted cuts.
	std::vector<LoopWatchdogCut> cuts;
};

// After a lost device the driver is not called: the mapping is read as it stands.
bool ReadLoopWatchdog(LoopWatchdogRecord& record, bool invalidate) {
	using Watchdog = ShaderRecompiler::IR::LoopWatchdog;
	auto* buffer   = g_loop_watchdog.load(std::memory_order_acquire);
	if (buffer == nullptr) {
		return false;
	}
	std::array<uint32_t, Watchdog::DwordCount> words {};
	if (invalidate) {
		buffer->Invalidate(0, sizeof(words));
	}
	std::memcpy(words.data(), buffer->Mapped().data(), sizeof(words));
	record.trips   = words[Watchdog::TripCount];
	record.claimed = words[Watchdog::Claimed] != 0;
	record.hash    = (static_cast<uint64_t>(words[Watchdog::HashHigh]) << 32u) |
	              words[Watchdog::HashLow];
	record.pc = words[Watchdog::LoopPc];
	record.cuts.clear();
	for (uint32_t slot = Watchdog::SlotBase; slot < Watchdog::DwordCount;
	     slot += Watchdog::SlotDwords) {
		if (words[slot + Watchdog::SlotTrips] != 0) {
			record.cuts.push_back(
			    {(static_cast<uint64_t>(words[slot + Watchdog::SlotHashHigh]) << 32u) |
			         words[slot + Watchdog::SlotHashLow],
			     words[slot + Watchdog::SlotLoopPc], words[slot + Watchdog::SlotTrips]});
		}
	}
	return true;
}

std::string DescribeLoopWatchdog(const LoopWatchdogRecord& record, uint32_t trips) {
	auto text = fmt::format("GPU loop watchdog: {} shader invocation(s) ran a guest loop for over "
	                        "{} iterations or its time budget and were cut short, as were the same "
	                        "shaders' invocations looping at the time, so their results are wrong",
	                        trips, ShaderRecompiler::IR::LoopWatchdog::IterationLimit);
	if (record.claimed) {
		text += fmt::format("; the first was shader 0x{:016x}, loop at pc 0x{:x}", record.hash,
		                    record.pc);
	}
	return text + "\n";
}

const char* KindName(GpuCheckpointKind kind) {
	switch (kind) {
		case GpuCheckpointKind::Dispatch: return "Dispatch";
		case GpuCheckpointKind::DispatchIndirect: return "DispatchIndirect";
		case GpuCheckpointKind::DispatchDone: return "DispatchDone";
		case GpuCheckpointKind::Draw: return "Draw";
		case GpuCheckpointKind::DrawMesh: return "DrawMesh";
		case GpuCheckpointKind::FaultProcess: return "Internal:FaultProcess";
		case GpuCheckpointKind::Tiler: return "Internal:Tiler";
		case GpuCheckpointKind::DccClear: return "Internal:DccClear";
		case GpuCheckpointKind::Blit: return "Internal:Blit";
	}
	return "Unknown";
}

void AppendRecord(std::string& text, uint64_t sequence, std::vector<uint64_t>& shaders) {
	const auto& record = Records()[sequence & (RecordCount - 1u)];
	if (record.sequence != sequence) {
		text += fmt::format("    #{}: record overwritten\n", sequence);
		return;
	}
	text += fmt::format("    #{} submit={} {}", sequence, record.submit_id, KindName(record.kind));
	for (uint32_t i = 0; i < record.shader_count; i++) {
		text += fmt::format(" shader=0x{:016x}", record.shaders[i]);
		if (std::ranges::find(shaders, record.shaders[i]) == shaders.end()) {
			shaders.push_back(record.shaders[i]);
		}
	}
	text += fmt::format(" args={},{},{}\n", record.args[0], record.args[1], record.args[2]);
	if (record.dispatch_threads[0] != 0) {
		text += fmt::format("      sized in threads: {}x{}x{}\n", record.dispatch_threads[0],
		                    record.dispatch_threads[1], record.dispatch_threads[2]);
	}
	if (record.user_data_count != 0) {
		text += "      user data:";
		for (uint32_t i = 0; i < record.user_data_count; i++) {
			text += fmt::format(" {:08x}", record.user_data[i]);
		}
		text += "\n";
	}
	if (record.srt_count != 0) {
		text += "      flattened SRT:";
		for (uint32_t i = 0; i < record.srt_count; i++) {
			text += fmt::format(" [{}]={:08x}", i, record.srt[i]);
		}
		text += "\n";
	}
	for (uint32_t i = 0; i < record.buffer_count; i++) {
		const auto& buffer = record.buffers[i];
		text += fmt::format("      constant buffer 0x{:010x} size={} {}:", buffer.address,
		                    buffer.size,
		                    !buffer.read ? "unreadable"
		                    : buffer.clean ? "guest"
		                                   : "guest, GPU-written since");
		if (buffer.read) {
			for (const auto dword: buffer.dwords) {
				text += fmt::format(" {:08x}", dword);
			}
		}
		text += "\n";
	}
}

void AppendDeviceFault(vk::Device device, std::string& text) {
	vk::DeviceFaultCountsEXT counts {};
	auto result = device.getFaultInfoEXT(&counts, nullptr);
	if (result != vk::Result::eSuccess) {
		text += fmt::format("  device fault: query failed: {}\n", vk::to_string(result));
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	counts.vendorBinarySize = 0;
	vk::DeviceFaultInfoEXT info {};
	info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
	result             = device.getFaultInfoEXT(&counts, &info);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		text += fmt::format("  device fault: query failed: {}\n", vk::to_string(result));
		return;
	}
	text += fmt::format("  device fault: {}\n", info.description.data());
	for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
		const auto& address = addresses[i];
		text += fmt::format("    {} address 0x{:016x} (precision 0x{:x})\n",
		                    vk::to_string(address.addressType), address.reportedAddress,
		                    address.addressPrecision);
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
		const auto& vendor = vendors[i];
		text += fmt::format("    vendor: {} code=0x{:x} data=0x{:x}\n", vendor.description.data(),
		                    vendor.vendorFaultCode, vendor.vendorFaultData);
	}
}

// The driver reports, per pipeline stage, the last checkpoint that reached it. Work marked after
// the one at the bottom of the pipe was still running, or had not started, when the device failed.
void AppendCheckpoints(vk::Queue queue, std::string& text, std::vector<uint64_t>& shaders) {
	uint32_t count = 0;
	queue.getCheckpointDataNV(&count, nullptr);
	std::vector<vk::CheckpointDataNV> data(count);
	queue.getCheckpointDataNV(&count, data.data());
	if (count == 0) {
		text += "  checkpoints: none reported\n";
		return;
	}
	uint64_t started  = 0;
	uint64_t finished = UINT64_MAX;
	for (uint32_t i = 0; i < count; i++) {
		const auto sequence = static_cast<uint64_t>(
		    reinterpret_cast<uintptr_t>(data[i].pCheckpointMarker));
		text += fmt::format("  last checkpoint at {}:\n", vk::to_string(data[i].stage));
		if (sequence == 0) {
			text += "    none\n";
			continue;
		}
		AppendRecord(text, sequence, shaders);
		started  = std::max(started, sequence);
		finished = std::min(finished, sequence);
	}
	if (finished != UINT64_MAX && started > finished + 1u) {
		const auto last = std::min(started, finished + MaxListedInFlight);
		text += fmt::format("  started but not finished ({} total):\n", started - finished);
		for (uint64_t sequence = finished + 1u; sequence <= last; sequence++) {
			AppendRecord(text, sequence, shaders);
		}
	}
}

} // namespace

void MarkGpuCheckpoint(const GraphicContext& graphics, vk::CommandBuffer buffer,
                       GpuCheckpointKind kind, uint64_t submit_id,
                       std::span<const uint64_t> shader_hashes, uint32_t arg0, uint32_t arg1,
                       uint32_t arg2, const GpuCheckpointDispatch* dispatch) {
	if (!graphics.device_checkpoints_enabled) {
		return;
	}
	const auto sequence = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1u;
	auto&      record   = Records()[sequence & (RecordCount - 1u)];
	record.submit_id    = submit_id;
	record.kind         = kind;
	record.shader_count =
	    static_cast<uint32_t>(std::min(shader_hashes.size(), record.shaders.size()));
	std::copy_n(shader_hashes.begin(), record.shader_count, record.shaders.begin());
	record.args = {arg0, arg1, arg2};
	const GpuCheckpointDispatch none;
	const auto&                 inputs = dispatch != nullptr ? *dispatch : none;
	const auto&                 buffers = inputs.buffers;
	record.user_data_count              = static_cast<uint32_t>(
	    std::min<size_t>(inputs.user_data.size(), MaxRecordedUserData));
	std::copy_n(inputs.user_data.begin(), record.user_data_count, record.user_data.begin());
	record.srt_count =
	    static_cast<uint32_t>(std::min<size_t>(inputs.flattened_srt.size(), MaxRecordedSrt));
	std::copy_n(inputs.flattened_srt.begin(), record.srt_count, record.srt.begin());
	record.dispatch_threads = inputs.dispatch_threads;
	record.buffer_count =
	    static_cast<uint32_t>(std::min<size_t>(buffers.size(), MaxRecordedBuffers));
	for (uint32_t i = 0; i < record.buffer_count; i++) {
		auto& recorded   = record.buffers[i];
		recorded.address = buffers[i].address;
		recorded.size    = buffers[i].size;
		recorded.dwords.fill(0);
		const auto bytes = std::min<uint64_t>(buffers[i].size, RecordedBufferBytes);
		recorded.clean   = LibKernel::Memory::TryReadGpuCleanBacking(buffers[i].address,
		                                                             recorded.dwords.data(), bytes);
		recorded.read    = recorded.clean || LibKernel::Memory::TryReadBacking(
		                                        buffers[i].address, recorded.dwords.data(), bytes);
	}
	record.sequence = sequence;
	buffer.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(sequence)));
}

void ReportGpuFailure(const GraphicContext& graphics, const char* what, vk::Result result) {
	static std::atomic_flag reported = ATOMIC_FLAG_INIT;
	if (reported.test_and_set(std::memory_order_acq_rel)) {
		return;
	}
	std::string text = fmt::format("GPU failure: {} returned {}\n", what, vk::to_string(result));
	if (graphics.device_fault_enabled) {
		AppendDeviceFault(graphics.device, text);
	} else {
		text += "  device fault: VK_EXT_device_fault is not available\n";
	}
	std::vector<uint64_t> shaders;
	if (graphics.device_checkpoints_enabled) {
		AppendCheckpoints(graphics.queue, text, shaders);
	} else {
		text += "  checkpoints: VK_NV_device_diagnostic_checkpoints is not available\n";
	}
	if (LoopWatchdogRecord record; ReadLoopWatchdog(record, false)) {
		if (record.trips == 0) {
			text += "  loop watchdog: no guest loop reached the limit\n";
		} else {
			text += "  " + DescribeLoopWatchdog(record, record.trips);
		}
	}
	Log::WriteToConsoleAndLog(text);
	for (const auto hash: shaders) {
		DumpDiagnosticShader(hash, "gpu_failure");
	}
}

void RegisterDiagnosticShader(uint64_t hash, std::span<const uint32_t> code) {
	auto&           registry = Shaders();
	std::lock_guard lock(registry.mutex);
	registry.code.try_emplace(hash, code.begin(), code.end());
}

void DumpDiagnosticShader(uint64_t hash, const char* reason) {
	std::vector<uint32_t> code;
	{
		auto&           registry = Shaders();
		std::lock_guard lock(registry.mutex);
		const auto      found = registry.code.find(hash);
		if (!registry.dumped.emplace(hash, reason).second) {
			return;
		}
		if (found == registry.code.end()) {
			Log::WriteToConsoleAndLog(
			    fmt::format("Shader 0x{:016x}: code was not registered, nothing dumped\n", hash));
			return;
		}
		code = found->second;
	}
	const auto folder = Config::GetShaderLogFolder() / "diagnostics";
	auto       base   = folder / fmt::format("{:016x}_{}", hash, reason);
	std::error_code error;
	std::filesystem::create_directories(folder, error);
	{
		auto path = base;
		path += ".bin";
		std::ofstream file(path, std::ios::binary);
		file.write(reinterpret_cast<const char*>(code.data()),
		           static_cast<std::streamsize>(code.size() * sizeof(uint32_t)));
	}
	// The binary is on disk before decoding, which exits on an instruction it does not know.
	ShaderRecompiler::Decoder::Program decoded;
	ShaderRecompiler::Decoder::DecodeProgram(code, decoded);
	{
		auto path = base;
		path += ".rdna2";
		std::ofstream file(path);
		file << ShaderRecompiler::Decoder::ProgramToString(decoded);
	}
	Log::WriteToConsoleAndLog(fmt::format("Shader 0x{:016x} ({}) dumped to {}.bin/.rdna2\n", hash,
	                                      reason,
	                                      std::filesystem::absolute(base, error).string()));
}

DiagnosticShaderScope::DiagnosticShaderScope(uint64_t hash): m_previous(g_translating_shader) {
	static const bool installed = (Common::SetFatalHook(DumpTranslatingShader), true);
	(void)installed;
	g_translating_shader = hash;
}

DiagnosticShaderScope::~DiagnosticShaderScope() {
	g_translating_shader = m_previous;
}

uint32_t RunawayShaderGeneration() {
	return g_runaway_generation.load(std::memory_order_acquire);
}

std::vector<uint64_t> RunawayShaders() {
	std::lock_guard lock(g_runaway_mutex);
	return g_runaway_shaders;
}

void RegisterLoopWatchdog(Buffer* buffer) {
	g_loop_watchdog.store(buffer, std::memory_order_release);
}

namespace {

struct DmaDispatchRecord {
	std::vector<uint32_t>                      user_data;
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	std::vector<uint8_t>                       input;
	std::array<uint32_t, 3>                    groups {};
};

// The latest distinct dispatches of each DMA shader. The watchdog names only the shader it cut,
// and a shader such as a BVH refit runs once per tree each frame.
constexpr size_t RecordedDispatches = 64;
std::mutex                                                  g_dma_dispatch_mutex;
std::unordered_map<uint64_t, std::deque<DmaDispatchRecord>> g_dma_dispatches;
DiagnosticMemoryReader                                      g_memory_reader;

// The latest compute dispatches and the ranges they wrote, oldest overwritten first.
struct DispatchWrites {
	uint64_t                                    sequence = 0;
	uint64_t                                    hash     = 0;
	bool                                        dma      = false;
	std::array<uint32_t, 3>                     groups {};
	uint32_t                                    count = 0;
	std::array<std::pair<uint64_t, uint64_t>, 4> ranges {};
};
constexpr uint64_t          WriteRecordCount = uint64_t {1} << 16u;
std::mutex                  g_write_mutex;
std::vector<DispatchWrites> g_writes(WriteRecordCount);
uint64_t                    g_write_sequence = 0;

// Lists the recorded dispatches that wrote any of `dumped` to <hash>_writers.txt, and dumps the
// code of each shader among them.
void DumpWriters(uint64_t hash, const std::map<uint64_t, uint64_t>& dumped) {
	struct Writer {
		uint64_t                   dispatches = 0;
		bool                       dma        = false;
		std::array<uint32_t, 3>    groups {};
		std::pair<uint64_t, uint64_t> example {};
		uint64_t                   last = 0;
	};
	std::map<uint64_t, Writer> writers;
	uint64_t                   newest = 0;
	{
		std::lock_guard lock(g_write_mutex);
		newest = g_write_sequence;
		for (const auto& record: g_writes) {
			if (record.sequence == 0) {
				continue;
			}
			for (uint32_t index = 0; index < record.count; index++) {
				const auto [address, size] = record.ranges[index];
				auto next = dumped.upper_bound(address + size - 1u);
				bool hit  = false;
				while (next != dumped.begin()) {
					--next;
					if (next->first + next->second > address) {
						hit = true;
					}
					break;
				}
				if (!hit) {
					continue;
				}
				auto& writer = writers[record.hash];
				writer.dispatches++;
				writer.dma = writer.dma || record.dma;
				if (record.sequence >= writer.last) {
					writer.last    = record.sequence;
					writer.groups  = record.groups;
					writer.example = record.ranges[index];
				}
				break;
			}
		}
	}
	const auto    folder = Config::GetShaderLogFolder() / "diagnostics";
	std::ofstream file(folder / fmt::format("{:016x}_writers.txt", hash));
	file << fmt::format("dispatches recorded so far: {}\n", newest);
	for (const auto& [writer, info]: writers) {
		file << fmt::format(
		    "shader {:016x}{}: {} recent dispatches wrote the dumped memory; the latest (#{}, groups "
		    "{} {} {}) wrote 0x{:010x}+0x{:x}\n",
		    writer, info.dma ? " (also stores through DMA)" : "", info.dispatches, info.last,
		    info.groups[0], info.groups[1], info.groups[2], info.example.first,
		    info.example.second);
	}
	for (const auto& [writer, info]: writers) {
		(void)info;
		if (writer != hash) {
			DumpDiagnosticShader(writer, "writer");
		}
	}
}

size_t NonzeroWords(const std::vector<uint8_t>& bytes) {
	size_t count = 0;
	for (size_t offset = 0; offset + sizeof(uint32_t) <= bytes.size(); offset += sizeof(uint32_t)) {
		uint32_t word = 0;
		std::memcpy(&word, bytes.data() + offset, sizeof(word));
		count += word != 0u ? 1u : 0u;
	}
	return count;
}

// Writes the shader's recorded dispatches (user data, group counts, input info as
// <hash>_input_<n>.bin) to <hash>_dispatch.txt, and every guest range they named, as the GPU and
// guest memory hold them after the frame, to <hash>_mem_<address>_gpu.bin and _cpu.bin.
void DumpDispatchMemory(uint64_t hash) {
	constexpr uint64_t              MaxRangeBytes = 32ull * 1024ull * 1024ull;
	constexpr uint64_t              MaxDumpBytes  = 256ull * 1024ull * 1024ull;
	std::deque<DmaDispatchRecord> records;
	DiagnosticMemoryReader        reader;
	{
		std::lock_guard lock(g_dma_dispatch_mutex);
		const auto      found = g_dma_dispatches.find(hash);
		if (found == g_dma_dispatches.end() || !g_memory_reader) {
			return;
		}
		records = found->second;
		reader  = g_memory_reader;
	}
	const auto      folder = Config::GetShaderLogFolder() / "diagnostics";
	std::error_code error;
	std::filesystem::create_directories(folder, error);
	const auto    summary_path = folder / fmt::format("{:016x}_dispatch.txt", hash);
	std::ofstream summary(summary_path);
	// Buffers can share a base and differ in size; each base is dumped once, at its largest.
	std::map<uint64_t, uint64_t> largest;
	for (size_t index = 0; index < records.size(); index++) {
		const auto& record = records[index];
		summary << fmt::format("dispatch {}:\nuser data:", index);
		for (size_t word = 0; word < record.user_data.size(); word++) {
			summary << fmt::format("{}{:08x}", word % 8u == 0u ? "\n  " : " ",
			                       record.user_data[word]);
		}
		summary << fmt::format("\ngroups: {} {} {}\n", record.groups[0], record.groups[1],
		                       record.groups[2]);
		if (!record.input.empty()) {
			std::ofstream file(folder / fmt::format("{:016x}_input_{}.bin", hash, index),
			                   std::ios::binary);
			file.write(reinterpret_cast<const char*>(record.input.data()),
			           static_cast<std::streamsize>(record.input.size()));
		}
		for (const auto& [address, range_size]: record.ranges) {
			if (address != 0 && range_size != 0) {
				auto& size = largest[address];
				size       = std::max(size, range_size);
			}
		}
	}
	summary << "ranges:\n";
	uint64_t dumped_bytes = 0;
	for (const auto& [address, range_size]: largest) {
		if (dumped_bytes >= MaxDumpBytes) {
			summary << fmt::format("  0x{:010x}+0x{:x}: not dumped, over the dump budget\n", address,
			                       range_size);
			continue;
		}
		dumped_bytes += std::min(range_size, MaxRangeBytes);
		const auto           size = std::min(range_size, MaxRangeBytes);
		std::vector<uint8_t> gpu;
		std::vector<uint8_t> cpu;
		std::string          note;
		reader(address, size, gpu, cpu, note);
		for (const auto& [bytes, kind]: {std::pair {&gpu, "gpu"}, std::pair {&cpu, "cpu"}}) {
			if (bytes->empty()) {
				continue;
			}
			std::ofstream file(folder / fmt::format("{:016x}_mem_{:010x}_{}.bin", hash, address, kind),
			                   std::ios::binary);
			file.write(reinterpret_cast<const char*>(bytes->data()),
			           static_cast<std::streamsize>(bytes->size()));
		}
		const auto same = std::min(gpu.size(), cpu.size());
		summary << fmt::format(
		    "  0x{:010x}+0x{:x} (of 0x{:x}): {}; gpu {} bytes {} nonzero words; cpu {} bytes {} "
		    "nonzero words; {}\n",
		    address, size, range_size, note, gpu.size(), NonzeroWords(gpu), cpu.size(),
		    NonzeroWords(cpu),
		    same == 0 ? "not compared"
		              : (std::memcmp(gpu.data(), cpu.data(), same) == 0 ? "gpu matches cpu"
		                                                                 : "gpu differs from cpu"));
	}
	DumpWriters(hash, largest);
	Log::WriteToConsoleAndLog(fmt::format("Shader 0x{:016x}: {} recent dispatches and their memory dumped to {}\n",
	                                      hash, records.size(),
	                                      std::filesystem::absolute(summary_path, error).string()));
}

} // namespace

void RecordDmaDispatch(uint64_t hash, std::span<const uint32_t> user_data,
                       std::vector<std::pair<uint64_t, uint64_t>> ranges) {
	std::lock_guard lock(g_dma_dispatch_mutex);
	auto&           records = g_dma_dispatches[hash];
	records.emplace_back();
	records.back().user_data.assign(user_data.begin(), user_data.end());
	records.back().ranges = std::move(ranges);
	if (records.size() > 2 * RecordedDispatches) {
		records.pop_front();
	}
}

void RecordDmaDispatchShape(uint64_t hash, std::span<const uint8_t> input,
                           std::array<uint32_t, 3> groups) {
	std::lock_guard lock(g_dma_dispatch_mutex);
	auto& records = g_dma_dispatches[hash];
	if (records.empty()) {
		return;
	}
	auto& record = records.back();
	record.input.assign(input.begin(), input.end());
	record.groups = groups;
	// A dispatch repeating an earlier one's user data and groups adds nothing to replay.
	const auto repeat = std::find_if(records.begin(), records.end() - 1, [&](const auto& other) {
		return other.user_data == record.user_data && other.groups == record.groups;
	});
	if (repeat != records.end() - 1) {
		records.erase(repeat);
	}
	if (records.size() > RecordedDispatches) {
		records.pop_front();
	}
}

void RecordDispatchWrites(uint64_t hash, std::span<const std::pair<uint64_t, uint64_t>> ranges,
                          bool dma_writer, std::array<uint32_t, 3> groups) {
	if (ranges.empty() && !dma_writer) {
		return;
	}
	std::lock_guard lock(g_write_mutex);
	auto&           record = g_writes[g_write_sequence % WriteRecordCount];
	record.sequence        = ++g_write_sequence;
	record.hash            = hash;
	record.dma             = dma_writer;
	record.groups          = groups;
	record.count           = static_cast<uint32_t>(std::min(ranges.size(), record.ranges.size()));
	std::copy_n(ranges.begin(), record.count, record.ranges.begin());
}

void RegisterDiagnosticMemoryReader(DiagnosticMemoryReader reader) {
	std::lock_guard lock(g_dma_dispatch_mutex);
	g_memory_reader = std::move(reader);
}

namespace {

std::mutex         g_dma_writer_mutex;
std::set<uint64_t> g_dma_writers;
std::set<uint64_t> g_recent_dma_writers;

// The recorded dispatches of every shader that stores through DMA, user data and group counts
// only, to dma_writers.txt: which of them built the memory a runaway loop reads.
void DumpDmaWriterDispatches() {
	std::set<uint64_t> writers;
	{
		std::lock_guard lock(g_dma_writer_mutex);
		writers = g_dma_writers;
	}
	const auto      folder = Config::GetShaderLogFolder() / "diagnostics";
	std::error_code error;
	std::filesystem::create_directories(folder, error);
	std::ofstream   summary(folder / "dma_writers.txt");
	std::lock_guard lock(g_dma_dispatch_mutex);
	for (const auto hash: writers) {
		const auto found = g_dma_dispatches.find(hash);
		if (found == g_dma_dispatches.end()) {
			continue;
		}
		summary << fmt::format("shader {:016x}: {} recent dispatches\n", hash, found->second.size());
		for (const auto& record: found->second) {
			summary << "  user data:";
			for (size_t word = 0; word < record.user_data.size(); word++) {
				summary << fmt::format("{}{:08x}", word % 8u == 0u ? "\n    " : " ",
				                       record.user_data[word]);
			}
			summary << fmt::format("\n  groups: {} {} {}\n", record.groups[0], record.groups[1],
			                       record.groups[2]);
		}
	}
}

} // namespace

void NoteDmaWriter(uint64_t hash) {
	bool first = false;
	{
		std::lock_guard lock(g_dma_writer_mutex);
		first = g_dma_writers.insert(hash).second;
		g_recent_dma_writers.insert(hash);
	}
	if (first) {
		DumpDiagnosticShader(hash, "dma_writer");
	}
}

std::vector<uint64_t> TakeDmaWriters() {
	std::lock_guard       lock(g_dma_writer_mutex);
	std::vector<uint64_t> writers(g_recent_dma_writers.begin(), g_recent_dma_writers.end());
	g_recent_dma_writers.clear();
	return writers;
}

void ReportDroppedDmaStores(std::span<const uint64_t> pages, std::span<const uint64_t> writers,
                            uint64_t page_size) {
	if (pages.empty()) {
		return;
	}
	// The first reports are all logged; later ones are sampled, with the running totals.
	static uint64_t reports       = 0;
	static uint64_t dropped_pages = 0;
	reports++;
	dropped_pages += pages.size();
	if (reports > 40 && reports % 100 != 0) {
		return;
	}
	std::vector<uint64_t> sorted(pages.begin(), pages.end());
	std::ranges::sort(sorted);
	std::string ranges;
	uint32_t    listed = 0;
	for (size_t index = 0; index < sorted.size() && listed < 12;) {
		auto end = index + 1;
		while (end < sorted.size() && sorted[end] == sorted[end - 1] + page_size) {
			end++;
		}
		ranges += fmt::format(" 0x{:010x}+0x{:x}", sorted[index], (end - index) * page_size);
		listed++;
		index = end;
	}
	std::string names;
	for (size_t index = 0; index < writers.size() && index < 8; index++) {
		names += fmt::format(" 0x{:016x}", writers[index]);
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "DMA stores dropped (report {}, {} pages so far): {} page(s) no cached buffer covered "
	    "were stored to:{}{}; DMA writers since the last report:{}\n",
	    reports, dropped_pages, sorted.size(), ranges, listed < sorted.size() ? " ..." : "",
	    names.empty() ? std::string(" none recorded") : names));
}

void ReportLoopWatchdog() {
	using Watchdog = ShaderRecompiler::IR::LoopWatchdog;
	LoopWatchdogRecord record;
	if (!ReadLoopWatchdog(record, true) || record.trips == 0) {
		return;
	}
	// Each shader and loop is described once; later cuts of it are not logged again.
	static std::set<std::pair<uint64_t, uint32_t>> described;
	if (!record.claimed || described.emplace(record.hash, record.pc).second) {
		Log::WriteToConsoleAndLog(DescribeLoopWatchdog(record, record.trips));
	}
	// The first shader cut, then every other shader whose slot counted cuts.
	std::vector<LoopWatchdogCut> cuts;
	if (record.claimed) {
		cuts.push_back({record.hash, record.pc, 0});
	}
	for (const auto& cut: record.cuts) {
		if (record.claimed && cut.hash == record.hash) {
			continue;
		}
		cuts.push_back(cut);
		if (described.emplace(cut.hash, cut.pc).second) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "GPU loop watchdog: shader 0x{:016x} was cut short too, {} invocation(s) in its "
			    "loop at pc 0x{:x}\n",
			    cut.hash, cut.trips, cut.pc));
		}
	}
	static bool writers_dumped = false;
	if (!cuts.empty() && !writers_dumped) {
		writers_dumped = true;
		DumpDmaWriterDispatches();
	}
	for (const auto& cut: cuts) {
		DumpDiagnosticShader(cut.hash, "loop_watchdog");
		static std::set<uint64_t> memory_dumped;
		if (memory_dumped.insert(cut.hash).second) {
			DumpDispatchMemory(cut.hash);
		}
		std::lock_guard lock(g_runaway_mutex);
		if (++g_runaway_reports[cut.hash] == RunawayShaderReports) {
			g_runaway_shaders.push_back(cut.hash);
			g_runaway_generation.fetch_add(1, std::memory_order_release);
		}
	}
	// Loops stop aborting once the counts are clear, and the next cut records its own shader.
	static_assert(Watchdog::Claimed == Watchdog::TripCount + 1u);
	auto* buffer = g_loop_watchdog.load(std::memory_order_acquire);
	auto* words  = reinterpret_cast<uint32_t*>(buffer->Mapped().data());
	words[Watchdog::TripCount] = 0;
	words[Watchdog::Claimed]   = 0;
	buffer->Flush(Watchdog::TripCount * sizeof(uint32_t), 2 * sizeof(uint32_t));
	std::fill(words + Watchdog::SlotBase, words + Watchdog::DwordCount, 0u);
	buffer->Flush(Watchdog::SlotBase * sizeof(uint32_t),
	              (Watchdog::DwordCount - Watchdog::SlotBase) * sizeof(uint32_t));
}

} // namespace Libs::Graphics
