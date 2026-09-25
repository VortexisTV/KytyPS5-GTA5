#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;
// Eager shadow flushes a readback-hot buffer may spend before a shadow has to prove useful again.
constexpr uint32_t EagerShadowBudget = 8;
// Readbacks up to this size are waited for by the faulting thread; bigger ones keep the old path.
constexpr uint64_t AsyncReadbackMax = 1024 * 1024;
// How long download ring contents stay valid, the bound the shadow path already relies on.
constexpr uint64_t ReadbackMaxAgeTicks = 64;
// Buffers up to this size get a host-visible shadow once the CPU has read them back.
constexpr uint64_t MaxHotReadbackSize = 1024 * 1024;

// GPU-written ranges that readbacks had to wait for, oldest first, so the writes that dirty them
// again can be named. Only the GPU thread touches it, so it needs no lock.
std::vector<std::pair<uint64_t, uint64_t>> g_read_back_dirty;
constexpr size_t                           ReadBackDirtyWatched = 16;

std::string DescribePacket(uint32_t packet) {
	return packet == GuestGpu::NoPacket
	           ? std::string("outside a packet")
	           : fmt::format("in PM4 op 0x{:02x} r {}", packet >> 8u, packet & 0xffu);
}

// A range the GPU rewrites every frame is read back every frame. Note the write that does it, the
// first time per writer and every 500th time after.
void NoteReadBackDirtied(uint64_t vaddr, uint64_t size) {
	const auto watched = std::find_if(
	    g_read_back_dirty.begin(), g_read_back_dirty.end(), [&](const auto& range) {
		    return vaddr < range.first + range.second && range.first < vaddr + size;
	    });
	if (watched == g_read_back_dirty.end()) {
		return;
	}
	static std::map<std::tuple<uint64_t, uint64_t, uint32_t>, uint64_t> writers;
	const auto packet = GuestGpu::CurrentPacket();
	auto&      seen   = writers[{vaddr, size, packet}];
	seen++;
	if ((seen == 1 && writers.size() <= 256) || seen % 500 == 0) {
		PerfStats::Note(fmt::format(
		    "gpu write {} ({}x this writer) 0x{:010x}+0x{:x} {} dirties read-back bytes "
		    "0x{:010x}+0x{:x}",
		    seen == 1 ? "first" : "sample", seen, vaddr, size, DescribePacket(packet),
		    watched->first, watched->second));
	}
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		// A fresh host buffer needs every dirty page uploaded; forget content hashes that would
		// otherwise let hot pages skip their first upload into it.
		m_memory_tracker.ClearHotPages(buffer.CpuAddress(), buffer.Size());
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, pages.first * sizeof(vk::DeviceAddress),
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(pages.first * sizeof(vk::DeviceAddress),
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
	// The set of buffers SynchronizeBuffersInRange walks has changed.
	m_layout_epoch.fetch_add(1, std::memory_order_release);
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}

	const auto [mapped, offset] = RecordDownload(buffer, copies, total_size);
	m_scheduler.DeferPriorityOperation([this, mapped, offset, total_size, buffer_address,
	                                    copies = std::move(copies)] {
		m_download_buffer.Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	});
	return true;
}

std::pair<uint8_t*, uint64_t> BufferCache::RecordDownload(Buffer&                      buffer,
                                                          std::vector<vk::BufferCopy>& copies,
                                                          uint64_t total_size) {
	PerfStats::Span span(PerfStats::SpanId::BufferDownload);
	if (PerfStats::Enabled()) {
		uint64_t bytes = 0;
		for (const auto& copy: copies) {
			bytes += copy.size;
		}
		PerfStats::Add(PerfStats::CounterId::BufferDownloadBytes, bytes);
	}
	const auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	if (mapped == nullptr) {
		EXIT("BufferCache: download exceeds 64 MiB staging buffer capacity\n");
	}
	m_download_buffer.Commit();
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), m_download_buffer.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = m_download_buffer.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	return {mapped, offset};
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	auto& gpu = m_scheduler.Context().GetGpu();
	if (GuestGpu::IsGpuThread()) {
		(void)ReadMemoryOnGpu(vaddr, size, is_write);
		return;
	}
	// Only recording and submitting the copy happen on the emulated GPU thread; this faulting
	// thread waits for the host GPU itself. Command translation keeps going meanwhile instead of
	// stalling until the host GPU has caught up with everything already recorded.
	ReadbackTicket ticket;
	gpu.SendCommandSync([&] { ticket = ReadMemoryOnGpu(vaddr, size, is_write, true); });
	if (ticket.id == 0) {
		return;
	}
	{
		PerfStats::Span wait(PerfStats::SpanId::ReadbackGuestWait);
		m_scheduler.GetMasterSemaphore().Wait(ticket.tick);
	}
	gpu.SendCommandSync([this, id = ticket.id] { FinishReadback(id); });
}

BufferCache::ReadbackTicket BufferCache::ReadMemoryOnGpu(uint64_t vaddr, uint64_t size,
                                                         bool is_write, bool allow_async) {
	const GpuWaitScope wait_scope(PerfStats::SpanId::GpuWaitReadback);
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return {};
	}
	const auto fault_vaddr = vaddr;
	const auto fault_size  = size;
	auto&      buffer      = m_slot_buffers[FindBuffer(vaddr, size)];

	// Widen nearby CPU reads so they share one GPU drain.
	constexpr uint64_t WindowSize   = 512 * 1024;
	const auto         buffer_begin = buffer.CpuAddress();
	const auto         buffer_end   = buffer_begin + buffer.Size();
	const auto window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
	const auto window_end = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);
	const auto window_size = window_end - window_begin;

	struct Range {
		uint64_t address = 0;
		uint64_t size    = 0;
	};
	std::vector<Range> ranges;
	uint64_t           download_bytes = 0;
	m_memory_tracker.ForEachDownloadRange<false>(
	    window_begin, window_size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer readback");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    ranges.push_back({start, end - start});
			    download_bytes += end - start;
		    });
	    });
	if (ranges.empty()) {
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
		return {};
	}

	// A buffer the CPU keeps reading back (GPU counters, query results) gets a host-visible
	// shadow recorded at the end of every slice that writes it. When that shadow is at least as
	// new as the last GPU write, the read only has to wait for a tick that usually finished long
	// ago instead of draining the whole GPU.
	Buffer*    hot_owner = buffer.Size() <= MaxHotReadbackSize ? &buffer : nullptr;
	const bool async     = allow_async && download_bytes <= AsyncReadbackMax;
	const auto overlaps_recent_write = [&](const Range& range) {
		return std::ranges::any_of(hot_owner->writes_since_shadow, [&](const auto& write) {
			return range.address < write.address + write.size &&
			       write.address < range.address + range.size;
		});
	};
	const bool shadow_usable =
	    hot_owner != nullptr && hot_owner->readback_hot && hot_owner->shadow_valid &&
	    hot_owner->writes_since_shadow.size() < 64 &&
	    m_scheduler.CurrentTick() - hot_owner->shadow_tick < ReadbackMaxAgeTicks &&
	    std::ranges::none_of(ranges, overlaps_recent_write);
	if (PerfStats::Enabled()) {
		// Which of the conditions above sent this readback down the draining path.
		PerfStats::Add([&] {
			if (shadow_usable) {
				return PerfStats::CounterId::ReadbacksShadow;
			}
			if (hot_owner == nullptr) {
				return PerfStats::CounterId::ReadbackNoOwner;
			}
			if (!hot_owner->readback_hot) {
				return PerfStats::CounterId::ReadbackNotHot;
			}
			if (!hot_owner->shadow_valid || hot_owner->writes_since_shadow.size() >= 64 ||
			    m_scheduler.CurrentTick() - hot_owner->shadow_tick >= ReadbackMaxAgeTicks) {
				return PerfStats::CounterId::ReadbackShadowStale;
			}
			return PerfStats::CounterId::ReadbackRecentWrite;
		}());
	}
	// Each readback stalls whoever waits for it, so the few that repeat every frame matter far more
	// than their byte count suggests. Note each distinct range the first time it is read back and
	// every 500th readback as a sample, with the thread and packet that asked: loading screens read
	// a range back once, while the ranges gameplay reads every frame keep reappearing in the
	// samples. Only the GPU thread runs this, so the statics need no lock.
	if (PerfStats::Enabled()) {
		static std::map<std::pair<uint64_t, uint64_t>, uint64_t> read_back;
		static uint64_t                                          readbacks = 0;
		auto& seen = read_back[{window_begin, window_size}];
		seen++;
		readbacks++;
		std::string dirty;
		for (const auto& range: ranges) {
			const std::pair watched {range.address, range.size};
			if (std::find(g_read_back_dirty.begin(), g_read_back_dirty.end(), watched) ==
			    g_read_back_dirty.end()) {
				if (g_read_back_dirty.size() == ReadBackDirtyWatched) {
					g_read_back_dirty.erase(g_read_back_dirty.begin());
				}
				g_read_back_dirty.push_back(watched);
			}
			if (dirty.size() < 64) {
				dirty += fmt::format(" 0x{:010x}+0x{:x}", range.address, range.size);
			}
		}
		if ((seen == 1 && read_back.size() <= 256) || readbacks % 500 == 0) {
			const auto asker = allow_async
			                       ? std::string("a game thread")
			                       : "the GPU thread " + DescribePacket(GuestGpu::CurrentPacket());
			PerfStats::Note(fmt::format(
			    "readback {} {} #{} ({}x this range) by {}: {} 0x{:010x}+0x{:x}, fault at "
			    "0x{:010x}, {} ranges, {} bytes:{} | owner 0x{:010x}+0x{:x} | {}",
			    shadow_usable ? "shadow" : "drain", seen == 1 ? "first" : "sample", readbacks, seen,
			    asker, is_write ? "write" : "read", window_begin, window_size, fault_vaddr,
			    ranges.size(), download_bytes, dirty, buffer.CpuAddress(), buffer.Size(),
			    hot_owner == nullptr
			        ? std::string("too big for a shadow")
			        : fmt::format("hot {} shadow {} writes since shadow {} shadow age {} ticks",
			                      hot_owner->readback_hot, hot_owner->shadow_valid,
			                      hot_owner->writes_since_shadow.size(),
			                      m_scheduler.CurrentTick() - hot_owner->shadow_tick)));
		}
	}

	const auto queue = [&](uint64_t tick, bool downloaded,
	                       std::vector<PendingReadback::Part>&& parts) {
		PerfStats::Add(PerfStats::CounterId::ReadbacksAsync);
		const auto id = m_next_readback_id++;
		m_pending_readbacks.push_back({.id          = id,
		                               .tick        = tick,
		                               .after_tick  = m_scheduler.CurrentTick(),
		                               .fault_vaddr = fault_vaddr,
		                               .fault_size  = fault_size,
		                               .vaddr       = window_begin,
		                               .size        = window_size,
		                               .is_write    = is_write,
		                               .downloaded  = downloaded,
		                               .valid       = true,
		                               .parts       = std::move(parts)});
		return ReadbackTicket {.id = id, .tick = tick};
	};
	if (shadow_usable) {
		hot_owner->eager_budget = EagerShadowBudget;
		if (async) {
			std::vector<PendingReadback::Part> parts;
			parts.reserve(ranges.size());
			for (const auto& range: ranges) {
				parts.push_back({range.address,
				                 hot_owner->shadow_offset + (range.address - buffer_begin),
				                 range.size});
			}
			if (hot_owner->shadow_tick >= m_scheduler.CurrentTick()) {
				m_scheduler.Flush();
			}
			return queue(hot_owner->shadow_tick, false, std::move(parts));
		}
		m_scheduler.Wait(hot_owner->shadow_tick);
		m_download_buffer.Invalidate(hot_owner->shadow_offset, hot_owner->Size());
		const auto* shadow = m_download_buffer.Mapped().data() + hot_owner->shadow_offset;
		for (const auto& range: ranges) {
			Libs::LibKernel::Memory::WriteBacking(range.address,
			                                      shadow + (range.address - buffer_begin), range.size);
			m_gpu_modified_ranges.Subtract(range.address, range.size);
		}
	} else {
		if (hot_owner != nullptr) {
			if (hot_owner->readback_hot && hot_owner->shadow_valid) {
				// The shadow existed but writes recorded after it were still unsubmitted.
				hot_owner->eager_budget = EagerShadowBudget;
			}
			hot_owner->readback_hot = true;
		}
		if (async) {
			// The dirty ranges stay until FinishReadback publishes the bytes, so a fault that
			// overlaps this one meanwhile still sees tracker pages and dirty bytes agree.
			std::vector<vk::BufferCopy> copies;
			uint64_t                    total_size = 0;
			for (const auto& range: ranges) {
				copies.emplace_back(range.address - buffer_begin, total_size, range.size);
				// Keep packed ranges on separate cache lines, as DownloadBufferMemory does.
				total_size += Common::AlignUp(range.size, 64);
			}
			const auto [mapped, offset] = RecordDownload(buffer, copies, total_size);
			(void)mapped;
			std::vector<PendingReadback::Part> parts;
			parts.reserve(copies.size());
			for (const auto& copy: copies) {
				parts.push_back({buffer_begin + copy.srcOffset, copy.dstOffset, copy.size});
			}
			const auto tick = m_scheduler.CurrentTick();
			m_scheduler.Flush();
			return queue(tick, true, std::move(parts));
		}
		EXIT_IF(!DownloadBufferMemory(buffer, window_begin, window_size));
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_size);
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
	return {};
}

void BufferCache::FinishReadback(uint64_t id) {
	const auto it = std::ranges::find(m_pending_readbacks, id, &PendingReadback::id);
	EXIT_IF(it == m_pending_readbacks.end());
	auto pending = std::move(*it);
	m_pending_readbacks.erase(it);
	// The bytes were copied when the fault arrived. A GPU write to the range since, or a copy old
	// enough for the download ring to have wrapped over it, means they may not be the latest: redo
	// the readback the synchronous way.
	if (!pending.valid || m_scheduler.CurrentTick() - pending.after_tick >= ReadbackMaxAgeTicks) {
		PerfStats::Add(PerfStats::CounterId::ReadbacksRetried);
		(void)ReadMemoryOnGpu(pending.fault_vaddr, pending.fault_size, pending.is_write);
		return;
	}
	m_scheduler.WaitPriorityOperations(pending.tick);
	for (const auto& part: pending.parts) {
		// Another fault on the same range may have completed first and handed the page to the
		// CPU, which may already have written it.
		if (!m_memory_tracker.IsRegionGpuModified(part.address, part.size)) {
			continue;
		}
		m_download_buffer.Invalidate(part.offset, part.size);
		Libs::LibKernel::Memory::WriteBacking(
		    part.address, m_download_buffer.Mapped().data() + part.offset, part.size);
		m_gpu_modified_ranges.Subtract(part.address, part.size);
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(pending.vaddr, pending.size);
	if (pending.is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(pending.fault_vaddr, pending.fault_size);
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, PageTable::kAddressSpaceSize - end);
			}
			if (expands_right) {
				const auto minimum = CACHING_PAGESIZE * 2;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	PerfStats::Add(PerfStats::CounterId::BufferCreates);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	uint64_t                    upload_start = 0;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept {
		    // Started here rather than above so the hot-page hashing in the range walk stays out.
		    upload_start = PerfStats::Enabled() ? PerfStats::Now() : 0;
		    source       = UploadCopies(buffer, copies, total_size);
	    });
	if (source) {
		PerfStats::Add(PerfStats::CounterId::BufferUploads);
		PerfStats::Add(PerfStats::CounterId::BufferUploadBytes, total_size);
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
		if (PerfStats::Enabled()) {
			PerfStats::Record(PerfStats::SpanId::BufferUpload, PerfStats::Now() - upload_start);
		}
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			std::memcpy(mapped + copy.srcOffset, reinterpret_cast<const void*>(address), copy.size);
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		std::memcpy(temporary->Mapped().data() + copy.srcOffset,
		            reinterpret_cast<const void*>(address), copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    m_memory_tracker.IsRegionCpuModifiedAndGpuClean(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr && Libs::LibKernel::Memory::TryReadBacking(vaddr, mapped, size)) {
			m_stream_buffer.Commit();
			PerfStats::Add(PerfStats::CounterId::StreamUploads);
			PerfStats::Add(PerfStats::CounterId::StreamUploadBytes, size);
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		if (PerfStats::Enabled()) {
			NoteReadBackDirtied(vaddr, size);
		}
		m_gpu_modified_ranges.Add(vaddr, size);
		if (!m_write_watches.empty()) {
			// No watch starting before this bound can reach vaddr.
			const auto lowest = vaddr >= m_write_watch_span ? vaddr - m_write_watch_span + 1 : 0;
			for (auto it = m_write_watches.lower_bound(lowest);
			     it != m_write_watches.end() && it->first < vaddr + size; ++it) {
				if (it->first + it->second.size > vaddr) {
					it->second.writes++;
				}
			}
		}
		// A readback copied before this write no longer holds the latest bytes of the range.
		for (auto& pending: m_pending_readbacks) {
			if (vaddr < pending.vaddr + pending.size && pending.vaddr < vaddr + size) {
				pending.valid = false;
			}
		}
		buffer.last_gpu_write_tick = m_scheduler.CurrentTick();
		if (buffer.readback_hot) {
			if (buffer.writes_since_shadow.size() < 64) {
				buffer.writes_since_shadow.push_back({vaddr, size});
			}
			if (!buffer.shadow_pending) {
				buffer.shadow_pending = true;
				m_hot_written.push_back(id);
			}
			if (buffer.eager_budget != 0) {
				buffer.eager_budget--;
				m_eager_shadow_pending = true;
			}
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

uint64_t BufferCache::GpuWriteCount(uint64_t vaddr, uint64_t size) {
	auto& watch        = m_write_watches[vaddr];
	watch.size         = std::max(watch.size, size);
	m_write_watch_span = std::max(m_write_watch_span, watch.size);
	return watch.writes;
}

void BufferCache::RecordHotShadows() {
	if (m_hot_written.empty()) {
		return;
	}
	auto& command = m_scheduler.Current();
	for (const auto id: m_hot_written) {
		auto* buffer = m_slot_buffers.try_get(id);
		if (buffer == nullptr || buffer->is_deleted) {
			continue;
		}
		buffer->shadow_pending = false;
		const auto [mapped, offset] = m_download_buffer.Map(buffer->Size(), 64);
		if (mapped == nullptr) {
			buffer->shadow_valid = false;
			continue;
		}
		m_download_buffer.CopyFrom(command, *buffer, 0, offset, buffer->Size(),
		                           vk::AccessFlagBits::eMemoryWrite, vk::AccessFlags {},
		                           vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                           vk::AccessFlagBits::eHostRead);
		m_download_buffer.Commit();
		buffer->shadow_offset = offset;
		buffer->shadow_tick   = m_scheduler.CurrentTick();
		buffer->shadow_valid  = true;
		buffer->writes_since_shadow.clear();
		PerfStats::Add(PerfStats::CounterId::ShadowsRecorded);
		// Once the GPU has produced this shadow, push it into guest memory proactively. A CPU
		// poll that arrives afterwards then reads current data without faulting at all.
		const auto tick = buffer->shadow_tick;
		m_scheduler.DeferOperation([this, id, tick] { WriteBackShadow(id, tick); });
	}
	m_hot_written.clear();
	m_eager_shadow_pending = false;
}

void BufferCache::WriteBackShadow(BufferId id, uint64_t tick) {
	auto* buffer = m_slot_buffers.try_get(id);
	if (buffer == nullptr || buffer->is_deleted || !buffer->readback_hot || !buffer->shadow_valid ||
	    buffer->shadow_tick != tick || !buffer->writes_since_shadow.empty()) {
		return;
	}
	const auto base = buffer->CpuAddress();
	const auto size = buffer->Size();
	struct Copy {
		uint64_t address = 0;
		uint64_t size    = 0;
	};
	std::vector<Copy> copies;
	m_memory_tracker.ForEachDownloadRange<false>(
	    base, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    start = std::max(start, base);
			    end   = std::min(end, base + size);
			    if (start < end) {
				    copies.push_back({start, end - start});
			    }
		    });
	    });
	if (copies.empty()) {
		return;
	}
	m_download_buffer.Invalidate(buffer->shadow_offset, size);
	const auto* shadow = m_download_buffer.Mapped().data() + buffer->shadow_offset;
	for (const auto& copy: copies) {
		Libs::LibKernel::Memory::WriteBacking(copy.address, shadow + (copy.address - base),
		                                      copy.size);
		m_gpu_modified_ranges.Subtract(copy.address, copy.size);
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(base, size);
	buffer->eager_budget = EagerShadowBudget;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr || (!Libs::LibKernel::Memory::TryReadBacking(vaddr, staging, size) &&
	                           !Libs::LibKernel::Memory::TryReadPrtBacking(vaddr, staging, size))) {
		EXIT("BufferCache: failed to read mapped guest image backing\n");
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

// Guest memory and every cached copy of it take the bytes directly, without the fault that a store
// through the guest mapping would take on a protected page.
void BufferCache::WriteHostMemory(uint64_t vaddr, std::span<const uint8_t> data) {
	if (vaddr == 0 || data.empty() || data.size() > UINT64_MAX - vaddr) {
		EXIT("BufferCache: invalid host write\n");
	}
	Libs::LibKernel::Memory::WriteBacking(vaddr, data.data(), data.size());

	const auto end   = vaddr + data.size();
	auto       first = m_buffers.upper_bound(vaddr);
	if (first != m_buffers.begin()) {
		first = std::prev(first);
	}
	for (auto it = first; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer    = m_slot_buffers[it->second];
		const auto begin     = std::max(vaddr, buffer.CpuAddress());
		const auto range_end = std::min(end, buffer.CpuAddress() + buffer.Size());
		if (begin >= range_end) {
			continue;
		}
		WriteDataBuffer(buffer, begin, data.data() + (begin - vaddr), range_end - begin);
		TouchBuffer(buffer);
	}
}

// A plain store into a page holding GPU-written bytes faults, and the page can only change hands
// once the GPU has finished writing them. When the stored bytes are not among them, guest memory is
// already the only copy of those bytes that matters: write it and every cached copy directly and
// leave the GPU's bytes and the page's protection alone.
bool BufferCache::TryWriteBesideGpu(uint64_t vaddr, std::span<const uint8_t> data) {
	if (data.empty() || !IsRegionGpuModified(vaddr, data.size()) ||
	    HasGpuDirtyBytes(vaddr, data.size())) {
		return false;
	}
	if (m_texture_cache.FindImageFromRange(vaddr, data.size(), false) ||
	    m_texture_cache.IsRegionGpuModified(vaddr, data.size())) {
		return false;
	}
	PerfStats::Add(PerfStats::CounterId::WritesBesideGpu);
	WriteHostMemory(vaddr, data);
	return true;
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	RecordHotShadows();
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	PerfStats::Set(PerfStats::GaugeId::GpuMemoryMb, m_total_used_memory >> 20u);
	PerfStats::Set(PerfStats::GaugeId::BufferGcTriggerMb, m_trigger_gc_memory >> 20u);
	PerfStats::Set(PerfStats::GaugeId::CachedBuffers, m_buffers.size());
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_IF(!DownloadBufferMemory(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			PerfStats::Add(PerfStats::CounterId::BuffersEvicted);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
	PerfStats::Add(PerfStats::CounterId::BuffersEvicted, dirty_buffers.size());
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	uint64_t   visited = 0;
	const auto end     = vaddr + size;
	auto       first   = m_buffers.upper_bound(vaddr);
	if (first != m_buffers.begin()) {
		const auto previous = std::prev(first);
		if (previous->first + m_slot_buffers[previous->second].Size() > vaddr) {
			first = previous;
		}
	}
	if (first == m_buffers.end() || first->first >= end) {
		return;
	}
	const auto last             = std::prev(m_buffers.lower_bound(end));
	const auto candidate_start  = std::max(vaddr, first->first);
	const auto candidate_end    = std::min(end, last->first + m_slot_buffers[last->second].Size());
	uint64_t   synchronized_end = vaddr;
	m_memory_tracker.ForEachUploadCandidateRange(
	    candidate_start, candidate_end - candidate_start, [&](uint64_t dirty, uint64_t bytes) {
		    const auto dirty_end = dirty + bytes;
		    dirty                = std::max(dirty, synchronized_end);
		    if (dirty >= dirty_end) {
			    return;
		    }
		    auto it = m_buffers.upper_bound(dirty);
		    if (it != m_buffers.begin()) {
			    --it;
		    }
		    for (; it != m_buffers.end() && it->first < dirty_end; ++it) {
			    auto&      buffer = m_slot_buffers[it->second];
			    const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
			    if (finish <= dirty) {
				    continue;
			    }
			    // Synchronize the whole mapped intersection once, even when several dirty runs or
			    // tracking regions intersect this buffer. The normal upload path rechecks the
			    // pages.
			    const auto start = std::max(buffer.CpuAddress(), vaddr);
			    (void)SynchronizeBuffer(buffer, start, finish - start, false, false);
			    synchronized_end = finish;
			    visited++;
		    }
	    });
	PerfStats::Add(PerfStats::CounterId::BdaBuffersVisited, visited);
}

} // namespace Libs::Graphics
