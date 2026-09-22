#include "graphics/host_gpu/renderer/cache/gpuResourceManager.h"

#include "common/assert.h"
#include "common/perfStats.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>

namespace Libs::Graphics {

GpuResourceManager::GpuResourceManager(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_scheduler(scheduler), m_buffer_cache(graphics, scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, scheduler, m_page_manager, m_buffer_cache) {}

GpuResourceManager::~GpuResourceManager() = default;

bool GpuResourceManager::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	PerfStats::Span span(PerfStats::SpanId::PageFault);
	PerfStats::Add(access == PageFaultAccess::Write ? PerfStats::CounterId::WriteFaults
	                                                : PerfStats::CounterId::ReadFaults);
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool GpuResourceManager::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool GpuResourceManager::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

uint64_t GpuResourceManager::MappedExtent(uint64_t vaddr, uint64_t max_size) const noexcept {
	if (vaddr == 0 || max_size == 0 || vaddr >= TRACKER_ADDRESS_SIZE) {
		return 0;
	}
	// Clip before vaddr + max_size could leave the tracker's address space, so a descriptor that
	// nominally spans terabytes cannot produce a range the tracker would later reject outright.
	const auto       limit = std::min(max_size, TRACKER_ADDRESS_SIZE - vaddr);
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.ContiguousExtent(vaddr, limit);
}

void GpuResourceManager::MapMemory(uint64_t vaddr, uint64_t size) {
	{
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Add(vaddr, size);
	}
	m_mapping_epoch.fetch_add(1, std::memory_order_release);
	m_page_manager.OnGpuMap(vaddr, size);
}

void GpuResourceManager::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		if (m_scheduler.Active()) {
			const auto tick = m_scheduler.CurrentTick();
			m_scheduler.Finish();
			m_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_page_manager.OnGpuUnmap(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		m_mapping_epoch.fetch_add(1, std::memory_order_release);
	};
	if (m_gpu == nullptr) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void GpuResourceManager::PrepareBda() {
	PerfStats::Span span(PerfStats::SpanId::BdaPrepare);
	// A DMA shader can read any cached buffer, so a pass synchronizes every one of them, and one
	// is requested before every DMA draw and dispatch. After a complete pass, another can only
	// find something to upload once a page turns CPU-dirty, hot pages come due, a buffer is added
	// or removed, or the GPU mappings change; each of those advances the epoch. The epoch is read
	// before the pass, so a change made while the pass runs makes the next request repeat it.
	const auto epoch =
	    m_buffer_cache.SynchronizationEpoch() + m_mapping_epoch.load(std::memory_order_acquire);
	if (m_bda_epoch == epoch) {
		PerfStats::Add(PerfStats::CounterId::BdaPassesSkipped);
	} else {
		std::shared_lock lock(m_mapped_ranges_mutex);
		m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
			m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
		});
		m_bda_epoch = epoch;
	}
	m_fault_process_pending = true;
}

void GpuResourceManager::RunGarbageCollector() {
	PerfStats::Span span(PerfStats::SpanId::GarbageCollect);
	RegionManager::AdvanceGeneration();
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

} // namespace Libs::Graphics
