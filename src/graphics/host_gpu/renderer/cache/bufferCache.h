#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <atomic>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = uint64_t {1} << (40 - CACHING_PAGEBITS);
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	// Counts GPU writes through the cache that overlap the range, so a caller can tell whether
	// GPU-owned bytes changed without reading them back. The first call starts watching the range;
	// only writes after it are counted.
	[[nodiscard]] uint64_t GpuWriteCount(uint64_t vaddr, uint64_t size);
	// Keeps CPU writes to the range faulting: a hot page stays writable, and the texture cache
	// learns about CPU writes to an image only from faults.
	void ClearHotPages(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.ClearHotPagesIfAny(vaddr, size);
	}

	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	// Stores bytes the GPU has not written into a page it owns without taking the page back.
	// Returns false when the store has to go through guest memory the usual way.
	[[nodiscard]] bool TryWriteBesideGpu(uint64_t vaddr, std::span<const uint8_t> data);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Advances whenever SynchronizeBuffersInRange could upload something over ranges it has
	// already synchronized: new work in the memory tracker, or a buffer registered or removed.
	[[nodiscard]] uint64_t SynchronizationEpoch() const noexcept {
		// Both counters only grow, so their sum changes whenever either does.
		return m_memory_tracker.UploadEpoch() + m_layout_epoch.load(std::memory_order_acquire);
	}
	void               RunGarbageCollector();
	// Records host-visible shadows of hot readback buffers written since the last call. Call
	// before every submit so a CPU read never has to drain the GPU for data already produced.
	void RecordHotShadows();
	// Whether a draw or dispatch just wrote a buffer that should be shadowed and submitted now,
	// rather than at the next flush.
	[[nodiscard]] bool EagerShadowPending() const noexcept { return m_eager_shadow_pending; }

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 40, 16>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Queues backing publication; callers wait before clearing dirty pages or reusing their data.
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Records copies of `buffer` into the download ring; returns where the ring maps them.
	[[nodiscard]] std::pair<uint8_t*, uint64_t> RecordDownload(Buffer&                     buffer,
	                                                           std::vector<vk::BufferCopy>& copies,
	                                                           uint64_t total_size);
	void WriteBackShadow(BufferId id, uint64_t tick);
	void WriteHostMemory(uint64_t vaddr, std::span<const uint8_t> data);
	// A readback the GPU thread has submitted and a faulting guest thread waits for; zero id when
	// the readback completed on the spot.
	struct ReadbackTicket {
		uint64_t id   = 0;
		uint64_t tick = 0;
	};
	struct PendingReadback {
		struct Part {
			uint64_t address = 0;
			uint64_t offset  = 0; // in m_download_buffer
			uint64_t size    = 0;
		};
		uint64_t          id          = 0;
		uint64_t          tick        = 0; // the submission that produces the bytes
		uint64_t          after_tick  = 0; // CurrentTick() once it was submitted
		uint64_t          fault_vaddr = 0; // for a synchronous retry
		uint64_t          fault_size  = 0;
		uint64_t          vaddr       = 0; // the range whose ownership passes to the CPU
		uint64_t          size        = 0;
		bool              is_write    = false;
		bool              downloaded  = false; // copied now rather than served by a shadow
		bool              valid       = true;  // no GPU write to the range since the copy
		std::vector<Part> parts;
	};
	ReadbackTicket ReadMemoryOnGpu(uint64_t vaddr, uint64_t size, bool is_write,
	                               bool allow_async = false);
	void           FinishReadback(uint64_t id);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	std::vector<BufferId>                             m_hot_written;
	bool                                              m_eager_shadow_pending = false;
	std::vector<PendingReadback>                      m_pending_readbacks;
	uint64_t                                          m_next_readback_id = 1;
	struct WriteWatch {
		uint64_t size   = 0;
		uint64_t writes = 0;
	};
	std::map<uint64_t, WriteWatch>                    m_write_watches;
	uint64_t                                          m_write_watch_span = 0; // largest watch
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	std::atomic_uint64_t m_layout_epoch {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
