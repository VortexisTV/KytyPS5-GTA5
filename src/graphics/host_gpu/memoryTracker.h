#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// One locked pass answering "CPU-dirty somewhere and GPU-dirty nowhere", the fast-path test
	// ObtainBuffer makes for every small read-only binding.
	[[nodiscard]] bool IsRegionCpuModifiedAndGpuClean(uint64_t vaddr, uint64_t size) {
		CheckNotInUploadCallback();
		bool cpu_any = false;
		bool gpu_any = false;
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
				gpu_any = true;
				return true;
			}
			cpu_any |= manager->IsModified<DirtySource::Cpu>(offset, bytes);
			return false;
		});
		return !gpu_any && cpu_any;
	}
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
	// Read-only candidate walk for BDA synchronization. Uncreated regions are CPU-dirty by
	// definition. Release each region lock before invoking the callback, which may upload buffers
	// and reenter the tracker. Changes after a snapshot are covered by the upload epoch.
	template <typename Func>
	void ForEachUploadCandidateRange(uint64_t vaddr, uint64_t size, Func&& func) {
		CheckNotInUploadCallback();
		ValidateRange(vaddr, size);
		const auto end = vaddr + size;
		while (vaddr < end) {
			const auto base   = vaddr & ~(TRACKER_REGION_SIZE - 1);
			const auto finish = std::min(end, base + TRACKER_REGION_SIZE);
			auto* manager = m_regions[base / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
			if (manager == nullptr) {
				func(vaddr, finish - vaddr);
			} else {
				const auto candidates = [&] {
					std::scoped_lock lock(manager->lock);
					return manager->UploadCandidates(vaddr, finish - vaddr);
				}();
				for (const auto [first, last]: candidates) {
					const auto start = std::max(vaddr, base + first * TRACKER_PAGE_SIZE);
					const auto stop  = std::min(finish, base + last * TRACKER_PAGE_SIZE);
					func(start, stop - start);
				}
			}
			vaddr = finish;
		}
	}
	// Advances whenever uploading a range again could copy something the previous upload of it
	// did not: a page turning CPU-dirty, hot-page state being reset, a new region (regions start
	// CPU-dirty) or a new submission generation (hot pages upload once per generation). Each
	// change is published after the state it describes, so an epoch read before an upload
	// accounts for everything that upload can see.
	[[nodiscard]] uint64_t UploadEpoch() const noexcept {
		// Both counters only grow, so their sum changes whenever either does.
		return m_upload_epoch.load(std::memory_order_acquire) + RegionManager::Generation();
	}
	// Drops hot-page state for a range that now belongs to a new host buffer.
	void ClearHotPages(uint64_t vaddr, uint64_t size) {
		Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			manager->ClearHot(manager->GetCpuAddr() + offset, bytes);
		});
		// A CPU-dirty page that stops being hot uploads again.
		AdvanceUploadEpoch();
	}
	// As ClearHotPages, for a range that must keep faulting on every CPU write; the upload epoch
	// moves only when a page actually stops being hot.
	void ClearHotPagesIfAny(uint64_t vaddr, uint64_t size) {
		bool cleared = false;
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			cleared |= manager->ClearHot(manager->GetCpuAddr() + offset, bytes);
		});
		if (cleared) {
			AdvanceUploadEpoch();
		}
	}
	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();

		bool newly_dirty = false;
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				newly_dirty |= !manager->IsFullyModified<DirtySource::Cpu>(offset, bytes);
				manager->ChangeState<DirtySource::Cpu, true, true>(manager->GetCpuAddr() + offset,
				                                                   bytes);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
		// Re-marking pages that are already CPU-dirty gives an upload nothing new to copy.
		if (newly_dirty) {
			AdvanceUploadEpoch();
		}
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			const auto       address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes, func);
			if constexpr (clear) {
				manager->template ChangeState<DirtySource::Gpu, false>(address, bytes);
			}
		});
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		Iterate<true>(vaddr, size, [](RegionManager*, uint64_t, uint64_t) {});
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			manager->lock.lock();
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(
			    manager->GetCpuAddr() + offset, bytes, range_func,
			    is_written ? RegionManager::HotPolicy::ClearAndUnhot
			               : RegionManager::HotPolicy::Preserve);
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		if (is_written) {
			Iterate<false>(vaddr, size,
			               [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               manager->template ChangeState<DirtySource::Gpu, true>(
				                   manager->GetCpuAddr() + offset, bytes);
				               manager->lock.unlock();
			               });
		}
		s_upload_owner = previous_upload_owner;
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;

	void CheckNotInUploadCallback() const noexcept {
		if (s_upload_owner == this) {
			EXIT("memory tracker re-entered from upload callback\n");
		}
	}

	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = vaddr / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);
	void           AdvanceUploadEpoch() noexcept {
		m_upload_epoch.fetch_add(1, std::memory_order_release);
	}

	std::atomic_uint64_t                           m_upload_epoch {0};
	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
