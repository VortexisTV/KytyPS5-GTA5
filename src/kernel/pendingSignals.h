#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_PENDINGSIGNALS_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_PENDINGSIGNALS_H_

#include <atomic>
#include <bit>
#include <cstdint>

namespace Libs::LibKernel {

// Takes the lowest pending signal below `limit` (at most 64) from a thread's pending-signal mask
// and returns its number, or -1 when none is pending.
//
// Guest spin loops poll this on every sceKernelUsleep and every blocking-call retry, and nearly
// always find nothing: then it costs one load. The legacy scan cleared all 64 bits one by one,
// 64 locked operations per call. When bits are set, the result is the same: bits are tried from
// the lowest up, each with an atomic fetch_and, so a signal taken by another thread is skipped.
[[nodiscard]] inline int TakeLowestPendingSignal(std::atomic<uint64_t>& mask, int limit) {
	if (limit <= 0) {
		return -1;
	}
	const uint64_t allowed = limit >= 64 ? ~uint64_t {0} : (uint64_t {1} << limit) - 1u;
	uint64_t       pending = mask.load(std::memory_order_acquire) & allowed;
	while (pending != 0) {
		const int      signum = std::countr_zero(pending);
		const uint64_t bit    = uint64_t {1} << static_cast<uint32_t>(signum);
		if ((mask.fetch_and(~bit, std::memory_order_acq_rel) & bit) != 0) {
			return signum;
		}
		pending &= ~bit;
	}
	return -1;
}

} // namespace Libs::LibKernel

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_PENDINGSIGNALS_H_ */
