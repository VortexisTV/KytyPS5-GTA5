#ifndef EMULATOR_INCLUDE_EMULATOR_AUDIO_INTERNAL_H_
#define EMULATOR_INCLUDE_EMULATOR_AUDIO_INTERNAL_H_

#include "common/common.h"

namespace Libs::Audio::AudioInternal {

enum class Format {
	Unknown,
	Signed16bitMono,
	Signed16bitStereo,
	Signed16bit8Ch,
	FloatMono,
	FloatStereo,
	Float8Ch,
	Signed16bit8ChStd,
	Float8ChStd,
};

struct OutputParam {
	int         handle = 0;
	const void* data   = nullptr;
};

static constexpr int OUT_PORTS_MAX = 32;

// Audio kept queued on a host device ahead of playback, so that a game thread that runs late does
// not leave the device without samples.
static constexpr uint64_t OUTPUT_LATENCY_TARGET_US = 40000;

// Bytes a blocking output lets its host queue drain to before adding a grain of grain_bytes that
// plays for grain_us: whole grains covering at least OUTPUT_LATENCY_TARGET_US, 2 to 16 of them.
constexpr uint32_t OutputQueueTargetBytes(uint32_t grain_bytes, uint64_t grain_us) {
	const uint64_t grains =
	    grain_us != 0 ? (OUTPUT_LATENCY_TARGET_US + grain_us - 1) / grain_us : 2;
	return grain_bytes * static_cast<uint32_t>(grains < 2 ? 2 : (grains > 16 ? 16 : grains));
}

// Silence to queue ahead of a grain that reaches a host queue holding queued_bytes. An empty queue
// means the device is starting or has run dry; the lead-in refills it to the target, so a stream
// that the game feeds at exactly the playback rate still has a cushion.
constexpr uint32_t OutputLeadInBytes(uint32_t queued_bytes, uint32_t grain_bytes,
                                     uint64_t grain_us) {
	return queued_bytes != 0 ? 0 : OutputQueueTargetBytes(grain_bytes, grain_us) - grain_bytes;
}

int      AudioOutOpen(int type, uint32_t samples_num, uint32_t freq, Format format);
void     AudioOutClose(int handle);
bool     AudioOutHasDevice(int handle);
uint32_t AudioOutOutputs(const OutputParam* params, uint32_t num, bool blocking = true);
// Audio the port's host device has yet to play, in microseconds; zero without a device.
uint64_t AudioOutQueuedMicros(int handle);

} // namespace Libs::Audio::AudioInternal

#endif // EMULATOR_INCLUDE_EMULATOR_AUDIO_INTERNAL_H_
