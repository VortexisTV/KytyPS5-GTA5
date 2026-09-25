#include "libs/audio.h"
#include "libs/audio_internal.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace {

namespace AudioOut2 = Libs::Audio::AudioOut2;

std::mutex              g_device_mutex;
std::condition_variable g_device_cv;
std::vector<int>        g_live_devices;
std::vector<int>        g_device_backed_handles;
std::vector<bool>       g_output_blocking;
int                     g_next_device  = 1;
int                     g_open_waiters = 0;
bool                    g_block_opens  = false;

// Modelled host device queue: every output adds g_output_micros of audio to it.
uint64_t g_device_queue_micros = 0;
uint64_t g_output_micros       = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "AudioOut2PortTests: failed: %s\n", text);
		std::abort();
	}
}

struct PortParam {
	uint16_t port_type;
	uint16_t pad;
	uint32_t data_format;
	uint32_t sampling_freq;
	uint32_t flags;
	uint64_t user_handle;
	uint32_t reserved[10];
};

struct ContextParam {
	uint32_t max_ports;
	uint32_t max_object_ports;
	uint32_t guarantee_object_ports;
	uint32_t queue_depth;
	uint32_t num_grains;
	uint32_t flags;
	uint32_t reserved[10];
};

struct PortState {
	uint16_t output;
	uint8_t  num_channels;
	uint8_t  pad1;
	int16_t  volume;
	uint16_t reroute_counter;
	uint32_t flags;
	uint32_t pad2;
	uint64_t reserved[6];
};

struct Attribute {
	uint32_t    attribute_id;
	int32_t     reserved;
	const void* value;
	size_t      value_size;
};

struct Pcm {
	const void* data;
};

const auto* AsParam(const PortParam* param) {
	return reinterpret_cast<const AudioOut2::AudioOut2PortParam*>(param);
}

const auto* AsParam(const ContextParam* param) {
	return reinterpret_cast<const AudioOut2::AudioOut2ContextParam*>(param);
}

auto* AsState(PortState* state) {
	return reinterpret_cast<AudioOut2::AudioOut2PortState*>(state);
}

const auto* AsAttribute(const Attribute* attribute) {
	return reinterpret_cast<const AudioOut2::AudioOut2Attribute*>(attribute);
}

PortParam MakeParam(uint32_t data_format = 0x200) {
	PortParam param {};
	param.data_format   = data_format;
	param.sampling_freq = 48000;
	return param;
}

AudioOut2::AudioOut2ContextHandle CreateContext(uint32_t queue_depth = 4) {
	ContextParam param {};
	param.queue_depth                         = queue_depth;
	param.num_grains                          = 512;
	AudioOut2::AudioOut2ContextHandle context = 0;
	Check(AudioOut2::AudioOut2ContextCreate(AsParam(&param), nullptr, 0, &context) == OK,
	      "context create failed");
	return context;
}

void BlockDeviceOpens() {
	std::lock_guard lock(g_device_mutex);
	g_open_waiters = 0;
	g_block_opens  = true;
}

void WaitForDeviceOpens(int count) {
	std::unique_lock lock(g_device_mutex);
	g_device_cv.wait(lock, [count]() { return g_open_waiters >= count; });
}

void ReleaseDeviceOpens() {
	std::lock_guard lock(g_device_mutex);
	g_block_opens = false;
	g_device_cv.notify_all();
}

int LiveDeviceCount() {
	std::lock_guard lock(g_device_mutex);
	return static_cast<int>(g_live_devices.size());
}

void SetPcm(AudioOut2::AudioOut2PortHandle port, const void* data) {
	const Pcm       pcm {data};
	const Attribute attribute {0, 0, &pcm, sizeof(pcm)};
	Check(AudioOut2::AudioOut2PortSetAttributes(port, AsAttribute(&attribute), 1) == OK,
	      "setting PCM failed");
}

void ResetOutputCalls() {
	std::lock_guard lock(g_device_mutex);
	g_output_blocking.clear();
}

std::vector<bool> OutputCalls() {
	std::lock_guard lock(g_device_mutex);
	return g_output_blocking;
}

void SetDeviceQueue(uint64_t queued_micros, uint64_t output_micros) {
	std::lock_guard lock(g_device_mutex);
	g_device_queue_micros = queued_micros;
	g_output_micros       = output_micros;
}

void TestSlotReuse() {
	const auto context = CreateContext();
	const auto param   = MakeParam();
	for (int i = 0; i < 300; i++) {
		AudioOut2::AudioOut2PortHandle port = 0;
		Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
		      "port slot was not reusable");
		Check(port != 0, "port handle is zero");
		AudioOut2::AudioOut2PortDestroy(port);
	}
	AudioOut2::AudioOut2ContextDestroy(context);
}

void TestFullTableRecovers() {
	const auto                                  context = CreateContext();
	const auto                                  param   = MakeParam();
	std::vector<AudioOut2::AudioOut2PortHandle> ports;
	ports.reserve(256);

	for (int i = 0; i < 256; i++) {
		AudioOut2::AudioOut2PortHandle port = 0;
		Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
		      "port table filled early");
		ports.push_back(port);
	}

	AudioOut2::AudioOut2PortHandle overflow = 0;
	Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &overflow) != OK,
	      "full port table accepted another port");

	for (auto port: ports) {
		AudioOut2::AudioOut2PortDestroy(port);
	}

	AudioOut2::AudioOut2PortHandle port = 0;
	Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
	      "port table did not recover");
	AudioOut2::AudioOut2PortDestroy(port);
	AudioOut2::AudioOut2ContextDestroy(context);
}

void TestConcurrentCreates() {
	constexpr int                               thread_count = 8;
	const auto                                  context      = CreateContext();
	const auto                                  param        = MakeParam(0x800);
	std::vector<AudioOut2::AudioOut2PortHandle> ports(thread_count);
	std::vector<int>                            results(thread_count);
	std::vector<std::thread>                    threads;

	BlockDeviceOpens();
	for (int i = 0; i < thread_count; i++) {
		threads.emplace_back([&, i]() {
			results[i] = AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &ports[i]);
		});
	}
	WaitForDeviceOpens(thread_count);
	ReleaseDeviceOpens();
	for (auto& thread: threads) {
		thread.join();
	}

	for (int i = 0; i < thread_count; i++) {
		Check(results[i] == OK, "concurrent port create failed");
		PortState state {};
		AudioOut2::AudioOut2PortGetState(ports[i], AsState(&state));
		Check(state.num_channels == 8, "concurrent create lost its reserved slot");
		AudioOut2::AudioOut2PortDestroy(ports[i]);
	}
	Check(LiveDeviceCount() == 0, "concurrent create leaked a device");
	AudioOut2::AudioOut2ContextDestroy(context);
}

void TestContextDestroyCancelsPendingCreate() {
	const auto                     context = CreateContext();
	const auto                     param   = MakeParam();
	AudioOut2::AudioOut2PortHandle port    = 0;
	int                            result  = OK;

	BlockDeviceOpens();
	std::thread creator(
	    [&]() { result = AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port); });
	WaitForDeviceOpens(1);
	AudioOut2::AudioOut2ContextDestroy(context);
	ReleaseDeviceOpens();
	creator.join();

	Check(result != OK, "destroyed context retained a pending port create");
	Check(LiveDeviceCount() == 0, "cancelled port create leaked a device");
}

void TestSynchronousDevicePushBypassesModelledQueue() {
	const auto context = CreateContext(1);
	const auto param   = MakeParam();
	AudioOut2::AudioOut2PortHandle port = 0;
	Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
	      "device port create failed");

	uint32_t pcm[512] {};
	SetPcm(port, pcm);
	ResetOutputCalls();

	Check(AudioOut2::AudioOut2ContextPush(context, 1) == OK, "first sync push failed");
	Check(AudioOut2::AudioOut2ContextPush(context, 1) == OK,
	      "device-paced sync push was blocked by modelled queue");
	const auto calls = OutputCalls();
	Check(calls.size() == 2, "sync pushes did not reach the device backend");
	Check(calls[0] && calls[1], "sync pushes lost their blocking mode");

	AudioOut2::AudioOut2PortDestroy(port);
	AudioOut2::AudioOut2ContextDestroy(context);
}

void TestFloat12ChannelPortOutputsPcm() {
	const auto context = CreateContext();
	const auto param   = MakeParam(0x0c00);
	AudioOut2::AudioOut2PortHandle port = 0;
	Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
	      "12-channel port create failed");
	Check(LiveDeviceCount() == 1, "12-channel port did not open an audio device");

	PortState state {};
	Check(AudioOut2::AudioOut2PortGetState(port, AsState(&state)) == OK,
	      "12-channel port state query failed");
	Check(state.num_channels == 12, "12-channel port lost its guest channel count");

	float pcm[512 * 12] {};
	SetPcm(port, pcm);
	ResetOutputCalls();
	Check(AudioOut2::AudioOut2ContextPush(context, 1) == OK, "12-channel PCM push failed");
	const auto calls = OutputCalls();
	Check(calls.size() == 1 && calls[0], "12-channel PCM did not reach the device backend");

	AudioOut2::AudioOut2PortDestroy(port);
	Check(LiveDeviceCount() == 0, "12-channel port leaked its audio device");
	AudioOut2::AudioOut2ContextDestroy(context);
}

void TestAsynchronousDevicePushFollowsDeviceQueue() {
	const auto context = CreateContext(1);
	const auto param   = MakeParam();
	AudioOut2::AudioOut2PortHandle port = 0;
	Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
	      "device port create failed");

	uint32_t pcm[512] {};
	SetPcm(port, pcm);
	ResetOutputCalls();
	// 512-sample grains last 10666 us; the device keeps 40000 us queued before refusing pushes.
	SetDeviceQueue(0, 10666);

	uint32_t accepted = 0;
	while (AudioOut2::AudioOut2ContextPush(context, 0) == OK) {
		accepted++;
		Check(accepted <= 16, "async pushes to a device were never refused");
	}
	Check(accepted == 4, "async pushes did not fill the device up to its latency cushion");
	const auto calls = OutputCalls();
	Check(calls.size() == 4 && std::ranges::none_of(calls, [](bool blocking) { return blocking; }),
	      "async pushes did not reach the device backend without blocking");

	uint32_t queued    = 0;
	uint32_t available = 0;
	Check(AudioOut2::AudioOut2ContextGetQueueLevel(context, &queued, &available) == OK,
	      "queue-level query failed");
	Check(queued == 1 && available == 0,
	      "queue level does not report the audio beyond the device cushion");

	SetDeviceQueue(30000, 10666);
	Check(AudioOut2::AudioOut2ContextPush(context, 0) == OK,
	      "async push was refused after the device drained");

	SetDeviceQueue(0, 0);
	AudioOut2::AudioOut2PortDestroy(port);
	AudioOut2::AudioOut2ContextDestroy(context);
}

void TestHandleWithoutPcmDoesNotBypassQueue() {
	const auto context = CreateContext(1);
	const auto param   = MakeParam();
	AudioOut2::AudioOut2PortHandle port = 0;
	Check(AudioOut2::AudioOut2PortCreate(context, AsParam(&param), &port) == OK,
	      "device port create failed");
	ResetOutputCalls();

	Check(AudioOut2::AudioOut2ContextPush(context, 1) == OK, "empty sync push failed");
	Check(AudioOut2::AudioOut2ContextPush(context, 0) != OK,
	      "handle without PCM bypassed queue backpressure");
	Check(OutputCalls().empty(), "empty push reached the device backend");

	AudioOut2::AudioOut2PortDestroy(port);
	AudioOut2::AudioOut2ContextDestroy(context);
}

void TestOutputQueueTarget() {
	using Libs::Audio::AudioInternal::OutputQueueTargetBytes;
	// 256-frame grains at 48 kHz last 5333 us: 8 of them cover the 40 ms target.
	Check(OutputQueueTargetBytes(2048, 5333) == 8 * 2048, "short grains miss the latency target");
	Check(OutputQueueTargetBytes(8192, 21333) == 2 * 8192, "long grains miss the latency target");
	Check(OutputQueueTargetBytes(4096, 50000) == 2 * 4096, "fewer than two grains are kept");
	Check(OutputQueueTargetBytes(64, 1000) == 16 * 64, "tiny grains are not capped at 16");
	Check(OutputQueueTargetBytes(512, 0) == 2 * 512, "unknown grain length is not two grains");
}

// A pseudo-random delay for the next grain: 0 to 239 frames, up to 5 ms at 48 kHz.
uint32_t NextLateness(uint32_t* state) {
	*state = *state * 1103515245u + 12345u;
	return (*state >> 16u) % 240u;
}

// A device that plays 480-frame periods (10 ms at 48 kHz, the Windows shared-mode period), fed
// 256-frame grains at exactly its playback rate, each up to 5 ms late, so the game never runs
// ahead of it. Returns the periods that found fewer frames queued than they play.
uint32_t CountStarvedPeriods(bool lead_in, uint32_t period_phase) {
	constexpr uint32_t GRAIN  = 256;
	constexpr uint32_t PERIOD = 480;

	uint32_t random    = 12345;
	uint32_t queued    = 0;
	uint32_t starved   = 0;
	uint32_t grains    = 0;
	uint32_t next_push = NextLateness(&random);
	for (uint32_t frame = 0; frame < 4 * 48000; frame++) {
		while (frame >= next_push) {
			if (lead_in) {
				queued += Libs::Audio::AudioInternal::OutputLeadInBytes(queued, GRAIN, 5333);
			}
			queued += GRAIN;
			next_push = ++grains * GRAIN + NextLateness(&random);
		}
		if (grains != 0 && frame % PERIOD == period_phase) {
			if (queued < PERIOD) {
				starved++;
			}
			queued -= std::min(queued, PERIOD);
		}
	}
	return starved;
}

void TestLeadInKeepsRealTimeStreamFed() {
	uint32_t starved_without = 0;
	for (uint32_t phase = 0; phase < 480; phase += 16) {
		starved_without += CountStarvedPeriods(false, phase);
		Check(CountStarvedPeriods(true, phase) == 0, "a stream fed at its playback rate ran dry");
	}
	Check(starved_without > 0, "the model no longer starves a stream that has no lead-in");
}

} // namespace

namespace Libs::Audio::AudioInternal {

int AudioOutOpen(int type, uint32_t /*samples_num*/, uint32_t /*freq*/, Format /*format*/) {
	std::unique_lock lock(g_device_mutex);
	const int        handle = g_next_device++;
	g_live_devices.push_back(handle);
	if (type != 10) {
		g_device_backed_handles.push_back(handle);
	}
	g_open_waiters++;
	g_device_cv.notify_all();
	g_device_cv.wait(lock, []() { return !g_block_opens; });
	return handle;
}

void AudioOutClose(int handle) {
	std::lock_guard lock(g_device_mutex);
	const auto      it = std::find(g_live_devices.begin(), g_live_devices.end(), handle);
	if (it != g_live_devices.end()) {
		g_live_devices.erase(it);
	}
	const auto device_it =
	    std::find(g_device_backed_handles.begin(), g_device_backed_handles.end(), handle);
	if (device_it != g_device_backed_handles.end()) {
		g_device_backed_handles.erase(device_it);
	}
}

bool AudioOutHasDevice(int handle) {
	std::lock_guard lock(g_device_mutex);
	return std::find(g_device_backed_handles.begin(), g_device_backed_handles.end(), handle) !=
	       g_device_backed_handles.end();
}

uint32_t AudioOutOutputs(const OutputParam* /*params*/, uint32_t /*num*/, bool blocking) {
	std::lock_guard lock(g_device_mutex);
	g_output_blocking.push_back(blocking);
	g_device_queue_micros += g_output_micros;
	return 0;
}

uint64_t AudioOutQueuedMicros(int /*handle*/) {
	std::lock_guard lock(g_device_mutex);
	return g_device_queue_micros;
}

} // namespace Libs::Audio::AudioInternal

namespace Libs::LibKernel {

uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
	static std::atomic_uint64_t now {0};
	return now.fetch_add(1000);
}

} // namespace Libs::LibKernel

int main() {
	TestSlotReuse();
	TestFullTableRecovers();
	TestConcurrentCreates();
	TestContextDestroyCancelsPendingCreate();
	TestSynchronousDevicePushBypassesModelledQueue();
	TestFloat12ChannelPortOutputsPcm();
	TestAsynchronousDevicePushFollowsDeviceQueue();
	TestHandleWithoutPcmDoesNotBypassQueue();
	TestOutputQueueTarget();
	TestLeadInKeepsRealTimeStreamFed();
	std::printf("AudioOut2PortTests: all cases passed\n");
	return 0;
}
