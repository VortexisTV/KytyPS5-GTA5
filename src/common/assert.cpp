#include "common/assert.h"

#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>

namespace Common {

// The failing thread's return addresses, for llvm-symbolizer --obj=kyty_emulator.exe (the exe
// loads at its preferred base, so they are its virtual addresses).
static std::string CaptureStack() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::array<void*, 48> frames {};
	const auto count = RtlCaptureStackBackTrace(2, static_cast<DWORD>(frames.size()), frames.data(),
	                                            nullptr);
	std::string text = "stack:";
	for (USHORT i = 0; i < count; i++) {
		text += fmt::format(" 0x{:x}", reinterpret_cast<uintptr_t>(frames[i]));
	}
	return text + "\n";
#else
	return {};
#endif
}

static std::atomic<FatalHook> g_fatal_hook {nullptr};

void SetFatalHook(FatalHook hook) {
	g_fatal_hook.store(hook, std::memory_order_release);
}

// A failure inside the hook reports itself but does not run the hook again.
static void RunFatalHook() {
	static std::atomic_flag ran = ATOMIC_FLAG_INIT;
	if (ran.test_and_set(std::memory_order_acq_rel)) {
		return;
	}
	if (const auto hook = g_fatal_hook.load(std::memory_order_acquire); hook != nullptr) {
		hook();
	}
}

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n{}", KYTY_BUILD_LABEL, title, text, file,
	                   line, CaptureStack());
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	Log::WriteFatal(BuildFatalReport(title, text, file, line));
	RunFatalHook();
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	Log::WriteFatal(BuildFatalReport("--- Error ---", text, file, line));
	RunFatalHook();
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	Log::WriteFatal(style, BuildFatalReport("--- Error ---", text, file, line));
	RunFatalHook();
	return 1;
}

void DbgExit(int status) {
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(status);
}

} // namespace Common
