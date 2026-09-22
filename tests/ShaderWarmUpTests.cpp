#include "graphics/host_gpu/renderer/pipeline/shaderWarmUp.h"

#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::ShaderWarmUp;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ShaderWarmUpTests: failed: %s\n", message);
		std::abort();
	}
}

void TestOffUnlessStarted() {
	ShaderWarmUp warm_up;
	Check(!warm_up.Active(), "warm-up was active before it was started");
	warm_up.Update(1, 0);
	warm_up.Update(2, 5);
	Check(!warm_up.Active(), "an unstarted warm-up turned itself on");
}

void TestQuietFramesBeforeTheFirstBuildDoNotEndIt() {
	ShaderWarmUp warm_up;
	warm_up.Start();
	// A game shows its loading screen for a while before it asks for anything new.
	for (uint64_t frame = 1; frame <= 120; frame++) {
		warm_up.Update(frame, 0);
	}
	Check(warm_up.Active(), "warm-up ended before a single shader was built");
}

void TestItEndsAtTheFirstQuietFrameAfterABuild() {
	ShaderWarmUp warm_up;
	warm_up.Start();
	warm_up.Update(1, 0);
	warm_up.Update(2, 40); // the first frame that had to build
	Check(warm_up.Active(), "warm-up ended on the frame that built");
	warm_up.Update(3, 73); // still building
	Check(warm_up.Active(), "warm-up ended while shaders were still being built");
	warm_up.Update(4, 73); // nothing new
	Check(!warm_up.Active(), "warm-up did not end after a frame that needed nothing");
	Check(warm_up.Builds() == 73, "warm-up did not keep what it had counted");
}

void TestItIgnoresLookupsInsideTheSameFrame() {
	ShaderWarmUp warm_up;
	warm_up.Start();
	warm_up.Update(1, 10);
	// Thousands of draws per frame each ask; only the frame boundary decides.
	for (int i = 0; i < 1000; i++) {
		warm_up.Update(1, 10);
	}
	Check(warm_up.Active(), "warm-up ended without the frame moving on");
	warm_up.Update(2, 10);
	Check(!warm_up.Active(), "warm-up did not end at the next frame");
}

void TestItStaysOffOnceItEnds() {
	ShaderWarmUp warm_up;
	warm_up.Start();
	warm_up.Update(1, 4);
	warm_up.Update(2, 4);
	Check(!warm_up.Active(), "warm-up did not end");
	// A later area builds more shaders; those draws are dropped, which is the point.
	warm_up.Update(3, 90);
	warm_up.Update(4, 90);
	Check(!warm_up.Active(), "warm-up came back after it had ended");
}

} // namespace

int main() {
	TestOffUnlessStarted();
	TestQuietFramesBeforeTheFirstBuildDoNotEndIt();
	TestItEndsAtTheFirstQuietFrameAfterABuild();
	TestItIgnoresLookupsInsideTheSameFrame();
	TestItStaysOffOnceItEnds();
	std::printf("ShaderWarmUpTests: ok\n");
	return 0;
}
