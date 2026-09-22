#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERWARMUP_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERWARMUP_H_

#include <cstdint>

namespace Libs::Graphics {

// Asynchronous compilation drops a draw whose shader or pipeline is not ready, which leaves a
// game's first frames missing whatever it is still building. Warm-up builds everything on the
// spot instead, and stops once the game has shown a frame that needed nothing new: from there on
// the run is asynchronous, so a later scene costs a dropped draw rather than a stall.
class ShaderWarmUp {
public:
	[[nodiscard]] bool Active() const noexcept { return m_active; }

	void Start() noexcept { m_active = true; }

	// `frames` counts the frames the guest has shown and `builds` everything compiled here so
	// far. Warm-up ends on the first frame boundary where `builds` did not move, but never
	// before something has been built: the frames before that show a loading screen the game
	// already has every shader for.
	void Update(uint64_t frames, uint64_t builds) noexcept {
		if (!m_active || frames == m_frames) {
			return;
		}
		if (builds != 0 && builds == m_builds) {
			m_active = false;
		}
		m_frames = frames;
		m_builds = builds;
	}

	[[nodiscard]] uint64_t Frames() const noexcept { return m_frames; }
	[[nodiscard]] uint64_t Builds() const noexcept { return m_builds; }

private:
	bool     m_active = false;
	uint64_t m_frames = 0;
	uint64_t m_builds = 0;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERWARMUP_H_ */
