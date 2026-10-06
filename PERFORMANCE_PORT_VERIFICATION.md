# GTA V performance port and shadow verification

Verified on 2026-10-06 using GTA V PPSA04264, version 01.005.000, from
`H:\Games\PPSA04264 [ 01.005.000 ]`.

## Source comparison

- Reference checkout commit:
  `8174f23cc4e06cef65c80a3732b65c023470c6d9`.
- Destination: `H:\Thomas-KytyPS5\KytyPS5-GTA5`, based on commit
  `724d929f7e081d5ce3e9385244431356264dd236`, with the fixes described below.
- Compared all 443 reference files under `src`, normalizing CRLF/LF. No reference
  source files are missing. 344 are identical; 99 differ in the destination.
- All 21 selected core performance files match the reference exactly after newline
  normalization. These cover draw speculation, command hooks, pipeline libraries,
  shader precompilation, memory tracking, region management, SRT walking, DCC clear
  resolution, stream buffers, samplers, indirect mesh arguments and presentation.
- The command scheduler retains the reference performance logic, with destination
  GPU failure diagnostics added. Destination GTA V compatibility changes remain.

The detailed file list and counts are saved in
`_Build/investigation/performance-file-comparison.json` and
`_Build/investigation/source-tree-audit.json`.

| Performance option | Reference default | Verified destination default |
| --- | --- | --- |
| Pipeline libraries | Enabled | Enabled |
| Draw speculation | Enabled | Enabled |
| Command recording thread | Enabled | Enabled |
| Asynchronous submission | Enabled | Enabled |
| GPU DCC clear | Enabled | Enabled |
| Hardware buffer bounds | Enabled | Enabled |
| Relaxed GPU readback | Disabled | Disabled |
| Asynchronous pipeline draw skipping | Disabled | Disabled (restored) |

## Shadow fix

The destination had changed asynchronous pipeline draw skipping to enabled by
default. A shadow map can be cleared and then miss its geometry while a shader or
pipeline is still compiling. This is consistent with the disappearing shadows in
the supplied destination recording.

The destination now restores the reference's disabled default. Depth-only draws
also wait for both shader translation and Vulkan pipeline library compilation when
asynchronous pipelines are explicitly enabled. Pipeline libraries and the other
ported performance features remain enabled.

Added `graphics_shadow_pipelines`, which uses real Vulkan graphics pipeline
libraries and background workers. It clears depth before each of three cold vertex
shader variants, enables asynchronous pipelines, submits through the production
draw path and checks that the depth output is present immediately. This passed on
the RTX 5070 Ti. Systems without fast pipeline library support skip this test.

## F2 overlay removal

Removed the emulator settings panel, its F2 toggle, drawing, navigation and settings
session state. F2 is no longer reserved against custom game input mappings. Game
IME and error dialogs remain. Manual INI and command-line options remain available.

Pressed F2 during live GTA V gameplay and confirmed that no settings panel appeared.
Evidence: `_Build/investigation/gta-runtime/f2-check.jpg`.

## Build and validation

- Release emulator and test targets built successfully with clang-cl and Ninja.
- 45 selected CTest checks passed: 22 performance/memory/command checks, nine CLI
  and shader checks, 13 graphics/texture checks and the new shadow pipeline check.
- The graphics checks include rasterization, HTile clears and depth readback.
- `git diff --check` passed.
- Fixed the ZArchive FetchContent patch step to accept an already-applied patch,
  which had prevented repeat CMake configuration. Verified fresh and repeated
  application separately.

Logs are in `_Build/investigation/{build-final,performance-tests,cli-shader-tests,
graphics-tests,shadow-test}.log`.

## Live GTA V result

Launched the rebuilt executable with 1920x1080 output, Immediate presentation and
two-second drain statistics. The test used separate copies of the destination
saves under `_Build/investigation/gta-runtime`; the original install saves were not
changed by this run. The old pipeline cache was invalidated by the new build and
rebuilt normally.

GTA V loaded the beach save successfully. Character and scene shadows remained
visible in the observed running, camera-turning and stationary views. Saved
screenshots: `_Build/investigation/gta-runtime/shadows-{01,02,03}.jpg`.
Warm samples reported about 60 game fps, with two-second samples at 59.9-60.4
frames/s and 16 ms median / 17 ms 95th percentile in the sampled stationary view.
Earlier loading and first-use shader compilation produced stalls.

This verifies the ported code, GPU regression and the observed beach scene. It is
not a controlled FPS comparison between checkouts or an exhaustive check of all
GTA V locations and graphics modes.

## Updated binary

The verified executable is copied to `_Build/windows/install/kyty_emulator.exe`
for the existing launcher. The previous executable is backed up as
`_Build/investigation/Thomas-before.exe`.

The build title retains `724d929-dirty` because these source fixes are uncommitted.

## Title cleanup

The title's counters now use `frame: <number>, fps: <number>`, with whole-number
game FPS. The presentation-rate counter is omitted from the heading. Removed the
old checkout name from this report. The Release emulator rebuilt successfully
after this change, and the installed copy matches the rebuilt executable.
