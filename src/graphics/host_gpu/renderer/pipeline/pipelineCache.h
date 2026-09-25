#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/pipeline/shaderWarmUp.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

struct GraphicContext;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;
class GraphicsPipelineLibraryCache;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       depth_bounds_test_enable = false;
	float                      depth_min_bounds         = 0.0f;
	float                      depth_max_bounds         = 0.0f;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       cull_front                                         = false;
	bool                       cull_back                                          = false;
	bool                       face                                               = false;
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 125);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	bool operator==(const PipelineVertexInputState&) const = default;
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	// Final save at shutdown: writes the driver cache and releases it.
	void Save();
	// Write the driver cache to disk when pipelines were added since the last write. Safe from
	// any thread while the cache stays in use; the background saver calls it periodically.
	void SaveSnapshot();
	// Best-effort save from a thread that is about to terminate the process on a fault: bounded
	// in time and never blocks on the crashing thread's own state.
	void EmergencySave() noexcept;

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
		// False when the layout is shared from the graphics pipeline library cache.
		bool                    owns_layout           = true;
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info);
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info);

	// Returns null while the pipeline is still being built on a worker thread (async mode); the
	// caller skips the draw and retries on a later frame.
	Pipeline* GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

	// The presenter counts the frames the guest shows, which is what ends warm-up. It is a
	// counter rather than a call on this object because the presenter outlives it.
	static void NoteGuestFrame() noexcept;

private:
	struct ProgramCache;

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}

		static void MixStaticParams(std::size_t& hash, const PipelineStaticParameters& params) {
			// The struct is packed with no padding, so hashing its bytes in one pass is exact;
			// this runs once per draw.
			Mix(hash, static_cast<std::size_t>(XXH3_64bits(&params, sizeof(params))));
		}

		static void MixRendering(std::size_t& hash, const PipelineRenderingState& rendering) {
			Mix(hash, rendering.color_count);
			for (uint32_t i = 0; i < rendering.color_count; i++) {
				Mix(hash, static_cast<uint32_t>(rendering.color_formats[i]));
			}
			Mix(hash, static_cast<uint32_t>(rendering.depth_format));
			Mix(hash, static_cast<uint32_t>(rendering.stencil_format));
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const {
			std::size_t hash = 0;
			PipelineKeyHash::MixRendering(hash, key.rendering);
			for (const auto id: key.vertex_shader_ids) {
				PipelineKeyHash::Mix(hash, id);
			}
			PipelineKeyHash::Mix(hash, key.ps_shader_id);
			PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
			for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
				PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
				PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
			}
			PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
			for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
				PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
				PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
			}
			PipelineKeyHash::MixStaticParams(hash, key.static_params);
			return hash;
		}
	};

	GraphicContext&               m_graphics;
	GraphicsPipelineKey           m_last_graphics_key {};
	Pipeline*                     m_last_graphics_pipeline = nullptr;
	std::unique_ptr<ProgramCache> m_program_cache;
	std::unique_ptr<GraphicsPipelineLibraryCache> m_graphics_library_cache;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	Common::Mutex m_mutex;

	// Asynchronous graphics pipeline construction. Keys being built live in m_pending_pipelines
	// (GPU thread only); finished pipelines wait in m_completed_pipelines until the GPU thread
	// drains them at its next lookup.
	struct CompletedPipeline {
		GraphicsPipelineKey       key;
		std::unique_ptr<Pipeline> pipeline;
	};
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> m_pending_pipelines;
	std::mutex                                                       m_completed_mutex;
	std::vector<CompletedPipeline>                                   m_completed_pipelines;
	std::mutex                                                       m_job_mutex;
	std::condition_variable                                          m_job_available;
	std::deque<std::function<void()>>                                m_jobs;
	std::vector<std::jthread>                                        m_workers;
	bool                                                             m_stop_workers = false;
	bool                                                             m_async        = false;
	ShaderWarmUp                                                     m_warm_up;
	uint64_t                                                         m_sync_pipeline_builds = 0;
	static std::atomic<uint64_t>                                     s_guest_frames;

	// Periodic and emergency saves of the driver cache. m_pipelines_created counts pipelines that
	// were really built (and therefore added to the driver cache); a save is only worth it when
	// it moved past m_pipelines_saved.
	std::mutex              m_save_mutex;
	std::atomic<uint64_t>   m_pipelines_created {0};
	uint64_t                m_pipelines_saved = 0; // guarded by m_save_mutex
	std::jthread            m_saver;
	std::mutex              m_saver_mutex;
	std::condition_variable m_saver_wake;
	bool                    m_stop_saver = false; // guarded by m_saver_mutex

	void InitializeDriverCache();
	bool WriteDriverCache();
	void StartSaver();
	void StopSaver();
	void StartWorkers();
	void StopWorkers();
	void EnqueueJob(std::function<void()> job);
	void DrainCompletedPipelines();
	void UpdateWarmUp();
};

// Builds plain vertex + pixel graphics pipelines from shared VK_EXT_graphics_pipeline_library
// parts, so a new combination links in microseconds instead of compiling from scratch.
class GraphicsPipelineLibraryCache {
public:
	GraphicsPipelineLibraryCache(GraphicContext& graphics, vk::PipelineCache driver_cache);
	~GraphicsPipelineLibraryCache();
	KYTY_CLASS_NO_COPY(GraphicsPipelineLibraryCache);

	void PrepareLayout(PipelineCache::Pipeline& pipeline, uint64_t vs_shader_id,
	                   uint64_t ps_shader_id, vk::ShaderStageFlags push_constant_stages,
	                   std::span<const vk::DescriptorSetLayoutBinding> bindings);
	bool CreatePipeline(PipelineCache::Pipeline& pipeline, uint64_t vs_shader_id,
	                    uint64_t ps_shader_id, const PipelineRenderingState& rendering,
	                    const PipelineStaticParameters&          static_params,
	                    const vk::GraphicsPipelineCreateInfo&    pipeline_info,
	                    uint32_t                                 pre_raster_stage_count,
	                    const vk::PipelineShaderStageCreateInfo* fragment_stage);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache,
                            GraphicsPipelineLibraryCache*          library_cache = nullptr);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
