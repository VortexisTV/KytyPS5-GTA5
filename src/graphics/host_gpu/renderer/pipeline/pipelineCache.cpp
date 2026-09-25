#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// The driver validates the blob itself (pipelineCacheUUID) and keys its entries by shader and
	// state contents, so the cache stays valid across emulator revisions. Tying it to one revision
	// threw away everything the game had compiled on every update.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n", properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// Guest reads resource materialization made since the last shader lookup reported them to
// PerfStats. Counting locally keeps atomics off the per-read path.
thread_local uint64_t g_materialize_reads = 0;

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	g_materialize_reads++;
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(), values.size_bytes());
}

bool ValidateShaderGuestMemoryRange(void*, uint64_t address, uint64_t size) {
	return Libs::LibKernel::Memory::TryClampRangeSize(address, size) != 0;
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code, const std::string& decoded_dump) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto base = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(base.parent_path());
	for (const auto& [suffix, data, size]: {
	         std::tuple {".bin", static_cast<const void*>(code.data()), code.size_bytes()},
	         std::tuple {".rdna2", static_cast<const void*>(decoded_dump.data()),
	                     decoded_dump.size()},
	     }) {
		if (size == 0) {
			continue;
		}
		auto path = base;
		path += suffix;
		Common::File file(path);
		if (file.IsInvalid()) {
			const auto path_text = Common::PathToString(path);
			LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		} else {
			file.Write(data, size);
		}
	}
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		explicit SourceEntry(
		    ShaderRecompiler::IR::ResourcePlan               plan,
		    std::optional<ShaderRecompiler::TranslateResult> initial_translation = std::nullopt)
		    : resource_plan(std::move(plan)), initial_translation(std::move(initial_translation)) {}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// The first async pass already translated the complete shader to extract resource_plan.
		// Preserve that move-only IR until the first specialization is known instead of decoding
		// and translating the same source again in the compilation continuation.
		std::optional<ShaderRecompiler::TranslateResult> initial_translation;
		// A deque keeps every Permutation at a stable address: draw state and pipeline jobs hold
		// pointers to permutation.program while new permutations are still being appended.
		std::deque<Permutation> permutations;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing up to 429 words first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 13 + ShaderVertexInputInfo::RES_MAX * 13;

	Permutation CompilePermutation(const ShaderParams&                          params,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		const char* stage_name = nullptr;
		switch (options.stage) {
			case ShaderType::Vertex: stage_name = "vs"; break;
			case ShaderType::Mesh: stage_name = "ms"; break;
			case ShaderType::Local: stage_name = "ls"; break;
			case ShaderType::TessellationControl: stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: stage_name = "ds"; break;
			case ShaderType::Pixel: stage_name = "ps"; break;
			case ShaderType::Compute: stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		DumpShaderOriginal(stage_name, options.shader_hash, params.code, result.decoded_dump);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
	}

	template <typename InputInfo>
	static ShaderType StageOf(const InputInfo& input_info) {
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			return input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			return ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			return ShaderType::Compute;
		}
	}

	// The compile options for one stage. `input_info` must outlive the returned options.
	template <typename InputInfo>
	static ShaderRecompiler::CompileOptions MakeOptions(ShaderType stage, const InputInfo& input_info,
	                                                    std::span<const uint32_t> user_data,
	                                                    std::span<const uint32_t> back_code,
	                                                    uint64_t                  hash) {
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = hash;
		options.user_data   = user_data;
		options.back_code   = back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size      = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		return options;
	}

	void PrintCounts() const {
		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
	}

	// Everything a worker needs to translate and compile one shader, copied out of guest memory
	// and draw state so it stays valid after the draw is gone.
	template <typename InputInfo>
	struct AsyncJob {
		ProgramKey                                       key;
		ShaderType                                       stage = ShaderType::Unknown;
		uint64_t                                         hash  = 0;
		std::vector<uint32_t>                            code;
		std::vector<uint32_t>                            back_code;
		std::vector<uint32_t>                            user_data;
		InputInfo                                        input_info {};
		bool                                             has_entry = false;
		std::optional<ShaderRecompiler::TranslateResult> translated;
		ShaderRecompiler::IR::ResourceSpecialization     specialization;
		uint32_t                                         push_data_cursor = 0;
	};

	struct AsyncResult {
		ProgramKey                                        key;
		ShaderRecompiler::IR::ResourceSpecialization      specialization;
		uint32_t                                          push_data_cursor = 0;
		std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
		std::optional<ShaderRecompiler::TranslateResult>  translated;
		std::optional<Permutation>                        permutation;
	};

	// A translation or permutation already handed to a worker.
	struct PendingJob {
		bool                                         has_entry = false;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint32_t                                     push_data_cursor = 0;

		bool operator==(const PendingJob&) const = default;
	};

	template <typename InputInfo>
	void RunAsyncJob(AsyncJob<InputInfo>& job) {
		PerfStats::Span    span(PerfStats::SpanId::ShaderCompileAsync);
		const auto         options = MakeOptions(job.stage, job.input_info, job.user_data,
		                                         job.back_code, job.hash);
		ShaderParams       params {};
		params.code      = job.code;
		params.hash      = job.hash;
		params.back_code = job.back_code;
		auto translated  = job.translated ? std::move(*job.translated)
		                                  : ShaderRecompiler::TranslateProgram(job.code, options);
		// Translation-only jobs are shared by every cursor, so they are recorded without one.
		AsyncResult result {.key              = job.key,
		                    .specialization   = job.specialization,
		                    .push_data_cursor = job.has_entry ? job.push_data_cursor : 0u};
		if (!job.has_entry) {
			result.plan       = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			result.translated = std::move(translated);
		} else {
			result.permutation = CompilePermutation(params, options, std::move(translated),
			                                        std::move(job.specialization),
			                                        job.push_data_cursor);
			PerfStats::Add(PerfStats::CounterId::ShadersCompiled);
		}
		std::lock_guard lock(completed_mutex);
		completed.push_back(std::move(result));
	}

	// GPU thread: adopt finished translations and permutations.
	void DrainCompleted() {
		std::vector<AsyncResult> done;
		{
			std::lock_guard lock(completed_mutex);
			done.swap(completed);
		}
		bool compiled = false;
		for (auto& result: done) {
			auto entry = programs.find(result.key);
			if (entry == programs.end()) {
				if (!result.plan || !result.translated) {
					continue;
				}
				entry = programs
				            .try_emplace(result.key, std::move(*result.plan),
				                         std::move(result.translated))
				            .first;
			}
			const PendingJob job {.has_entry        = result.permutation.has_value(),
			                      .specialization   = result.specialization,
			                      .push_data_cursor = result.push_data_cursor};
			if (result.permutation) {
				entry->second.permutations.push_back(std::move(*result.permutation));
				compiled = true;
			}
			if (const auto pending_it = pending.find(result.key); pending_it != pending.end()) {
				std::erase(pending_it->second, job);
				if (pending_it->second.empty()) {
					pending.erase(pending_it);
				}
			}
		}
		if (compiled) {
			PrintCounts();
		}
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor, bool allow_async = false) {
		PerfStats::Span key_span(PerfStats::SpanId::ShaderKey);
		if (enqueue) {
			DrainCompleted();
		}
		const ShaderType stage = StageOf(input_info);

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		const ShaderRecompiler::IR::SrtRuntime       runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .validate_memory_range      = ValidateShaderGuestMemoryRange,
		};
		key_span.Stop();
		if (entry != programs.end()) {
			if (PerfStats::Enabled()) {
				CountRepeatedInputs(entry->second, params);
			}
			PerfStats::Span materialize_span(PerfStats::SpanId::ShaderMaterialize);
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, entry->second.resources,
			    entry->second.specialization));
			materialize_span.Stop();
			PerfStats::Add(PerfStats::CounterId::ShaderGuestReads,
			               std::exchange(g_materialize_reads, 0));
			PerfStats::Span permutation_span(PerfStats::SpanId::ShaderPermutation);
			if (const auto permutation = std::ranges::find_if(
			        entry->second.permutations, [&](const Permutation& candidate) {
				        const auto& layout = candidate.program.bindings;
				        return layout.push_data_start_dword ==
				                   ShaderRecompiler::IR::PushData::StartFor(
				                       push_data_cursor, layout.ShaderDataDwords()) &&
				               candidate.specialization == entry->second.specialization;
			        });
			    permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		if (enqueue && allow_async) {
			// Hand translation (and, once the plan exists, permutation compilation) to a worker.
			// The draw that needed this shader is skipped until the result has been adopted.
			const bool has_entry = entry != programs.end();
			PendingJob pending_job {.has_entry        = has_entry,
			                        .push_data_cursor = has_entry ? push_data_cursor : 0u};
			if (has_entry) {
				pending_job.specialization = entry->second.specialization;
			}
			auto& jobs = pending[lookup_key];
			if (std::ranges::find(jobs, pending_job) != jobs.end()) {
				return {};
			}
			jobs.push_back(pending_job);
			auto job   = std::make_shared<AsyncJob<InputInfo>>();
			job->key   = lookup_key;
			job->stage = stage;
			job->hash  = params.hash;
			job->code.assign(params.code.begin(), params.code.end());
			job->back_code.assign(params.back_code.begin(), params.back_code.end());
			job->user_data.assign(user_data.begin(), user_data.end());
			job->input_info       = input_info;
			job->has_entry        = has_entry;
			job->push_data_cursor = push_data_cursor;
			if (has_entry) {
				job->translated = std::move(entry->second.initial_translation);
				entry->second.initial_translation.reset();
				job->specialization = entry->second.specialization;
			}
			enqueue([this, job] { RunAsyncJob<InputInfo>(*job); });
			return {};
		}

		PerfStats::Span compile_span(PerfStats::SpanId::ShaderCompileSync);
		const auto options = MakeOptions(stage, input_info, user_data, params.back_code, params.hash);
		auto translated = ShaderRecompiler::TranslateProgram(params.code, options);
		if (entry == programs.end()) {
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program)).first;
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, entry->second.resources,
			    entry->second.specialization));
		}
		entry->second.permutations.push_back(CompilePermutation(
		    params, options, std::move(translated), entry->second.specialization, push_data_cursor));
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		PerfStats::Add(PerfStats::CounterId::ShadersCompiled);
		sync_builds++;
		PrintCounts();
		return permutation.handle;
	}

	// Counts lookups whose program, shader base and user data match a recent lookup: the work a
	// cache keyed on those inputs could skip, provided the guest memory behind them is unchanged.
	// A direct-mapped table remembers roughly the last 64k input sets.
	void CountRepeatedInputs(const SourceEntry& source, const ShaderParams& params) {
		static constexpr size_t RecentInputsSize = size_t {1} << 16u;
		if (recent_inputs.empty()) {
			recent_inputs.resize(RecentInputsSize);
		}
		const auto seed   = reinterpret_cast<uintptr_t>(&source) ^ params.Base();
		const auto inputs = XXH3_64bits_withSeed(params.user_data.data(),
		                                         params.user_data_count * sizeof(uint32_t), seed);
		auto&      slot   = recent_inputs[inputs & (RecentInputsSize - 1u)];
		if (slot == inputs) {
			PerfStats::Add(PerfStats::CounterId::ShaderInputRepeats);
		}
		slot = inputs;
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	vk::Device                                                  device;
	std::atomic<uint64_t>                                       next_shader_id = 0;
	std::vector<uint64_t>                                       recent_inputs;

	// Async translation: set by PipelineCache when workers are enabled.
	std::function<void(std::function<void()>)> enqueue;
	// Shaders this thread had to build itself, which is what warm-up watches.
	uint64_t                                   sync_builds = 0;
	std::mutex                                 completed_mutex;
	std::vector<AsyncResult>                   completed;
	std::unordered_map<ProgramKey, std::vector<PendingJob>, ProgramKeyHash> pending;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	if (graphics.graphics_pipeline_library_enabled &&
	    graphics.graphics_pipeline_library_fast_linking) {
		m_graphics_library_cache =
		    std::make_unique<GraphicsPipelineLibraryCache>(graphics, m_driver_cache);
		PipelineCacheLog("Vulkan graphics pipeline libraries: fast-link path enabled");
	} else {
		PipelineCacheLog("Vulkan graphics pipeline libraries: monolithic fallback");
	}
	const auto async_mode = Config::GetAsyncShaders();
	m_async               = async_mode != Config::AsyncShaders::Off;
	if (async_mode == Config::AsyncShaders::WarmUp) {
		m_warm_up.Start();
	}
	// Which mode is running is worth stating in every one of them: the difference is visible on
	// screen, and a run that was meant to warm up but did not looks like the mode doing nothing.
	PipelineCacheLog(
	    "Shaders: {}",
	    async_mode == Config::AsyncShaders::Off ? "synchronous, every draw waits for what it needs"
	    : async_mode == Config::AsyncShaders::WarmUp
	        ? "warming up, nothing is skipped until a frame needs nothing new"
	        : "asynchronous, a draw is skipped until its shader and pipeline are ready");
	if (m_async) {
		StartWorkers();
		m_program_cache->enqueue = [this](std::function<void()> job) { EnqueueJob(std::move(job)); };
	}
	if (m_driver_cache != nullptr) {
		StartSaver();
	}
}

std::atomic<uint64_t> PipelineCache::s_guest_frames {0};

void PipelineCache::NoteGuestFrame() noexcept {
	s_guest_frames.fetch_add(1, std::memory_order_relaxed);
}

// Called before every graphics lookup, so it sees the first lookup of each frame.
void PipelineCache::UpdateWarmUp() {
	if (!m_warm_up.Active()) {
		return;
	}
	m_warm_up.Update(s_guest_frames.load(std::memory_order_relaxed),
	                 m_sync_pipeline_builds + m_program_cache->sync_builds);
	if (!m_warm_up.Active()) {
		PipelineCacheLog("Shaders: warm-up finished after {} frames and {} builds",
		                 m_warm_up.Frames(), m_warm_up.Builds());
	}
}

void PipelineCache::StartSaver() {
	m_saver = std::jthread([this] {
		std::unique_lock lock(m_saver_mutex);
		for (;;) {
			// A session that ends in a crash or a forced exit would otherwise lose everything
			// compiled since launch; write what is new every half minute.
			if (m_saver_wake.wait_for(lock, std::chrono::seconds(30),
			                          [this] { return m_stop_saver; })) {
				return;
			}
			lock.unlock();
			SaveSnapshot();
			lock.lock();
		}
	});
}

void PipelineCache::StopSaver() {
	{
		std::lock_guard lock(m_saver_mutex);
		m_stop_saver = true;
	}
	m_saver_wake.notify_all();
	if (m_saver.joinable()) {
		m_saver.join();
	}
}

void PipelineCache::StartWorkers() {
	const auto hardware = std::thread::hardware_concurrency();
	const auto count    = hardware > 2 ? hardware / 2 : 1;
	for (uint32_t i = 0; i < count; i++) {
		m_workers.emplace_back([this] {
			for (;;) {
				std::function<void()> job;
				{
					std::unique_lock lock(m_job_mutex);
					m_job_available.wait(lock, [this] { return m_stop_workers || !m_jobs.empty(); });
					if (m_stop_workers && m_jobs.empty()) {
						return;
					}
					job = std::move(m_jobs.front());
					m_jobs.pop_front();
				}
				job();
			}
		});
	}
	PipelineCacheLog("Async shader pipelines: {} worker threads", count);
}

void PipelineCache::StopWorkers() {
	{
		std::lock_guard lock(m_job_mutex);
		m_stop_workers = true;
	}
	m_job_available.notify_all();
	m_workers.clear(); // joins
}

void PipelineCache::EnqueueJob(std::function<void()> job) {
	{
		std::lock_guard lock(m_job_mutex);
		m_jobs.push_back(std::move(job));
	}
	m_job_available.notify_one();
}

void PipelineCache::DrainCompletedPipelines() {
	std::vector<CompletedPipeline> completed;
	{
		std::lock_guard lock(m_completed_mutex);
		completed.swap(m_completed_pipelines);
	}
	for (auto& done: completed) {
		m_pending_pipelines.erase(done.key);
		auto [iter, inserted] =
		    m_graphics_pipelines.emplace(std::move(done.key), std::move(done.pipeline));
		EXIT_IF(!inserted);
	}
}

PipelineCache::~PipelineCache() {
	StopSaver();
	if (m_async) {
		StopWorkers();
		DrainCompletedPipelines();
		m_program_cache->DrainCompleted();
	}
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			if (pipeline->owns_layout) {
				m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
				m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout,
				                                             nullptr);
			}
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	m_graphics_library_cache.reset();
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	if (git_hash.ends_with("-dirty")) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::Save() {
	StopSaver();
	std::lock_guard lock(m_save_mutex);
	if (m_driver_cache == nullptr) {
		return;
	}
	if (m_pipelines_created.load(std::memory_order_acquire) != m_pipelines_saved &&
	    WriteDriverCache()) {
		m_pipelines_saved = m_pipelines_created.load(std::memory_order_acquire);
	}
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

void PipelineCache::SaveSnapshot() {
	std::lock_guard lock(m_save_mutex);
	if (m_driver_cache == nullptr) {
		return;
	}
	const auto created = m_pipelines_created.load(std::memory_order_acquire);
	if (created == m_pipelines_saved) {
		return;
	}
	if (WriteDriverCache()) {
		m_pipelines_saved = created;
	}
}

void PipelineCache::EmergencySave() noexcept {
	if (m_driver_cache == nullptr) {
		return; // disabled, or already written and released by Save()
	}
	// Other threads may be inside the driver or holding the save lock when the process dies on
	// a fault. Do the work on a helper and give it a bounded time so a wedged driver cannot turn
	// the crash into a hang.
	auto        done = std::make_shared<std::atomic<bool>>(false);
	std::thread worker([this, done] {
		SaveSnapshot();
		done->store(true, std::memory_order_release);
	});
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
	while (!done->load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	if (done->load(std::memory_order_acquire)) {
		worker.join();
	} else {
		PipelineCacheLog("Vulkan pipeline cache: emergency save did not finish in time");
		worker.detach();
	}
}

// Caller holds m_save_mutex and m_driver_cache is valid.
bool PipelineCache::WriteDriverCache() {
	PerfStats::Span      span(PerfStats::SpanId::PipelineCacheSave);
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return false;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return false;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return false;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
	return true;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	PerfStats::Span prepare_span(PerfStats::SpanId::ShaderPrepare);
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		const auto& blend          = context.GetBlendControl(0);
		const auto  is_dual_source = [](uint8_t factor) {
			return factor >= static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color) &&
			       factor <= static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		};
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (is_dual_source(blend.color_srcblend) || is_dual_source(blend.color_destblend) ||
		     (blend.separate_alpha_blend &&
		      (is_dual_source(blend.alpha_srcblend) || is_dual_source(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies a second blend source for the same render target as MRT0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		}
	}
	prepare_span.Stop();
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	Common::LockGuard lock(m_mutex);
	UpdateWarmUp();
	const bool        allow_async = !m_warm_up.Active();
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	if (pixel_active) {
		result.pixel =
		    m_program_cache->Get(pixel_params, pixel_info, push_data_cursor, allow_async);
		if (!result.pixel) {
			// Still compiling; the vertex lookups would otherwise use a wrong push-data cursor.
			return result;
		}
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor,
		                                        allow_async);
		if (!result.vertex[i]) {
			result.vertex = {};
			return result;
		}
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor = 0;
	// Compute results feed later passes (indirect arguments, skinning); skipping a dispatch is
	// riskier than a stall, so compute shaders stay synchronous.
	return m_program_cache->Get(params, input_info, push_data_cursor, /*allow_async=*/false);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline* PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	Common::LockGuard lock(m_mutex);
	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		// A PS5 color target is written only where CB_TARGET_MASK and the pixel shader's
		// CB_SHADER_MASK agree. Vulkan leaves a channel undefined when the fragment shader has no
		// output for it, so channels the shader does not export are masked out.
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_write_mask_slot(ctx.GetRenderTargetMask(),
		                                  ctx.GetShaderRegisters().m_cbShaderMask,
		                                  colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		static_params.color_srcblend[slot]       = bc.color_srcblend;
		static_params.color_comb_fcn[slot]       = bc.color_comb_fcn;
		static_params.color_destblend[slot]      = bc.color_destblend;
		static_params.alpha_srcblend[slot]       = bc.alpha_srcblend;
		static_params.alpha_comb_fcn[slot]       = bc.alpha_comb_fcn;
		static_params.alpha_destblend[slot]      = bc.alpha_destblend;
		static_params.separate_alpha_blend[slot] = bc.separate_alpha_blend;
		static_params.blend_enable[slot]         = bc.enable && !rt.info.blend_bypass;
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
#if defined(__APPLE__)
	static_params.depth_min_bounds = depth.depth_min_bounds;
	static_params.depth_max_bounds = depth.depth_max_bounds;
#else
	// The bounds themselves are dynamic state (see SetGraphicsDynamicParams); keep the key stable.
	static_params.depth_min_bounds = 0.0f;
	static_params.depth_max_bounds = 1.0f;
#endif
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		uint32_t attributes_num          = 0;
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			EXIT_IF(buffer.attr_num < 0 || buffer.attr_num > ShaderVertexInputBuffer::ATTR_MAX);
			attributes_num += static_cast<uint32_t>(buffer.attr_num);
			EXIT_IF(attributes_num > static_cast<uint32_t>(vs_input_info.resources_num));
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
			for (int attribute = 0; attribute < buffer.attr_num; attribute++) {
				const auto index = buffer.attr_indices[attribute];
				EXIT_IF(index < 0 || index >= vs_input_info.resources_num);
				key.vertex_input.attributes[index] = {
				    .offset  = buffer.attr_offsets[attribute],
				    .binding = static_cast<uint8_t>(binding),
				};
			}
		}
		EXIT_IF(attributes_num != static_cast<uint32_t>(vs_input_info.resources_num));
	}

	// Consecutive draws very often share a pipeline; a full key compare is far cheaper than
	// hashing the key and probing the map.
	if (m_last_graphics_pipeline != nullptr && key == m_last_graphics_key) {
		return m_last_graphics_pipeline;
	}
	if (m_async) {
		DrainCompletedPipelines();
	}
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		m_last_graphics_key      = key;
		m_last_graphics_pipeline = iter->second.get();
		return iter->second.get();
	}
	if (m_async && !m_warm_up.Active()) {
		if (m_pending_pipelines.contains(key)) {
			return nullptr;
		}
		// Build on a worker. Everything the builder reads is copied; the program pointers inside
		// the input infos stay valid because permutations live in a deque.
		m_pending_pipelines.insert(key);
		auto vertex_copy =
		    std::make_shared<std::vector<ShaderVertexInputInfo>>(vertex_info.begin(), vertex_info.end());
		std::shared_ptr<ShaderPixelInputInfo> ps_copy;
		if (ps_active) {
			ps_copy = std::make_shared<ShaderPixelInputInfo>(*ps_input_info);
		}
		EnqueueJob([this, key, static_params, rendering, vertex_copy, ps_copy,
		            programs]() mutable {
			PerfStats::Span span(PerfStats::SpanId::PipelineCreateAsync);
			auto cached = std::make_unique<Pipeline>();
			CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, *vertex_copy,
			                       ps_copy ? ps_copy.get() : nullptr, programs, static_params,
			                       m_driver_cache, m_graphics_library_cache.get());
			EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
			EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
			m_pipelines_created.fetch_add(1, std::memory_order_acq_rel);
			PerfStats::Add(PerfStats::CounterId::PipelinesCreated);
			span.Stop();
			std::lock_guard lock(m_completed_mutex);
			m_completed_pipelines.push_back({std::move(key), std::move(cached)});
		});
		return nullptr;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	m_sync_pipeline_builds++;
	PerfStats::Span create_span(PerfStats::SpanId::PipelineCreateSync);
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                       ps_input_info, programs, static_params, m_driver_cache,
	                       m_graphics_library_cache.get());
	create_span.Stop();
	PerfStats::Add(PerfStats::CounterId::PipelinesCreated);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	m_pipelines_created.fetch_add(1, std::memory_order_acq_rel);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	m_last_graphics_key      = iter->first;
	m_last_graphics_pipeline = iter->second.get();

	return iter->second.get();
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	Common::LockGuard lock(m_mutex);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	PerfStats::Span create_span(PerfStats::SpanId::PipelineCreateSync);
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);
	create_span.Stop();
	PerfStats::Add(PerfStats::CounterId::PipelinesCreated);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	m_pipelines_created.fetch_add(1, std::memory_order_acq_rel);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}
} // namespace Libs::Graphics
