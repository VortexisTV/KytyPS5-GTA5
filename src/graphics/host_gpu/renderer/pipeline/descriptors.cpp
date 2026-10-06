#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/gpuCrashDiagnostics.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandHooks.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fmt/format.h>
#include <limits>
#include <new>
#include <optional>
#include <map>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <xxhash.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::LoopWatchdog: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

uint32_t NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding) {
	return binding.resources.empty() ? 1u : static_cast<uint32_t>(binding.resources.size());
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(!texture.image_id || view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

// Bytes to ask RenderContext::MappedExtent for when binding a shader buffer descriptor, so that the
// binding is resolved against the ranges the guest mapped for GPU access. Zero means bind nothing
// at all.
//
// A GNM descriptor's num_records is routinely over-provisioned - commonly close to UINT32_MAX -
// because the shader bounds-checks its own accesses. The nominal footprint is therefore not a claim
// that any of it is resident, and binding it whole asks the buffer cache to stage, track and map
// memory the guest never mapped.
//
// The binding is not grown past that footprint either. A read-only descriptor with a stride was
// once given the mapped extent up to 64 KiB, for meshes whose bone-matrix descriptor describes
// fewer records than the mesh indexes. ObtainBuffer tracks whatever range it is given, which
// write-protects the guest pages underneath and makes the range count as a use of anything else
// that lives there, so the extra 64 KiB behind every such descriptor has to stay out of it.
static uint64_t StorageBufferRequest(RenderContext& context, const ShaderBufferResource& descriptor) {
	if (descriptor.Base48() == 0) {
		return 0;
	}
	const auto max_range = static_cast<uint64_t>(
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	return std::min<uint64_t>(descriptor.GetSize(), max_range);
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset) {
	buffer_offset = 0;

	const auto address = source.address;
	const auto size    = source.size;
	const auto id      = source.id;
	if (address == 0 || size == 0) {
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	std::optional<DrawPhaseTimer::ProbeScope> obtain_probe;
	obtain_probe.emplace(g_draw_phases, resource.written ? DrawPhaseTimer::BufferWritten
	                                                     : DrawPhaseTimer::BufferRead);
	auto [buffer, offset] = context.GetBufferCache().ObtainBuffer(address, size, resource.written,
	                                                              resource.formatted, id);
	obtain_probe.reset();
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range =
	    static_cast<uint64_t>(graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	if (adjustment % sizeof(uint32_t) != 0 || adjustment >= 256) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	// Aligning the offset down puts `adjustment` bytes in front of the binding, so the described
	// range has to leave room for them inside the device limit. Trimming the tail is safe for the
	// same reason the clamp in StorageBufferRequest is.
	buffer_offset = static_cast<uint32_t>(adjustment);
	auto range = std::min(size, max_range - adjustment) + adjustment;
	if (graphics.hardware_storage_buffer_bounds) {
		// The device checks word accesses against the range, which the shader's own checks took
		// in whole dwords: a trailing partial dword is out of range. A range without one whole
		// dword is a null descriptor, which the device treats as empty.
		range &= ~vk::DeviceSize {sizeof(uint32_t) - 1};
		if (range == 0) {
			buffer_offset = 0;
			return {nullptr, 0, VK_WHOLE_SIZE};
		}
	}
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, range};
	if (resource.written) {
		DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::BufferInvalidate);
		context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
	}
	return result;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	const bool         common          = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                                     (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                                     descriptor.fields[3] == field3_expected;
	if (r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	const bool full = common && descriptor.fields[4] == field4_expected &&
	                  descriptor.fields[5] == field5_expected;
	if (!full ||
	    (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type()))) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	const bool encoding_ok = IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d) && supported_tile &&
	       descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

static bool IsSupportedStorageTextureEncoding(const ShaderRecompiler::IR::ImageResource& resource,
                                              const ShaderTextureResource& descriptor) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	constexpr uint32_t field5_expected      = 0x00700000u;
	constexpr uint32_t field5_max_mip_mask  = 0x000000f0u;
	const uint32_t     expected_field3 = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     expected_field4 =
	    descriptor.Depth() | (static_cast<uint32_t>(descriptor.BaseArray5()) << 16u);
	const bool common = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                    (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                    descriptor.fields[3] == expected_field3;
	if (resource.r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	return common && descriptor.fields[4] == expected_field4 &&
	       (descriptor.fields[5] & ~field5_max_mip_mask) == field5_expected;
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsSupportedStorageTextureEncoding(resource, descriptor);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = format == Prospero::BufferFormat::k32SInt && uint_resource &&
	                              resource.written && !resource.read && !resource.atomic;
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == Prospero::BufferFormat::k32UInt));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float:
			desc.info.guest_format = Prospero::BufferFormat::k32Float;
			break;
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	desc.info.pixel_format    = VulkanFormat(desc.info.guest_format);
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = 4;
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = desc.info.pixel_format;
	desc.view_info.type       = vk::ImageViewType::e2D;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	desc.type                 = binding;
	return desc;
}

static void PopulateTextureMipLayout(ImageInfo& info) {
	if (info.IsVolume() && info.tile_mode != Prospero::TileMode::kLinear) {
		TileSurfaceLayout            surface {};
		const TileSurfaceDescription description {
		    info.guest_format,  info.tile_mode,    TileSurfaceDimension::Dim3D, info.extent.width,
		    info.extent.height, info.extent.depth, info.resources.levels,       1};
		if (!TileGetTiledTextureLayout(description, surface)) {
			EXIT("unsupported normalized volume texture layout\n");
		}
		for (uint32_t level = 0; level < info.resources.levels; level++) {
			const auto& mip        = surface.mips[level];
			info.mip_layout[level] = {
			    mip.offset,
			    mip.size,
			    mip.padded_width,
			    mip.padded_height,
			};
		}
		return;
	}

	TileSizeOffset levels[16] {};
	TilePaddedSize padded[16] {};
	TileGetTextureSize(info.guest_format, info.extent.width, info.extent.height,
	                   info.resources.levels, info.tile_mode, nullptr, levels, padded);
	const auto texel_shift = info.IsBlock() ? 2u : 0u;
	for (uint32_t level = 0; level < info.resources.levels; level++) {
		const auto offset =
		    levels[level].src_size != 0 ? levels[level].src_offset : levels[level].offset;
		auto size = static_cast<uint64_t>(levels[level].src_size != 0 ? levels[level].src_size
		                                                              : levels[level].size);
		if (info.IsVolume()) {
			size *= std::max(info.extent.depth >> level, 1u);
		} else {
			size *= info.resources.layers;
		}
		info.mip_layout[level] = {
		    offset,
		    size,
		    padded[level].width >> texel_shift,
		    padded[level].height >> texel_shift,
		};
	}
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || surface_format.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(const TileSurfaceDescription& description, bool metadata,
                                   uint32_t view_levels, uint32_t& levels, uint32_t& base_level) {
	TileSurfaceLayout physical {};
	TileSurfaceLayout view {};
	auto              view_description = description;
	view_description.levels            = levels;
	if (!TileGetTiledTextureLayout(description, physical) ||
	    !TileGetTiledTextureLayout(view_description, view)) {
		return false;
	}
	if (physical.first_tail_level == view.first_tail_level &&
	    physical.block_slice_size == view.block_slice_size &&
	    physical.total_size == view.total_size &&
	    std::equal(std::begin(physical.mips), std::begin(physical.mips) + description.levels,
	               std::begin(view.mips))) {
		return true;
	}
	if (metadata || ((description.layers > 1 || description.depth > 1) &&
	                 physical.block_slice_size != view.block_slice_size)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= description.levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (physical.mips[target] != view.mips[source] ||
			    (target >= physical.first_tail_level) != (source >= view.first_tail_level) ||
			    std::max(description.width >> target, 1u) != std::max(description.width >> source, 1u) ||
			    std::max(description.height >> target, 1u) != std::max(description.height >> source, 1u) ||
			    std::max(description.depth >> target, 1u) != std::max(description.depth >> source, 1u)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			levels     = description.levels;
			base_level = base;
			return true;
		}
	}
	return false;
}

namespace {

// Everything ResolveTexture derives from an image resource and its descriptor alone.
struct TextureDescription {
	TextureCache::ImageDesc desc;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint64_t                size              = 0;
	bool                    shader_conversion = false;
};

TextureDescription DescribeTexture(const ShaderRecompiler::IR::ImageResource& resource,
                                   const ShaderTextureResource&               descriptor) {
	const bool storage         = resource.written;
	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::DynamicStorage;
	const auto view_levels = multisampled || single_storage_mip
	                             ? 1u
	                             : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class == Prospero::SampledTextureNumericClass(format);
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	if (levels > physical_levels) {
		const TileSurfaceDescription physical {
		    format, tile, volume ? TileSurfaceDimension::Dim3D : TileSurfaceDimension::Dim2D,
		    width, height, volume ? depth : 1u, physical_levels, image_layers};
		if (!ResolveTextureMipView(physical, !resource.r128 && descriptor.MetaCompress(),
		                           view_levels, levels, view_base)) {
			EXIT("unsupported texture mip view changes physical layout: base=%u last=%u max=%u "
			     "extent=%ux%ux%u tile=%u\n",
			     base_level, last_level, max_mip, width, height, depth,
			     static_cast<uint32_t>(tile));
		}
	}
	uint32_t      pitch = 0;
	TileSizeAlign size {};
	if (multisampled) {
		const auto bytes = Prospero::NumBytesPerElement(format);
		pitch            = depth_tile ? TileGetDepthPitch(width, bytes, last_level)
		                              : TileGetRenderTargetPitch(width, bytes, last_level);
		if (pitch == 0 || !TileGetRenderTargetSize(width, height, pitch, bytes, size, last_level) ||
		    size.size > UINT32_MAX / image_layers) {
			EXIT("unsupported multisample texture layout\n");
		}
		size.size *= image_layers;
	} else {
		pitch = TileGetTexturePitch(format, width, tile);
		TileGetTextureTotalSize(format, width, height, volume ? depth : image_layers,
		                        physical_levels, tile, volume, size);
	}
	EXIT_NOT_IMPLEMENTED(size.size == 0 || size.align == 0 ||
	                     (address & (static_cast<uint64_t>(size.align) - 1u)) != 0);
	if (storage) {
		ValidateStorageTexture(resource, descriptor, size.size);
	}

	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	const auto storage_view_format = storage && (resource.atomic ||
	                                            format == Prospero::BufferFormat::k32SInt)
	                                     ? vk::Format::eR32Uint
	                                     : SrgbStorageViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, size.size};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.pitch        = pitch;
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth &&
	    !desc.info.IsDepth()) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.kind          = ImageMetadataKind::Dcc;
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	if (samples > 1) {
		desc.info.mip_layout[0] = {0, size.size, pitch, height};
	} else {
		PopulateTextureMipLayout(desc.info);
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	return {std::move(desc), pixel_format, view_format, size.size, shader_conversion};
}

// A texture description is a pure function of its inputs, and draws bind the same textures over
// and over, so recent descriptions are kept. An entry holds copies of both inputs: a hit is exact.
// A scene can bind a few thousand distinct textures (Sky Garden missed half its lookups in a
// 256-entry direct-mapped table), so the table is set-associative and evicts the least recently
// used entry of a set.
class TextureDescriptionCache {
public:
	struct Entry {
		uint32_t                              hash     = 0;
		uint64_t                              last_use = 0; // 0: empty.
		ShaderRecompiler::IR::DescriptorValue value;
		ShaderRecompiler::IR::ImageResource   resource;
		TextureDescription                    description;
		// The description's last FindImage result (see TextureCache::RefindImage) and the image
		// ResolveTexture bound for it; generation 0: none.
		uint64_t                              generation = 0;
		ImageId                               found;
		ImageId                               bound;
		// The view RebindImages acquired for a binding of this description.
		PreparedBindings::ViewMemo            view;
	};

	// The entry stays valid until the next call.
	Entry& Get(const ShaderRecompiler::IR::ImageResource&   resource,
	           const ShaderRecompiler::IR::DescriptorValue& value,
	           const ShaderTextureResource&                 descriptor) {
		uint32_t hash = 2166136261u;
		for (const auto dword: value.dwords) {
			hash = (hash ^ dword) * 16777619u;
		}
		hash = (hash ^ resource.source) * 16777619u;
		// Mix the high bits into the set index (MurmurHash3's finalizer).
		hash ^= hash >> 16u;
		hash *= 0x85ebca6bu;
		hash ^= hash >> 13u;
		hash *= 0xc2b2ae35u;
		hash ^= hash >> 16u;
		auto* const set    = &m_entries[(hash % Sets) * Ways];
		Entry*      victim = set;
		for (size_t way = 0; way < Ways; way++) {
			auto& entry = set[way];
			if (entry.last_use != 0 && entry.hash == hash && entry.value == value &&
			    entry.resource == resource) {
				entry.last_use = ++m_clock;
				return entry;
			}
			if (entry.last_use < victim->last_use) {
				victim = &entry;
			}
		}
		victim->description = DescribeTexture(resource, descriptor);
		victim->hash        = hash;
		victim->last_use    = ++m_clock;
		victim->value       = value;
		victim->resource    = resource;
		victim->generation  = 0;
		victim->view        = {};
		return *victim;
	}

private:
	static constexpr size_t Ways = 4;
	static constexpr size_t Sets = 1024;
	uint64_t                m_clock   = 0;
	std::vector<Entry>      m_entries = std::vector<Entry>(Sets * Ways);
};

// KYTY_DEBUG_TEXTURE_REUSE=0 resolves every texture binding from scratch, for A/B runs (see also
// AbFeatureOff).
bool TextureReuseEnabled() {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_DEBUG_TEXTURE_REUSE");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("reuse");
	return enabled && !(ab && AbFeatureOff());
}

// A null texture binding keeps its image and view while its descriptor repeats;
// KYTY_DEBUG_AB=nullreuse alternates.
bool NullReuseEnabled() {
	static const bool ab = AbSelected("nullreuse");
	return TextureReuseEnabled() && !(ab && AbFeatureOff());
}

// A resolved binding starts from the view its description's last binding acquired;
// KYTY_DEBUG_AB=viewmemo alternates.
bool ViewMemoEnabled() {
	static const bool ab = AbSelected("viewmemo");
	return TextureReuseEnabled() && !(ab && AbFeatureOff());
}

} // namespace

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              PreparedBindings::ImageSource*               source) {
	TextureBinding binding;
	ResolveTextureInto(resource, value, source, binding);
	return binding;
}

void RenderExecutor::ResolveTextureInto(const ShaderRecompiler::IR::ImageResource&   resource,
                                        const ShaderRecompiler::IR::DescriptorValue& value,
                                        PreparedBindings::ImageSource*               source,
                                        TextureBinding&                              out) {
	if (source != nullptr) {
		source->generation = 0;
		source->null_image = false;
		source->view_memo  = nullptr;
	}
	out.image_view = nullptr;
	out.layout     = vk::ImageLayout::eUndefined;
	out.mip_views.clear();
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		out.desc     = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                 : TextureCache::BindingType::Texture);
		out.image_id = texture_cache.FindImage(out.desc);
		if (source != nullptr && !storage) {
			source->resource   = resource;
			source->value      = value;
			source->found      = out.image_id;
			source->null_image = true;
		}
		return;
	}

	thread_local TextureDescriptionCache descriptions;
	std::optional<DrawPhaseTimer::ProbeScope> describe_probe;
	describe_probe.emplace(g_draw_phases, DrawPhaseTimer::TextureDescribe);
	auto&       entry     = descriptions.Get(resource, value, descriptor);
	const auto& described = entry.description;
	out.desc              = described.desc;
	auto&       desc      = out.desc;
	describe_probe.reset();
	// Debug: KYTY_DEBUG_VOLUME_MIP0=1 views volume textures as their first level only, so every
	// sample clamps to it: tells wrong lower levels (their layout) from wrong level 0.
	static const bool volume_mip0 = std::getenv("KYTY_DEBUG_VOLUME_MIP0") != nullptr;
	if (volume_mip0 && desc.info.IsVolume()) [[unlikely]] {
		desc.view_info.level_count = 1;
	}

	const auto metadata_base_layer = desc.view_info.base_layer;
	const auto remember            = [&](ImageId found, uint64_t generation) {
		if (source != nullptr && generation != 0) {
			source->resource            = resource;
			source->value               = value;
			source->found               = found;
			source->generation          = generation;
			source->metadata_base_layer = metadata_base_layer;
		}
	};
	// The description's previous lookup still holds while the image set is unchanged, and the
	// validation below passed for the image it bound while that image's stencil association is
	// unchanged (see ReuseTexture).
	if (TextureReuseEnabled() && entry.generation != 0 &&
	    texture_cache.RefindImage(entry.found, entry.generation, desc, metadata_base_layer)) {
		const auto depth_id = texture_cache.GetImage(entry.found).depth_id;
		if ((depth_id ? depth_id : entry.found) == entry.bound) {
			remember(entry.found, entry.generation);
			out.image_id = entry.bound;
			if (source != nullptr) {
				source->view_memo = &entry.view;
				// A binding of this description acquired a view before, maybe in another slot.
				if (const auto& memo = entry.view;
				    ViewMemoEnabled() && memo.generation != 0 && memo.image == entry.bound) {
					source->view            = memo.view;
					source->view_image      = memo.image;
					source->view_info       = memo.info;
					source->view_generation = memo.generation;
				}
			}
			return;
		}
	}
	entry.generation = 0;
	if (source != nullptr) {
		source->view_memo = &entry.view;
	}

	uint64_t generation = 0;
	ImageId  id;
	{
		DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::TextureImage);
		id = texture_cache.FindImage(desc, described.shader_conversion, &generation);
	}
	remember(id, generation);
	const auto found = id;
	auto*      image = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, described.pixel_format,
		                            described.size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, described.view_format,
		                         descriptor.DstSelXYZW());
	} else {
		(void)SelectSampledColorView(image->info.pixel_format, described.pixel_format,
		                             descriptor.DstSelXYZW());
	}
	if (generation != 0) {
		entry.generation = generation;
		entry.found      = found;
		entry.bound      = id;
	}
	out.image_id = id;
}

static vk::Sampler NativeSampler(RenderContext&                       context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t index,
                                 const ShaderRecompiler::IR::DescriptorValue& value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::Upload);
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.info.data.Empty()) {
		return;
	}
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			image->binding = {};
		}
	}
	m_bound_images.clear();
}

// A sampled texture whose descriptor enables mip statistics counts in its counter, which texture
// streamers read back (RenderContext::ReportMipStats).
static void MarkMipStats(RenderContext& context, const ShaderRecompiler::IR::ImageResource& resource,
                         const ShaderRecompiler::IR::DescriptorValue& value) {
	if (resource.written || value.dword_count < 8) {
		return;
	}
	const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	if (descriptor.MipStatsCntEn()) {
		context.MarkMipStatsCounter(descriptor.MipStatsCntId());
	}
}

// Whether a stage slot's previous image binding stands for ResolveTexture(resource, value): the
// same resource and descriptor find the same image while the image set is unchanged, and the
// binding follows that image's stencil association, which must not have changed either.
static bool ReuseTexture(TextureCache& cache, const PreparedBindings::ImageSource& source,
                         const TextureBinding&                        binding,
                         const ShaderRecompiler::IR::ImageResource&   resource,
                         const ShaderRecompiler::IR::DescriptorValue& value) {
	if (source.null_image) {
		// The same resource's null descriptor finds the same null image, which never changes.
		return NullReuseEnabled() && source.value == value && source.resource == resource &&
		       source.found == binding.image_id;
	}
	if (!TextureReuseEnabled() || source.generation == 0 || !(source.value == value) ||
	    !(source.resource == resource) ||
	    !cache.RefindImage(source.found, source.generation, binding.desc,
	                       source.metadata_base_layer)) {
		return false;
	}
	const auto depth_id = cache.GetImage(source.found).depth_id;
	return (depth_id ? depth_id : source.found) == binding.image_id;
}

// KYTY_DEBUG_IMAGE_GROUPS=0 checks every image of every draw; KYTY_DEBUG_AB=imagegroups
// alternates.
static bool ImageGroupsEnabled() {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_DEBUG_IMAGE_GROUPS");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("imagegroups");
	return enabled && TextureReuseEnabled() && !(ab && AbFeatureOff());
}

// KYTY_VERIFY_IMAGE_GROUPS=1: a draw keeping its stage's image group also resolves each image and
// acquires its view, and reports what came out differently.
static bool VerifyImageGroups() {
	static const bool enabled = std::getenv("KYTY_VERIFY_IMAGE_GROUPS") != nullptr;
	return enabled;
}

static void ReportImageGroupCheck(bool same, uint64_t address) {
	static std::atomic<uint64_t> checked {0};
	static std::atomic<uint64_t> missed {0};
	const auto                   count = checked.fetch_add(1, std::memory_order_relaxed) + 1;
	if (!same && missed.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("image-group verify: 0x%016" PRIx64 " kept binding differs\n", address);
	}
	if (count % 100000 == 0) {
		std::printf("image-group verify: images=%" PRIu64 " missed=%" PRIu64 "\n", count,
		            missed.load(std::memory_order_relaxed));
		std::fflush(stdout);
	}
}

// Whether the stage's recorded images still stand: no image was registered or unregistered and
// the tick and GC tick are the ones they were checked under, so every lookup and view acquisition
// would repeat itself (RefindImage, IsTextureCurrent), and each image is still clean, bound for
// the image its lookup finds, and has no color clear to apply for its request.
static bool ImageGroupStands(TextureCache& cache, const PreparedBindings& prepared) {
	if (!(prepared.image_group.state == cache.CurrentTargetState())) {
		return false;
	}
	for (size_t i = 0; i < prepared.images.size(); i++) {
		const auto& source = prepared.image_sources[i];
		const auto  bound  = prepared.images[i].image_id;
		if (source.null_image) {
			continue; // The null image never changes (see ReuseTexture).
		}
		const auto depth = cache.GetImage(source.found).depth_id;
		if ((depth ? depth : source.found) != bound || !cache.IsTextureClean(bound) ||
		    !cache.IsColorClearCurrent(source.found, prepared.images[i].desc,
		                               source.metadata_base_layer)) {
			return false;
		}
	}
	return true;
}

// Records the stage's images as RebindImages left them, for the next draw of the same program and
// image descriptors (see PreparedBindings::ImageGroup). Only sampled textures that a lookup can
// repeat (or null textures) and that keep their views qualify; ImageGroupStands also asks each
// color-compressed surface whether its color clear check for the request still holds.
static void RecordImageGroup(RenderContext& context, PreparedBindings& prepared,
                             const ShaderRecompiler::IR::CompiledShaderInfo& program,
                             const std::vector<ShaderRecompiler::IR::DescriptorValue>& values) {
	auto& group   = prepared.image_group;
	auto& cache   = context.GetTextureCache();
	group.program = nullptr;
	if (!ImageGroupsEnabled()) {
		return;
	}
	group.mip_counters.clear();
	for (size_t i = 0; i < prepared.images.size(); i++) {
		const auto& binding = prepared.images[i];
		const auto& source  = prepared.image_sources[i];
		if (binding.desc.type != TextureCache::BindingType::Texture ||
		    (source.generation == 0 && !(source.null_image && NullReuseEnabled())) ||
		    source.view_generation == 0 || source.view_image != binding.image_id ||
		    source.view != binding.image_view ||
		    program.info.images[i].mip_mode == ShaderRecompiler::IR::ImageMipMode::DynamicStorage) {
			return;
		}
		if (values[i].dword_count >= 8) {
			const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(values[i]);
			if (descriptor.MipStatsCntEn()) {
				group.mip_counters.push_back(descriptor.MipStatsCntId());
			}
		}
	}
	group.images.assign(values.begin(), values.end());
	group.state   = cache.CurrentTargetState();
	group.program = &program;
}

// Debugging aid for the draw-phases line: counts stages whose program and descriptors repeat the
// previous draw's.
static void NoteBindingRepeat(PreparedBindings& prepared, const ShaderStageRuntime& runtime) {
	const auto& snapshot = *runtime.resources;
	const auto  hash     = [](const std::vector<ShaderRecompiler::IR::DescriptorValue>& values) {
		return values.empty() ? uint64_t {0}
		                      : XXH3_64bits(values.data(), values.size() * sizeof(values[0]));
	};
	const PreparedBindings::RepeatKey key {runtime.program, hash(snapshot.images),
	                                      hash(snapshot.buffers), hash(snapshot.samplers)};
	const auto&                       last    = prepared.repeat_key;
	const bool                        program = last.program == key.program;
	auto&                             stats   = g_binding_repeats;
	stats.stages.fetch_add(1, std::memory_order_relaxed);
	stats.same_program.fetch_add(program ? 1 : 0, std::memory_order_relaxed);
	stats.same_images.fetch_add(program && last.images == key.images ? 1 : 0,
	                           std::memory_order_relaxed);
	stats.same_buffers.fetch_add(program && last.buffers == key.buffers ? 1 : 0,
	                            std::memory_order_relaxed);
	stats.same_all.fetch_add(program && last.images == key.images && last.buffers == key.buffers &&
	                                last.samplers == key.samplers
	                            ? 1
	                            : 0,
	                        std::memory_order_relaxed);
	prepared.repeat_key = key;
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	PerfStats::Span span(PerfStats::SpanId::BindResolve);
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	if (g_draw_phases.Active()) [[unlikely]] {
		NoteBindingRepeat(prepared, runtime);
	}
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	// buffer_sources keeps the previous draw's entries of this program's slots for FindBuffers,
	// which replaces them all.
	prepared.buffer_sources.resize(program.info.buffers.size());
	prepared.buffers.clear();
	prepared.images.resize(program.info.images.size());
	prepared.image_sources.resize(program.info.images.size());
	prepared.samplers.clear();
	prepared.shader_data.clear();
	auto& texture_cache = m_context.GetTextureCache();
	// A draw repeating the stage's program and image descriptors binds its last images while they
	// stand (see ImageGroupStands); RebindImages checks them again.
	const auto& group = prepared.image_group;
	prepared.image_group_kept =
	    ImageGroupsEnabled() && group.program == &program &&
	    std::ranges::equal(snapshot.images, group.images) && ImageGroupStands(texture_cache, prepared);
	if (prepared.image_group_kept) {
		for (const auto counter: group.mip_counters) {
			m_context.MarkMipStatsCounter(counter);
		}
		for (const auto& binding: prepared.images) {
			BindImage(binding.image_id, false);
		}
		if (VerifyImageGroups()) [[unlikely]] {
			for (uint32_t i = 0; i < program.info.images.size(); i++) {
				auto           source = prepared.image_sources[i];
				TextureBinding check;
				ResolveTextureInto(program.info.images[i], snapshot.images[i], &source, check);
				const auto& kept = prepared.images[i];
				ReportImageGroupCheck(check.image_id == kept.image_id &&
				                          check.desc.view_info == kept.desc.view_info,
				                      kept.desc.info.data.address);
			}
		}
	}
	for (uint32_t i = 0; !prepared.image_group_kept && i < program.info.images.size(); i++) {
		MarkMipStats(m_context, program.info.images[i], snapshot.images[i]);
		// Consecutive draws mostly bind the same textures.
		auto& previous = prepared.images[i];
		if (ReuseTexture(texture_cache, prepared.image_sources[i], previous,
		                 program.info.images[i], snapshot.images[i])) {
			// As ResolveTexture returns it; RebindImages acquires the views.
			previous.image_view = nullptr;
			previous.layout     = vk::ImageLayout::eUndefined;
			previous.mip_views.clear();
			BindImage(previous.image_id,
			          previous.desc.type == TextureCache::BindingType::Storage);
			continue;
		}
		ResolveTextureInto(program.info.images[i], snapshot.images[i], &prepared.image_sources[i],
		                   previous);
		BindImage(previous.image_id, previous.desc.type == TextureCache::BindingType::Storage);
	}
	g_draw_phases.Mark(DrawPhaseTimer::StageTextures);
	prepared.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(m_context, program, i, snapshot.samplers[i]));
	}
	g_draw_phases.Mark(DrawPhaseTimer::StageSamplers);
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.bindings.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

void RenderExecutor::FindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	PerfStats::Span span(PerfStats::SpanId::BindFindBuffers);
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       cache    = m_context.GetBufferCache();

	// The previous draw's sources of this stage: consecutive draws mostly bind buffers the same
	// cached buffer still covers, which RefindBuffer confirms without the page table.
	auto& sources = prepared.buffer_sources;
	const auto previous_count = sources.size();
	sources.resize(program.info.buffers.size());
	// Read before any MappedExtent below, so an extent remembered with it is never newer than it.
	const auto mapping_epoch = m_context.MappingEpoch();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
		const auto address = descriptor.Base48();
		const auto requested = StorageBufferRequest(m_context, descriptor);
		// A base that maps nothing binds the null buffer rather than faulting; reads then return
		// zero, which is the same answer robust access gives past the end of a short binding.
		const bool remembered = i < previous_count && sources[i].address == address &&
		                        sources[i].requested == requested &&
		                        sources[i].mapping_epoch == mapping_epoch;
		const auto size = requested == 0 ? 0
		                  : remembered   ? sources[i].size
		                                 : m_context.MappedExtent(address, requested);
		if (address == 0 || size == 0) {
			sources[i] = {};
			continue;
		}
		const auto id   = i < previous_count && sources[i].id ? sources[i].id : BufferId {};
		sources[i]      = {address, size,
		                   id ? cache.RefindBuffer(id, address, size) : cache.FindBuffer(address, size),
		                   requested, mapping_epoch};
	}
}

// A DMA read resolves its guest page through the BDA page table. A page no cached buffer covers reads
// as zero and is registered only after the fault buffer is processed, so the first pass over newly
// referenced memory consumes zeros. GTA V hands rasterized glyph bitmaps to its atlas blit shader as
// a 64-bit pointer in user data; a glyph whose bitmap sits on an unseen page is blitted empty and the
// game never redraws it. Cache the pages named by the user data that forms DMA address bases before
// the shader runs.
//
// A DMA store to an uncached page is dropped, and its data never arrives: GTA V's ray-tracing BVH
// builder writes each tree once, through buffer descriptors built from a user-data pointer, and a
// tree missing its nodes sends every later traversal and refit around its lists forever. A base
// that is written through caches WriteWindow bytes from its page. GTA V's vertex writers
// instead pick each destination descriptor from a table at run time; every range the table's
// descriptors name is cached.
void RenderExecutor::PrepareDmaSources(const PreparedBindings& prepared) {
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program = *prepared.runtime->program;
	if (!program.info.uses_dma) {
		return;
	}
	PerfStats::Span span(PerfStats::SpanId::BindDmaSources);
	// Covers a glyph bitmap or a small record run; reads beyond it still take the fault path.
	constexpr uint64_t SourceWindow = 256ull * 1024ull;
	// Covers the largest BVH seen (7.7 MB). Every cached page is write-protected, so a wider window
	// would slow the CPU writing whatever else shares the memory.
	constexpr uint64_t WriteWindow = 16ull * 1024ull * 1024ull;
	auto&              cache        = m_context.GetBufferCache();
	const auto&        user_data    = prepared.runtime->resources->user_data;
	const auto&        writes       = program.info.dma_write_address_bases;
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	for (const auto& source: prepared.buffer_sources) {
		ranges.emplace_back(source.address, source.size);
	}
	std::string cached_writes;
	const auto  in_user_data = [&](uint32_t reg) {
		return reg >= program.user_data_base && reg - program.user_data_base < user_data.size();
	};
	for (const auto& base: program.info.dma_address_bases) {
		if (!in_user_data(base.low) || !in_user_data(base.high)) {
			continue;
		}
		const bool write = std::ranges::find(writes, base) != writes.end();
		const auto low   = user_data[base.low - program.user_data_base];
		// A buffer descriptor's second dword keeps its stride and swizzle above the base.
		const auto high = user_data[base.high - program.user_data_base] & (write ? 0xffffu : ~0u);
		// Guest GPU memory lies below 2^40. A value this close to a 4 GiB boundary is more likely a
		// pair of small integers than a base, and caching the wrong range write-protects live pages.
		if (high == 0 || high > 0xffu || low < 0x10000u) {
			continue;
		}
		const uint64_t address = (static_cast<uint64_t>(high) << 32u) | low;
		const uint64_t page    = address & ~(BufferCache::CACHING_PAGESIZE - 1u);
		const uint64_t start   = page;
		const uint64_t extent =
		    m_context.MappedExtent(start, write ? WriteWindow : SourceWindow);
		if (extent == 0) {
			continue;
		}
		(void)cache.FindBuffer(start, extent);
		ranges.emplace_back(address, extent - (address - start));
		if (write) {
			cached_writes += fmt::format(" 0x{:010x}+0x{:x}", start, extent);
		}
	}
	const auto table_ranges = CacheStoreDescriptorTables(prepared);
	if (table_ranges != 0) {
		cached_writes += fmt::format(" {} ranges its descriptor tables name", table_ranges);
		for (const auto table: program.info.dma_store_descriptor_tables) {
			if (table < prepared.buffer_sources.size()) {
				cached_writes += fmt::format(" (table 0x{:010x}+0x{:x})",
				                             prepared.buffer_sources[table].address,
				                             prepared.buffer_sources[table].size);
			}
		}
	}
	// For a replay of a dump: the descriptor tables behind pointers in the user data and the SRT
	// values the shader reads, which bound buffers do not cover.
	constexpr uint64_t TableWindow = 64ull * 1024ull;
	const auto         add_tables  = [&](std::span<const uint32_t> words) {
		for (size_t index = 0; index + 1u < words.size() && ranges.size() < 96u; index++) {
			const auto low  = words[index];
			const auto high = words[index + 1u];
			if (high == 0 || high > 0xffu || low < 0x10000u) {
				continue;
			}
			const uint64_t pointer = (static_cast<uint64_t>(high) << 32u) | low;
			const uint64_t start   = (pointer & ~uint64_t {0xfff}) - 0x1000u;
			const uint64_t extent  = m_context.MappedExtent(start, TableWindow);
			if (extent != 0) {
				ranges.emplace_back(start, extent);
			}
		}
	};
	add_tables(user_data);
	add_tables(prepared.runtime->resources->flattened_srt);
	RecordDmaDispatch(program.shader_hash, user_data, std::move(ranges));
	if (program.info.indirect_buffer_writes) {
		NoteDmaWriter(program.shader_hash);
	}
	// Each shader that stores through DMA says once what was cached ahead of its stores.
	static std::set<uint64_t> described;
	if (program.info.indirect_buffer_writes && described.insert(program.shader_hash).second) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader 0x{:016x} stores through DMA; cached ahead of its stores:{}\n",
		    program.shader_hash,
		    cached_writes.empty() ? std::string(" nothing, no user-data base found")
		                          : cached_writes));
	}
}

// Caches every range the descriptors in the shader's store-descriptor tables name, and returns how
// many there were. A table is read from guest memory at most every RescanInterval, and its ranges
// are looked up again only when its contents changed.
uint32_t RenderExecutor::CacheStoreDescriptorTables(const PreparedBindings& prepared) {
	const auto& program = *prepared.runtime->program;
	// Covers 262144 descriptors; a destination picked beyond that still takes the fault path.
	constexpr uint64_t MaxTableBytes  = 4ull * 1024ull * 1024ull;
	constexpr uint64_t MaxRangeBytes  = 256ull * 1024ull * 1024ull;
	constexpr auto     RescanInterval = std::chrono::milliseconds(16);
	struct Scan {
		uint64_t                              hash  = 0;
		uint32_t                              named = 0;
		std::chrono::steady_clock::time_point time;
	};
	static std::map<std::pair<uint64_t, uint64_t>, Scan> scans;
	auto&    cache = m_context.GetBufferCache();
	uint32_t total = 0;
	for (const auto table: program.info.dma_store_descriptor_tables) {
		if (table >= prepared.buffer_sources.size()) {
			continue;
		}
		const auto& source = prepared.buffer_sources[table];
		const auto  size   = std::min<uint64_t>(source.size, MaxTableBytes) & ~uint64_t {15};
		if (source.address == 0 || size == 0) {
			continue;
		}
		const std::pair key {source.address, size};
		const auto      now   = std::chrono::steady_clock::now();
		auto            found = scans.find(key);
		if (found != scans.end() && now - found->second.time < RescanInterval) {
			total += found->second.named;
			continue;
		}
		std::vector<uint32_t> words(size / sizeof(uint32_t));
		if (!Libs::LibKernel::Memory::TryReadBacking(source.address, words.data(), size)) {
			continue;
		}
		uint64_t hash = 0xcbf29ce484222325ull;
		for (const auto word: words) {
			hash = (hash ^ word) * 0x100000001b3ull;
		}
		auto& scan = scans[key];
		scan.time  = now;
		if (found != scans.end() && scan.hash == hash) {
			total += scan.named;
			continue;
		}
		scan.hash      = hash;
		uint32_t count = 0;
		for (size_t index = 0; index + 4u <= words.size(); index += 4u) {
			const auto high   = words[index + 1u] & 0xffffu;
			const auto stride = (words[index + 1u] >> 16u) & 0x3fffu;
			const auto records = words[index + 2u];
			const auto flags   = words[index + 3u];
			// A buffer descriptor (type 0) with a base in guest GPU memory.
			if (high == 0 || high > 0xffu || flags == 0 || (flags >> 30u) != 0u || records == 0) {
				continue;
			}
			const uint64_t address = (static_cast<uint64_t>(high) << 32u) | words[index];
			const uint64_t bytes =
			    stride != 0 ? static_cast<uint64_t>(stride) * records : static_cast<uint64_t>(records);
			if (bytes > MaxRangeBytes) {
				continue;
			}
			const auto extent = m_context.MappedExtent(address, bytes);
			if (extent == 0) {
				continue;
			}
			(void)cache.FindBuffer(address, extent);
			count++;
		}
		scan.named = count;
		total += count;
	}
	return total;
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	PerfStats::Span span(PerfStats::SpanId::BindRebindBuffers);
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != program.info.buffers.size());

	prepared.buffers.clear();
	prepared.buffers.reserve(program.info.buffers.size());
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		uint32_t buffer_offset = 0;
		prepared.buffers.push_back(NativeStorageBuffer(m_context, prepared.buffer_sources[i],
		                                               program.info.buffers[i], buffer_offset));
		pack_memory_offset(i, buffer_offset);
	}
	g_draw_phases.Mark(DrawPhaseTimer::BufferViews);
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::FlattenedSrt) != nullptr) {
		prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
	}
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::ShaderData) != nullptr) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

// KYTY_DEBUG_STORAGE_REUSE=0 acquires storage images every draw; KYTY_DEBUG_AB=storagereuse
// alternates.
static bool StorageReuseEnabled() {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_DEBUG_STORAGE_REUSE");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("storagereuse");
	return enabled && !(ab && AbFeatureOff());
}

// KYTY_VERIFY_STORAGE_REUSE=1: a storage binding that keeps its view acquires it anyway and
// reports what FindTexture returned differently or changed (see TextureCache::IsStorageCurrent).
static bool VerifyStorageReuse() {
	static const bool enabled = std::getenv("KYTY_VERIFY_STORAGE_REUSE") != nullptr;
	return enabled;
}

// KYTY_VERIFY_VIEW_REUSE=1: the same for sampled textures, including views that another slot's
// binding of the description acquired (the view memo, see ResolveTextureInto).
static bool VerifyViewReuse() {
	static const bool enabled = std::getenv("KYTY_VERIFY_VIEW_REUSE") != nullptr;
	return enabled;
}

static void CheckViewReuse(TextureCache& cache, const TextureBinding& binding) {
	const bool  storage         = binding.desc.type == TextureCache::BindingType::Storage;
	const auto& image           = cache.GetImage(binding.image_id);
	const bool  gpu_modified    = image.IsGpuModified();
	const bool  buffer_modified = image.IsBufferModified();
	const bool  cpu_dirty       = image.IsCpuDirty();
	const auto  view            = cache.FindTexture(binding.image_id, binding.desc);
	const auto& after           = cache.GetImage(binding.image_id);
	const bool  same            = view == binding.image_view &&
	                  after.IsGpuModified() == gpu_modified &&
	                  after.IsBufferModified() == buffer_modified && after.IsCpuDirty() == cpu_dirty;
	static std::atomic<uint64_t> checked_counts[2] {};
	static std::atomic<uint64_t> missed_counts[2] {};
	auto&       checked = checked_counts[storage ? 1 : 0];
	auto&       missed  = missed_counts[storage ? 1 : 0];
	const char* kind    = storage ? "storage" : "texture";
	const auto  count   = checked.fetch_add(1, std::memory_order_relaxed) + 1;
	if (!same && missed.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("%s-reuse verify: 0x%016" PRIx64
		            " kept view would have missed a change (view %s, gpu %d->%d, buffer %d->%d, "
		            "cpu %d->%d)\n",
		            kind, image.info.data.address, view == binding.image_view ? "same" : "differs",
		            gpu_modified, after.IsGpuModified(), buffer_modified, after.IsBufferModified(),
		            cpu_dirty, after.IsCpuDirty());
	}
	if (count % 100000 == 0) {
		std::printf("%s-reuse verify: reuses=%" PRIu64 " missed=%" PRIu64 "\n", kind, count,
		            missed.load(std::memory_order_relaxed));
		std::fflush(stdout);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	PerfStats::Span span(PerfStats::SpanId::BindRebindImages);
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	// Bindings assembled without PrepareBindings have no recorded sources.
	prepared.image_sources.resize(images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// PrepareBindings kept the group; BDA synchronization since may have dirtied an image.
	const bool group_kept     = prepared.image_group_kept;
	prepared.image_group_kept = false;
	if (group_kept && ImageGroupStands(texture_cache, prepared)) {
		if (VerifyImageGroups()) [[unlikely]] {
			for (const auto& binding: images) {
				CheckViewReuse(texture_cache, binding);
			}
		}
		return;
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			ResolveTextureInto(program.info.images[i], snapshot.images[i],
			                   &prepared.image_sources[i], images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::DynamicStorage) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else if (auto& source = prepared.image_sources[i];
		           TextureReuseEnabled() && source.view_image == binding.image_id &&
		           source.view_info == binding.desc.view_info &&
		           (binding.desc.type == TextureCache::BindingType::Texture
		                ? (source.null_image && source.view_generation != 0 && NullReuseEnabled()) ||
		                      texture_cache.IsTextureCurrent(binding.image_id, source.view_generation)
		                : binding.desc.type == TextureCache::BindingType::Storage &&
		                      StorageReuseEnabled() &&
		                      texture_cache.IsStorageCurrent(binding.image_id,
		                                                     source.view_generation))) {
			// A clean image keeps the view FindTexture returned for this binding before.
			binding.image_view = source.view;
			if (binding.desc.type == TextureCache::BindingType::Storage ? VerifyStorageReuse()
			                                                              : VerifyViewReuse())
			    [[unlikely]] {
				CheckViewReuse(texture_cache, binding);
			}
		} else {
			const auto generation = texture_cache.ImageGeneration(binding.image_id);
			binding.image_view    = texture_cache.FindTexture(binding.image_id, binding.desc);
			source.view_generation =
			    binding.desc.type == TextureCache::BindingType::Texture ||
			            binding.desc.type == TextureCache::BindingType::Storage
			        ? generation
			        : 0;
			source.view       = binding.image_view;
			source.view_image = binding.image_id;
			source.view_info  = binding.desc.view_info;
			if (source.view_memo != nullptr && source.view_generation != 0) {
				*source.view_memo = {source.view, source.view_image, source.view_info,
				                     source.view_generation};
			}
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
		image.usage.texture |= !storage;
	}
	RecordImageGroup(m_context, prepared, program, snapshot.images);
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool uses_dma = false;
	for (auto* stage: stages) {
		FindBuffers(*stage);
		uses_dma |= stage->runtime->program->info.uses_dma;
	}
	if (uses_dma) {
		DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::Bda);
		for (auto* stage: stages) {
			PrepareDmaSources(*stage);
		}
		m_context.PrepareBda();
	}
	g_draw_phases.Mark(DrawPhaseTimer::FindBuffers);
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	g_draw_phases.Mark(DrawPhaseTimer::RebindImages);
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

namespace {

// A draw's push constants and push descriptor sets as CommitBindings hands them to the thread
// recording its command buffer (see ReserveRecordedCall): this header, then the writes of set 0
// and set 1, then each write's image or buffer infos in turn.
struct RecordedBindings {
	VkDevice                       device      = VK_NULL_HANDLE;
	VkPipelineLayout               layout      = VK_NULL_HANDLE;
	VkPipelineBindPoint            bind_point  = VK_PIPELINE_BIND_POINT_GRAPHICS;
	VkShaderStageFlags             push_stages = 0; // 0: no push constants.
	std::array<uint32_t, 2>        set_writes {};
	// A set that does not push: allocated by CommitBindings, updated and bound by the recording
	// thread. VK_NULL_HANDLE: the set pushes.
	std::array<VkDescriptorSet, 2> sets {};
	ShaderRecompiler::IR::PushData push_data;
};

struct RecordedWrite {
	uint32_t         binding = 0;
	uint32_t         count   = 0;
	VkDescriptorType type    = VK_DESCRIPTOR_TYPE_SAMPLER;
	uint32_t         images  = 0; // Whether the infos are image infos rather than buffer infos.
};

static_assert(sizeof(vk::DescriptorImageInfo) == sizeof(vk::DescriptorBufferInfo));
static_assert(sizeof(RecordedBindings) % alignof(vk::DescriptorImageInfo) == 0 &&
              sizeof(RecordedWrite) % alignof(vk::DescriptorImageInfo) == 0);
constexpr size_t RecordedInfoBytes = sizeof(vk::DescriptorImageInfo);

// On the recording thread: the calls CommitBindings would have made.
void RunRecordedBindings(VkCommandBuffer command_buffer, const uint8_t* payload) {
	const auto& header = *reinterpret_cast<const RecordedBindings*>(payload);
	const auto* writes = reinterpret_cast<const RecordedWrite*>(payload + sizeof(RecordedBindings));
	const auto  count  = header.set_writes[0] + header.set_writes[1];
	const auto* infos  = payload + sizeof(RecordedBindings) + count * sizeof(RecordedWrite);
	thread_local std::vector<vk::WriteDescriptorSet> vk_writes;
	vk_writes.resize(count);
	for (uint32_t i = 0; i < count; i++) {
		const auto& write     = writes[i];
		auto&       vk_write  = vk_writes[i];
		vk_write              = vk::WriteDescriptorSet {};
		vk_write.dstBinding      = write.binding;
		vk_write.descriptorCount = write.count;
		vk_write.descriptorType  = static_cast<vk::DescriptorType>(write.type);
		if (write.images != 0) {
			vk_write.pImageInfo = reinterpret_cast<const vk::DescriptorImageInfo*>(infos);
		} else {
			vk_write.pBufferInfo = reinterpret_cast<const vk::DescriptorBufferInfo*>(infos);
		}
		infos += write.count * RecordedInfoBytes;
	}
	const vk::CommandBuffer  buffer(command_buffer);
	const vk::PipelineLayout layout(header.layout);
	const auto               bind_point = static_cast<vk::PipelineBindPoint>(header.bind_point);
	if (header.push_stages != 0) {
		buffer.pushConstants(layout, static_cast<vk::ShaderStageFlags>(header.push_stages), 0,
		                     sizeof(header.push_data), header.push_data.dwords.data());
	}
	auto* set_writes = vk_writes.data();
	for (uint32_t set_index = 0; set_index < 2; set_index++) {
		const auto count_in_set = header.set_writes[set_index];
		if (count_in_set != 0) {
			const vk::DescriptorSet set(header.sets[set_index]);
			if (!set) {
				buffer.pushDescriptorSetKHR(bind_point, layout, set_index, count_in_set, set_writes);
			} else {
				// A freshly allocated set, bound nowhere yet: updating it here, before its bind,
				// is what CommitBindings did on its own thread.
				for (uint32_t i = 0; i < count_in_set; i++) {
					set_writes[i].dstSet = set;
				}
				vk::Device(header.device).updateDescriptorSets(count_in_set, set_writes, 0, nullptr);
				buffer.bindDescriptorSets(bind_point, layout, set_index, 1, &set, 0, nullptr);
			}
		}
		set_writes += count_in_set;
	}
}

// Hands the push constants (with nonempty stages) and both descriptor sets to the thread that
// records `buffer`, as one payload call instead of one deep-copied call each; false when its
// recording is not deferred, and the caller makes the calls. A set that does not push
// (push[set] false) gets its set from allocate(set) here, once the call is certain, and the
// recording thread updates and binds it. KYTY_DEBUG_AB=pushpayload has the caller make the
// calls in every other window.
template <typename Allocate>
bool RecordPushedBindings(vk::CommandBuffer buffer, vk::Device device,
                          vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                          vk::ShaderStageFlags push_stages,
                          const ShaderRecompiler::IR::PushData&    push_data,
                          std::span<const vk::WriteDescriptorSet> set0,
                          std::span<const vk::WriteDescriptorSet> set1, std::array<bool, 2> push,
                          Allocate&& allocate) {
	static const bool ab = AbSelected("pushpayload");
	if (ab && AbFeatureOff()) {
		return false;
	}
	size_t info_count = 0;
	for (const auto* set: {&set0, &set1}) {
		for (const auto& write: *set) {
			// Each write carries its infos in one of the two arrays; leave anything else to the
			// calls as they were.
			if ((write.pImageInfo == nullptr) == (write.pBufferInfo == nullptr)) {
				return false;
			}
			info_count += write.descriptorCount;
		}
	}
	const auto write_count = set0.size() + set1.size();
	if (write_count == 0 && !push_stages) {
		return true;
	}
	auto* payload =
	    ReserveRecordedCall(static_cast<VkCommandBuffer>(buffer),
	                        sizeof(RecordedBindings) + write_count * sizeof(RecordedWrite) +
	                            info_count * RecordedInfoBytes);
	if (payload == nullptr) {
		return false;
	}
	auto& header       = *::new (payload) RecordedBindings;
	header.device      = static_cast<VkDevice>(device);
	header.layout      = static_cast<VkPipelineLayout>(layout);
	header.bind_point  = static_cast<VkPipelineBindPoint>(bind_point);
	header.push_stages = static_cast<VkShaderStageFlags>(push_stages);
	header.set_writes  = {static_cast<uint32_t>(set0.size()), static_cast<uint32_t>(set1.size())};
	for (uint32_t set_index = 0; set_index < 2; set_index++) {
		if (header.set_writes[set_index] != 0 && !push[set_index]) {
			header.sets[set_index] = static_cast<VkDescriptorSet>(allocate(set_index));
		}
	}
	header.push_data   = push_data;
	auto* writes = reinterpret_cast<RecordedWrite*>(payload + sizeof(RecordedBindings));
	auto* infos  = payload + sizeof(RecordedBindings) + write_count * sizeof(RecordedWrite);
	for (const auto* set: {&set0, &set1}) {
		for (const auto& write: *set) {
			const bool images = write.pImageInfo != nullptr;
			*writes++         = {write.dstBinding, write.descriptorCount,
			                     static_cast<VkDescriptorType>(write.descriptorType), images ? 1u : 0u};
			const size_t bytes = write.descriptorCount * RecordedInfoBytes;
			std::memcpy(infos,
			            images ? static_cast<const void*>(write.pImageInfo)
			                   : static_cast<const void*>(write.pBufferInfo),
			            bytes);
			infos += bytes;
		}
	}
	CommitRecordedCall(RunRecordedBindings);
	return true;
}

} // namespace

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	auto   vk_buffer        = buffer.Handle();
	size_t descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		write_count += program.bindings.descriptors.size();
		for (const auto& binding: program.bindings.descriptors) {
			descriptor_count += NativeDescriptorCount(binding);
		}
		const auto shader_stage = NativeShaderStage(program.stage);
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	m_descriptor_buffers.clear();
	m_descriptor_images.clear();
	m_descriptor_writes.clear();
	m_descriptor_buffers.reserve(descriptor_count);
	m_descriptor_images.reserve(descriptor_count);
	m_descriptor_writes.reserve(write_count);
	size_t pixel_write_start = write_count;

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		// The pixel stage comes last, so its writes form the tail committed to set 1.
		EXIT_IF(pixel_write_start != write_count);
		if (ShaderRecompiler::IR::NativeDescriptorSet(program.stage) != 0) {
			pixel_write_start = m_descriptor_writes.size();
		}
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& image   = m_context.GetTextureCache().GetImage(descriptors.images[i].image_id);
			auto& binding = descriptors.images[i];
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.info.data.Empty()) {
				image.Transit(vk::ImageLayout::eGeneral,
				              storage ? vk::AccessFlagBits2::eShaderRead |
				                            vk::AccessFlagBits2::eShaderWrite
				                      : vk::AccessFlagBits2::eShaderRead,
				              range, vk_buffer);
			} else if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                   : vk::ImageLayout::eShaderReadOnlyOptimal,
				              vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		}

		m_image_occurrences.assign(descriptors.images.size(), 0);
		for (const auto& binding: program.bindings.descriptors) {
			vk::WriteDescriptorSet write {};
			write.dstBinding     = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			write.descriptorType = NativeDescriptorType(binding.kind);
			write.descriptorCount   = NativeDescriptorCount(binding);
			const auto buffer_start = m_descriptor_buffers.size();
			const auto image_start  = m_descriptor_images.size();
			if (ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				for (const auto resource: binding.resources) {
					m_descriptor_images.push_back(MakeImageInfo(
					    descriptors.images.at(resource), m_image_occurrences.at(resource)++));
				}
			} else {
				switch (binding.kind) {
					case BindingKind::Buffers:
						for (const auto resource: binding.resources) {
							const auto& view = descriptors.buffers.at(resource);
							// With hardware bounds, a range without a whole dword is a null
							// descriptor (see NativeStorageBuffer).
							EXIT_IF(view.buffer == nullptr &&
							        !m_context.GetGraphics().hardware_storage_buffer_bounds);
							m_descriptor_buffers.push_back(view);
						}
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer:
					case BindingKind::LoopWatchdog: {
						auto&       cache      = m_context.GetBufferCache();
						const auto* bda_buffer = binding.kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                         : binding.kind == BindingKind::FaultBuffer
						                             ? cache.GetFaultBuffer()
						                             : cache.GetLoopWatchdogBuffer();
						m_descriptor_buffers.emplace_back(bda_buffer->Handle(), 0,
						                                  bda_buffer->Size());
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (binding.kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (binding.kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						}
						EXIT_IF(view->buffer == nullptr);
						m_descriptor_buffers.push_back(*view);
						break;
					}
					case BindingKind::Samplers:
						for (const auto resource: binding.resources) {
							const auto sampler = descriptors.samplers.at(resource);
							EXIT_IF(sampler == nullptr);
							m_descriptor_images.emplace_back(sampler, nullptr,
							                                 vk::ImageLayout::eUndefined);
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (m_descriptor_buffers.size() != buffer_start) {
				write.pBufferInfo = m_descriptor_buffers.data() + buffer_start;
			}
			if (m_descriptor_images.size() != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < descriptors.images.size(); i++) {
			const auto expected =
			    descriptors.images[i].mip_views.empty()
			        ? 1u
			        : static_cast<uint32_t>(descriptors.images[i].mip_views.size());
			EXIT_IF(m_image_occurrences[i] != expected);
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	if (has_push_data && pipeline.push_constant_stages) {
		// The stages must match the layout's push constant range exactly.
		push_stages = pipeline.push_constant_stages;
	}
	// Set 0 holds compute or vertex-side descriptors; the pixel shader's tail goes to set 1.
	const auto set0 = std::span(m_descriptor_writes).first(pixel_write_start);
	const auto set1 = std::span(m_descriptor_writes).subspan(pixel_write_start);
	EXIT_IF((!set0.empty() && pipeline.descriptor_set_layout == nullptr) ||
	        (!set1.empty() && pipeline.pixel_set_layout == nullptr));
	// A graphics layout pushes at most one of its two sets, so the payload carries the other as
	// an allocated set (see RecordPushedBindings).
	if (RecordPushedBindings(
	        vk_buffer, m_context.GetGraphics().device, pipeline_bind_point, pipeline.pipeline_layout,
	        has_push_data ? push_stages : vk::ShaderStageFlags {}, push_data, set0, set1,
	        {pipeline.uses_push_descriptors, pipeline.pixel_uses_push}, [&](uint32_t set_index) {
		        return m_context.GetDescriptorHeap().Commit(
		            set_index == 0 ? pipeline.descriptor_set_layout : pipeline.pixel_set_layout);
	        })) {
		return;
	}
	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	}

	const auto commit_set = [&](uint32_t set_index, vk::DescriptorSetLayout layout, bool push,
	                            std::span<vk::WriteDescriptorSet> writes) {
		if (writes.empty()) {
			return;
		}
		EXIT_IF(layout == nullptr);
		if (push) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, set_index,
			                               static_cast<uint32_t>(writes.size()), writes.data());
			return;
		}
		const auto set = m_context.GetDescriptorHeap().Commit(layout);
		for (auto& write: writes) {
			write.dstSet = set;
		}
		m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()),
		                                                    writes.data(), 0, nullptr);
		vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, set_index, 1,
		                             &set, 0, nullptr);
	};
	commit_set(0, pipeline.descriptor_set_layout, pipeline.uses_push_descriptors, set0);
	commit_set(1, pipeline.pixel_set_layout, pipeline.pixel_uses_push, set1);
}

} // namespace Libs::Graphics
