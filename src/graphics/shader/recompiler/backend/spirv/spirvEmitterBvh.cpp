#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include <array>
#include <limits>

// RDNA2 IMAGE_BVH_INTERSECT_RAY and IMAGE_BVH64_INTERSECT_RAY test a ray against one node of a BVH
// in memory. The node layout and the results follow the hardware as Mesa's RADV describes it, and
// its software fallback for the instruction:
// - a node pointer's low three bits are the node type (0-3 triangle, 4 box16, 5 box32); the rest
//   is the node's offset from the BVH base in 8-byte units;
// - a box node returns its four child pointers, the ones the ray hits first, nearest first when
//   the descriptor enables sorting, and 0xffffffff for the rest;
// - a triangle node returns t and the barycentrics as numerators over one denominator, or its
//   triangle id and hit status when the descriptor asks for that instead.
namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

constexpr uint32_t Box16Node      = 4;
constexpr uint32_t Box32Node      = 5;
constexpr uint32_t InvalidNode    = 0xffffffffu;
constexpr uint32_t InfinityBits   = 0x7f800000u;
constexpr uint32_t OneBits        = 0x3f800000u;
constexpr uint32_t Box32BoundsAt  = 4;
constexpr uint32_t Box16BoundsAt  = 4;
constexpr uint32_t TriangleIdWord = 12;

using Vec3 = std::array<uint32_t, 3>;

struct Ray {
	uint32_t extent = 0;
	Vec3     origin {};
	Vec3     dir {};
	Vec3     inv_dir {};
};

uint32_t U64Constant(EmitterState& state, uint64_t value) {
	return state.builder.Constant(spv::OpConstant, TypeScalarU64(state),
	                              static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32u));
}

uint32_t Extract(EmitterState& state, uint32_t type, uint32_t composite, uint32_t index) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, type, result, composite, index);
	return result;
}

uint32_t FloatOp(EmitterState& state, spv::Op op, uint32_t lhs, uint32_t rhs) {
	return Binary(state, op, TypeF32(state), lhs, rhs);
}

uint32_t Compare(EmitterState& state, spv::Op op, uint32_t lhs, uint32_t rhs) {
	return Binary(state, op, TypeBool(state), lhs, rhs);
}

uint32_t And(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return Binary(state, spv::OpLogicalAnd, TypeBool(state), lhs, rhs);
}

uint32_t Or(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return Binary(state, spv::OpLogicalOr, TypeBool(state), lhs, rhs);
}

uint32_t NMin(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return EmitGlsl<GLSLstd450NMin, IR::Type::F32>(state, lhs, rhs);
}

uint32_t NMax(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return EmitGlsl<GLSLstd450NMax, IR::Type::F32>(state, lhs, rhs);
}

uint32_t AsFloat(EmitterState& state, uint32_t bits) {
	return EmitBitcastU32ToF32(state, bits);
}

uint32_t AsBits(EmitterState& state, uint32_t value) {
	return EmitBitcastF32ToU32(state, value);
}

uint32_t HighHalfToF32(EmitterState& state, uint32_t bits) {
	return EmitF16BitsToF32(state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), bits,
	                                      ConstantU32(state, 16)));
}

uint32_t ConstructU32x4(EmitterState& state, const std::array<uint32_t, 4>& values) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Composite(state, 4), result,
	                          values[0], values[1], values[2], values[3]);
	return result;
}

// Component k (0, 1 or 2, known only at run time) of a vector.
uint32_t Component(EmitterState& state, const Vec3& vector, uint32_t k) {
	const auto is_x = Compare(state, spv::OpIEqual, k, ConstantU32(state, 0));
	const auto is_y = Compare(state, spv::OpIEqual, k, ConstantU32(state, 1));
	return Select(state, TypeF32(state), is_x, vector[0],
	              Select(state, TypeF32(state), is_y, vector[1], vector[2]));
}

// The DWORDs of one node, through the host address of its page. Nodes are at least 64-byte
// aligned, so a node never spans two pages.
struct NodeReader {
	EmitterState& state;
	uint32_t      host = 0;

	uint32_t Word(uint32_t index) const {
		const auto address = Binary(state, spv::OpIAdd, TypeScalarU64(state), host,
		                            U64Constant(state, index * 4ull));
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
		                          address);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
		                          spv::MemoryAccessAlignedMask, 4u);
		return value;
	}

	uint32_t Float(uint32_t index) const { return AsFloat(state, Word(index)); }
};

uint32_t BoxResult(EmitterState& state, const NodeReader& node, bool half, const Ray& ray,
                   uint32_t sort) {
	const auto infinity = AsFloat(state, ConstantU32(state, InfinityBits));
	const auto zero     = ConstantF32Value(state, 0.0f);
	std::array<uint32_t, 4> keys {};
	std::array<uint32_t, 4> children {};
	for (uint32_t i = 0; i < 4u; i++) {
		const auto child = node.Word(i);
		// Minimum x, y, z, then maximum x, y, z.
		std::array<uint32_t, 6> bounds {};
		if (half) {
			for (uint32_t word = 0; word < 3u; word++) {
				const auto packed     = node.Word(Box16BoundsAt + i * 3u + word);
				bounds[word * 2u]     = EmitF16BitsToF32(state, packed);
				bounds[word * 2u + 1u] = HighHalfToF32(state, packed);
			}
		} else {
			for (uint32_t k = 0; k < 6u; k++) {
				bounds[k] = node.Float(Box32BoundsAt + i * 6u + k);
			}
		}
		std::array<uint32_t, 3> near {};
		std::array<uint32_t, 3> far {};
		for (uint32_t axis = 0; axis < 3u; axis++) {
			const auto t0 = FloatOp(state, spv::OpFMul,
			                        FloatOp(state, spv::OpFSub, bounds[axis], ray.origin[axis]),
			                        ray.inv_dir[axis]);
			const auto t1 = FloatOp(state, spv::OpFMul,
			                        FloatOp(state, spv::OpFSub, bounds[3u + axis], ray.origin[axis]),
			                        ray.inv_dir[axis]);
			near[axis] = NMin(state, t0, t1);
			far[axis]  = NMax(state, t0, t1);
		}
		const auto t_min = NMax(state, NMax(state, near[0], near[1]), near[2]);
		const auto t_max = NMin(state, NMin(state, far[0], far[1]), far[2]);
		// An unused child has a NaN box or no pointer.
		const auto used = And(state, Unary(state, spv::OpLogicalNot, TypeBool(state),
		                                   Unary(state, spv::OpIsNan, TypeBool(state), bounds[0])),
		                      Compare(state, spv::OpINotEqual, child,
		                              ConstantU32(state, InvalidNode)));
		const auto hit = And(
		    state, used,
		    And(state,
		        Compare(state, spv::OpFOrdGreaterThanEqual, t_max, NMax(state, t_min, zero)),
		        Compare(state, spv::OpFOrdLessThanEqual, t_min, ray.extent)));
		keys[i]     = Select(state, TypeF32(state), hit, t_min, infinity);
		children[i] = Select(state, TypeU32(state), hit, child, ConstantU32(state, InvalidNode));
	}
	auto       sorted_keys     = keys;
	auto       sorted_children = children;
	const auto order           = [&](uint32_t a, uint32_t b) {
		const auto swap = Compare(state, spv::OpFOrdLessThan, sorted_keys[b], sorted_keys[a]);
		const auto key_a   = Select(state, TypeF32(state), swap, sorted_keys[b], sorted_keys[a]);
		const auto key_b   = Select(state, TypeF32(state), swap, sorted_keys[a], sorted_keys[b]);
		const auto child_a =
		    Select(state, TypeU32(state), swap, sorted_children[b], sorted_children[a]);
		const auto child_b =
		    Select(state, TypeU32(state), swap, sorted_children[a], sorted_children[b]);
		sorted_keys[a]     = key_a;
		sorted_keys[b]     = key_b;
		sorted_children[a] = child_a;
		sorted_children[b] = child_b;
	};
	order(0, 1);
	order(2, 3);
	order(0, 2);
	order(1, 3);
	order(1, 2);
	std::array<uint32_t, 4> result {};
	for (uint32_t i = 0; i < 4u; i++) {
		result[i] = Select(state, TypeU32(state), sort, sorted_children[i], children[i]);
	}
	return ConstructU32x4(state, result);
}

// Watertight ray/triangle intersection (Woop, Benthin and Wald, JCGT 2013), as RADV emulates it.
uint32_t TriangleResult(EmitterState& state, const NodeReader& node, const Ray& ray,
                        uint32_t return_ij) {
	const auto f32  = TypeF32(state);
	const auto u32  = TypeU32(state);
	const auto zero = ConstantF32Value(state, 0.0f);
	std::array<Vec3, 3> vertices {};
	for (uint32_t vertex = 0; vertex < 3u; vertex++) {
		for (uint32_t axis = 0; axis < 3u; axis++) {
			vertices[vertex][axis] = node.Float(vertex * 3u + axis);
		}
	}
	// The dimension where the ray direction is largest becomes z.
	Vec3 abs_dir {};
	for (uint32_t axis = 0; axis < 3u; axis++) {
		abs_dir[axis] = EmitFAbsValue(state, ray.dir[axis]);
	}
	const auto x_ge_y = Compare(state, spv::OpFOrdGreaterThanEqual, abs_dir[0], abs_dir[1]);
	const auto x_ge_z = Compare(state, spv::OpFOrdGreaterThanEqual, abs_dir[0], abs_dir[2]);
	const auto y_ge_z = Compare(state, spv::OpFOrdGreaterThanEqual, abs_dir[1], abs_dir[2]);
	const auto kz =
	    Select(state, u32, x_ge_y,
	           Select(state, u32, x_ge_z, ConstantU32(state, 0), ConstantU32(state, 2)),
	           Select(state, u32, y_ge_z, ConstantU32(state, 1), ConstantU32(state, 2)));
	const auto next = [&](uint32_t k) {
		return Select(state, u32, Compare(state, spv::OpIEqual, k, ConstantU32(state, 2)),
		              ConstantU32(state, 0),
		              Binary(state, spv::OpIAdd, u32, k, ConstantU32(state, 1)));
	};
	auto       kx    = next(kz);
	auto       ky    = next(kx);
	const auto dir_z = Component(state, ray.dir, kz);
	// Swapping x and y keeps the winding order when the ray points along -z.
	const auto swap  = Compare(state, spv::OpFOrdLessThan, dir_z, zero);
	const auto swapped_x = Select(state, u32, swap, ky, kx);
	ky                   = Select(state, u32, swap, kx, ky);
	kx                   = swapped_x;

	const auto sz = FloatOp(state, spv::OpFDiv, ConstantF32Value(state, 1.0f), dir_z);
	const auto sx = FloatOp(state, spv::OpFMul, Component(state, ray.dir, kx), sz);
	const auto sy = FloatOp(state, spv::OpFMul, Component(state, ray.dir, ky), sz);

	std::array<Vec3, 3> relative {};
	for (uint32_t vertex = 0; vertex < 3u; vertex++) {
		for (uint32_t axis = 0; axis < 3u; axis++) {
			relative[vertex][axis] =
			    FloatOp(state, spv::OpFSub, vertices[vertex][axis], ray.origin[axis]);
		}
	}
	std::array<uint32_t, 3> px {};
	std::array<uint32_t, 3> py {};
	std::array<uint32_t, 3> pz {};
	for (uint32_t vertex = 0; vertex < 3u; vertex++) {
		const auto z = Component(state, relative[vertex], kz);
		px[vertex]   = FloatOp(state, spv::OpFSub, Component(state, relative[vertex], kx),
		                       FloatOp(state, spv::OpFMul, sx, z));
		py[vertex]   = FloatOp(state, spv::OpFSub, Component(state, relative[vertex], ky),
		                       FloatOp(state, spv::OpFMul, sy, z));
		pz[vertex]   = FloatOp(state, spv::OpFMul, sz, z);
	}
	const auto cross = [&](uint32_t a, uint32_t b) {
		return FloatOp(state, spv::OpFSub, FloatOp(state, spv::OpFMul, px[a], py[b]),
		               FloatOp(state, spv::OpFMul, py[a], px[b]));
	};
	const auto u = cross(2, 1);
	const auto v = cross(0, 2);
	const auto w = cross(1, 0);

	const auto any = [&](spv::Op op) {
		return Or(state, Or(state, Compare(state, op, u, zero), Compare(state, op, v, zero)),
		          Compare(state, op, w, zero));
	};
	const auto inside = Unary(state, spv::OpLogicalNot, TypeBool(state),
	                          And(state, any(spv::OpFOrdLessThan), any(spv::OpFOrdGreaterThan)));
	const auto det = FloatOp(state, spv::OpFAdd, u, FloatOp(state, spv::OpFAdd, v, w));
	const auto t   = FloatOp(state, spv::OpFAdd,
	                         FloatOp(state, spv::OpFAdd, FloatOp(state, spv::OpFMul, u, pz[0]),
	                                 FloatOp(state, spv::OpFMul, v, pz[1])),
	                         FloatOp(state, spv::OpFMul, w, pz[2]));
	const auto t_signed =
	    FloatOp(state, spv::OpFMul, EmitGlsl<GLSLstd450FSign, IR::Type::F32>(state, det), t);
	const auto ahead = Unary(state, spv::OpLogicalNot, TypeBool(state),
	                         Compare(state, spv::OpFOrdLessThan, t_signed, zero));
	const auto hit   = And(state, inside, ahead);

	const auto hit_t   = Select(state, u32, hit, AsBits(state, t), ConstantU32(state, InfinityBits));
	const auto hit_det = Select(state, u32, hit, AsBits(state, det), ConstantU32(state, OneBits));
	const auto hit_i   = Select(state, u32, hit, AsBits(state, v), ConstantU32(state, 0));
	const auto hit_j   = Select(state, u32, hit, AsBits(state, w), ConstantU32(state, 0));
	const auto hit_status =
	    Select(state, u32, hit, ConstantU32(state, 1), ConstantU32(state, 0));
	const auto triangle_id = node.Word(TriangleIdWord);
	return ConstructU32x4(state, {hit_t, hit_det, Select(state, u32, return_ij, hit_i, triangle_id),
	                              Select(state, u32, return_ij, hit_j, hit_status)});
}

} // namespace

uint32_t EmitBvhIntersectRay(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto flags  = inst.Arg(2).U32();
	const bool bvh64  = (flags & 1u) != 0u;
	const bool a16    = (flags & 2u) != 0u;
	const auto u32    = TypeU32(state);
	const auto u64    = TypeScalarU64(state);
	const auto result_type = TypeU32Composite(state, 4);

	const auto descriptor = ctx.Arg(inst, 0);
	std::array<uint32_t, 4> desc {};
	for (uint32_t i = 0; i < 4u; i++) {
		desc[i] = Extract(state, u32, descriptor, i);
	}
	const auto* address = ctx.ImageAddress(inst.Arg(1));
	uint32_t    cursor  = 0;
	const auto  next    = [&]() { return ctx.Arg(*address, cursor++); };
	const auto  node_lo = next();
	const auto  node_hi = bvh64 ? next() : ConstantU32(state, 0);
	Ray         ray;
	ray.extent = AsFloat(state, next());
	for (auto& component: ray.origin) {
		component = AsFloat(state, next());
	}
	if (a16) {
		// dir.xy, dir.z and inv_dir.x, inv_dir.yz as packed halves.
		std::array<uint32_t, 3> packed {next(), next(), next()};
		std::array<uint32_t, 6> halves {};
		for (uint32_t word = 0; word < 3u; word++) {
			halves[word * 2u]      = EmitF16BitsToF32(state, packed[word]);
			halves[word * 2u + 1u] = HighHalfToF32(state, packed[word]);
		}
		ray.dir     = {halves[0], halves[1], halves[2]};
		ray.inv_dir = {halves[3], halves[4], halves[5]};
	} else {
		for (auto& component: ray.dir) {
			component = AsFloat(state, next());
		}
		for (auto& component: ray.inv_dir) {
			component = AsFloat(state, next());
		}
	}

	const auto zero = ConstantU32(state, 0);
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, 3), result_type, ConstructU32x4(state, {zero, zero, zero, zero}),
	    [&]() {
		    // The descriptor holds the 256-byte-aligned base address in bits [39:0] >> 8.
		    const auto base_high = Binary(state, spv::OpBitwiseAnd, u32, desc[1],
		                                  ConstantU32(state, 0xffu));
		    const auto base_words =
		        Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, desc[0]),
		               Binary(state, spv::OpShiftLeftLogical, u64,
		                      Unary(state, spv::OpUConvert, u64, base_high),
		                      U64Constant(state, 32)));
		    const auto base =
		        Binary(state, spv::OpShiftLeftLogical, u64, base_words, U64Constant(state, 8));
		    const auto node =
		        Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, node_lo),
		               Binary(state, spv::OpShiftLeftLogical, u64,
		                      Unary(state, spv::OpUConvert, u64, node_hi),
		                      U64Constant(state, 32)));
		    const auto offset = Binary(
		        state, spv::OpShiftLeftLogical, u64,
		        Binary(state, spv::OpBitwiseAnd, u64, node, U64Constant(state, ~uint64_t {7})),
		        U64Constant(state, 3));
		    const auto guest = Binary(state, spv::OpIAdd, u64, base, offset);
		    const auto type =
		        Binary(state, spv::OpBitwiseAnd, u32, node_lo, ConstantU32(state, 7));
		    const auto host    = EmitGuestToHostAddress(ctx, guest);
		    const auto present = Compare(state, spv::OpINotEqual, host, U64Constant(state, 0));
		    const NodeReader reader {state, host};
		    const auto sort = Compare(state, spv::OpINotEqual,
		                              Binary(state, spv::OpShiftRightLogical, u32, desc[1],
		                                     ConstantU32(state, 31)),
		                              zero);
		    const auto return_ij =
		        Compare(state, spv::OpINotEqual,
		                Binary(state, spv::OpBitwiseAnd, u32, desc[3], ConstantU32(state, 1u << 24u)),
		                zero);
		    const auto invalid  = ConstantU32(state, InvalidNode);
		    const auto box_miss = ConstructU32x4(state, {invalid, invalid, invalid, invalid});
		    const auto triangle_miss = ConstructU32x4(
		        state, {ConstantU32(state, InfinityBits), ConstantU32(state, OneBits), zero, zero});
		    const auto is_type = [&](uint32_t value) {
			    return Compare(state, spv::OpIEqual, type, ConstantU32(state, value));
		    };
		    const auto box16 = EmitValueOrDefaultIfCondition(
		        state, And(state, present, is_type(Box16Node)), result_type, box_miss,
		        [&]() { return BoxResult(state, reader, true, ray, sort); });
		    const auto boxes = EmitValueOrDefaultIfCondition(
		        state, And(state, present, is_type(Box32Node)), result_type, box16,
		        [&]() { return BoxResult(state, reader, false, ray, sort); });
		    return EmitValueOrDefaultIfCondition(
		        state, Compare(state, spv::OpULessThan, type, ConstantU32(state, 4)), result_type,
		        boxes, [&]() {
			        return EmitValueOrDefaultIfCondition(
			            state, present, result_type, triangle_miss,
			            [&]() { return TriangleResult(state, reader, ray, return_ij); });
		        });
	    });
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
