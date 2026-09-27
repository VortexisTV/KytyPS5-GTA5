#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

// Double-precision values travel through the IR as their U64 bit patterns; only these operations
// see them as doubles.
namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t TypeF64(EmitterState& state) {
	state.builder.RequireCapability(spv::CapabilityFloat64);
	return state.builder.Type(spv::OpTypeFloat, 64);
}

uint32_t AsDouble(EmitterState& state, uint32_t bits) {
	return Unary(state, spv::OpBitcast, TypeF64(state), bits);
}

uint32_t AsBits(EmitterState& state, uint32_t value) {
	return Unary(state, spv::OpBitcast, TypeU64(state), value);
}

} // namespace

uint32_t EmitFPFma64(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF64(state), result, GlslStd450(state),
	                          GLSLstd450Fma, AsDouble(state, a), AsDouble(state, b),
	                          AsDouble(state, c));
	return AsBits(state, result);
}

uint32_t EmitFPFract64(EmitterState& state, uint32_t value) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF64(state), result, GlslStd450(state),
	                          GLSLstd450Fract, AsDouble(state, value));
	return AsBits(state, result);
}

uint32_t EmitConvertF64S32(EmitterState& state, uint32_t value) {
	return AsBits(state, Unary(state, spv::OpConvertSToF, TypeF64(state), value));
}

uint32_t EmitConvertF64F32(EmitterState& state, uint32_t value) {
	return AsBits(state, Unary(state, spv::OpFConvert, TypeF64(state), value));
}

uint32_t EmitConvertF32F64(EmitterState& state, uint32_t value) {
	return Unary(state, spv::OpFConvert, TypeF32(state), AsDouble(state, value));
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
