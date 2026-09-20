#include "libs/saveDataFiles.h"

#include "common/file.h"

#include <chrono>
#include <cstring>
#include <system_error>

namespace Libs::SaveData::Files {

namespace {

constexpr char     ParamDirName[]   = "sce_sys";
constexpr char     ParamFileName[]  = "param.bin";
constexpr char     MemoryDirName[]  = "sce_sdmemory";
constexpr char     MemoryFileName[] = "memory.bin";
constexpr uint32_t ParamMagic       = 0x5044534Bu; // "KSDP"
constexpr uint32_t ParamVersion     = 1;
// Long enough for the guest fields (128, 128 and 1024 bytes) and short enough that a damaged
// length cannot ask for an unreasonable allocation.
constexpr uint32_t StringLengthMax = 4096;

void Append(std::vector<uint8_t>& out, const void* data, size_t size) {
	const auto* bytes = static_cast<const uint8_t*>(data);
	out.insert(out.end(), bytes, bytes + size);
}

void AppendString(std::vector<uint8_t>& out, const std::string& value) {
	const auto length = static_cast<uint32_t>(value.size());
	Append(out, &length, sizeof(length));
	Append(out, value.data(), value.size());
}

// Every read is bounds checked: a file from an older build, or a half-written one, reads as a
// save whose parameters were never stored.
bool Take(std::span<const uint8_t>& in, void* data, size_t size) {
	if (in.size() < size) {
		return false;
	}
	std::memcpy(data, in.data(), size);
	in = in.subspan(size);
	return true;
}

bool TakeString(std::span<const uint8_t>& in, std::string& value) {
	uint32_t length = 0;
	if (!Take(in, &length, sizeof(length)) || length > StringLengthMax || in.size() < length) {
		return false;
	}
	value.assign(reinterpret_cast<const char*>(in.data()), length);
	in = in.subspan(length);
	return true;
}

// Directories have a time too, and a save without stored parameters is dated by its own.
int64_t LastWriteUnixTime(const std::filesystem::path& path) {
	std::error_code error;
	const auto      written = std::filesystem::last_write_time(path, error);
	if (error) {
		return 0;
	}
	const auto system = std::chrono::clock_cast<std::chrono::system_clock>(written);
	return std::chrono::duration_cast<std::chrono::seconds>(system.time_since_epoch()).count();
}

bool WriteWholeFile(const std::filesystem::path& path, const void* data, size_t size) {
	std::error_code error;
	std::filesystem::create_directories(path.parent_path(), error);
	if (!std::filesystem::is_directory(path.parent_path(), error)) {
		return false;
	}
	// A crash between the two would otherwise leave a file that is neither the old nor the new
	// contents.
	auto temp = path;
	temp += ".tmp";
	// Mode::Write opens an existing file; creating one is the single-argument constructor.
	Common::File file(temp);
	if (file.IsInvalid()) {
		return false;
	}
	uint32_t written = 0;
	file.Write(data, static_cast<uint32_t>(size), &written);
	file.Flush();
	file.Close();
	if (written != size) {
		Common::File::DeleteFile(temp);
		return false;
	}
	Common::File::DeleteFile(path);
	return Common::File::RenameFile(temp, path);
}

bool ReadWholeFile(const std::filesystem::path& path, std::vector<uint8_t>& out) {
	if (!Common::File::IsFileExisting(path)) {
		return false;
	}
	Common::File file(path, Common::File::Mode::Read);
	if (file.IsInvalid()) {
		return false;
	}
	const auto size = file.Size();
	out.resize(static_cast<size_t>(size));
	uint32_t read = 0;
	if (size != 0) {
		file.Read(out.data(), static_cast<uint32_t>(size), &read);
	}
	file.Close();
	if (read != size) {
		out.clear();
		return false;
	}
	return true;
}

} // namespace

std::filesystem::path ParamPath(const std::filesystem::path& save_dir) {
	return save_dir / ParamDirName / ParamFileName;
}

std::filesystem::path MemoryPath(const std::filesystem::path& title_dir) {
	return title_dir / MemoryDirName / MemoryFileName;
}

bool WriteParam(const std::filesystem::path& save_dir, const Param& param) {
	std::vector<uint8_t> out;
	Append(out, &ParamMagic, sizeof(ParamMagic));
	Append(out, &ParamVersion, sizeof(ParamVersion));
	Append(out, &param.user_param, sizeof(param.user_param));
	Append(out, &param.mtime, sizeof(param.mtime));
	AppendString(out, param.title);
	AppendString(out, param.sub_title);
	AppendString(out, param.detail);
	return WriteWholeFile(ParamPath(save_dir), out.data(), out.size());
}

Param ReadParam(const std::filesystem::path& save_dir) {
	Param                param;
	const auto           path = ParamPath(save_dir);
	std::vector<uint8_t> bytes;
	if (ReadWholeFile(path, bytes)) {
		std::span<const uint8_t> in {bytes};
		uint32_t                 magic   = 0;
		uint32_t                 version = 0;
		if (Take(in, &magic, sizeof(magic)) && magic == ParamMagic &&
		    Take(in, &version, sizeof(version)) && version == ParamVersion &&
		    Take(in, &param.user_param, sizeof(param.user_param)) &&
		    Take(in, &param.mtime, sizeof(param.mtime)) && TakeString(in, param.title) &&
		    TakeString(in, param.sub_title) && TakeString(in, param.detail)) {
			if (param.mtime == 0) {
				param.mtime = LastWriteUnixTime(path);
			}
			return param;
		}
		param = {};
	}
	// A save copied in by hand, or one written before parameters were kept, still has a time.
	param.mtime = LastWriteUnixTime(save_dir);
	return param;
}

bool WriteMemory(const std::filesystem::path& title_dir, std::span<const uint8_t> data) {
	return WriteWholeFile(MemoryPath(title_dir), data.data(), data.size());
}

bool ReadMemory(const std::filesystem::path& title_dir, std::vector<uint8_t>& data) {
	return ReadWholeFile(MemoryPath(title_dir), data);
}

} // namespace Libs::SaveData::Files
