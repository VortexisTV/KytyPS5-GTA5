#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_SAVEDATAFILES_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_SAVEDATAFILES_H_

#include "common/common.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

// A save holds more than the files a game writes into it: every save carries the parameters
// sceSaveDataSetParam stores, and a title may keep one save data memory image. Both outlive the
// process, so they are kept on disk beside the save itself.
namespace Libs::SaveData::Files {

// What sceSaveDataGetParam and sceSaveDataDirNameSearch report about a save.
struct Param {
	std::string title;
	std::string sub_title;
	std::string detail;
	uint32_t    user_param = 0;
	int64_t     mtime      = 0; // seconds since the Unix epoch, 0 when nothing is known
};

// Beside the game's own files, where the system keeps what it knows about the save.
[[nodiscard]] std::filesystem::path ParamPath(const std::filesystem::path& save_dir);

// One title's save data memory. Its directory name is reserved, so a search skips it.
[[nodiscard]] std::filesystem::path MemoryPath(const std::filesystem::path& title_dir);

bool WriteParam(const std::filesystem::path& save_dir, const Param& param);

// A save whose parameters were never stored still reports when it last changed.
[[nodiscard]] Param ReadParam(const std::filesystem::path& save_dir);

bool WriteMemory(const std::filesystem::path& title_dir, std::span<const uint8_t> data);

// False when the title has no save data memory at all, which the guest has to be able to tell
// apart from an image that happens to be empty.
[[nodiscard]] bool ReadMemory(const std::filesystem::path& title_dir, std::vector<uint8_t>& data);

} // namespace Libs::SaveData::Files

#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_SAVEDATAFILES_H_ */
