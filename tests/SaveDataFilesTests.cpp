#include "common/file.h"
#include "libs/saveDataFiles.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace Files = Libs::SaveData::Files;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "SaveDataFilesTests: failed: %s\n", message);
		std::abort();
	}
}

class TempDirectory {
public:
	TempDirectory() {
		const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
		m_path            = std::filesystem::temp_directory_path() /
		                    ("kyty_save_data_files_" + std::to_string(unique));
		Check(std::filesystem::create_directories(m_path), "create the temporary directory");
	}

	~TempDirectory() {
		std::error_code error;
		std::filesystem::remove_all(m_path, error);
	}

	[[nodiscard]] const std::filesystem::path& Path() const { return m_path; }

	KYTY_CLASS_NO_COPY(TempDirectory);

private:
	std::filesystem::path m_path;
};

void TestParametersSurviveAWrite() {
	TempDirectory temp;
	const auto    save = temp.Path() / "SAVEDATAPROFILE";
	Check(std::filesystem::create_directories(save), "create the save directory");

	Files::Param param;
	param.title      = "Grand Theft Auto V";
	param.sub_title  = "Profile";
	param.detail     = "";
	param.user_param = 7;
	param.mtime      = 1758300000;
	Check(Files::WriteParam(save, param), "write the parameters");

	const auto read = Files::ReadParam(save);
	Check(read.title == param.title, "the title did not come back");
	Check(read.sub_title == param.sub_title, "the subtitle did not come back");
	Check(read.detail.empty(), "an empty detail did not come back empty");
	Check(read.user_param == param.user_param, "the user parameter did not come back");
	Check(read.mtime == param.mtime, "the modification time did not come back");
}

void TestASaveWithoutParametersStillHasATime() {
	TempDirectory temp;
	const auto    save = temp.Path() / "SAVEDATASGTA50000";
	Check(std::filesystem::create_directories(save), "create the save directory");

	const auto read = Files::ReadParam(save);
	Check(read.title.empty() && read.sub_title.empty(), "an unwritten save reported a title");
	Check(read.user_param == 0, "an unwritten save reported a user parameter");
	// A game that sorts its saves by time must not see them all at the epoch.
	Check(read.mtime > 0, "an unwritten save reported no modification time");
}

void TestDamagedParametersReadAsUnwritten() {
	TempDirectory temp;
	const auto    save = temp.Path() / "SAVEDATAPROFILE";
	Check(std::filesystem::create_directories(save), "create the save directory");

	Files::Param param;
	param.title = "Grand Theft Auto V";
	param.mtime = 1758300000;
	Check(Files::WriteParam(save, param), "write the parameters");

	const auto path = Files::ParamPath(save);
	Check(Common::File::IsFileExisting(path), "the parameter file is not where it belongs");
	Check(std::filesystem::file_size(path) > 8, "the parameter file is too small to truncate");
	std::filesystem::resize_file(path, 8);

	const auto read = Files::ReadParam(save);
	Check(read.title.empty(), "a truncated parameter file was read as a title");
	Check(read.mtime > 0, "a truncated parameter file left the save without a time");
}

void TestSaveDataMemorySurvivesAWrite() {
	TempDirectory        temp;
	std::vector<uint8_t> read;
	Check(!Files::ReadMemory(temp.Path(), read),
	      "a title with no save data memory reported having one");

	std::vector<uint8_t> written(4096);
	for (size_t i = 0; i < written.size(); i++) {
		written[i] = static_cast<uint8_t>(i * 7);
	}
	Check(Files::WriteMemory(temp.Path(), written), "write the save data memory");
	Check(Files::ReadMemory(temp.Path(), read), "read the save data memory back");
	Check(read == written, "the save data memory came back changed");

	// A game may set up a smaller image later; what it reads back has to be that one.
	const std::vector<uint8_t> smaller(16, 0xAB);
	Check(Files::WriteMemory(temp.Path(), smaller), "write a smaller save data memory");
	Check(Files::ReadMemory(temp.Path(), read), "read the smaller save data memory back");
	Check(read == smaller, "the smaller save data memory came back changed");
}

} // namespace

int main() {
	TestParametersSurviveAWrite();
	TestASaveWithoutParametersStillHasATime();
	TestDamagedParametersReadAsUnwritten();
	TestSaveDataMemorySurvivesAWrite();
	std::printf("SaveDataFilesTests: ok\n");
	return 0;
}
