#include <c10/util/tempfile.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#if !defined(_WIN32)
TEST(TempFileTest, MatchesExpectedPattern) {
  c10::TempFile pattern = c10::make_tempfile("test-pattern-");
  ASSERT_NE(pattern.name.find("test-pattern-"), std::string::npos);
}
#endif // !defined(_WIN32)

#ifdef __cilksan__
#ifdef __cplusplus
extern "C" {
#endif
void __csan_default_libhook(uint64_t call_id, uint64_t func_id, unsigned count);
void __csan_rmdir(uint64_t call_id, uint64_t func_id, unsigned count) {
  __csan_default_libhook(call_id, func_id, count);
}
#ifdef __cplusplus
}
#endif
#endif

static bool directory_exists(const char* path) {
  struct stat st;
  return (stat(path, &st) == 0 && (st.st_mode & S_IFDIR));
}

TEST(TempDirTest, tryMakeTempdir) {
  c10::optional<c10::TempDir> tempdir = c10::make_tempdir("test-dir-");
  std::string tempdir_name = tempdir->name;

  // directory should exist while tempdir is alive
  ASSERT_TRUE(directory_exists(tempdir_name.c_str()));

  // directory should not exist after tempdir destroyed
  tempdir.reset();
  ASSERT_FALSE(directory_exists(tempdir_name.c_str()));
}
