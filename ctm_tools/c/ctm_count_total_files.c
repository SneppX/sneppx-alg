/*
 * CTM Tool: Count total files
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

int is_ignored_dir(const char *name) {
    if (strcmp(name, ".git") == 0) return 1;
    if (strcmp(name, ".svn") == 0) return 1;
    if (strcmp(name, ".hg") == 0) return 1;
    if (strcmp(name, "__pycache__") == 0) return 1;
    if (strcmp(name, "node_modules") == 0) return 1;
    if (strcmp(name, ".venv") == 0) return 1;
    if (strcmp(name, "venv") == 0) return 1;
    if (strcmp(name, "build") == 0) return 1;
    if (strcmp(name, "dist") == 0) return 1;
    if (strcmp(name, ".next") == 0) return 1;
    if (strcmp(name, ".cache") == 0) return 1;
    if (strcmp(name, ".pytest_cache") == 0) return 1;
    if (strcmp(name, ".ruff_cache") == 0) return 1;
    return 0;
}

int is_ignored_ext(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) return 0;
    if (strcmp(ext, ".pyc") == 0) return 1;
    if (strcmp(ext, ".pyo") == 0) return 1;
    if (strcmp(ext, ".o") == 0) return 1;
    if (strcmp(ext, ".obj") == 0) return 1;
    if (strcmp(ext, ".exe") == 0) return 1;
    if (strcmp(ext, ".dll") == 0) return 1;
    if (strcmp(ext, ".so") == 0) return 1;
    if (strcmp(ext, ".dylib") == 0) return 1;
    if (strcmp(ext, ".class") == 0) return 1;
    if (strcmp(ext, ".jar") == 0) return 1;
    return 0;
}

#ifdef _WIN32
long count_files_win(const char *base) {
    char path[MAX_PATH];
    snprintf(path, MAX_PATH, \"%s\\\\*\", base);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(path, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    long count = 0;
    do {
        if (strcmp(fd.cFileName, \".\") == 0 || strcmp(fd.cFileName, \"..\") == 0) continue;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if (is_ignored_dir(fd.cFileName)) continue;
            char sub[MAX_PATH];
            snprintf(sub, MAX_PATH, \"%s\\\\%s\", base, fd.cFileName);
            count += count_files_win(sub);
        } else {
            if (is_ignored_ext(fd.cFileName)) continue;
            count++;
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    return count;
}
#endif

int main(int argc, char **argv) {
    const char *root = argc > 1 ? argv[1] : \".\";
#ifdef _WIN32
    printf(\"%ld\\n\", count_files_win(root));
#else
    // simplified for non-WIN32 if needed
    printf(\"0\\n\");
#endif
    return 0;
}
