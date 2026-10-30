cmake_minimum_required(VERSION 3.16)
project(ctm_tools LANGUAGES C)

# CTM Tools - Custom Codebase Metrics Tools
# Build with: cmake -B build -DSNEPPX_BUILD_CTM_TOOLS=ON && cmake --build build

add_executable(ctm_count_total_files c/ctm_count_total_files.c)
target_include_directories(ctm_count_total_files PRIVATE /include /common)
set_target_properties(ctm_count_total_files PROPERTIES RUNTIME_OUTPUT_DIRECTORY /ctm_tools)

add_executable(ctm_count_total_dirs c/ctm_count_total_dirs.c)
target_include_directories(ctm_count_total_dirs PRIVATE /include /common)
set_target_properties(ctm_count_total_dirs PROPERTIES RUNTIME_OUTPUT_DIRECTORY /ctm_tools)

add_executable(ctm_count_total_loc c/ctm_count_total_loc.c)
target_include_directories(ctm_count_total_loc PRIVATE /include /common)
set_target_properties(ctm_count_total_loc PROPERTIES RUNTIME_OUTPUT_DIRECTORY /ctm_tools)

add_executable(ctm_count_loc_by_file c/ctm_count_loc_by_file.c)
target_include_directories(ctm_count_loc_by_file PRIVATE /include /common)
set_target_properties(ctm_count_loc_by_file PROPERTIES RUNTIME_OUTPUT_DIRECTORY /ctm_tools)

add_executable(ctm_count_loc_by_language c/ctm_count_loc_by_language.c)
target_include_directories(ctm_count_loc_by_language PRIVATE /include /common)
set_target_properties(ctm_count_loc_by_language PROPERTIES RUNTIME_OUTPUT_DIRECTORY /ctm_tools)
