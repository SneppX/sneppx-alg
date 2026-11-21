#include "anti_debugging_countermeasure.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <string>
#endif

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <algorithm>

/*
 * SNEPPX - Debug Detect
 *
 * WHAT
 *   Debug Detect.
 *
 * CONCEPT
 *   Provides the Debug Detect.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
/**
 * @brief Perform Secure Zero.
 *
 * @param ptr [out] Ptr value.
 * @param len [in] Len value.
 */
void SNEPPX_secure_zero(void* ptr, size_t len);
}

namespace SNEPPX {

SNEPPXAntiDebug::SNEPPXAntiDebug() : action(SNEPPXAntiDebugAction::WIPE_AND_EXIT), debugger_detected(false) {}

SNEPPXAntiDebug::~SNEPPXAntiDebug() {}

void SNEPPXAntiDebug::set_action(SNEPPXAntiDebugAction act) {
    action = act;
}

SNEPPXAntiDebugAction SNEPPXAntiDebug::get_action() const {
    return action;
}

uint64_t SNEPPXAntiDebug::read_timestamp() {
#ifdef _WIN32
    return __rdtsc();
#else
    uint32_t lo, hi;
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
}

bool SNEPPXAntiDebug::detect_ptrace() {
#ifdef _WIN32
    return false;
#else
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) return false;
    std::string line;
    while (std::getline(status, line)) {
        if (line.find("TracerPid:") != std::string::npos) {
            std::string pid_str = line.substr(line.find(':') + 1);
            pid_str.erase(0, pid_str.find_first_not_of(" \t"));
            int tracer_pid = std::stoi(pid_str);
            if (tracer_pid != 0) {
                debugger_detected = true;
                return true;
            }
            break;
        }
    }
    return false;
#endif
}

bool SNEPPXAntiDebug::detect_debugger_present() {
#ifdef _WIN32
    if (IsDebuggerPresent()) {
        debugger_detected = true;
        return true;
    }
#else
    (void)0;
#endif
    return false;
}

bool SNEPPXAntiDebug::detect_timing_anomaly() {
    uint64_t t1 = read_timestamp();
    uint64_t t2 = read_timestamp();
    uint64_t diff = t2 - t1;
    if (diff > 1000) {
        debugger_detected = true;
        return true;
    }
    return false;
}

bool SNEPPXAntiDebug::detect_breakpoint(const uint8_t* code_start, size_t code_len) {
    if (code_start == nullptr || code_len == 0) return false;
    for (size_t i = 0; i < code_len; ++i) {
        if (code_start[i] == 0xCC) {
            debugger_detected = true;
            return true;
        }
    }
    return false;
}

bool SNEPPXAntiDebug::detect_vm() {
#ifdef _WIN32
    int cpuinfo[4] = {0};
    __cpuidex(cpuinfo, 1, 0);
    if (cpuinfo[2] & (1 << 31)) {
        debugger_detected = true;
        return true;
    }
#else
    uint32_t eax, ebx, ecx, edx;
    asm volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1));
    if (ecx & (1 << 31)) {
        debugger_detected = true;
        return true;
    }
#endif
    return false;
}

void SNEPPXAntiDebug::wipe_sensitive_data() {
    volatile char dummy[256];
    SNEPPX_secure_zero(const_cast<char*>(dummy), sizeof(dummy));
}

void SNEPPXAntiDebug::react() {
    switch (action) {
        case SNEPPXAntiDebugAction::CRASH:
            wipe_sensitive_data();
            {
                volatile int* p = nullptr;
                *p = 0;
            }
            break;

        case SNEPPXAntiDebugAction::FAKE_DATA:
            break;

        case SNEPPXAntiDebugAction::DELAY:
            for (volatile uint64_t i = 0; i < 100000000; ++i) {}
            break;

        case SNEPPXAntiDebugAction::SILENT_EXIT:
            wipe_sensitive_data();
#ifdef _WIN32
            ExitProcess(0);
#else
            _exit(0);
#endif
            break;

        case SNEPPXAntiDebugAction::WIPE_AND_EXIT:
            wipe_sensitive_data();
#ifdef _WIN32
            TerminateProcess(GetCurrentProcess(), 0);
#else
            raise(SIGKILL);
#endif
            break;
    }
}

void SNEPPXAntiDebug::full_scan() {
    if (detect_ptrace()) { react(); return; }
    if (detect_debugger_present()) { react(); return; }
    if (detect_timing_anomaly()) { react(); return; }
    if (detect_vm()) { react(); return; }
}

uint32_t SNEPPXAntiDebug::compute_code_hash(const uint8_t* code, size_t len) {
    uint32_t hash = 0x811C9DC5;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint32_t>(code[i]);
        hash *= 0x01000193;
    }
    return hash;
}

} // namespace SNEPPX
