#pragma once
#ifndef SNEPPX_OBF_ANTI_H
#define SNEPPX_OBF_ANTI_H

#include <cstdint>
#include <string>
#include <functional>

/*
 * SNEPPX - Anti Debugging Countermeasure
 *
 * WHAT
 *   Anti Debugging Countermeasure.
 *
 * CONCEPT
 *   Provides the Anti Debugging Countermeasure.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace SNEPPX {

enum class SNEPPXAntiDebugAction {
    CRASH,
    FAKE_DATA,
    DELAY,
    SILENT_EXIT,
    WIPE_AND_EXIT
};

class SNEPPXAntiDebug {
public:
    SNEPPXAntiDebug();
    ~SNEPPXAntiDebug();

    void set_action(SNEPPXAntiDebugAction action);
    SNEPPXAntiDebugAction get_action() const;

    bool detect_ptrace();
    bool detect_debugger_present();
    bool detect_timing_anomaly();
    bool detect_breakpoint(const uint8_t* code_start, size_t code_len);
    bool detect_vm();

    void react();
    void full_scan();

    bool is_debugger_attached() const { return debugger_detected; }

private:
    SNEPPXAntiDebugAction action;
    bool debugger_detected;

    uint64_t read_timestamp();
    uint32_t compute_code_hash(const uint8_t* code, size_t len);
    void wipe_sensitive_data();
};

} // namespace SNEPPX

#endif // SNEPPX_OBF_ANTI_H
