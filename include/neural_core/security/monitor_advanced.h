#ifndef SNEPPX_MONITOR_ADVANCED_H
#define SNEPPX_MONITOR_ADVANCED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Monitor Advanced
 *
 * WHAT
 *   Monitor Advanced.
 *
 * CONCEPT
 *   Provides the Monitor Advanced.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

#define SNEPPX_MON_MAX_REGIONS 128
#define SNEPPX_MON_MAX_EVENTS 1024
#define SNEPPX_MON_ML_FEATURES 8

/* Code segment tamper detection */
typedef struct {
    const void* code_addr;
    size_t code_size;
    uint8_t baseline_hash[32];
    int enabled;
} SNEPPXCodeTamperDetector;

/**
 * @brief Initialize Code Tamper.
 *
 * @param ctd [out] Ctd value.
 * @param addr [in] Addr value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_code_tamper_init(SNEPPXCodeTamperDetector* ctd, const void* addr, size_t size);
/**
 * @brief Perform Code Tamper Check.
 *
 * @param ctd [out] Ctd value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_code_tamper_check(SNEPPXCodeTamperDetector* ctd);

/* Function pointer hook detection */
typedef struct {
    const void** func_ptrs[64];
    uintptr_t original_values[64];
    int count;
} SNEPPXFuncPtrDetector;

/**
 * @brief Initialize Func Ptr Detector.
 *
 * @param fpd [out] Fpd value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_func_ptr_detector_init(SNEPPXFuncPtrDetector* fpd);
/**
 * @brief Perform Func Ptr Detector Watch.
 *
 * @param fpd [out] Fpd value.
 * @param func_ptr [in] Func Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_func_ptr_detector_watch(SNEPPXFuncPtrDetector* fpd, const void** func_ptr);
/**
 * @brief Perform Func Ptr Detector Scan.
 *
 * @param fpd [out] Fpd value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_func_ptr_detector_scan(SNEPPXFuncPtrDetector* fpd);

/* Heap corruption detector */
typedef struct {
    uint64_t sentinel_value;
    int enabled;
} SNEPPXHeapCorruptionDetector;

/**
 * @brief Initialize Heap Corruption.
 *
 * @param hcd [out] Hcd value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_heap_corruption_init(SNEPPXHeapCorruptionDetector* hcd);
/**
 * @brief Perform Heap Corruption Apply Sentinel.
 *
 * @param hcd [out] Hcd value.
 * @param alloc [out] Alloc value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_heap_corruption_apply_sentinel(SNEPPXHeapCorruptionDetector* hcd, void* alloc, size_t size);
/**
 * @brief Perform Heap Corruption Check.
 *
 * @param hcd [out] Hcd value.
 * @param alloc [out] Alloc value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_heap_corruption_check(SNEPPXHeapCorruptionDetector* hcd, void* alloc, size_t size);

/* Stack overflow detection */
/**
 * @brief Perform Stack Overflow Guard Install.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_stack_overflow_guard_install(void);
/**
 * @brief Perform Stack Overflow Check.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_stack_overflow_check(void);

/* Return address verification */
/**
 * @brief Verify Ret Addr.
 *
 * @param ret_addr [out] Ret Addr value.
 * @param expected_ret_addr [out] Expected Ret Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ret_addr_verify(void* ret_addr, void* expected_ret_addr);

/* Instruction-level tracing (stub: platform-specific) */
typedef struct {
    int enabled;
    uint64_t trace_buffer_size;
    void* trace_buffer;
} SNEPPXInstructionTracer;

/**
 * @brief Initialize Inst Tracer.
 *
 * @param tracer [out] Tracer value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_inst_tracer_init(SNEPPXInstructionTracer* tracer);
/**
 * @brief Start Inst Tracer.
 *
 * @param tracer [out] Tracer value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_inst_tracer_start(SNEPPXInstructionTracer* tracer);
/**
 * @brief Stop Inst Tracer.
 *
 * @param tracer [out] Tracer value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_inst_tracer_stop(SNEPPXInstructionTracer* tracer);

/* ML anomaly detector */
typedef struct {
    double means[SNEPPX_MON_ML_FEATURES];
    double stds[SNEPPX_MON_ML_FEATURES];
    int trained;
    double threshold;
} SNEPPXMLAnomalyDetector;

/**
 * @brief Initialize Ml Anomaly.
 *
 * @param ml [out] Ml value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ml_anomaly_init(SNEPPXMLAnomalyDetector* ml);
/**
 * @brief Train Ml Anomaly.
 *
 * @param ml [out] Ml value.
 * @param n_samples [in] N Samples value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ml_anomaly_train(SNEPPXMLAnomalyDetector* ml, const double features[][SNEPPX_MON_ML_FEATURES], int n_samples);
/**
 * @brief Perform Ml Anomaly Score.
 *
 * @param ml [out] Ml value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_ml_anomaly_score(SNEPPXMLAnomalyDetector* ml, const double features[SNEPPX_MON_ML_FEATURES]);
/**
 * @brief Perform Ml Anomaly Is Anomaly.
 *
 * @param ml [out] Ml value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ml_anomaly_is_anomaly(SNEPPXMLAnomalyDetector* ml, const double features[SNEPPX_MON_ML_FEATURES]);

/* File system integrity */
typedef struct {
    char paths[64][256];
    uint8_t hashes[64][32];
    int count;
    int enabled;
} SNEPPXFSIntegrity;

/**
 * @brief Initialize Fs Integrity.
 *
 * @param fsi [out] Fsi value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_fs_integrity_init(SNEPPXFSIntegrity* fsi);
/**
 * @brief Perform Fs Integrity Watch.
 *
 * @param fsi [out] Fsi value.
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_fs_integrity_watch(SNEPPXFSIntegrity* fsi, const char* path);
/**
 * @brief Perform Fs Integrity Scan.
 *
 * @param fsi [out] Fsi value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_fs_integrity_scan(SNEPPXFSIntegrity* fsi);

/* Registry key monitoring (Windows) / file monitoring (Linux) */
/**
 * @brief Initialize Persistence Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_persistence_monitor_init(void);
/**
 * @brief Perform Persistence Monitor Scan.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_persistence_monitor_scan(void);

/* Process injection detection */
/**
 * @brief Perform Proc Injection Detect.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_proc_injection_detect(void);

/* Network connection monitoring */
/**
 * @brief Initialize Net Conn Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_net_conn_monitor_init(void);
/**
 * @brief Perform Net Conn Monitor Check.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_net_conn_monitor_check(void);

/* USB/device insertion detection */
/**
 * @brief Perform Device Insertion Detect.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_device_insertion_detect(void);

/* Kernel object reference monitor */
/**
 * @brief Initialize Kernel Obj Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_kernel_obj_monitor_init(void);
/**
 * @brief Perform Kernel Obj Monitor Check.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_kernel_obj_monitor_check(void);

/* TOCTOU detection */
typedef struct {
    uint8_t baseline[32];
    int initialized;
} SNEPPXTOCTOUDetector;

/**
 * @brief Initialize Toctou.
 *
 * @param td [out] Td value.
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_toctou_init(SNEPPXTOCTOUDetector* td, const char* path);
/**
 * @brief Perform Toctou Check.
 *
 * @param td [out] Td value.
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_toctou_check(SNEPPXTOCTOUDetector* td, const char* path);

/* IMA-style integrity */
/**
 * @brief Perform Ima Measure.
 *
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ima_measure(const char* path, uint8_t hash[32]);
/**
 * @brief Perform Ima Appraise.
 *
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ima_appraise(const char* path, const uint8_t hash[32]);

/* Alert correlation engine */
typedef struct {
    struct { uint64_t timestamp; int type; const char* desc; } events[SNEPPX_MON_MAX_EVENTS];
    int count;
    int alerts_triggered;
} SNEPPXAlertCorrelator;

/**
 * @brief Initialize Alert Correlator.
 *
 * @param ac [out] Ac value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_alert_correlator_init(SNEPPXAlertCorrelator* ac);
/**
 * @brief Add Alert Correlator.
 *
 * @param ac [out] Ac value.
 * @param type [in] Type value.
 * @param desc [in] Desc value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_alert_correlator_add(SNEPPXAlertCorrelator* ac, int type, const char* desc);
/**
 * @brief Perform Alert Correlator Evaluate.
 *
 * @param ac [out] Ac value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_alert_correlator_evaluate(SNEPPXAlertCorrelator* ac);

#ifdef __cplusplus
}
#endif
#endif
