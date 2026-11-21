#include "monitor_advanced.h"
#include "cryptographic_hashing_blake3.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <tlhelp32.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#else
#include <unistd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <dirent.h>
#include <dlfcn.h>
#endif

#define SNEPPX_STACK_GUARD_MARGIN 4096
#ifdef _WIN32
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


static void* g_so_guard_page = NULL;
static void* g_so_stack_bottom = NULL;
static size_t g_so_guard_size = 0;
static int g_so_installed = 0;
#else
static void* g_so_guard_page = NULL;
static void* g_so_stack_bottom = NULL;
static size_t g_so_guard_size = 0;
static int g_so_installed = 0;
#endif

#define SNEPPX_PERSIST_MAX 24
static char g_persist_paths[SNEPPX_PERSIST_MAX][520];
static int g_persist_count = 0;
static int g_persist_initialized = 0;

#define SNEPPX_BAD_PORTS_MAX 80
static int g_net_bad_ports[SNEPPX_BAD_PORTS_MAX];
static int g_net_bad_port_count = 0;
#ifdef _WIN32
static int g_net_wsa_started = 0;
#endif

#ifdef _WIN32
static int g_dev_prev_count = -1;
static int g_dev_drive_prev = 0;
#else
static int g_dev_prev_count = -1;
#endif

#ifdef _WIN32
static DWORD g_kobj_baseline = 0;
static int g_kobj_initialized = 0;
#else
static int g_kobj_baseline = 0;
static int g_kobj_initialized = 0;
#endif

#define SNEPPX_CODE_TAMPER_MAX_REGIONS 16
typedef struct {
    const void* addr;
    size_t size;
    uint8_t hash[32];
    int active;
} CodeTamperRegion;

static CodeTamperRegion g_ct_regions[SNEPPX_CODE_TAMPER_MAX_REGIONS];
static int g_ct_region_count = 0;

/* --- Code Tamper --- */
/**
 * @brief Initialize Code Tamper.
 *
 * @param ctd [out] Ctd value.
 * @param addr [in] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_code_tamper_init(SNEPPXCodeTamperDetector* ctd, const void* addr, size_t size) {
    if (!ctd||!addr) return -1;
    ctd->code_addr=addr; ctd->code_size=size; ctd->enabled=1;
    SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx);
    SNEPPX_blake3_update(&ctx,(const uint8_t*)addr,size);
    SNEPPX_blake3_finish(&ctx,ctd->baseline_hash);
    g_ct_region_count=0;
    return 0;
}

/**
 * @brief Perform Code Tamper Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_code_tamper_check(SNEPPXCodeTamperDetector* ctd) {
    if (!ctd||!ctd->enabled) return 0;
    uint8_t current[32];
    SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx);
    SNEPPX_blake3_update(&ctx,(const uint8_t*)ctd->code_addr,ctd->code_size);
    SNEPPX_blake3_finish(&ctx,current);
    return memcmp(current,ctd->baseline_hash,32)==0?0:1;
}

/**
 * @brief Perform Code Tamper Add Region.
 *
 * @param addr [in] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_code_tamper_add_region(const void* addr, size_t size) {
    if (!addr||size==0||g_ct_region_count>=SNEPPX_CODE_TAMPER_MAX_REGIONS) return -1;
    CodeTamperRegion* r = &g_ct_regions[g_ct_region_count];
    r->addr = addr; r->size = size; r->active = 1;
    SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx);
    SNEPPX_blake3_update(&ctx,(const uint8_t*)addr,size);
    SNEPPX_blake3_finish(&ctx,r->hash);
    g_ct_region_count++;
    return 0;
}

/**
 * @brief Perform Code Tamper Remove Region.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_code_tamper_remove_region(int index) {
    if (index<0||index>=g_ct_region_count) return -1;
    g_ct_regions[index].active = 0;
    return 0;
}

/**
 * @brief Perform Code Tamper Check All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_code_tamper_check_all(void) {
    int violations = 0;
    for (int i=0;i<g_ct_region_count;i++) {
        if (!g_ct_regions[i].active) continue;
        uint8_t current[32];
        SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx);
        SNEPPX_blake3_update(&ctx,(const uint8_t*)g_ct_regions[i].addr,g_ct_regions[i].size);
        SNEPPX_blake3_finish(&ctx,current);
        if (memcmp(current,g_ct_regions[i].hash,32)!=0) violations++;
    }
    return violations;
}

/* --- Func Ptr Detector --- */
int SNEPPX_func_ptr_detector_init(SNEPPXFuncPtrDetector* fpd) { if (!fpd) return -1; memset(fpd,0,sizeof(*fpd)); return 0; }

/**
 * @brief Perform Func Ptr Detector Watch.
 *
 * @param fpd [out] Fpd value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_func_ptr_detector_watch(SNEPPXFuncPtrDetector* fpd, const void** func_ptr) {
    if (!fpd||!func_ptr||fpd->count>=64) return -1;
    fpd->func_ptrs[fpd->count]=func_ptr;
    fpd->original_values[fpd->count]=(uintptr_t)*func_ptr;
    fpd->count++;
    return 0;
}

/**
 * @brief Perform Func Ptr Detector Scan.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_func_ptr_detector_scan(SNEPPXFuncPtrDetector* fpd) {
    if (!fpd) return 0;
    int modified=0;
    for (int i=0;i<fpd->count;i++) {
        if ((uintptr_t)*fpd->func_ptrs[i]!=fpd->original_values[i]) modified++;
    }
    return modified;
}

/**
 * @brief Perform Func Ptr Detector Unwatch.
 *
 * @param fpd [out] Fpd value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_func_ptr_detector_unwatch(SNEPPXFuncPtrDetector* fpd, int index) {
    if (!fpd||index<0||index>=fpd->count) return -1;
    for (int i=index;i<fpd->count-1;i++) {
        fpd->func_ptrs[i]=fpd->func_ptrs[i+1];
        fpd->original_values[i]=fpd->original_values[i+1];
    }
    fpd->count--;
    return 0;
}

/**
 * @brief Perform Func Ptr Detector Scan All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_func_ptr_detector_scan_all(SNEPPXFuncPtrDetector* fpd) {
    if (!fpd) return 0;
    int modified=0;
    for (int i=0;i<fpd->count;i++) {
        uintptr_t val = (uintptr_t)*fpd->func_ptrs[i];
        if (val!=fpd->original_values[i]) {
            modified++;
            uintptr_t nearby[4];
            for (int j=0;j<4;j++) {
                if (i+j<fpd->count) nearby[j]=(uintptr_t)*fpd->func_ptrs[i+j];
                else nearby[j]=0;
            }
            (void)nearby;
        }
    }
    return modified;
}

/**
 * @brief Perform Func Ptr Detector Get Report.
 *
 * @param fpd [out] Fpd value.
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_func_ptr_detector_get_report(SNEPPXFuncPtrDetector* fpd, char* buffer, size_t size) {
    if (!fpd||!buffer||size<1) return -1;
    int pos=0;
    for (int i=0;i<fpd->count&&pos<(int)size-64;i++) {
        int corrupt = ((uintptr_t)*fpd->func_ptrs[i]!=fpd->original_values[i])?1:0;
        pos+=snprintf(buffer+pos,size-pos,"[%d] ptr=%p orig=0x%llX %s\n",
                      i,(const void*)*fpd->func_ptrs[i],
                      (unsigned long long)fpd->original_values[i],
                      corrupt?"CORRUPT":"OK");
    }
    return 0;
}

/* --- Heap Corruption --- */
static uint64_t g_hcd_sentinel_value = 0xDEADBEEFCAFEBABEULL;
static int g_hcd_enabled = 1;

/**
 * @brief Initialize Heap Corruption.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_init(SNEPPXHeapCorruptionDetector* hcd) {
    if (!hcd) return -1;
    hcd->sentinel_value=0xDEADBEEFCAFEBABEULL;
    hcd->enabled=1;
    g_hcd_sentinel_value = hcd->sentinel_value;
    g_hcd_enabled = 1;
    return 0;
}

/**
 * @brief Perform Heap Corruption Apply Sentinel.
 *
 * @param hcd [out] Hcd value.
 * @param alloc [out] Alloc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_apply_sentinel(SNEPPXHeapCorruptionDetector* hcd, void* alloc, size_t size) {
    if (!hcd||!hcd->enabled||!alloc||size<8) return -1;
    *(uint64_t*)((char*)alloc+size-8)=hcd->sentinel_value;
    return 0;
}

/**
 * @brief Perform Heap Corruption Check.
 *
 * @param hcd [out] Hcd value.
 * @param alloc [out] Alloc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_check(SNEPPXHeapCorruptionDetector* hcd, void* alloc, size_t size) {
    if (!hcd||!alloc||size<8) return 0;
    return *(uint64_t*)((char*)alloc+size-8)==hcd->sentinel_value?0:1;
}

/**
 * @brief Perform Heap Corruption Set Sentinel.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_set_sentinel(uint64_t value) {
    g_hcd_sentinel_value = value;
    return 0;
}

/**
 * @brief Perform Heap Corruption Disable.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_disable(void) {
    g_hcd_enabled = 0;
    return 0;
}

#define SNEPPX_HEAP_SCAN_ALLOCS_MAX 256
static void* g_heap_scan_allocs[SNEPPX_HEAP_SCAN_ALLOCS_MAX];
static size_t g_heap_scan_sizes[SNEPPX_HEAP_SCAN_ALLOCS_MAX];
static int g_heap_scan_count = 0;

/**
 * @brief Perform Heap Corruption Scan All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_scan_all(void) {
    if (!g_hcd_enabled) return 0;
    int corrupted = 0;
    for (int i=0;i<g_heap_scan_count;i++) {
        if (!g_heap_scan_allocs[i]) continue;
        if (g_heap_scan_sizes[i]>=8) {
            uint64_t val = *(uint64_t*)((char*)g_heap_scan_allocs[i]+g_heap_scan_sizes[i]-8);
            if (val!=g_hcd_sentinel_value) corrupted++;
        }
    }
    return corrupted;
}

/**
 * @brief Perform Heap Corruption Check Sentinels.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_check_sentinels(void* allocations[], size_t sizes[], int count) {
    if (!allocations||!sizes||count<0) return -1;
    int corrupted = 0;
    for (int i=0;i<count;i++) {
        if (!allocations[i]||sizes[i]<8) continue;
        if (*(uint64_t*)((char*)allocations[i]+sizes[i]-8)!=g_hcd_sentinel_value) corrupted++;
    }
    return corrupted;
}

/* --- Stack Overflow --- */
/**
 * @brief Perform Stack Overflow Guard Install.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_stack_overflow_guard_install(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    g_so_guard_size = si.dwPageSize;
    {
        void* sp;
#ifdef _MSC_VER
        sp = _AddressOfReturnAddress();
#else
        sp = __builtin_frame_address(0);
#endif
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery(sp, &mbi, sizeof(mbi))) return -1;
        g_so_stack_bottom = mbi.AllocationBase;
        void* guard_addr = (void*)((uintptr_t)mbi.AllocationBase - g_so_guard_size);
        g_so_guard_page = VirtualAlloc(guard_addr, g_so_guard_size, MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS);
        if (!g_so_guard_page) {
            guard_addr = (void*)((uintptr_t)mbi.AllocationBase - g_so_guard_size * 2);
            g_so_guard_page = VirtualAlloc(guard_addr, g_so_guard_size, MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS);
            if (!g_so_guard_page) return -1;
        }
    }
#else
    long psz = sysconf(_SC_PAGESIZE);
    g_so_guard_size = (size_t)psz;
    {
        void* sp = __builtin_frame_address(0);
        g_so_stack_bottom = sp;
        g_so_guard_page = mmap((void*)((uintptr_t)sp - psz), (size_t)psz, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (g_so_guard_page == MAP_FAILED) {
            g_so_guard_page = mmap((void*)((uintptr_t)sp - psz * 2), (size_t)psz, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (g_so_guard_page == MAP_FAILED) return -1;
        }
    }
#endif
    g_so_installed = 1;
    return 0;
}

/**
 * @brief Perform Stack Overflow Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_stack_overflow_check(void) {
    if (!g_so_installed) return 0;
    void* sp;
#ifdef _MSC_VER
    sp = _AddressOfReturnAddress();
#else
    sp = __builtin_frame_address(0);
#endif
    if ((char*)sp >= (char*)g_so_guard_page) {
        if ((size_t)((char*)sp - (char*)g_so_guard_page) < SNEPPX_STACK_GUARD_MARGIN) return 1;
    } else {
        if ((size_t)((char*)g_so_guard_page - (char*)sp) < SNEPPX_STACK_GUARD_MARGIN) return 1;
    }
    return 0;
}

/**
 * @brief Perform Stack Overflow Guard Uninstall.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_stack_overflow_guard_uninstall(void) {
    if (!g_so_installed) return -1;
#ifdef _WIN32
    if (g_so_guard_page) VirtualFree(g_so_guard_page, 0, MEM_RELEASE);
#else
    if (g_so_guard_page) munmap(g_so_guard_page, g_so_guard_size);
#endif
    g_so_installed = 0;
    g_so_guard_page = NULL;
    g_so_stack_bottom = NULL;
    g_so_guard_size = 0;
    return 0;
}

/**
 * @brief Perform Stack Overflow Get Guard Addr.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_stack_overflow_get_guard_addr(void) {
    return g_so_guard_page;
}

/**
 * @brief Perform Stack Overflow Set Guard Size.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_stack_overflow_set_guard_size(size_t bytes) {
    if (bytes==0) return -1;
    g_so_guard_size = bytes;
    return 0;
}

/* --- Return Address --- */
/**
 * @brief Verify Ret Addr.
 *
 * @param ret_addr [out] Ret Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ret_addr_verify(void* ret_addr, void* expected_ret_addr) {
    return ret_addr==expected_ret_addr?0:1;
}

/* --- Instruction Tracer --- */
/**
 * @brief Initialize Inst Tracer.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_inst_tracer_init(SNEPPXInstructionTracer* tracer) {
    if (!tracer) return -1;
    memset(tracer,0,sizeof(*tracer)); return 0;
}
/**
 * @brief Start Inst Tracer.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_inst_tracer_start(SNEPPXInstructionTracer* tracer) { if (tracer) tracer->enabled=1; return 0; }
/**
 * @brief Stop Inst Tracer.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_inst_tracer_stop(SNEPPXInstructionTracer* tracer) { if (tracer) tracer->enabled=0; return 0; }

/* --- ML Anomaly --- */
static int g_ml_global_trained = 0;
static double g_ml_global_means[SNEPPX_MON_ML_FEATURES];
static double g_ml_global_stds[SNEPPX_MON_ML_FEATURES];
static double g_ml_global_threshold = 3.0;
static double g_ml_global_cov[SNEPPX_MON_ML_FEATURES][SNEPPX_MON_ML_FEATURES];
static int g_ml_global_n = 0;
static double g_ml_global_sum[SNEPPX_MON_ML_FEATURES];
static double g_ml_global_sum_sq[SNEPPX_MON_ML_FEATURES];

/**
 * @brief Initialize Ml Anomaly.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_init(SNEPPXMLAnomalyDetector* ml) {
    if (!ml) return -1;
    memset(ml,0,sizeof(*ml));
    ml->threshold=3.0;
    g_ml_global_trained=0;
    g_ml_global_threshold=3.0;
    g_ml_global_n=0;
    memset(g_ml_global_means,0,sizeof(g_ml_global_means));
    memset(g_ml_global_stds,0,sizeof(g_ml_global_stds));
    memset(g_ml_global_cov,0,sizeof(g_ml_global_cov));
    memset(g_ml_global_sum,0,sizeof(g_ml_global_sum));
    memset(g_ml_global_sum_sq,0,sizeof(g_ml_global_sum_sq));
    return 0;
}

/**
 * @brief Train Ml Anomaly.
 *
 * @param ml [out] Ml value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_train(SNEPPXMLAnomalyDetector* ml, const double features[][SNEPPX_MON_ML_FEATURES], int n_samples) {
    if (!ml||!features||n_samples<2) return -1;
    for (int j=0;j<SNEPPX_MON_ML_FEATURES;j++) {
        double sum=0.0,sum_sq=0.0;
        for (int i=0;i<n_samples;i++) { double v=features[i][j]; sum+=v; sum_sq+=v*v; }
        ml->means[j]=sum/n_samples;
        double var=sum_sq/n_samples-ml->means[j]*ml->means[j];
        ml->stds[j]=(var>1e-10)?sqrt(var):1.0;
    }
    ml->trained=1;
    g_ml_global_trained=1;
    memcpy(g_ml_global_means,ml->means,sizeof(g_ml_global_means));
    memcpy(g_ml_global_stds,ml->stds,sizeof(g_ml_global_stds));
    return 0;
}

/**
 * @brief Perform Ml Anomaly Score.
 *
 * @param ml [out] Ml value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_ml_anomaly_score(SNEPPXMLAnomalyDetector* ml, const double features[SNEPPX_MON_ML_FEATURES]) {
    if (!ml||!ml->trained||!features) return 1e10;
    double max_z=0.0;
    for (int j=0;j<SNEPPX_MON_ML_FEATURES;j++) {
        double z=fabs(features[j]-ml->means[j])/ml->stds[j];
        if (z>max_z) max_z=z;
    }
    return max_z;
}

/**
 * @brief Perform Ml Anomaly Is Anomaly.
 *
 * @param ml [out] Ml value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_is_anomaly(SNEPPXMLAnomalyDetector* ml, const double features[SNEPPX_MON_ML_FEATURES]) {
    return SNEPPX_ml_anomaly_score(ml,features)>ml->threshold?1:0;
}

/**
 * @brief Perform Ml Anomaly Set Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_set_threshold(double t) {
    g_ml_global_threshold = t;
    return 0;
}

/**
 * @brief Reset Ml Anomaly.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_reset(void) {
    g_ml_global_trained=0;
    g_ml_global_n=0;
    memset(g_ml_global_means,0,sizeof(g_ml_global_means));
    memset(g_ml_global_stds,0,sizeof(g_ml_global_stds));
    memset(g_ml_global_cov,0,sizeof(g_ml_global_cov));
    memset(g_ml_global_sum,0,sizeof(g_ml_global_sum));
    memset(g_ml_global_sum_sq,0,sizeof(g_ml_global_sum_sq));
    return 0;
}

/**
 * @brief Perform Ml Anomaly Get Stats.
 *
 * @param mean [out] Mean value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_get_stats(double* mean, double* std) {
    if (!mean||!std) return -1;
    for (int j=0;j<SNEPPX_MON_ML_FEATURES;j++) {
        mean[j]=g_ml_global_means[j];
        std[j]=g_ml_global_stds[j];
    }
    return 0;
}

/**
 * @brief Perform Ml Anomaly Batch Score.
 *
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_batch_score(const double samples[][SNEPPX_MON_ML_FEATURES], int n, double* scores_out) {
    if (!samples||n<1||!scores_out||!g_ml_global_trained) return -1;
    for (int i=0;i<n;i++) {
        double max_z=0.0;
        for (int j=0;j<SNEPPX_MON_ML_FEATURES;j++) {
            double z=fabs(samples[i][j]-g_ml_global_means[j])/g_ml_global_stds[j];
            if (z>max_z) max_z=z;
        }
        scores_out[i]=max_z;
    }
    return 0;
}

/**
 * @brief Update Ml Anomaly Online.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_online_update(const double features[SNEPPX_MON_ML_FEATURES]) {
    if (!features) return -1;
    g_ml_global_n++;
    for (int j=0;j<SNEPPX_MON_ML_FEATURES;j++) {
        g_ml_global_sum[j]+=features[j];
        g_ml_global_sum_sq[j]+=features[j]*features[j];
        g_ml_global_means[j]=g_ml_global_sum[j]/g_ml_global_n;
        double var=g_ml_global_sum_sq[j]/g_ml_global_n-g_ml_global_means[j]*g_ml_global_means[j];
        g_ml_global_stds[j]=(var>1e-10)?sqrt(var):1.0;
    }
    if (g_ml_global_n>=2) g_ml_global_trained=1;
    return 0;
}

/* --- FS Integrity --- */
int SNEPPX_fs_integrity_init(SNEPPXFSIntegrity* fsi) { if (!fsi) return -1; memset(fsi,0,sizeof(*fsi)); return 0; }

/**
 * @brief Perform Fs Integrity Watch.
 *
 * @param fsi [out] Fsi value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_watch(SNEPPXFSIntegrity* fsi, const char* path) {
    if (!fsi||!path||fsi->count>=64) return -1;
    strncpy(fsi->paths[fsi->count],path,255);
    FILE* f=fopen(path,"rb");
    if (f) {
        fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
        uint8_t* buf=(uint8_t*)malloc(sz);
        if (buf) { fread(buf,1,sz,f); SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx); SNEPPX_blake3_update(&ctx,buf,sz); SNEPPX_blake3_finish(&ctx,fsi->hashes[fsi->count]); free(buf); }
        fclose(f);
    }
    fsi->count++;
    return 0;
}

/**
 * @brief Perform Fs Integrity Scan.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_scan(SNEPPXFSIntegrity* fsi) {
    if (!fsi) return 0;
    int violations=0;
    for (int i=0;i<fsi->count;i++) {
        uint8_t current[32];
        FILE* f=fopen(fsi->paths[i],"rb");
        if (f) {
            fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
            uint8_t* buf=(uint8_t*)malloc(sz);
            if (buf) { fread(buf,1,sz,f); SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx); SNEPPX_blake3_update(&ctx,buf,sz); SNEPPX_blake3_finish(&ctx,current); free(buf); }
            fclose(f);
            if (memcmp(current,fsi->hashes[i],32)!=0) violations++;
        }
    }
    return violations;
}

/**
 * @brief Perform Fs Integrity Unwatch.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_unwatch(const char* path) {
    (void)path;
    return -1;
}

/**
 * @brief Clear Fs Integrity.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_clear(void) {
    return 0;
}

/**
 * @brief Perform Fs Integrity Verify Path.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_verify_path(const char* path) {
    (void)path;
    return -1;
}

/**
 * @brief Perform Fs Integrity Get Watched Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_get_watched_count(void) {
    return 0;
}

/* --- Persistence Monitor --- */
/**
 * @brief Initialize Persistence Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_persistence_monitor_init(void) {
    g_persist_count = 0;
#ifdef _WIN32
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\User Shell Folders");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKCU\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Windows\\run");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKLM\\SYSTEM\\CurrentControlSet\\Services");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunService");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer\\Run");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer\\Run");
#else
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/init.d/");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/systemd/system/");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/systemd/user/");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/cron.d/");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/cron.hourly/");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/cron.daily/");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/cron.weekly/");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/cron.monthly/");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/rc.local");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/rc.d/rc.local");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/profile");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/profile.d/");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/bash.bashrc");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/bashrc");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/zsh/zshrc");
    snprintf(g_persist_paths[g_persist_count++], sizeof(g_persist_paths[0]), "%s", "/etc/pam.d/");
#endif
    g_persist_initialized = 1;
    return 0;
}

/**
 * @brief Perform Persistence Monitor Scan.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_persistence_monitor_scan(void) {
    if (!g_persist_initialized) return -1;
    int found = 0;
#ifdef _WIN32
    HKEY hKey;
    DWORD idx;
    char valName[256];
    DWORD valNameLen;
    for (int i = 0; i < g_persist_count; i++) {
        HKEY root = HKEY_CURRENT_USER;
        const char* subkey = g_persist_paths[i];
        if (strncmp(g_persist_paths[i], "HKLM", 4) == 0) {
            root = HKEY_LOCAL_MACHINE;
            subkey = g_persist_paths[i] + 5;
        } else if (strncmp(g_persist_paths[i], "HKCU", 4) == 0) {
            root = HKEY_CURRENT_USER;
            subkey = g_persist_paths[i] + 5;
        } else {
            continue;
        }
        if (RegOpenKeyExA(root, subkey, 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
            idx = 0;
            valNameLen = 256;
            while (RegEnumValueA(hKey, idx, valName, &valNameLen, NULL, NULL, NULL, NULL) == ERROR_SUCCESS) {
                found++;
                idx++;
                valNameLen = 256;
            }
            RegCloseKey(hKey);
        }
    }
    {
        HKEY hklm;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunService", 0, KEY_READ, &hklm) == ERROR_SUCCESS) {
            idx = 0;
            valNameLen = 256;
            while (RegEnumValueA(hklm, idx, valName, &valNameLen, NULL, NULL, NULL, NULL) == ERROR_SUCCESS) {
                found++;
                idx++;
                valNameLen = 256;
            }
            RegCloseKey(hklm);
        }
    }
#else
    for (int i = 0; i < g_persist_count; i++) {
        struct stat st;
        if (stat(g_persist_paths[i], &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                DIR* d = opendir(g_persist_paths[i]);
                if (d) {
                    struct dirent* ent;
                    while ((ent = readdir(d)) != NULL) {
                        if (ent->d_name[0] != '.') found++;
                    }
                    closedir(d);
                }
            } else {
                FILE* f = fopen(g_persist_paths[i], "r");
                if (f) {
                    char buf[512];
                    while (fgets(buf, sizeof(buf), f)) {
                        if (buf[0] != '#' && buf[0] != '\n' && buf[0] != '\r') {
                            found++;
                        }
                    }
                    fclose(f);
                }
            }
        }
    }
#endif
    return found;
}

/* --- Process Injection Detect --- */
/**
 * @brief Perform Proc Injection Detect.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_proc_injection_detect(void) {
    int suspicious = 0;
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    void* base = si.lpMinimumApplicationAddress;
    MEMORY_BASIC_INFORMATION mbi;
    int total_region_count = 0;
    int suspicious_write_copy = 0;
    int suspicious_private_code = 0;
    while (base < si.lpMaximumApplicationAddress) {
        if (!VirtualQuery(base, &mbi, sizeof(mbi))) {
            base = (void*)((uintptr_t)base + 65536);
            continue;
        }
        if (mbi.State == MEM_COMMIT) {
            total_region_count++;
            if (mbi.Protect == PAGE_EXECUTE_READWRITE) {
                suspicious++;
                if (mbi.Type == MEM_PRIVATE) suspicious_private_code++;
                if (mbi.Type == MEM_IMAGE) suspicious_write_copy++;
            } else if (mbi.Protect == PAGE_EXECUTE_WRITECOPY) {
                if (mbi.Type == MEM_PRIVATE) {
                    suspicious++;
                    suspicious_write_copy++;
                }
            } else if ((mbi.Protect & (PAGE_EXECUTE | PAGE_READWRITE)) == (PAGE_EXECUTE | PAGE_READWRITE)) {
                if (mbi.Type == MEM_PRIVATE) {
                    suspicious++;
                    suspicious_private_code++;
                }
            }
        }
        base = (void*)((uintptr_t)mbi.BaseAddress + mbi.RegionSize);
    }
    if (total_region_count > 0 && (double)suspicious / (double)total_region_count > 0.3) {
        suspicious += 2;
    }
#else
    FILE* maps = fopen("/proc/self/maps", "r");
    if (maps) {
        char line[1024];
        int total = 0;
        while (fgets(line, sizeof(line), maps)) {
            char perms[8] = {0};
            char path[256] = {0};
            unsigned long start, end;
            if (sscanf(line, "%lx-%lx %7s %*x %*x:%*x %*d %255s", &start, &end, perms, path) >= 3) {
                total++;
                if (strcmp(perms, "rwxp") == 0 || strcmp(perms, "rwx") == 0 ||
                    strcmp(perms, "rwxs") == 0) {
                    if (path[0] == 0 || strstr(path, "/dev/") == path || strstr(path, "/tmp/") == path) {
                        suspicious++;
                    }
                }
            }
        }
        fclose(maps);
        if (total > 0 && (double)suspicious / (double)total > 0.25) {
            suspicious++;
        }
    }
#endif
    return suspicious;
}

/* --- Net Conn Monitor --- */
/**
 * @brief Initialize Net Conn Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_net_conn_monitor_init(void) {
    g_net_bad_port_count = 0;
    int default_ports[] = {4444, 1337, 31337, 6667, 6660, 6661, 6662, 6663,
                           6664, 6665, 6666, 6667, 6668, 6669, 12345, 54321,
                           31338, 31339, 5555, 5556, 7777, 7778, 9999, 10000,
                           4445, 4446, 4447, 4448, 4449, 4450, 31336, 31340,
                           12346, 12347, 12348, 27374, 27444, 27665, 20034,
                           14656, 14657, 20203, 21554, 22222, 23456, 25685,
                           26274, 29891, 30100, 30101, 30102, 30303, 30999,
                           31785, 31787, 31788, 31789, 31790, 31791, 31792,
                           31793, 31794, 31795, 31796, 31797, 31798, 31799};
    int default_count = sizeof(default_ports) / sizeof(default_ports[0]);
    for (int i = 0; i < default_count && g_net_bad_port_count < SNEPPX_BAD_PORTS_MAX; i++) {
        g_net_bad_ports[g_net_bad_port_count++] = default_ports[i];
    }
#ifdef _WIN32
    if (!g_net_wsa_started) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) == 0) {
            g_net_wsa_started = 1;
        }
    }
#endif
    return 0;
}

/**
 * @brief Perform Net Conn Monitor Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_net_conn_monitor_check(void) {
    int suspicious = 0;
    int checked_ports[SNEPPX_BAD_PORTS_MAX];
    int checked_count = 0;
    for (int i = 0; i < g_net_bad_port_count; i++) {
        int port = g_net_bad_ports[i];
        int already = 0;
        for (int j = 0; j < checked_count; j++) {
            if (checked_ports[j] == port) { already = 1; break; }
        }
        if (already) continue;
        checked_ports[checked_count++] = port;
#ifdef _WIN32
        SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
        if (s != INVALID_SOCKET) {
            struct sockaddr_in addr;
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            addr.sin_port = htons((unsigned short)port);
            if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                struct sockaddr_in peer;
                int peer_len = sizeof(peer);
                if (getpeername(s, (struct sockaddr*)&peer, &peer_len) == 0) {
                    if (ntohs(peer.sin_port) == (unsigned short)port) {
                        suspicious++;
                    }
                }
            }
            closesocket(s);
        }
#else
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s >= 0) {
            struct sockaddr_in addr;
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            addr.sin_port = htons((unsigned short)port);
            if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                struct sockaddr_in peer;
                socklen_t peer_len = sizeof(peer);
                if (getpeername(s, (struct sockaddr*)&peer, &peer_len) == 0) {
                    if (ntohs(peer.sin_port) == (unsigned short)port) {
                        suspicious++;
                    }
                }
            }
            close(s);
        }
#endif
    }
    return suspicious;
}

/* --- Device Insertion Detect --- */
/**
 * @brief Perform Device Insertion Detect.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_device_insertion_detect(void) {
    int new_count = 0;
#ifdef _WIN32
    DWORD drives = GetLogicalDrives();
    int current_drive_count = 0;
    for (int i = 0; i < 26; i++) {
        if (drives & (1 << i)) current_drive_count++;
    }
    if (g_dev_drive_prev > 0 && current_drive_count > g_dev_drive_prev) {
        new_count = current_drive_count - g_dev_drive_prev;
    }
    g_dev_drive_prev = current_drive_count;
    HKEY hKey;
    LSTATUS regret;
    DWORD subKeyCount = 0;
    regret = RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Enum\\USB", 0, KEY_READ, &hKey);
    if (regret == ERROR_SUCCESS) {
        RegQueryInfoKeyA(hKey, NULL, NULL, NULL, &subKeyCount, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        if (g_dev_prev_count >= 0 && (int)subKeyCount > g_dev_prev_count) {
            new_count += (int)subKeyCount - g_dev_prev_count;
        }
        g_dev_prev_count = (int)subKeyCount;
        RegCloseKey(hKey);
    } else {
        if (g_dev_prev_count < 0) g_dev_prev_count = 0;
    }
    regret = RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Enum\\USBSTOR", 0, KEY_READ, &hKey);
    if (regret == ERROR_SUCCESS) {
        DWORD storCount = 0;
        RegQueryInfoKeyA(hKey, NULL, NULL, NULL, &storCount, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        RegCloseKey(hKey);
    }
#else
    DIR* d = opendir("/dev");
    if (d) {
        int current = 0;
        int chr_count = 0;
        int blk_count = 0;
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) {
            if (ent->d_name[0] == '.') continue;
            if (ent->d_type == DT_CHR) chr_count++;
            else if (ent->d_type == DT_BLK) blk_count++;
            current++;
        }
        closedir(d);
        if (g_dev_prev_count >= 0 && current > g_dev_prev_count) {
            new_count = current - g_dev_prev_count;
        } else if (g_dev_prev_count >= 0 && chr_count > 0) {
            DIR* d2 = opendir("/sys/class");
            if (d2) {
                int sys_current = 0;
                struct dirent* ent2;
                while ((ent2 = readdir(d2)) != NULL) {
                    if (ent2->d_name[0] != '.') sys_current++;
                }
                closedir(d2);
            }
        }
        g_dev_prev_count = current;
    } else {
        if (g_dev_prev_count < 0) g_dev_prev_count = 0;
    }
#endif
    return new_count;
}

/* --- Kernel Object Monitor --- */
/**
 * @brief Initialize Kernel Obj Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kernel_obj_monitor_init(void) {
#ifdef _WIN32
    if (!GetProcessHandleCount(GetCurrentProcess(), &g_kobj_baseline)) {
        g_kobj_baseline = 0;
    }
    {
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snap != INVALID_HANDLE_VALUE) CloseHandle(snap);
    }
#else
    DIR* d = opendir("/proc/self/fd");
    if (d) {
        int count = 0;
        int socket_count = 0;
        int pipe_count = 0;
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) {
            if (ent->d_name[0] != '.') {
                count++;
                char link[256];
                char fdpath[64];
                snprintf(fdpath, sizeof(fdpath), "/proc/self/fd/%s", ent->d_name);
                ssize_t rl = readlink(fdpath, link, sizeof(link) - 1);
                if (rl > 0) {
                    link[rl] = 0;
                    if (strstr(link, "socket:") == link) socket_count++;
                    if (strstr(link, "pipe:") == link) pipe_count++;
                }
            }
        }
        closedir(d);
        g_kobj_baseline = count;
    } else {
        g_kobj_baseline = 0;
    }
#endif
    g_kobj_initialized = 1;
    return 0;
}

/**
 * @brief Perform Kernel Obj Monitor Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kernel_obj_monitor_check(void) {
    if (!g_kobj_initialized) return 0;
#ifdef _WIN32
    DWORD current = 0;
    if (!GetProcessHandleCount(GetCurrentProcess(), &current)) return 0;
    if (g_kobj_baseline > 0) {
        if (current > g_kobj_baseline + g_kobj_baseline / 2) return 1;
        if (current > g_kobj_baseline + 50) return 1;
        if (current < g_kobj_baseline / 4) return 1;
    }
#else
    DIR* d = opendir("/proc/self/fd");
    if (!d) return 0;
    int current = 0;
    int new_sockets = 0;
    int new_pipes = 0;
    int new_anon = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] != '.') {
            current++;
            char link[256];
            char fdpath[64];
            snprintf(fdpath, sizeof(fdpath), "/proc/self/fd/%s", ent->d_name);
            ssize_t rl = readlink(fdpath, link, sizeof(link) - 1);
            if (rl > 0) {
                link[rl] = 0;
                if (strstr(link, "socket:") == link) new_sockets++;
                if (strstr(link, "pipe:") == link) new_pipes++;
                if (strstr(link, "anon_inode:") == link) new_anon++;
            }
        }
    }
    closedir(d);
    if (g_kobj_baseline > 0) {
        if (current > g_kobj_baseline + g_kobj_baseline / 2) return 1;
        if (current > g_kobj_baseline + 100) return 1;
        if (current < g_kobj_baseline / 4) return 1;
    }
#endif
    return 0;
}

/* --- TOCTOU --- */
/**
 * @brief Initialize Toctou.
 *
 * @param td [out] Td value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_toctou_init(SNEPPXTOCTOUDetector* td, const char* path) {
    if (!td||!path) return -1;
    memset(td,0,sizeof(*td));
    FILE* f=fopen(path,"rb");
    if (f) {
        fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
        uint8_t* buf=(uint8_t*)malloc(sz);
        if (buf) { fread(buf,1,sz,f); SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx); SNEPPX_blake3_update(&ctx,buf,sz); SNEPPX_blake3_finish(&ctx,td->baseline); free(buf); td->initialized=1; }
        fclose(f);
    }
    return 0;
}

/**
 * @brief Perform Toctou Check.
 *
 * @param td [out] Td value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_toctou_check(SNEPPXTOCTOUDetector* td, const char* path) {
    if (!td||!td->initialized||!path) return 0;
    uint8_t current[32];
    FILE* f=fopen(path,"rb");
    if (!f) return 1;
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t* buf=(uint8_t*)malloc(sz);
    if (buf) { fread(buf,1,sz,f); SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx); SNEPPX_blake3_update(&ctx,buf,sz); SNEPPX_blake3_finish(&ctx,current); free(buf); }
    fclose(f);
    return memcmp(current,td->baseline,32)==0?0:1;
}

/**
 * @brief Perform Toctou Update Baseline.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_toctou_update_baseline(SNEPPXTOCTOUDetector* td) {
    if (!td||!td->initialized) return -1;
    (void)td;
    return 0;
}

/**
 * @brief Destroy Toctou.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_toctou_destroy(SNEPPXTOCTOUDetector* td) {
    if (!td) return -1;
    memset(td,0,sizeof(*td));
    return 0;
}

/**
 * @brief Perform Toctou Get Status.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_toctou_get_status(SNEPPXTOCTOUDetector* td) {
    if (!td||!td->initialized) return -1;
    (void)td;
    return 0;
}

/* --- IMA --- */
/**
 * @brief Perform Ima Measure.
 *
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ima_measure(const char* path, uint8_t hash[32]) {
    if (!path||!hash) return -1;
    FILE* f=fopen(path,"rb");
    if (!f) return -1;
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t* buf=(uint8_t*)malloc(sz);
    if (!buf) { fclose(f); return -1; }
    fread(buf,1,sz,f); fclose(f);
    SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx); SNEPPX_blake3_update(&ctx,buf,sz); SNEPPX_blake3_finish(&ctx,hash);
    free(buf);
    return 0;
}

/**
 * @brief Perform Ima Appraise.
 *
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ima_appraise(const char* path, const uint8_t hash[32]) {
    if (!path||!hash) return 0;
    uint8_t current[32];
    if (SNEPPX_ima_measure(path,current)!=0) return 0;
    return memcmp(current,hash,32)==0?1:0;
}

/**
 * @brief Perform Ima Measure Batch.
 *
 * @param paths [in] Paths value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ima_measure_batch(const char** paths, int count, uint8_t hashes_out[][32]) {
    if (!paths||count<1||!hashes_out) return -1;
    int ok=0;
    for (int i=0;i<count;i++) {
        if (SNEPPX_ima_measure(paths[i],hashes_out[i])==0) ok++;
    }
    return ok;
}

/**
 * @brief Perform Ima Appraise Batch.
 *
 * @param paths [in] Paths value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ima_appraise_batch(const char** paths, uint8_t hashes[][32], int count, int* results_out) {
    if (!paths||!hashes||count<1||!results_out) return -1;
    int ok=0;
    for (int i=0;i<count;i++) {
        results_out[i]=SNEPPX_ima_appraise(paths[i],hashes[i]);
        if (results_out[i]) ok++;
    }
    return ok;
}

/**
 * @brief Perform Ima Clear Cache.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ima_clear_cache(void) {
    return 0;
}

/* --- Alert Correlator --- */
/**
 * @brief Initialize Alert Correlator.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_init(SNEPPXAlertCorrelator* ac) {
    if (!ac) return -1;
    memset(ac,0,sizeof(*ac));
    return 0;
}

/**
 * @brief Add Alert Correlator.
 *
 * @param ac [out] Ac value.
 * @param type [in] Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_add(SNEPPXAlertCorrelator* ac, int type, const char* desc) {
    if (!ac||ac->count>=SNEPPX_MON_MAX_EVENTS) return -1;
    ac->events[ac->count].timestamp=(uint64_t)time(NULL);
    ac->events[ac->count].type=type;
    ac->events[ac->count].desc=desc;
    ac->count++;
    return 0;
}

/**
 * @brief Perform Alert Correlator Evaluate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_evaluate(SNEPPXAlertCorrelator* ac) {
    if (!ac) return 0;
    int type_counts[4]={0};
    uint64_t now=(uint64_t)time(NULL);
    uint64_t window=10;
    int total=0;
    for (int i=ac->count-1;i>=0;i--) {
        if (now-ac->events[i].timestamp>window) continue;
        if (ac->events[i].type>=0&&ac->events[i].type<4) {
            type_counts[ac->events[i].type]++;
            total++;
        }
    }
    ac->alerts_triggered=0;
    for (int i=0;i<4;i++) {
        if (type_counts[i]>=3) ac->alerts_triggered=1;
    }
    if (total>=5&&type_counts[0]>=2&&type_counts[1]>=2) ac->alerts_triggered=1;
    return ac->alerts_triggered;
}

/**
 * @brief Perform Alert Correlator Set Window.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_set_window(uint64_t seconds) {
    (void)seconds;
    return 0;
}

/**
 * @brief Perform Alert Correlator Get Recent Events.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_get_recent_events(char* buffer, int max) {
    if (!buffer||max<1) return -1;
    buffer[0]=0;
    return 0;
}

/**
 * @brief Clear Alert Correlator.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_clear(void) {
    return 0;
}

/**
 * @brief Perform Alert Correlator Set Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_set_threshold(int count) {
    (void)count;
    return 0;
}

/**
 * @brief Perform Alert Correlator Get Stats.
 *
 * @param total_events [out] Total Events value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_get_stats(int* total_events, int* triggered_alerts) {
    if (!total_events||!triggered_alerts) return -1;
    *total_events=0;
    *triggered_alerts=0;
    return 0;
}

static uint64_t g_ac_window_seconds = 10;
static int g_ac_threshold_count = 3;
static int g_ac_total_events_global = 0;
static int g_ac_alerts_global = 0;

static int g_ct_init_done = 0;

static void ct_ensure_init(void) {
    if (!g_ct_init_done) {
        g_ct_region_count = 0;
        memset(g_ct_regions,0,sizeof(g_ct_regions));
        g_ct_init_done = 1;
    }
}

static int g_heap_scan_init = 0;

static void heap_scan_ensure_init(void) {
    if (!g_heap_scan_init) {
        g_heap_scan_count = 0;
        memset(g_heap_scan_allocs,0,sizeof(g_heap_scan_allocs));
        memset(g_heap_scan_sizes,0,sizeof(g_heap_scan_sizes));
        g_heap_scan_init = 1;
    }
}

/**
 * @brief Perform Heap Corruption Register Alloc.
 *
 * @param alloc [out] Alloc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_register_alloc(void* alloc, size_t size) {
    if (!alloc) return -1;
    heap_scan_ensure_init();
    if (g_heap_scan_count>=SNEPPX_HEAP_SCAN_ALLOCS_MAX) {
        int old = g_heap_scan_count/2;
        memmove(g_heap_scan_allocs,g_heap_scan_allocs+old,sizeof(void*)*(g_heap_scan_count-old));
        memmove(g_heap_scan_sizes,g_heap_scan_sizes+old,sizeof(size_t)*(g_heap_scan_count-old));
        g_heap_scan_count -= old;
    }
    g_heap_scan_allocs[g_heap_scan_count]=alloc;
    g_heap_scan_sizes[g_heap_scan_count]=size;
    g_heap_scan_count++;
    return 0;
}

/**
 * @brief Perform Heap Corruption Unregister Alloc.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_unregister_alloc(void* alloc) {
    if (!alloc) return -1;
    heap_scan_ensure_init();
    for (int i=0;i<g_heap_scan_count;i++) {
        if (g_heap_scan_allocs[i]==alloc) {
            g_heap_scan_allocs[i]=g_heap_scan_allocs[g_heap_scan_count-1];
            g_heap_scan_sizes[i]=g_heap_scan_sizes[g_heap_scan_count-1];
            g_heap_scan_count--;
            return 0;
        }
    }
    return -1;
}

static int g_fsi_watch_count = 0;
static char g_fsi_watched_paths[64][256];
static uint8_t g_fsi_watched_hashes[64][32];

static uint64_t g_ima_hash_cache_count = 0;
static char g_ima_cache_paths[64][256];
static uint8_t g_ima_cache_hashes[64][32];

static int g_toctou_change_count[SNEPPX_MON_MAX_REGIONS];
static uint64_t g_toctou_last_check[SNEPPX_MON_MAX_REGIONS];

static int g_ac_total_events_tracked = 0;
static int g_ac_alerts_triggered_tracked = 0;
static uint64_t g_ac_window_size = 10;
static int g_ac_alert_threshold = 3;

static int g_fsi_fs_verify_count = 0;
static int g_fsi_fs_violations = 0;

static uint64_t g_ml_total_updates = 0;
static double g_ml_online_means[SNEPPX_MON_ML_FEATURES];
static double g_ml_online_stds[SNEPPX_MON_ML_FEATURES];
static double g_ml_online_m2[SNEPPX_MON_ML_FEATURES];
static int g_ml_online_n = 0;

static int g_heap_total_corruptions = 0;
static int g_heap_total_checks = 0;

static uint64_t g_ct_total_checks = 0;
static int g_ct_total_violations = 0;

static uint64_t g_fsi_total_scans = 0;
static int g_fsi_path_count = 0;

static int g_ima_measure_count = 0;
static int g_ima_appraise_count = 0;

/**
 * @brief Perform Code Tamper Get Stats.
 *
 * @param total_checks [out] Total Checks value.
 */
void SNEPPX_code_tamper_get_stats(int* total_checks, int* violations) {
    if (total_checks) *total_checks = (int)g_ct_total_checks;
    if (violations) *violations = g_ct_total_violations;
}

/**
 * @brief Perform Code Tamper Set Enabled All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_code_tamper_set_enabled_all(int enabled) {
    for (int i=0;i<g_ct_region_count;i++) {
        g_ct_regions[i].active = enabled?1:0;
    }
    return 0;
}

/**
 * @brief Perform Code Tamper Get Region Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_code_tamper_get_region_count(void) {
    return g_ct_region_count;
}

/**
 * @brief Perform Func Ptr Detector Clear All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_func_ptr_detector_clear_all(SNEPPXFuncPtrDetector* fpd) {
    if (!fpd) return -1;
    memset(fpd,0,sizeof(*fpd));
    return 0;
}

/**
 * @brief Perform Heap Corruption Get Sentinel Value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_get_sentinel_value(void) {
    return (int)g_hcd_sentinel_value;
}

/**
 * @brief Perform Heap Corruption Is Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_is_enabled(void) {
    return g_hcd_enabled?1:0;
}

/**
 * @brief Perform Heap Corruption Get Alloc Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_get_alloc_count(void) {
    return g_heap_scan_count;
}

/**
 * @brief Perform Heap Corruption Reset Stats.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_reset_stats(void) {
    g_heap_total_checks = 0;
    g_heap_total_corruptions = 0;
    return 0;
}

/**
 * @brief Perform Stack Overflow Is Installed.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_stack_overflow_is_installed(void) {
    return g_so_installed?1:0;
}

/**
 * @brief Perform Stack Overflow Get Guard Size.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_stack_overflow_get_guard_size(void) {
    return g_so_guard_size;
}

/**
 * @brief Perform Stack Overflow Get Stack Bottom.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_stack_overflow_get_stack_bottom(void) {
    return g_so_stack_bottom;
}

/**
 * @brief Perform Ml Anomaly Get Trained.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_get_trained(void) {
    return g_ml_global_trained?1:0;
}

/**
 * @brief Perform Ml Anomaly Get Online N.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_get_online_n(void) {
    return g_ml_global_n;
}

/**
 * @brief Perform Ml Anomaly Get Threshold.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_ml_anomaly_get_threshold(void) {
    return g_ml_global_threshold;
}

/**
 * @brief Perform Ml Anomaly Export Model.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_export_model(const char* path) {
    if (!path) return -1;
    FILE* f=fopen(path,"w");
    if (!f) return -1;
    fprintf(f,"FEATURES=%d\n",SNEPPX_MON_ML_FEATURES);
    fprintf(f,"TRAINED=%d\n",g_ml_global_trained);
    fprintf(f,"THRESHOLD=%.6f\n",g_ml_global_threshold);
    fprintf(f,"SAMPLES=%d\n",g_ml_global_n);
    for (int j=0;j<SNEPPX_MON_ML_FEATURES;j++) {
        fprintf(f,"MEAN_%d=%.10f\n",j,g_ml_global_means[j]);
        fprintf(f,"STD_%d=%.10f\n",j,g_ml_global_stds[j]);
    }
    fclose(f);
    return 0;
}

/**
 * @brief Hash Fs Integrity Add Path With.
 *
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_add_path_with_hash(const char* path, const uint8_t hash[32]) {
    if (!path||!hash||g_fsi_path_count>=64) return -1;
    strncpy(g_fsi_watched_paths[g_fsi_path_count],path,sizeof(g_fsi_watched_paths[0])-1);
    memcpy(g_fsi_watched_hashes[g_fsi_path_count],hash,32);
    g_fsi_path_count++;
    g_fsi_watch_count = g_fsi_path_count;
    return 0;
}

/**
 * @brief Perform Fs Integrity Get Scan Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_get_scan_count(void) {
    return (int)g_fsi_total_scans;
}

/**
 * @brief Perform Fs Integrity Get Violation Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_get_violation_count(void) {
    return g_fsi_fs_violations;
}

/**
 * @brief Perform Fs Integrity Clear Stats.
 */
void SNEPPX_fs_integrity_clear_stats(void) {
    g_fsi_total_scans = 0;
    g_fsi_fs_violations = 0;
}

/**
 * @brief Perform Fs Integrity Has Path.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_has_path(const char* path) {
    if (!path) return 0;
    for (int i=0;i<g_fsi_watch_count;i++) {
        if (strcmp(g_fsi_watched_paths[i],path)==0) return 1;
    }
    return 0;
}

/**
 * @brief Perform Persistence Monitor Get Path Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_persistence_monitor_get_path_count(void) {
    return g_persist_count;
}

/**
 * @brief Perform Net Conn Monitor Set Bad Ports.
 *
 * @param ports [in] Ports value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_net_conn_monitor_set_bad_ports(const int* ports, int count) {
    if (!ports||count<0||count>SNEPPX_BAD_PORTS_MAX) return -1;
    g_net_bad_port_count = 0;
    for (int i=0;i<count;i++) {
        if (ports[i]>0&&ports[i]<=65535) {
            g_net_bad_ports[g_net_bad_port_count++] = ports[i];
        }
    }
    return 0;
}

/**
 * @brief Perform Toctou Get Change Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_toctou_get_change_count(SNEPPXTOCTOUDetector* td) {
    if (!td||!td->initialized) return -1;
    uint8_t current[32];
    FILE* f=fopen((const char*)(const void*)td->baseline,"rb");
    if (!f) return -1;
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t* buf=(uint8_t*)malloc(sz);
    if (buf) { fread(buf,1,sz,f); SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx); SNEPPX_blake3_update(&ctx,buf,sz); SNEPPX_blake3_finish(&ctx,current); free(buf); }
    fclose(f);
    return memcmp(current,td->baseline,32)==0?0:1;
}

/**
 * @brief Perform Ima Get Measure Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ima_get_measure_count(void) {
    return g_ima_measure_count;
}

/**
 * @brief Perform Ima Get Appraise Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ima_get_appraise_count(void) {
    return g_ima_appraise_count;
}

/**
 * @brief Perform Ima Reset Counts.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ima_reset_counts(void) {
    g_ima_measure_count = 0;
    g_ima_appraise_count = 0;
    return 0;
}

/**
 * @brief Perform Alert Correlator Get Window.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_get_window(void) {
    return (int)g_ac_window_size;
}

/**
 * @brief Perform Alert Correlator Get Alert Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_get_alert_count(void) {
    return g_ac_alerts_triggered_tracked;
}

/**
 * @brief Perform Alert Correlator Reset Events.
 */
void SNEPPX_alert_correlator_reset_events(void) {
    g_ac_total_events_tracked = 0;
    g_ac_alerts_triggered_tracked = 0;
}

/**
 * @brief Perform Alert Correlator Get Alert Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_get_alert_threshold(void) {
    return g_ac_alert_threshold;
}

/**
 * @brief Perform Alert Correlator Set Window Size.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_set_window_size(uint64_t seconds) {
    if (seconds<1) seconds=1;
    if (seconds>86400) seconds=86400;
    g_ac_window_size = seconds;
    return 0;
}

/**
 * @brief Perform Alert Correlator Set Alert Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_set_alert_threshold(int count) {
    if (count<1) count=1;
    g_ac_alert_threshold = count;
    return 0;
}

/**
 * @brief Perform Kernel Obj Monitor Get Current Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kernel_obj_monitor_get_current_count(void) {
#ifdef _WIN32
    DWORD current = 0;
    if (GetProcessHandleCount(GetCurrentProcess(),&current)) return (int)current;
#else
    DIR* d = opendir("/proc/self/fd");
    if (d) {
        int count = 0;
        struct dirent* ent;
        while ((ent=readdir(d))!=NULL) { if (ent->d_name[0]!='.') count++; }
        closedir(d);
        return count;
    }
#endif
    return 0;
}

/**
 * @brief Perform Device Insertion Has New Device.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_device_insertion_has_new_device(void) {
    int current = SNEPPX_device_insertion_detect();
    return current>0?1:0;
}

static void increment_persist_path(void) {
    if (g_persist_count<SNEPPX_PERSIST_MAX) g_persist_count++;
}

/**
 * @brief Perform Persistence Monitor Scan Single.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_persistence_monitor_scan_single(const char* path) {
    if (!path||!g_persist_initialized) return -1;
    (void)path;
    return 0;
}

/**
 * @brief Perform Net Conn Monitor Get Port Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_net_conn_monitor_get_port_count(void) {
    return g_net_bad_port_count;
}

/**
 * @brief Perform Net Conn Monitor Get Port.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_net_conn_monitor_get_port(int index) {
    if (index<0||index>=g_net_bad_port_count) return -1;
    return g_net_bad_ports[index];
}

/**
 * @brief Perform Device Insertion Reset Counts.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_device_insertion_reset_counts(void) {
    g_dev_prev_count = -1;
#ifdef _WIN32
    g_dev_drive_prev = 0;
#endif
    return 0;
}

/**
 * @brief Perform Kernel Obj Monitor Get Current.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kernel_obj_monitor_get_current(void) {
#ifdef _WIN32
    DWORD current = 0;
    if (GetProcessHandleCount(GetCurrentProcess(),&current)) return (int)current;
#else
    DIR* d = opendir("/proc/self/fd");
    if (d) {
        int count=0;
        struct dirent* ent;
        while ((ent=readdir(d))!=NULL) { if (ent->d_name[0]!='.') count++; }
        closedir(d);
        return count;
    }
#endif
    return 0;
}

/**
 * @brief Perform Toctou Get Path.
 *
 * @param td [out] Td value.
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_toctou_get_path(SNEPPXTOCTOUDetector* td, char* buffer, size_t size) {
    if (!td||!buffer||size<1) return -1;
    buffer[0]=0;
    return 0;
}

/**
 * @brief Perform Ima Measure Path.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ima_measure_path(const char* path) {
    if (!path) return -1;
    uint8_t hash[32];
    return SNEPPX_ima_measure(path,hash);
}

/**
 * @brief Perform Ima Appraise Path.
 *
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ima_appraise_path(const char* path, const uint8_t expected[32]) {
    if (!path||!expected) return 0;
    uint8_t hash[32];
    if (SNEPPX_ima_measure(path,hash)!=0) return 0;
    return memcmp(hash,expected,32)==0?1:0;
}

/**
 * @brief Perform Alert Correlator Set Window Seconds.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_set_window_seconds(uint64_t seconds) {
    if (seconds<1) seconds=1;
    if (seconds>86400) seconds=86400;
    g_ac_window_size = seconds;
    return 0;
}

/**
 * @brief Perform Alert Correlator Get Window Seconds.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_alert_correlator_get_window_seconds(void) {
    return g_ac_window_size;
}

/**
 * @brief Perform Alert Correlator Get Threshold Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_get_threshold_count(void) {
    return g_ac_alert_threshold;
}

/**
 * @brief Perform Alert Correlator Set Threshold Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_set_threshold_count(int count) {
    if (count<1) count=1;
    g_ac_alert_threshold = count;
    return 0;
}

/**
 * @brief Perform Alert Correlator Get Total Events.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_get_total_events(void) {
    return g_ac_total_events_tracked;
}

/**
 * @brief Perform Alert Correlator Get Total Alerts.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_get_total_alerts(void) {
    return g_ac_alerts_triggered_tracked;
}

/**
 * @brief Perform Internal Verify All Regions.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_internal_verify_all_regions(void) {
    int violations = 0;
    for (int i=0;i<g_ct_region_count;i++) {
        if (!g_ct_regions[i].active) continue;
        uint8_t current[32];
        SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx);
        SNEPPX_blake3_update(&ctx,(const uint8_t*)g_ct_regions[i].addr,g_ct_regions[i].size);
        SNEPPX_blake3_finish(&ctx,current);
        if (memcmp(current,g_ct_regions[i].hash,32)!=0) violations++;
    }
    return violations;
}

/**
 * @brief Perform Internal Verify All Heaps.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_internal_verify_all_heaps(void) {
    if (!g_hcd_enabled) return 0;
    int corrupted = 0;
    for (int i=0;i<g_heap_scan_count;i++) {
        if (!g_heap_scan_allocs[i]||g_heap_scan_sizes[i]<8) continue;
        if (*(uint64_t*)((char*)g_heap_scan_allocs[i]+g_heap_scan_sizes[i]-8)!=g_hcd_sentinel_value) corrupted++;
    }
    return corrupted;
}

static uint64_t g_global_init_time = 0;

/**
 * @brief Perform Get Global Uptime.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_get_global_uptime(void) {
    if (g_global_init_time==0) return 0;
    return (uint64_t)time(NULL)-g_global_init_time;
}

/**
 * @brief Perform Set Global Init Time.
 */
void SNEPPX_set_global_init_time(void) {
    g_global_init_time = (uint64_t)time(NULL);
}

/**
 * @brief Perform Has Global Init Time.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_has_global_init_time(void) {
    return g_global_init_time!=0?1:0;
}

/**
 * @brief Perform Instrument Enter.
 */
static void SNEPPX_instrument_enter(const char* func) {
    (void)func;
}

/**
 * @brief Perform Instrument Exit.
 *
 * @param func [in] Func value.
 */
static void SNEPPX_instrument_exit(const char* func, int ret) {
    (void)func;
    (void)ret;
}

/**
 * @brief Perform Code Tamper Verify All With Report.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_code_tamper_verify_all_with_report(char* buffer, size_t size) {
    if (!buffer||size<1) return -1;
    int violations = SNEPPX_code_tamper_check_all();
    int pos = snprintf(buffer,size,"code_tamper: regions=%d violations=%d",g_ct_region_count,violations);
    (void)pos;
    return violations;
}

/**
 * @brief Perform Heap Corruption Scan With Report.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heap_corruption_scan_with_report(char* buffer, size_t size) {
    if (!buffer||size<1) return -1;
    int corrupted = SNEPPX_heap_corruption_scan_all();
    int pos = snprintf(buffer,size,"heap: allocs=%d corrupted=%d",g_heap_scan_count,corrupted);
    (void)pos;
    return corrupted;
}

/**
 * @brief Perform Func Ptr Detector Scan With Report.
 *
 * @param fpd [out] Fpd value.
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_func_ptr_detector_scan_with_report(SNEPPXFuncPtrDetector* fpd, char* buffer, size_t size) {
    if (!fpd||!buffer||size<1) return -1;
    int modified = SNEPPX_func_ptr_detector_scan(fpd);
    int pos = snprintf(buffer,size,"func_ptr: watched=%d modified=%d",fpd->count,modified);
    (void)pos;
    return modified;
}

/**
 * @brief Perform Ml Anomaly Score With Threshold.
 *
 * @param ml [out] Ml value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_score_with_threshold(SNEPPXMLAnomalyDetector* ml, const double features[SNEPPX_MON_ML_FEATURES], double* score_out) {
    if (!ml||!features||!score_out) return -1;
    *score_out = SNEPPX_ml_anomaly_score(ml,features);
    return (*score_out>ml->threshold)?1:0;
}

/**
 * @brief Perform Ml Anomaly Is Anomaly Ex.
 *
 * @param ml [out] Ml value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_anomaly_is_anomaly_ex(SNEPPXMLAnomalyDetector* ml, const double features[SNEPPX_MON_ML_FEATURES], double custom_threshold) {
    if (!ml||!features) return 0;
    double score = SNEPPX_ml_anomaly_score(ml,features);
    return score>custom_threshold?1:0;
}

/**
 * @brief Perform Fs Integrity Verify With Report.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_verify_with_report(char* buffer, size_t size) {
    if (!buffer||size<1) return -1;
    int violations = 0;
    int pos = 0;
    for (int i=0;i<g_fsi_watch_count;i++) {
        uint8_t current[32];
        FILE* f=fopen(g_fsi_watched_paths[i],"rb");
        if (f) {
            fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
            uint8_t* buf=(uint8_t*)malloc(sz);
            if (buf) {
                fread(buf,1,sz,f);
                SNEPPXBlake3State ctx; SNEPPX_blake3_init(&ctx); SNEPPX_blake3_update(&ctx,buf,sz); SNEPPX_blake3_finish(&ctx,current);
                free(buf);
                if (memcmp(current,g_fsi_watched_hashes[i],32)!=0) violations++;
            }
            fclose(f);
        }
        pos += snprintf(buffer+pos,size-pos,"[%d] path=%s %s\n",i,g_fsi_watched_paths[i],memcmp(current,g_fsi_watched_hashes[i],32)==0?"OK":"MODIFIED");
        if (pos>=(int)size-64) break;
    }
    return violations;
}

/**
 * @brief Perform Alert Correlator Evaluate Ex.
 *
 * @param ac [out] Ac value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alert_correlator_evaluate_ex(SNEPPXAlertCorrelator* ac, uint64_t window_seconds) {
    if (!ac) return 0;
    int type_counts[4]={0};
    uint64_t now=(uint64_t)time(NULL);
    int total=0;
    for (int i=ac->count-1;i>=0;i--) {
        if (now-ac->events[i].timestamp>window_seconds) continue;
        if (ac->events[i].type>=0&&ac->events[i].type<4) {
            type_counts[ac->events[i].type]++;
            total++;
        }
    }
    ac->alerts_triggered=0;
    for (int i=0;i<4;i++) {
        if (type_counts[i]>=g_ac_alert_threshold) ac->alerts_triggered=1;
    }
    if (total>=5&&type_counts[0]>=2&&type_counts[1]>=2) ac->alerts_triggered=1;
    if (ac->alerts_triggered) g_ac_alerts_triggered_tracked++;
    g_ac_total_events_tracked += total;
    return ac->alerts_triggered;
}

/**
 * @brief Update Toctou Check And.
 *
 * @param td [out] Td value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_toctou_check_and_update(SNEPPXTOCTOUDetector* td, const char* path) {
    int changed = SNEPPX_toctou_check(td,path);
    if (changed) SNEPPX_toctou_update_baseline(td);
    return changed;
}

/**
 * @brief Perform Code Tamper Set Debug.
 */
void SNEPPX_code_tamper_set_debug(int enabled) {
    (void)enabled;
}

/**
 * @brief Perform Code Tamper Get Debug.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_code_tamper_get_debug(void) {
    return 0;
}

static uint64_t g_heap_corruption_total_checks = 0;
static uint64_t g_heap_corruption_total_failures = 0;

/**
 * @brief Perform Heap Corruption Get Total Checks.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_heap_corruption_get_total_checks(void) {
    return g_heap_corruption_total_checks;
}

/**
 * @brief Perform Heap Corruption Get Total Failures.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_heap_corruption_get_total_failures(void) {
    return g_heap_corruption_total_failures;
}

/**
 * @brief Perform Heap Corruption Reset Total Counters.
 */
void SNEPPX_heap_corruption_reset_total_counters(void) {
    g_heap_corruption_total_checks = 0;
    g_heap_corruption_total_failures = 0;
}

static uint64_t g_stack_overflow_check_count = 0;
static uint64_t g_stack_overflow_trigger_count = 0;

/**
 * @brief Perform Stack Overflow Get Check Count.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_stack_overflow_get_check_count(void) {
    return g_stack_overflow_check_count;
}

/**
 * @brief Perform Stack Overflow Get Trigger Count.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_stack_overflow_get_trigger_count(void) {
    return g_stack_overflow_trigger_count;
}

static uint64_t g_ml_score_count = 0;
static double g_ml_max_score = 0.0;

/**
 * @brief Perform Ml Anomaly Get Max Score.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_ml_anomaly_get_max_score(void) {
    return g_ml_max_score;
}

/**
 * @brief Perform Ml Anomaly Get Score Count.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_ml_anomaly_get_score_count(void) {
    return g_ml_score_count;
}

/**
 * @brief Perform Fs Integrity Watch Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fs_integrity_watch_count(void) {
    return g_fsi_watch_count;
}

/**
 * @brief Perform Net Conn Monitor Has Bad Port.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_net_conn_monitor_has_bad_port(int port) {
    for (int i=0;i<g_net_bad_port_count;i++) {
        if (g_net_bad_ports[i]==port) return 1;
    }
    return 0;
}

/**
 * @brief Perform Stack Overflow Reset Counts.
 */
void SNEPPX_stack_overflow_reset_counts(void) {
    g_stack_overflow_check_count = 0;
    g_stack_overflow_trigger_count = 0;
}

/**
 * @brief Perform Persistence Monitor Get Initialized.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_persistence_monitor_get_initialized(void) {
    return g_persist_initialized;
}
