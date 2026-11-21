#include "vmem.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#endif

/*
 * SNEPPX - Vmem
 *
 * WHAT
 *   Vmem.
 *
 * CONCEPT
 *   Provides the Vmem.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Initialize Vmem.
 */
void SNEPPX_vmem_init(SNEPPXVMemAllocator* alloc) {
    if (alloc) {
        memset(alloc, 0, sizeof(*alloc));
        alloc->page_size_default = 4096;
        alloc->evict_strategy = NULL;
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si);
        alloc->page_size_default = (size_t)si.dwPageSize;
#endif
    }
}

/**
 * @brief Perform Vmem Cleanup.
 */
void SNEPPX_vmem_cleanup(SNEPPXVMemAllocator* alloc) {
    (void)alloc;
}

/**
 * @brief Perform Vmem Reserve.
 *
 * @param alloc [out] Alloc value.
 * @param bytes [in] Bytes value.
 * @param alignment [in] Alignment value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_vmem_reserve(SNEPPXVMemAllocator* alloc, size_t bytes, size_t alignment, int flags) {
    if (bytes == 0) return NULL;
    size_t page_size = alloc ? alloc->page_size_default : 4096;
    size_t aligned = (bytes + page_size - 1) & ~(page_size - 1);
    if (alignment > page_size) {
        size_t overalloc = aligned + alignment;
#ifdef _WIN32
        void* base = VirtualAlloc(NULL, overalloc, MEM_RESERVE, PAGE_NOACCESS);
        if (!base) { if (alloc && alloc->on_oom) alloc->on_oom(overalloc); return NULL; }
        uintptr_t addr = ((uintptr_t)base + alignment - 1) & ~(uintptr_t)(alignment - 1);
        VirtualFree(base, 0, MEM_RELEASE);
        void* result = VirtualAlloc((void*)addr, aligned, MEM_RESERVE, PAGE_NOACCESS);
        if (!result && alloc && alloc->on_oom) alloc->on_oom(aligned);
        if (result && alloc) alloc->total_reserved += aligned;
        return result;
#else
        void* base = mmap(NULL, overalloc, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) { if (alloc && alloc->on_oom) alloc->on_oom(overalloc); return NULL; }
        uintptr_t addr = ((uintptr_t)base + alignment - 1) & ~(uintptr_t)(alignment - 1);
        munmap(base, overalloc);
        void* result = mmap((void*)addr, aligned, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (result == MAP_FAILED && alloc && alloc->on_oom) alloc->on_oom(aligned);
        if (result != MAP_FAILED && alloc) alloc->total_reserved += aligned;
        return (result != MAP_FAILED) ? result : NULL;
#endif
    }
#ifdef _WIN32
    DWORD protect = PAGE_NOACCESS;
    if (flags & SNEPPX_VMEM_FLAG_EXECUTABLE) protect = PAGE_EXECUTE;
    int mem_type = MEM_RESERVE;
    if (flags & SNEPPX_VMEM_FLAG_HUGEPAGE) mem_type |= MEM_LARGE_PAGES;
    void* result = VirtualAlloc(NULL, aligned, mem_type, protect);
    if (!result) { if (alloc && alloc->on_oom) alloc->on_oom(aligned); return NULL; }
    if (alloc) alloc->total_reserved += aligned;
    return result;
#else
    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_HUGETLB
    if (flags & SNEPPX_VMEM_FLAG_HUGEPAGE) mmap_flags |= MAP_HUGETLB;
#endif
    int prot = PROT_NONE;
    void* result = mmap(NULL, aligned, prot, mmap_flags, -1, 0);
    if (result == MAP_FAILED) { if (alloc && alloc->on_oom) alloc->on_oom(aligned); return NULL; }
    if (alloc) alloc->total_reserved += aligned;
    return result;
#endif
}

/**
 * @brief Perform Vmem Commit.
 *
 * @param alloc [out] Alloc value.
 * @param addr [out] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vmem_commit(SNEPPXVMemAllocator* alloc, void* addr, size_t bytes) {
    if (!addr || bytes == 0) return -1;
#ifdef _WIN32
    void* result = VirtualAlloc(addr, bytes, MEM_COMMIT, PAGE_READWRITE);
    if (!result) return -1;
#else
    void* result = mmap(addr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (result == MAP_FAILED) return -1;
#endif
    if (alloc) { alloc->total_committed += bytes; if (alloc->total_committed > alloc->peak_committed) alloc->peak_committed = alloc->total_committed; }
    return 0;
}

/**
 * @brief Perform Vmem Decommit.
 *
 * @param alloc [out] Alloc value.
 * @param addr [out] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vmem_decommit(SNEPPXVMemAllocator* alloc, void* addr, size_t bytes) {
    if (!addr || bytes == 0) return -1;
#ifdef _WIN32
    if (!VirtualFree(addr, bytes, MEM_DECOMMIT)) return -1;
#else
    if (mmap(addr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED) return -1;
    madvise(addr, bytes, MADV_DONTNEED);
#endif
    if (alloc) { if (bytes > alloc->total_committed) alloc->total_committed = 0; else alloc->total_committed -= bytes; }
    return 0;
}

/**
 * @brief Perform Vmem Release.
 *
 * @param alloc [out] Alloc value.
 * @param addr [out] Addr value.
 */
void SNEPPX_vmem_release(SNEPPXVMemAllocator* alloc, void* addr, size_t bytes) {
    if (!addr || bytes == 0) return;
#ifdef _WIN32
    VirtualFree(addr, 0, MEM_RELEASE);
#else
    munmap(addr, bytes);
#endif
    if (alloc) { if (bytes > alloc->total_reserved) alloc->total_reserved = 0; else alloc->total_reserved -= bytes; }
}

/**
 * @brief Perform Vmem Advise Hugepage.
 *
 * @param addr [out] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vmem_advise_hugepage(void* addr, size_t bytes) {
#ifdef _WIN32
    (void)addr; (void)bytes;
    HANDLE hToken; TOKEN_PRIVILEGES tp;
    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &hToken)) {
        LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid);
        tp.PrivilegeCount = 1; tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, NULL);
        CloseHandle(hToken);
    }
    return 0;
#else
#ifdef MADV_HUGEPAGE
    return madvise(addr, bytes, MADV_HUGEPAGE);
#else
    (void)addr; (void)bytes; return 0;
#endif
#endif
}

/**
 * @brief Perform Vmem Advise Nohugepage.
 *
 * @param addr [out] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vmem_advise_nohugepage(void* addr, size_t bytes) {
#ifdef _WIN32
    (void)addr; (void)bytes; return 0;
#else
#ifdef MADV_NOHUGEPAGE
    return madvise(addr, bytes, MADV_NOHUGEPAGE);
#else
    (void)addr; (void)bytes; return 0;
#endif
#endif
}

/**
 * @brief Perform Vmem Register Evict Strategy.
 *
 * @param alloc [out] Alloc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vmem_register_evict_strategy(SNEPPXVMemAllocator* alloc, SNEPPXEvictStrategy* strat) {
    if (!alloc || !strat) return -1;
    alloc->evict_strategy = strat;
    return 0;
}

/**
 * @brief Perform Vmem Evict Page.
 *
 * @param alloc [out] Alloc value.
 * @param addr [out] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vmem_evict_page(SNEPPXVMemAllocator* alloc, void* addr, size_t size) {
    if (!alloc || !addr) return -1;
    if (alloc->evict_strategy && alloc->evict_strategy->select_victim) {
        void* victim_addr = NULL;
        size_t victim_size = 0;
        if (alloc->evict_strategy->select_victim(alloc->evict_strategy, &victim_addr, &victim_size) == 0
            && victim_addr && victim_size > 0) {
            return SNEPPX_vmem_decommit(alloc, victim_addr, victim_size);
        }
    }
    return SNEPPX_vmem_decommit(alloc, addr, size);
}

void SNEPPX_vmem_set_oom_handler(SNEPPXVMemAllocator* alloc, int (*handler)(size_t)) {
    if (alloc) alloc->on_oom = handler;
}
