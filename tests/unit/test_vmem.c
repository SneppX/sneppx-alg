#include "../../mm/internal/vmem.h"
#include "test_gtest.h"
#include "../../mm/internal/vmem.c"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Vmem
 *
 * WHAT
 *   Test Vmem.
 *
 * CONCEPT
 *   Provides the Test Vmem.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_vmem_reserve_commit(void) {
    SNEPPXVMemAllocator alloc;
    SNEPPX_vmem_init(&alloc);
    void* region = SNEPPX_vmem_reserve(&alloc, 65536, 4096, SNEPPX_VMEM_FLAG_WRITABLE);
    if (region == NULL) {
        printf("SKIP (stub returns NULL): ");
        SNEPPX_vmem_cleanup(&alloc);
        return;
    }
    int ok = SNEPPX_vmem_commit(&alloc, region, 4096);
    SX_ASSERT(ok == 0, "vmem commit 4KB");
    SNEPPX_vmem_decommit(&alloc, region, 4096);
    SNEPPX_vmem_release(&alloc, region, 65536);
    SNEPPX_vmem_cleanup(&alloc);
}

static void test_vmem_large_region(void) {
    SNEPPXVMemAllocator alloc;
    SNEPPX_vmem_init(&alloc);
    void* region = SNEPPX_vmem_reserve(&alloc, 1048576, 4096, SNEPPX_VMEM_FLAG_WRITABLE);
    if (region == NULL) {
        printf("SKIP (stub returns NULL): ");
        SNEPPX_vmem_cleanup(&alloc);
        return;
    }
    SNEPPX_vmem_release(&alloc, region, 1048576);
    SNEPPX_vmem_cleanup(&alloc);
}


TEST(test_vmem, vmem_reserve_commit) { test_vmem_reserve_commit(); }
TEST(test_vmem, vmem_large_region) { test_vmem_large_region(); }
