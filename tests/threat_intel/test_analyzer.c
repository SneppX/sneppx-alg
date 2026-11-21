#include "security/threat_intel/analyzer.h"
#include <assert.h>
#include <string.h>

/*
 * SNEPPX - Test Analyzer
 *
 * WHAT
 *   Test Analyzer.
 *
 * CONCEPT
 *   Provides the Test Analyzer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_create_ioc(void) {
    SNEPPXTIIoC ioc;
    memset(&ioc, 0, sizeof(ioc));
    const uint8_t* ip = "192.168.1.1";
    int ret = snepx_ti_analyzer_create_ioc(&ioc, SNEPPX_TI_IOC_TYPE_IPV4, ip, strlen(ip));
    assert(ret == 0);
    assert(ioc.ioc_type == SNEPPX_TI_IOC_TYPE_IPV4);
    assert(ioc.indicator_len == strlen(ip));
    snepx_ti_analyzer_ioc_destroy(&ioc);
}

static void test_add_tag_and_category(void) {
    SNEPPXTIIoC ioc;
    memset(&ioc, 0, sizeof(ioc));
    snepx_ti_analyzer_create_ioc(&ioc, SNEPPX_TI_IOC_TYPE_IPV4, "10.0.0.1", 8);
    snepx_ti_analyzer_add_tag(&ioc, "malicious", 9);
    assert(ioc.num_tags == 1);
    snepx_ti_analyzer_add_category(&ioc, "c2", 2);
    assert(ioc.num_categories == 1);
    snepx_ti_analyzer_ioc_destroy(&ioc);
}

int main(void) {
    test_create_ioc();
    test_add_tag_and_category();
    return 0;
}
