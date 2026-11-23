#include "entropy_pool.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond)                                                            \
    do                                                                         \
    {                                                                          \
        if (!(cond))                                                           \
        {                                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(void)
{
    SNEPPXEntropyPool ep;
    uint8_t first[32];
    uint8_t second[32];
    uint8_t small[8];
    int nonzero = 0;

    CHECK(SNEPPX_entropy_pool_init(&ep) == 0);

    /* Skip automatic collection so this exercises state evolution only. */
    ep.entropy_estimate = 256;
    memset(ep.pool, 0xA5, sizeof(ep.pool));

    CHECK(SNEPPX_entropy_pool_get(&ep, first, sizeof(first)) == 0);
    CHECK(SNEPPX_entropy_pool_get(&ep, second, sizeof(second)) == 0);
    CHECK(memcmp(first, second, sizeof(first)) != 0);

    for (size_t i = 0; i < sizeof(first); ++i)
        nonzero |= first[i];
    CHECK(nonzero != 0);

    CHECK(SNEPPX_entropy_pool_add(&ep, (SNEPPXEntropySource) 99, small, 1) == -1);
    CHECK(SNEPPX_entropy_pool_get(&ep, small, (size_t) INT_MAX + 1) == -1);

    printf("entropy_pool_native: ok\n");
    return 0;
}
