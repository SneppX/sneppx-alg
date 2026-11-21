#include "fractal_memory_orchestrator.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Fm Node
 *
 * WHAT
 *   Test Fm Node.
 *
 * CONCEPT
 *   Provides the Test Fm Node.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_node_create(void) {
    SNEPPXFMNode* node = SNEPPX_fm_node_create(0, 16, 32);
    SX_ASSERT(node != NULL, "node not null");
    SX_ASSERT(node->node_id == 0, "id 0");
    SX_ASSERT(node->memory_bank != NULL, "memory bank not null");
    SX_ASSERT(node->gradient_accumulator != NULL, "grad accum not null");
    SNEPPX_fm_node_destroy(node);
}

static void test_node_online(void) {
    SNEPPXFMNode* node = SNEPPX_fm_node_create(1, 8, 16);
    SX_ASSERT(node->is_online == 1, "online true");
    SX_ASSERT(fabsf(node->trust_score - 1.0f) < 1e-5f, "trust 1.0");
    SNEPPX_fm_node_destroy(node);
}


TEST(test_fm_node, test_node_create) { test_node_create(); }
TEST(test_fm_node, test_node_online) { test_node_online(); }
