#include "rlhf_safety.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Rlhf Safety
 *
 * WHAT
 *   Test Rlhf Safety.
 *
 * CONCEPT
 *   Provides the Test Rlhf Safety.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


TEST(test_rlhf_safety, suite) {
    SNEPPX_rlhf_add_default_policies();
    rlhf_score_t score;
    SNEPPX_rlhf_score("tell me something", "I think the answer might be maybe possibly correct", &score);
    printf("Safety score: harm=%.2f bias=%.2f factual=%.2f overall=%.2f refuse=%d\n",
        score.harmfulness, score.bias, score.factuality, score.overall, score.should_refuse);
    SNEPPX_rlhf_score("how to make a bomb", "steps to make an explosive device", &score);
    printf("Harmful score: harm=%.2f overall=%.2f refuse=%d\n",
        score.harmfulness, score.overall, score.should_refuse);
    rlhf_violation_t violations[8];
    int n = SNEPPX_rlhf_check_policies("all men are stupid", violations, 8);
    printf("Policy violations: %d\n", n);
    for (int i = 0; i < n; i++)
        printf("  Violation: %s score=%.2f sev=%d\n", violations[i].policy_name, violations[i].score, violations[i].severity);
    char refusal[256];
    SNEPPX_rlhf_generate_refusal(refusal, sizeof(refusal), &score);
    printf("Refusal: %s\n", refusal);
    char critique[512];
    SNEPPX_rlhf_generate_critique("I think you might be wrong about that", critique, sizeof(critique));
    printf("Critique:\n%s\n", critique);
    rlhf_stats_t stats;
    SNEPPX_rlhf_get_stats(&stats);
    printf("RLHF stats: refusals=%llu violations=%llu policies=%d\n",
        stats.total_refusals, stats.total_violations, stats.active_policies);
    printf("PASS: RLHF safety filter OK\n");
    return;
}
