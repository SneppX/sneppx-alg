#ifndef _COVERAGE_H
#define _COVERAGE_H

/*
 * SNEPPX - Coverage
 *
 * WHAT
 *   Coverage.
 *
 * CONCEPT
 *   Provides the Coverage.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// Standalone coverage instrumentation header.
// Usage: include in source files, then link with --coverage.

// Coverage counters — compile with -DCOVERAGE_ENABLED
#ifdef COVERAGE_ENABLED
    #define COV_HIT()   do { static long long __cov_counter = 0; __cov_counter++; } while(0)
    #define COV_BRANCH(c) do { static long long __cov_branch_taken = 0; static long long __cov_branch_not = 0; if(c) __cov_branch_taken++; else __cov_branch_not++; } while(0)
#else
    #define COV_HIT()
    #define COV_BRANCH(c) ((void)(c))
#endif

#endif
