#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/**
 * @brief Create Tape.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTape* SNEPPX_tape_create(void) {
    SNEPPXTape* tape = (SNEPPXTape*)SNEPPX_malloc(sizeof(SNEPPXTape), 64);
    if (!tape) return NULL;
    memset(tape, 0, sizeof(SNEPPXTape));
    tape->capacity = 64;
    tape->num_vars = 0;
    tape->vars = (SNEPPXVariable**)SNEPPX_malloc(tape->capacity * sizeof(SNEPPXVariable*), 64);
    if (!tape->vars) { SNEPPX_free(tape, sizeof(SNEPPXTape)); return NULL; }
    memset(tape->vars, 0, tape->capacity * sizeof(SNEPPXVariable*));
    return tape;
}

/**
 * @brief Destroy Tape.
 */
void SNEPPX_tape_destroy(SNEPPXTape* tape) {
    if (!tape) return;
    for (size_t i = 0; i < tape->num_vars; i++) {
        if (tape->vars[i]) SNEPPX_variable_destroy(tape->vars[i]);
    }
    if (tape->vars) SNEPPX_free(tape->vars, tape->capacity * sizeof(SNEPPXVariable*));
    SNEPPX_free(tape, sizeof(SNEPPXTape));
}

/**
 * @brief Perform Tape Record.
 *
 * @param tape [out] Tape value.
 */
void SNEPPX_tape_record(SNEPPXTape* tape, SNEPPXVariable* var) {
    if (!tape || !var) return;
    /* The tape OWNS every recorded variable (destroyed in tape_destroy).
     * Recording the same variable twice would register two destroy entries
     * for one pointer -> double free at teardown. Skip duplicates. */
    for (size_t i = 0; i < tape->num_vars; i++) {
        if (tape->vars[i] == var) return;
    }
    if (tape->num_vars >= tape->capacity) {
        tape->capacity *= 2;
        SNEPPXVariable** new_vars = (SNEPPXVariable**)SNEPPX_malloc(tape->capacity * sizeof(SNEPPXVariable*), 64);
        if (!new_vars) return;
        memcpy(new_vars, tape->vars, tape->num_vars * sizeof(SNEPPXVariable*));
        SNEPPX_free(tape->vars, tape->capacity / 2 * sizeof(SNEPPXVariable*));
        tape->vars = new_vars;
    }
    tape->vars[tape->num_vars++] = var;
    if (tape->checkpointing && var->backward_ctx && var->recompute_ctx) {
        var->checkpointed = 1;
        var->free_ctx(var->backward_ctx);
        var->backward_ctx = NULL;
    }
}

/**
 * @brief Perform Tape Checkpoint Begin.
 */
void SNEPPX_tape_checkpoint_begin(SNEPPXTape* tape) {
    if (tape) tape->checkpointing = 1;
}

/**
 * @brief Perform Tape Checkpoint End.
 */
void SNEPPX_tape_checkpoint_end(SNEPPXTape* tape) {
    if (tape) tape->checkpointing = 0;
}

static void tape_topological_sort(SNEPPXTape* tape, SNEPPXVariable** sorted, size_t* num_sorted) {
    if (!tape || !sorted) return;
    size_t n = tape->num_vars;
    if (n == 0) return;
    int* visited = (int*)SNEPPX_malloc(n * sizeof(int), 64);
    if (!visited) return;
    memset(visited, 0, n * sizeof(int));
    *num_sorted = 0;

    /* DFS-based topological sort: for each node, visit all parents first */
    for (size_t i = 0; i < n; i++) {
        if (visited[i]) continue;
        /* Simple stack-based DFS post-order */
        size_t stack_cap = 64;
        size_t* stack = (size_t*)SNEPPX_malloc(stack_cap * sizeof(size_t), 64);
        int* on_stack = (int*)SNEPPX_malloc(n * sizeof(int), 64);
        if (!stack || !on_stack) { SNEPPX_free(stack, stack_cap*sizeof(size_t)); SNEPPX_free(on_stack, n*sizeof(int)); break; }
        memset(on_stack, 0, n * sizeof(int));
        size_t sp = 0;
        stack[sp++] = i;
        on_stack[i] = 1;

        size_t loop_guard = 0;
        while (sp > 0) {
            if (++loop_guard > 1000000) { fprintf(stderr, "ERROR: tape_topological_sort infinite loop at i=%zu sp=%zu n=%zu\n", i, sp, n); break; }
            size_t idx = stack[sp - 1];
            SNEPPXVariable* var = tape->vars[idx];
            int all_parents_visited = 1;
            if (var && var->parents) {
                for (size_t p = 0; p < var->num_parents; p++) {
                    SNEPPXVariable* parent = var->parents[p];
                    if (!parent) continue;
                    /* Find parent index in tape vars */
                    int found = 0;
                    for (size_t pi = 0; pi < n; pi++) {
                        if (tape->vars[pi] == parent) {
                            found = 1;
                            if (!visited[pi]) {
                                if (!on_stack[pi]) {
                                    /* Grow stack if needed */
                                    if (sp >= stack_cap) {
                                        stack_cap *= 2;
                                        size_t* ns = (size_t*)SNEPPX_malloc(stack_cap * sizeof(size_t), 64);
                                        if (!ns) break;
                                        memcpy(ns, stack, sp * sizeof(size_t));
                                        SNEPPX_free(stack, stack_cap/2 * sizeof(size_t));
                                        stack = ns;
                                    }
                                    stack[sp++] = pi;
                                    on_stack[pi] = 1;
                                    all_parents_visited = 0;
                                }
                            }
                            break;
                        }
                    }
                    /* If parent not found on tape, treat as visited (no backward needed) */
                    if (!found) continue;
                    if (!all_parents_visited) break;
                }
            }
            if (all_parents_visited) {
                visited[idx] = 1;
                sorted[(*num_sorted)++] = var;
                on_stack[idx] = 0;
                sp--;
            }
        }
        SNEPPX_free(on_stack, n * sizeof(int));
        SNEPPX_free(stack, stack_cap * sizeof(size_t));
    }
    SNEPPX_free(visited, n * sizeof(int));
}

/**
 * @brief Run the backward pass for Tape.
 *
 * @param tape [out] Tape value.
 */
void SNEPPX_tape_backward(SNEPPXTape* tape, SNEPPXVariable* loss) {
    if (!tape || !loss) return;

    if (loss->grad) SNEPPX_tensor_destroy(loss->grad);
    loss->grad = SNEPPX_tensor_ones(loss->data->shape, loss->data->ndim, SNEPPX_FLOAT32);

    SNEPPXVariable** sorted = NULL;
    size_t n = tape->num_vars;
    size_t num_sorted = 0;
    if (n > 0) {
        sorted = (SNEPPXVariable**)SNEPPX_malloc(n * sizeof(SNEPPXVariable*), 64);
        if (sorted) {
            tape_topological_sort(tape, sorted, &num_sorted);
        }
    }

    for (size_t i = num_sorted; i > 0; i--) {
        SNEPPXVariable* var = sorted ? sorted[i - 1] : tape->vars[i - 1];
        if (!var || !var->backward_fn || !var->grad) continue;
        if (!var->backward_ctx && var->checkpointed && var->recompute_ctx) {
            var->backward_ctx = var->recompute_ctx(var, var->params, var->param_count);
        }
        if (!var->backward_ctx) continue;
        var->backward_fn(var->backward_ctx, var->grad);
        if (var->backward_ctx && var->free_ctx) {
            var->free_ctx(var->backward_ctx);
            var->backward_ctx = NULL;
        }
        for (size_t p = 0; p < var->num_parents; p++) {
            SNEPPXVariable* parent = var->parents[p];
            if (!parent) continue;
            parent->ref_count--;
        }
    }

    if (sorted) SNEPPX_free(sorted, n * sizeof(SNEPPXVariable*));
}

/**
 * @brief Perform Tape Zero Grad.
 */
void SNEPPX_tape_zero_grad(SNEPPXTape* tape) {
    if (!tape) return;
    for (size_t i = 0; i < tape->num_vars; i++) {
        if (tape->vars[i] && tape->vars[i]->grad) {
            SNEPPX_tensor_destroy(tape->vars[i]->grad);
            tape->vars[i]->grad = NULL;
        }
    }
}

/**
 * @brief Perform Tape Global Norm.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_tape_global_norm(SNEPPXTape* tape) {
    if (!tape) return 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < tape->num_vars; i++) {
        SNEPPXVariable* var = tape->vars[i];
        if (!var || !var->grad) continue;
        float* gd = (float*)var->grad->data;
        size_t sz = var->grad->size;
        for (size_t j = 0; j < sz; j++) sum_sq += gd[j] * gd[j];
    }
    return sqrtf(sum_sq);
}

/* ---------- no_grad context ---------- */
static int g_no_grad_depth = 0;
/**
 * @brief Perform No Grad Enter.
 */
void SNEPPX_no_grad_enter(void) { g_no_grad_depth++; }
/**
 * @brief Perform No Grad Exit.
 */
void SNEPPX_no_grad_exit(void) { if (g_no_grad_depth > 0) g_no_grad_depth--; }
/**
 * @brief Perform No Grad Is Active.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_no_grad_is_active(void) { return g_no_grad_depth > 0; }

/**
 * @brief Perform Tape Clip Grad Norm.
 *
 * @param tape [out] Tape value.
 */
void SNEPPX_tape_clip_grad_norm(SNEPPXTape* tape, float max_norm) {
    if (!tape || max_norm <= 0.0f) return;
    float total_norm = SNEPPX_tape_global_norm(tape);
    if (total_norm <= max_norm) return;
    float scale = max_norm / (total_norm + 1e-7f);
    for (size_t i = 0; i < tape->num_vars; i++) {
        SNEPPXVariable* var = tape->vars[i];
        if (!var || !var->grad) continue;
        float* gd = (float*)var->grad->data;
        for (size_t j = 0; j < var->grad->size; j++) gd[j] *= scale;
    }
}
