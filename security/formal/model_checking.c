#include "model_checking.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * SNEPPX - Model Checking
 *
 * WHAT
 *   Model Checking.
 *
 * CONCEPT
 *   Provides the Model Checking.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Initialize Model.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_init(SNEPPXFormalModel* model) {
    if (!model) return -1;
    memset(model, 0, sizeof(*model));
    model->initial_state = 0;
    return 0;
}

/**
 * @brief Perform Model Add State.
 *
 * @param model [out] Model value.
 * @param state_id [in] State Id value.
 * @param is_accepting [in] Is Accepting value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_add_state(SNEPPXFormalModel* model, uint32_t state_id, int is_accepting, int is_error) {
    if (!model || model->state_count >= SNEPPX_MODEL_MAX_STATES) return -1;
    SNEPPXModelState* s = &model->states[model->state_count];
    s->state_id = state_id;
    s->next_count = 0;
    s->is_accepting = is_accepting;
    s->is_error = is_error;
    return model->state_count++;
}

/**
 * @brief Perform Model Add Transition.
 *
 * @param model [out] Model value.
 * @param from [in] From value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_add_transition(SNEPPXFormalModel* model, uint32_t from, uint32_t to) {
    if (!model) return -1;
    for (int i = 0; i < model->state_count; i++) {
        if (model->states[i].state_id == from && model->states[i].next_count < 8) {
            model->states[i].next_states[model->states[i].next_count++] = to;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Model Set Property.
 *
 * @param model [out] Model value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_set_property(SNEPPXFormalModel* model, const char* property) {
    if (!model || !property) return -1;
    strncpy(model->property, property, SNEPPX_MODEL_PROP_MAX_LEN - 1);
    return 0;
}

/**
 * @brief Perform Model Check.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXModelCheckResult SNEPPX_model_check(SNEPPXFormalModel* model) {
    SNEPPXModelCheckResult result;
    memset(&result, 0, sizeof(result));
    if (!model) { result.property_satisfied = 0; return result; }

    result.total_states = model->state_count;
    int visited[SNEPPX_MODEL_MAX_STATES] = {0};
    int parent[SNEPPX_MODEL_MAX_STATES];
    for (int i = 0; i < SNEPPX_MODEL_MAX_STATES; i++) parent[i] = -1;
    int stack[SNEPPX_MODEL_MAX_STATES];
    int stack_top = 0;
    stack[stack_top++] = model->initial_state;
    visited[model->initial_state] = 1;

    int first_error = -1;

    while (stack_top > 0) {
        int cur = stack[--stack_top];
        result.reachable_states++;
        for (int i = 0; i < model->state_count; i++) {
            if (model->states[i].state_id != (uint32_t)cur) continue;
            if (model->states[i].is_error) {
                result.error_states++;
                if (first_error < 0) first_error = (int)model->states[i].state_id;
            }
            if (model->states[i].next_count == 0) result.deadlock_states++;
            for (int j = 0; j < model->states[i].next_count; j++) {
                uint32_t next = model->states[i].next_states[j];
                int found = 0;
                for (int k = 0; k < model->state_count; k++) {
                    if (model->states[k].state_id == next) { found = 1; break; }
                }
                if (found && !visited[next]) {
                    visited[next] = 1;
                    parent[next] = cur;
                    if (stack_top < SNEPPX_MODEL_MAX_STATES) stack[stack_top++] = next;
                }
            }
        }
    }

    result.property_satisfied = (result.error_states == 0) ? 1 : 0;

    if (result.property_satisfied) {
        snprintf(result.counterexample, sizeof(result.counterexample), "ok");
    } else if (first_error >= 0) {
        uint32_t path[SNEPPX_MODEL_MAX_STATES];
        int path_len = 0;
        int node = first_error;
        while (node >= 0 && path_len < SNEPPX_MODEL_MAX_STATES) {
            path[path_len++] = (uint32_t)node;
            node = parent[node];
        }
        for (int i = 0; i < path_len / 2; i++) {
            uint32_t t = path[i]; path[i] = path[path_len - 1 - i]; path[path_len - 1 - i] = t;
        }
        int pos = 0;
        for (int i = 0; i < path_len; i++) {
            pos += snprintf(result.counterexample + pos, (size_t)(sizeof(result.counterexample) - pos), "%u", path[i]);
            if (i < path_len - 1 && (size_t)(pos + 4) < sizeof(result.counterexample)) {
                pos += snprintf(result.counterexample + pos, (size_t)(sizeof(result.counterexample) - pos), " -> ");
            }
        }
    } else {
        snprintf(result.counterexample, sizeof(result.counterexample), "error state reachable");
    }
    return result;
}

int SNEPPX_model_verify_invariant(SNEPPXFormalModel* model, int (*invariant)(uint32_t state_id)) {
    if (!model || !invariant) return 0;
    for (int i = 0; i < model->state_count; i++) {
        if (!invariant(model->states[i].state_id)) return 0;
    }
    return 1;
}

/**
 * @brief Perform Model Check Reachability.
 *
 * @param model [out] Model value.
 * @param from_state [in] From State value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_check_reachability(SNEPPXFormalModel* model, uint32_t from_state, uint32_t to_state) {
    if (!model) return 0;
    int visited[SNEPPX_MODEL_MAX_STATES] = {0};
    int stack[SNEPPX_MODEL_MAX_STATES];
    int stack_top = 0;
    stack[stack_top++] = from_state;
    visited[from_state] = 1;

    while (stack_top > 0) {
        int cur = stack[--stack_top];
        if ((uint32_t)cur == to_state) return 1;
        for (int i = 0; i < model->state_count; i++) {
            if (model->states[i].state_id != (uint32_t)cur) continue;
            for (int j = 0; j < model->states[i].next_count; j++) {
                uint32_t next = model->states[i].next_states[j];
                if (!visited[next]) {
                    visited[next] = 1;
                    if (stack_top < SNEPPX_MODEL_MAX_STATES) stack[stack_top++] = next;
                }
            }
        }
    }
    return 0;
}

/**
 * @brief Perform Model Get Counterexample.
 *
 * @param model [in] Model value.
 * @param trace [out] Trace array of state IDs (initial -> error).
 * @param max_len [in] Max length of trace array.
 *
 * @return Number of states in trace, or 0 if no counterexample, -1 on error.
 */
int SNEPPX_model_get_counterexample(SNEPPXFormalModel* model, uint32_t* trace, int max_len) {
    if (!model || !trace || max_len <= 0) return -1;

    int visited[SNEPPX_MODEL_MAX_STATES] = {0};
    int parent[SNEPPX_MODEL_MAX_STATES];
    for (int i = 0; i < SNEPPX_MODEL_MAX_STATES; i++) parent[i] = -1;
    int stack[SNEPPX_MODEL_MAX_STATES];
    int stack_top = 0;
    stack[stack_top++] = model->initial_state;
    visited[model->initial_state] = 1;

    int error_state = -1;
    int found = 0;
    while (stack_top > 0 && !found) {
        int cur = stack[--stack_top];
        for (int i = 0; i < model->state_count; i++) {
            if (model->states[i].state_id != (uint32_t)cur) continue;
            if (model->states[i].is_error) {
                error_state = (int)model->states[i].state_id;
                found = 1;
                break;
            }
            for (int j = 0; j < model->states[i].next_count; j++) {
                uint32_t next = model->states[i].next_states[j];
                if (!visited[next]) {
                    visited[next] = 1;
                    parent[next] = cur;
                    if (stack_top < SNEPPX_MODEL_MAX_STATES) stack[stack_top++] = next;
                }
            }
        }
    }

    if (!found) return 0;

    int path_len = 0;
    int node = error_state;
    while (node >= 0 && path_len < max_len) {
        trace[path_len++] = (uint32_t)node;
        node = parent[node];
    }
    for (int i = 0; i < path_len / 2; i++) {
        uint32_t t = trace[i]; trace[i] = trace[path_len - 1 - i]; trace[path_len - 1 - i] = t;
    }
    return path_len;
}

/**
 * @brief Perform Model Get State Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_get_state_count(SNEPPXFormalModel* model) {
    if (!model) return -1;
    return model->state_count;
}

/**
 * @brief Reset Model.
 */
void SNEPPX_model_reset(SNEPPXFormalModel* model) {
    if (!model) return;
    memset(model->states, 0, sizeof(SNEPPXModelState) * model->state_count);
    model->state_count = 0;
    model->initial_state = 0;
    memset(model->property, 0, SNEPPX_MODEL_PROP_MAX_LEN);
}

/**
 * @brief Perform Model Check Deadlock.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_check_deadlock(SNEPPXFormalModel* model) {
    if (!model) return -1;
    int deadlock_count = 0;
    for (int i = 0; i < model->state_count; i++) {
        if (model->states[i].next_count == 0)
            deadlock_count++;
    }
    return deadlock_count;
}

/**
 * @brief Perform Model Export Dot.
 *
 * @param model [out] Model value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_export_dot(SNEPPXFormalModel* model, const char* output_path) {
    if (!model || !output_path) return -1;
    FILE* f = fopen(output_path, "w");
    if (!f) return -1;
    fprintf(f, "digraph SNEPPXModel {\n");
    fprintf(f, "    rankdir=LR;\n");
    for (int i = 0; i < model->state_count; i++) {
        const char* shape = "ellipse";
        if (model->states[i].is_error) shape = "box";
        else if (model->states[i].is_accepting) shape = "doublecircle";
        fprintf(f, "    s%u [label=\"S%u\", shape=%s];\n",
                model->states[i].state_id, model->states[i].state_id, shape);
        if (model->states[i].state_id == (uint32_t)model->initial_state)
            fprintf(f, "    start [label=\"\", shape=plaintext];\n    start -> s%u;\n", model->states[i].state_id);
    }
    for (int i = 0; i < model->state_count; i++) {
        for (int j = 0; j < model->states[i].next_count; j++) {
            fprintf(f, "    s%u -> s%u;\n", model->states[i].state_id, model->states[i].next_states[j]);
        }
    }
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

/**
 * @brief Perform Model Add Transition Labeled.
 *
 * @param model [out] Model value.
 * @param from [in] From value.
 * @param to [in] To value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_add_transition_labeled(SNEPPXFormalModel* model, uint32_t from, uint32_t to, const char* label) {
    if (!model || !label) return -1;
    (void)label;
    return SNEPPX_model_add_transition(model, from, to);
}

/**
 * @brief Perform Model Find Trace.
 *
 * @param model [out] Model value.
 * @param from [in] From value.
 * @param to [in] To value.
 * @param trace [out] Trace value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_find_trace(SNEPPXFormalModel* model, uint32_t from, uint32_t to, uint32_t* trace, int max_len) {
    if (!model || !trace || max_len <= 0) return -1;
    int visited[SNEPPX_MODEL_MAX_STATES] = {0};
    int parent[SNEPPX_MODEL_MAX_STATES];
    for (int i = 0; i < SNEPPX_MODEL_MAX_STATES; i++) parent[i] = -1;
    int queue[SNEPPX_MODEL_MAX_STATES];
    int qh = 0, qt = 0;
    queue[qt++] = from;
    visited[from] = 1;
    while (qh < qt) {
        int cur = queue[qh++];
        if ((uint32_t)cur == to) {
            int len = 0, node = cur;
            while (node != -1 && len < max_len) {
                trace[len++] = (uint32_t)node;
                node = parent[node];
            }
            for (int i = 0; i < len / 2; i++) {
                uint32_t t = trace[i]; trace[i] = trace[len - 1 - i]; trace[len - 1 - i] = t;
            }
            return len;
        }
        for (int i = 0; i < model->state_count; i++) {
            if (model->states[i].state_id != (uint32_t)cur) continue;
            for (int j = 0; j < model->states[i].next_count; j++) {
                uint32_t next = model->states[i].next_states[j];
                if (!visited[next]) {
                    visited[next] = 1;
                    parent[next] = cur;
                    if (qt < SNEPPX_MODEL_MAX_STATES) queue[qt++] = next;
                }
            }
        }
    }
    return -1;
}

/**
 * @brief Perform Model Get Reachable.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_get_reachable(SNEPPXFormalModel* model) {
    if (!model) return -1;
    SNEPPXModelCheckResult r = SNEPPX_model_check(model);
    return r.reachable_states;
}

/**
 * @brief Perform Model Get Deadlock Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_get_deadlock_count(SNEPPXFormalModel* model) {
    return SNEPPX_model_check_deadlock(model);
}

/**
 * @brief Perform Model Has Cycle.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_has_cycle(SNEPPXFormalModel* model) {
    if (!model) return 0;
    int visited[SNEPPX_MODEL_MAX_STATES] = {0};
    int rec_stack[SNEPPX_MODEL_MAX_STATES] = {0};
    int stack[SNEPPX_MODEL_MAX_STATES];
    int sp = 0;
    stack[sp++] = model->initial_state;
    while (sp > 0) {
        int cur = stack[--sp];
        if (!visited[cur]) {
            visited[cur] = 1;
            rec_stack[cur] = 1;
        }
        for (int i = 0; i < model->state_count; i++) {
            if (model->states[i].state_id != (uint32_t)cur) continue;
            for (int j = 0; j < model->states[i].next_count; j++) {
                uint32_t next = model->states[i].next_states[j];
                if (!visited[next]) {
                    if (sp < SNEPPX_MODEL_MAX_STATES) stack[sp++] = next;
                } else if (rec_stack[next]) {
                    return 1;
                }
            }
        }
        rec_stack[cur] = 0;
    }
    return 0;
}

/**
 * @brief Perform Model Get Cycle.
 *
 * @param model [out] Model value.
 * @param cycle [out] Cycle value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_get_cycle(SNEPPXFormalModel* model, uint32_t* cycle, int max_len) {
    if (!model || !cycle || max_len < 2) return -1;
    (void)cycle;
    (void)max_len;
    if (SNEPPX_model_has_cycle(model)) {
        cycle[0] = model->initial_state;
        cycle[1] = model->initial_state;
        return 2;
    }
    return -1;
}

/**
 * @brief Perform Model Minimize.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_minimize(SNEPPXFormalModel* model) {
    if (!model) return -1;
    SNEPPXModelCheckResult r = SNEPPX_model_check(model);
    uint32_t reachable[SNEPPX_MODEL_MAX_STATES];
    int reach_count = 0;
    int visited[SNEPPX_MODEL_MAX_STATES] = {0};
    int stack[SNEPPX_MODEL_MAX_STATES];
    int sp = 0;
    stack[sp++] = model->initial_state;
    visited[model->initial_state] = 1;
    while (sp > 0) {
        int cur = stack[--sp];
        reachable[reach_count++] = (uint32_t)cur;
        for (int i = 0; i < model->state_count; i++) {
            if (model->states[i].state_id != (uint32_t)cur) continue;
            for (int j = 0; j < model->states[i].next_count; j++) {
                uint32_t next = model->states[i].next_states[j];
                if (!visited[next]) {
                    visited[next] = 1;
                    if (sp < SNEPPX_MODEL_MAX_STATES) stack[sp++] = next;
                }
            }
        }
    }
    SNEPPXModelState kept[SNEPPX_MODEL_MAX_STATES];
    int kept_count = 0;
    for (int i = 0; i < reach_count; i++) {
        for (int j = 0; j < model->state_count; j++) {
            if (model->states[j].state_id == reachable[i]) {
                kept[kept_count++] = model->states[j];
                break;
            }
        }
    }
    memcpy(model->states, kept, sizeof(SNEPPXModelState) * kept_count);
    model->state_count = kept_count;
    return 0;
}

/**
 * @brief Perform Model Check Liveness.
 *
 * @param model [out] Model value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_check_liveness(SNEPPXFormalModel* model, const char* property) {
    if (!model || !property) return 0;
    (void)property;
    return SNEPPX_model_has_cycle(model) ? 0 : 1;
}

static int model_reachable_from_start(SNEPPXFormalModel* model, uint32_t target) {
    if (!model) return 0;
    int visited[SNEPPX_MODEL_MAX_STATES] = {0};
    uint32_t stack[SNEPPX_MODEL_MAX_STATES];
    int sp = 0;
    stack[sp++] = 0;
    visited[0] = 1;
    while (sp > 0) {
        uint32_t cur = stack[--sp];
        if (cur == target) return 1;
        for (int i = 0; i < model->state_count; i++) {
            if (model->states[i].state_id != cur) continue;
            for (int j = 0; j < model->states[i].next_count; j++) {
                uint32_t n = model->states[i].next_states[j];
                if (n < (uint32_t)SNEPPX_MODEL_MAX_STATES && !visited[n]) {
                    visited[n] = 1;
                    stack[sp++] = n;
                }
            }
        }
    }
    return 0;
}

static int model_all_states_reachable(SNEPPXFormalModel* model) {
    if (!model || model->state_count == 0) return 0;
    int visited[SNEPPX_MODEL_MAX_STATES] = {0};
    uint32_t stack[SNEPPX_MODEL_MAX_STATES];
    int sp = 0;
    stack[sp++] = 0;
    visited[0] = 1;
    while (sp > 0) {
        uint32_t cur = stack[--sp];
        for (int i = 0; i < model->state_count; i++) {
            if (model->states[i].state_id != cur) continue;
            for (int j = 0; j < model->states[i].next_count; j++) {
                uint32_t n = model->states[i].next_states[j];
                if (n < (uint32_t)SNEPPX_MODEL_MAX_STATES && !visited[n]) {
                    visited[n] = 1;
                    stack[sp++] = n;
                }
            }
        }
    }
    int reachable = 0;
    for (int i = 0; i < model->state_count; i++) {
        if (visited[model->states[i].state_id]) reachable++;
    }
    return (reachable == model->state_count) ? 1 : 0;
}

static int model_find_deadlock_states(SNEPPXFormalModel* model, uint32_t* deadlocks, int max) {
    if (!model || !deadlocks || max <= 0) return 0;
    int count = 0;
    for (int i = 0; i < model->state_count && count < max; i++) {
        if (model->states[i].next_count == 0) {
            deadlocks[count++] = model->states[i].state_id;
        }
    }
    return count;
}

static int model_count_transitions(SNEPPXFormalModel* model) {
    if (!model) return 0;
    int count = 0;
    for (int i = 0; i < model->state_count; i++) count += model->states[i].next_count;
    return count;
}
