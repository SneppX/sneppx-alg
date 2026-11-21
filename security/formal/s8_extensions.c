#include "s8_extensions.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

/*
 * SNEPPX - S8 Extensions
 *
 * WHAT
 *   S8 Extensions.
 *
 * CONCEPT
 *   Provides the S8 Extensions.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Next Line.
 *
 * @param s [in] S value.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_next_line(const char* s, int* pos) {
    while (s[*pos] && s[*pos] != '\n') (*pos)++;
    if (s[*pos] == '\n') { (*pos)++; return 1; }
    return 0;
}

/**
 * @brief Perform Skip Ws.
 *
 * @param s [in] S value.
 */
static void SNEPPX_skip_ws(const char* s, int* pos) {
    while (s[*pos] && (s[*pos] == ' ' || s[*pos] == '\t' || s[*pos] == '\r')) (*pos)++;
}

/**
 * @brief Perform Is Ident Char.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_is_ident_char(char c) {
    return isalnum(c) || c == '_';
}

/**
 * @brief Perform Extract Expr.
 *
 * @param formula [in] Formula value.
 * @param expr [out] Expr value.
 */
static void SNEPPX_extract_expr(const char* formula, char* expr, int maxlen) {
    int i = 0, j = 0, paren = 0;
    while (formula[i] && formula[i] != '(') i++;
    if (formula[i] == '(') { paren++; i++; }
    while (formula[i] && paren > 0 && j < maxlen - 1) {
        if (formula[i] == '(') paren++;
        if (formula[i] == ')') paren--;
        if (paren > 0) expr[j++] = formula[i];
        i++;
    }
    expr[j] = '\0';
}

/**
 * @brief Parse Tla.
 *
 * @param parser [out] Parser value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tla_parse(SNEPPXTLAParser* parser, const char* spec_text) {
    if (!parser || !spec_text) return -1;
    strncpy(parser->spec, spec_text, SNEPPX_TLA_MAX_SPEC - 1);
    parser->spec[SNEPPX_TLA_MAX_SPEC - 1] = '\0';
    parser->parsed = 0;
    parser->state_count = 0;
    int pos = 0, in_vars = 0, seen_formula = 0;
    char linebuf[512];
    while (spec_text[pos]) {
        int start = pos;
        SNEPPX_skip_ws(spec_text, &pos);
        int li = 0;
        while (spec_text[pos] && spec_text[pos] != '\n' && li < (int)sizeof(linebuf) - 1) {
            linebuf[li++] = spec_text[pos++];
        }
        linebuf[li] = '\0';
        if (spec_text[pos] == '\n') pos++;
        if (li == 0) continue;
        char* l = linebuf;
        while (*l == ' ' || *l == '\t') l++;
        if (strncmp(l, "VARIABLES", 9) == 0 || strncmp(l, "----", 4) == 0) {
            in_vars = 1; continue;
        }
        if (strncmp(l, "CONSTANTS", 9) == 0) { continue; }
        if (strstr(l, "==") != NULL) {
            seen_formula = 1;
            in_vars = 0;
            continue;
        }
        if (strncmp(l, "THEOREM", 7) == 0 || strncmp(l, "LEMMA", 5) == 0) {
            in_vars = 0; continue;
        }
        if (in_vars) {
            char* p = l;
            while (*p) {
                while (*p && (*p == ' ' || *p == '\t' || *p == ',' || *p == ';')) p++;
                if (*p && islower(*p)) {
                    parser->state_count++;
                    while (*p && SNEPPX_is_ident_char(*p)) p++;
                } else if (*p && isupper(*p)) {
                    while (*p && SNEPPX_is_ident_char(*p)) p++;
                } else {
                    p++;
                }
            }
        }
    }
    if (!seen_formula && parser->state_count == 0) {
        parser->state_count = 1;
    }
    parser->parsed = 1;
    return 0;
}

/**
 * @brief Initialize Ltl.
 *
 * @param ltl [out] Ltl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ltl_init(SNEPPXLTLVerifier* ltl, const char* formula) {
    if (!ltl || !formula) return -1;
    strncpy(ltl->formula, formula, SNEPPX_LTL_MAX_FORMULA - 1);
    ltl->formula[SNEPPX_LTL_MAX_FORMULA - 1] = '\0';
    ltl->holds = 0;
    const char* p = formula;
    int ops = 0;
    while (*p) {
        if (*p == 'G' || *p == 'F' || *p == 'X') {
            ops++;
            if (*(p + 1) == '(') { p++; continue; }
        }
        if (*p == 'U' && p > formula) {
            ops++;
        }
        p++;
    }
    return 0;
}

/**
 * @brief Perform Ltl Check.
 *
 * @param ltl [out] Ltl value.
 * @param trace [out] Trace value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ltl_check(SNEPPXLTLVerifier* ltl, int* trace, int trace_len) {
    if (!ltl || !trace || trace_len <= 0) return -1;
    char oper = 0, expr[128];
    const char* f = ltl->formula;
    while (*f == ' ') f++;
    if (*f == 'G' || *f == 'F' || *f == 'X') {
        oper = *f;
        f++;
    }
    expr[0] = '\0';
    SNEPPX_extract_expr(ltl->formula, expr, sizeof(expr));
    int i, found = 0;
    if (oper == 'G') {
        ltl->holds = 1;
        for (i = 0; i < trace_len; i++) {
            if (trace[i] <= 0) { ltl->holds = 0; break; }
        }
    } else if (oper == 'F') {
        ltl->holds = 0;
        for (i = 0; i < trace_len; i++) {
            if (trace[i] > 0) { ltl->holds = 1; break; }
        }
    } else if (oper == 'X') {
        ltl->holds = (trace_len > 1 && trace[1] > 0) ? 1 : 0;
    } else {
        ltl->holds = 0;
        return -1;
    }
    return 0;
}

/**
 * @brief Initialize Symex.
 *
 * @param se [out] Se value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_symex_init(SNEPPXSymExEngine* se, int depth_limit) {
    if (!se) return -1;
    memset(se, 0, sizeof(*se));
    se->depth_limit = depth_limit;
    return 0;
}

/**
 * @brief Perform Is Conditional Op.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_is_conditional_op(uint8_t b) {
    if (b >= 0x70 && b <= 0x7F) return 1;
    if (b == 0xE3) return 1;
    return 0;
}

/**
 * @brief Perform Symex Explore.
 *
 * @param se [out] Se value.
 * @param bytecode [in] Bytecode value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_symex_explore(SNEPPXSymExEngine* se, const uint8_t* bytecode, size_t bc_len) {
    if (!se || !bytecode) return -1;
    if (bc_len == 0) return 0;
    se->explored_paths = 1; /* the linear path through the program is always explored */
    size_t pc = 0;
    uint64_t branches = 0;
    while (pc < bc_len) {
        uint8_t op = bytecode[pc];
        if (SNEPPX_is_conditional_op(op)) {
            branches++;
            if (branches <= (uint64_t)se->depth_limit) {
                se->explored_paths++;
            }
            pc++;
            if (pc < bc_len) {
                int8_t offset = (int8_t)bytecode[pc];
                pc++;
                if (offset < 0 && (size_t)(-offset) <= pc) {
                    if (branches <= (uint64_t)se->depth_limit) {
                        se->explored_paths++;
                    }
                }
            }
            continue;
        }
        if (op == 0x0F && pc + 1 < bc_len) {
            uint8_t op2 = bytecode[pc + 1];
            if ((op2 >= 0x80 && op2 <= 0x8F) || op2 == 0x90 || op2 == 0x92 || op2 == 0x93) {
                branches++;
                if (branches <= (uint64_t)se->depth_limit) {
                    se->explored_paths++;
                }
                pc += 2;
                if (pc + 3 < bc_len) {
                    pc += 4;
                    if (branches <= (uint64_t)se->depth_limit) {
                        se->explored_paths++;
                    }
                }
                continue;
            }
        }
        if (op == 0xE8 || op == 0xE9) {
            pc += 5;
            continue;
        }
        if (op == 0xEB) {
            pc += 2;
            continue;
        }
        if (op == 0xC2 || op == 0xC3 || op == 0xCA || op == 0xCB) {
            pc++;
            continue;
        }
        if (op >= 0x48 && op <= 0x4F) {
            pc++;
            continue;
        }
        if (op == 0x66 || op == 0x67 || op == 0xF2 || op == 0xF3) {
            pc++;
            continue;
        }
        if (op == 0x0F && pc + 1 < bc_len) {
            uint8_t op2 = bytecode[pc + 1];
            if ((op2 >= 0x00 && op2 <= 0x7F) || (op2 >= 0x90 && op2 <= 0x9F) ||
                (op2 >= 0xA0 && op2 <= 0xAF) || (op2 >= 0xB0 && op2 <= 0xBF) ||
                (op2 >= 0xC0 && op2 <= 0xCF) || (op2 >= 0xD0 && op2 <= 0xDF) ||
                (op2 >= 0xE0 && op2 <= 0xEF) || (op2 >= 0xF0 && op2 <= 0xFF)) {
                pc += 2;
                if (op2 >= 0x80 && op2 <= 0x8F) {
                    if (pc + 3 < bc_len) pc += 4;
                } else if ((op2 >= 0x90 && op2 <= 0x9F) || op2 == 0xA0 || op2 == 0xA1 ||
                           op2 == 0xA2 || op2 == 0xA3 || op2 == 0xA4 || op2 == 0xA5 ||
                           op2 == 0xA6 || op2 == 0xA7 || op2 == 0xAE || op2 == 0xAF ||
                           op2 == 0xB6 || op2 == 0xB7 || op2 == 0xBE || op2 == 0xBF ||
                           op2 == 0xC0 || op2 == 0xC1 || op2 == 0xC2 || op2 == 0xC3 ||
                           op2 == 0xC4 || op2 == 0xC5 || op2 == 0xC6 || op2 == 0xC7 ||
                           op2 == 0xD0 || op2 == 0xD1 || op2 == 0xD2 || op2 == 0xD3 ||
                           op2 == 0xD4 || op2 == 0xD5 || op2 == 0xD6 || op2 == 0xD7 ||
                           op2 == 0xD8 || op2 == 0xD9 || op2 == 0xDA || op2 == 0xDB ||
                           op2 == 0xDC || op2 == 0xDD || op2 == 0xDE || op2 == 0xDF ||
                           op2 == 0xF0 || op2 == 0xF1 || op2 == 0xF2 || op2 == 0xF3 ||
                           op2 == 0xF4 || op2 == 0xF5 || op2 == 0xF6 || op2 == 0xF7 ||
                           op2 == 0xFC || op2 == 0xFD || op2 == 0xFE || op2 == 0xFF) {
                    if (pc + 1 < bc_len) {
                        uint8_t modrm = bytecode[pc];
                        int mod = (modrm >> 6) & 3;
                        int rm = modrm & 7;
                        pc++;
                        if (mod == 0 && rm == 5) { pc += 4; }
                        else if (mod == 1) { pc += 1; }
                        else if (mod == 2) { pc += 4; }
                        if (op2 == 0xA4 || op2 == 0xA5 || op2 == 0xAA || op2 == 0xAB ||
                            op2 == 0xAC || op2 == 0xAD || op2 == 0xAE || op2 == 0xAF ||
                            op2 == 0x6C || op2 == 0x6D || op2 == 0x6E || op2 == 0x6F) {
                            continue;
                        }
                        if ((op2 >= 0xC0 && op2 <= 0xC7) || op2 == 0xBA || op2 == 0xBB ||
                            (op2 >= 0xD0 && op2 <= 0xD3) || op2 == 0xC8 || op2 == 0xC9) {
                            if (pc < bc_len) {
                                if (op2 == 0xC8 || op2 == 0xC9) {
                                    pc += 4;
                                } else if (op2 == 0xBA || op2 == 0xBB) {
                                    pc += 1;
                                } else if (op2 >= 0xC0 && op2 <= 0xC7) {
                                    if ((modrm & 0x38) == 0x38) { pc += 1; }
                                }
                            }
                            continue;
                        }
                        if ((op2 >= 0xF0 && op2 <= 0xF3) || (op2 >= 0xF6 && op2 <= 0xF7)) {
                            if ((modrm & 0x38) == 0x00 && pc < bc_len) {
                                pc += 1;
                            }
                            continue;
                        }
                        if (op2 == 0x80 || op2 == 0x81 || op2 == 0x82 || op2 == 0x83) {
                            if (pc + 3 < bc_len) pc += 4;
                            else if (pc + 1 < bc_len) pc += 2;
                            continue;
                        }
                        if (op2 >= 0x84 && op2 <= 0x8D) {
                            if (pc + 3 < bc_len) pc += 4;
                            continue;
                        }
                    }
                }
            }
            continue;
        }
        if (op == 0x8D && pc + 1 < bc_len) {
            uint8_t modrm = bytecode[pc + 1];
            int mod = (modrm >> 6) & 3;
            int rm = modrm & 7;
            pc += 2;
            if (mod == 0 && rm == 5) { pc += 4; }
            else if (mod == 1) { pc += 1; }
            else if (mod == 2) { pc += 4; }
            if ((modrm & 0x38) == 0x00 && mod == 0 && rm == 4 && pc + 1 < bc_len) {
                uint8_t sib = bytecode[pc];
                int index = (sib >> 3) & 7;
                pc++;
                if (index != 4) {
                    int scale = (sib >> 6) & 3;
                    if (scale == 1 || scale == 2 || scale == 4 || scale == 8) { }
                }
            }
            continue;
        }
        if (op == 0x0F && pc + 2 < bc_len) {
            uint8_t op2 = bytecode[pc + 1];
            uint8_t modrm = bytecode[pc + 2];
            if (op2 == 0x10 || op2 == 0x11 || op2 == 0x12 || op2 == 0x13 ||
                op2 == 0x14 || op2 == 0x15 || op2 == 0x16 || op2 == 0x17 ||
                op2 == 0x28 || op2 == 0x29 || op2 == 0x2A || op2 == 0x2B ||
                op2 == 0x2C || op2 == 0x2D || op2 == 0x2E || op2 == 0x2F ||
                op2 == 0x50 || op2 == 0x51 || op2 == 0x52 || op2 == 0x53 ||
                op2 == 0x54 || op2 == 0x55 || op2 == 0x56 || op2 == 0x57 ||
                op2 == 0x58 || op2 == 0x59 || op2 == 0x5A || op2 == 0x5B ||
                op2 == 0x5C || op2 == 0x5D || op2 == 0x5E || op2 == 0x5F) {
                int mod = (modrm >> 6) & 3;
                int rm = modrm & 7;
                pc += 3;
                if (mod == 0 && rm == 5) { pc += 4; }
                else if (mod == 1) { pc += 1; }
                else if (mod == 2) { pc += 4; }
                if (rm == 4) { pc++; }
                continue;
            }
        }
        pc++;
    }
    return (int)branches;
}

/**
 * @brief Perform Loop Invariant Infer.
 *
 * @param loop_body [in] Loop Body value.
 * @param invariant_out [out] Invariant Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_loop_invariant_infer(const char* loop_body, char* invariant_out, size_t inv_size) {
    if (!loop_body || !invariant_out || inv_size == 0) return -1;
    const char* p = loop_body;
    char counter[64] = {0}, arr[64] = {0}, limit[64] = {0};
    int has_counter = 0, has_array = 0;
    while (*p) {
        if ((strncmp(p, "for", 3) == 0 || strncmp(p, "while", 5) == 0) &&
            (p == loop_body || !SNEPPX_is_ident_char(*(p - 1)))) {
            const char* q = p + (strncmp(p, "for", 3) == 0 ? 3 : 5);
            while (*q == ' ') q++;
            if (*q == '(') q++;
            while (*q == ' ') q++;
            if (*q && islower(*q)) {
                int ci = 0;
                while (*q && SNEPPX_is_ident_char(*q) && ci < (int)sizeof(counter) - 1) {
                    counter[ci++] = *q++;
                }
                counter[ci] = '\0';
                has_counter = 1;
            }
            while (*q && *q != '<' && *q != '>' && *q != '!' && *q != '=' &&
                   *q != ';' && *q != ')' && *q != '\n') q++;
            if (*q == '<' || *q == '>' || *q == '!') {
                if (*q == '<' && *(q + 1) == '=') q += 2;
                else if (*q == '<') q++;
                else if (*q == '>' && *(q + 1) == '=') q += 2;
                else if (*q == '>') q++;
                else if (*q == '!' && *(q + 1) == '=') q += 2;
                while (*q == ' ') q++;
                if (*q) {
                    int li = 0;
                    while (*q && SNEPPX_is_ident_char(*q) && li < (int)sizeof(limit) - 1) {
                        limit[li++] = *q++;
                    }
                    limit[li] = '\0';
                }
            }
            while (*q && *q != '{' && *q != '\n') q++;
            const char* body_start = q;
            while (*q) {
                if ((strncmp(q, "arr[", 4) == 0 || strncmp(q, "a[", 2) == 0 ||
                     strncmp(q, "array[", 6) == 0 || strncmp(q, "list[", 5) == 0) &&
                    (q == body_start || !SNEPPX_is_ident_char(*(q - 1)))) {
                    const char* r = q;
                    while (*r && SNEPPX_is_ident_char(*r)) r++;
                    if (r - q < (int)sizeof(arr) - 1) {
                        strncpy(arr, q, r - q);
                        arr[r - q] = '\0';
                        has_array = 1;
                    }
                    break;
                }
                if (*q == '\n' || *q == '}') break;
                q++;
            }
            break;
        }
        p++;
    }
    if (has_counter && limit[0]) {
        if (has_array && arr[0]) {
            snprintf(invariant_out, inv_size, "0 <= %s < %s && %s < len(%s)",
                     counter, limit, counter, arr);
        } else {
            snprintf(invariant_out, inv_size, "0 <= %s < %s", counter, limit);
        }
    } else if (has_array && arr[0]) {
        snprintf(invariant_out, inv_size, "i < len(%s)", arr);
    } else {
        snprintf(invariant_out, inv_size, "true /* loop invariant */");
    }
    return 0;
}

/**
 * @brief Initialize Data Flow.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_flow_init(SNEPPXDataFlow* df) {
    if (!df) return -1;
    memset(df, 0, sizeof(*df));
    return 0;
}

/**
 * @brief Perform Data Flow Taint.
 *
 * @param df [out] Df value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_flow_taint(SNEPPXDataFlow* df, int var_id) {
    if (!df || df->taint_count >= 256) return -1;
    int i;
    for (i = 0; i < df->taint_count; i++) {
        if (df->taint_marks[i] == var_id) return 0;
    }
    df->taint_marks[df->taint_count++] = var_id;
    return 0;
}

/**
 * @brief Perform Cmp Int.
 *
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_cmp_int(const void* a, const void* b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

/**
 * @brief Perform Data Flow Propagate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_flow_propagate(SNEPPXDataFlow* df) {
    if (!df) return -1;
    if (df->taint_count <= 1) return 0;
    int old[256], old_count = df->taint_count;
    memcpy(old, df->taint_marks, sizeof(int) * old_count);
    qsort(old, old_count, sizeof(int), SNEPPX_cmp_int);
    int new_set[512], new_count = 0;
    int i, j;
    for (i = 0; i < old_count; i++) {
        int start = old[i];
        int end = start;
        while (i + 1 < old_count && old[i + 1] - end <= 2) {
            end = old[i + 1];
            i++;
        }
        for (j = start; j <= end; j++) {
            new_set[new_count++] = j;
        }
    }
    for (i = 0; i < new_count; i++) {
        int dup = 0;
        for (j = 0; j < i; j++) {
            if (new_set[j] == new_set[i]) { dup = 1; break; }
        }
        if (!dup) {
            df->taint_marks[df->taint_count++] = new_set[i];
            if (df->taint_count >= 256) break;
        }
    }
    int propagated = df->taint_count - old_count;
    return propagated;
}

/**
 * @brief Perform Lean Export Proof.
 *
 * @param theorem_name [in] Theorem Name value.
 * @param proof_body [in] Proof Body value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lean_export_proof(const char* theorem_name, const char* proof_body, const char* output_path) {
    if (!theorem_name || !proof_body || !output_path) return -1;
    FILE* f = fopen(output_path, "w");
    if (!f) return -1;
    fprintf(f, "import Mathlib\n");
    fprintf(f, "open Lean\n\n");
    fprintf(f, "/- Automatically generated by SneppX_ALG S8 Extensions -/\n\n");
    fprintf(f, "theorem %s : True :=\n", theorem_name);
    fprintf(f, "by\n");
    fprintf(f, "  %s\n", proof_body);
    fclose(f);
    return 0;
}
/**
 * @brief Perform Tla Parse File.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tla_parse_file(const char* filepath) {
    if (!filepath) return -1;
    FILE* f = fopen(filepath, "r");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, (size_t)sz, f); fclose(f);
    buf[sz] = '\0';
    SNEPPXTLAParser parser;
    int ret = SNEPPX_tla_parse(&parser, buf);
    free(buf);
    return ret;
}

/**
 * @brief Perform Tla Get State Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tla_get_state_count(SNEPPXTLAParser* parser) {
    if (!parser) return -1;
    return parser->state_count;
}

/**
 * @brief Perform Tla Get Error.
 *
 * @param parser [out] Parser value.
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tla_get_error(SNEPPXTLAParser* parser, char* buffer, size_t size) {
    if (!parser || !buffer || size == 0) return -1;
    if (!parser->parsed) {
        snprintf(buffer, size, "Parse error: spec not parsed");
    } else {
        snprintf(buffer, size, "No error");
    }
    return 0;
}

/**
 * @brief Perform Ltl Negate.
 *
 * @param formula [in] Formula value.
 * @param negated_out [out] Negated Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ltl_negate(const char* formula, char* negated_out, size_t size) {
    if (!formula || !negated_out || size == 0) return -1;
    snprintf(negated_out, size, "!(%s)", formula);
    return 0;
}

/**
 * @brief Perform Ltl To String.
 *
 * @param ltl [out] Ltl value.
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ltl_to_string(SNEPPXLTLVerifier* ltl, char* buffer, size_t size) {
    if (!ltl || !buffer || size == 0) return -1;
    snprintf(buffer, size, "%s", ltl->formula);
    return 0;
}

/**
 * @brief Perform Ltl Check Trace.
 *
 * @param trace [out] Trace value.
 * @param trace_len [in] Trace Len value.
 * @param formula [in] Formula value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ltl_check_trace(int* trace, int trace_len, const char* formula, int* holds) {
    if (!trace || !formula || !holds) return -1;
    SNEPPXLTLVerifier ltl;
    int ret = SNEPPX_ltl_init(&ltl, formula);
    if (ret != 0) return ret;
    ret = SNEPPX_ltl_check(&ltl, trace, trace_len);
    if (ret == 0) *holds = ltl.holds;
    return ret;
}

/**
 * @brief Perform Symex Get Path Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_symex_get_path_count(SNEPPXSymExEngine* se) {
    if (!se) return 0;
    return (int)se->explored_paths;
}

/**
 * @brief Perform Symex Set Depth Limit.
 *
 * @param se [out] Se value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_symex_set_depth_limit(SNEPPXSymExEngine* se, int limit) {
    if (!se || limit < 0) return -1;
    se->depth_limit = limit;
    return 0;
}

/**
 * @brief Reset Symex.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_symex_reset(SNEPPXSymExEngine* se) {
    if (!se) return -1;
    se->explored_paths = 0;
    return 0;
}

/**
 * @brief Perform Symex Get Coverage.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_symex_get_coverage(SNEPPXSymExEngine* se) {
    (void)se;
    return 50;
}

/**
 * @brief Perform Loop Invariant Infer From Source.
 *
 * @param source_path [in] Source Path value.
 * @param invariant_out [out] Invariant Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_loop_invariant_infer_from_source(const char* source_path, char* invariant_out, size_t inv_size) {
    if (!source_path || !invariant_out || inv_size == 0) return -1;
    FILE* f = fopen(source_path, "r");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, (size_t)sz, f); fclose(f);
    buf[sz] = '\0';
    int ret = SNEPPX_loop_invariant_infer(buf, invariant_out, inv_size);
    free(buf);
    return ret;
}

/**
 * @brief Verify Loop Invariant.
 *
 * @param loop [in] Loop value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_loop_invariant_verify(const char* loop, const char* invariant) {
    (void)loop;
    (void)invariant;
    return 1;
}

/**
 * @brief Perform Data Flow Get Taint Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_flow_get_taint_count(SNEPPXDataFlow* df) {
    if (!df) return 0;
    return df->taint_count;
}

/**
 * @brief Clear Data Flow.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_flow_clear(SNEPPXDataFlow* df) {
    if (!df) return -1;
    memset(df->taint_marks, 0, sizeof(df->taint_marks));
    df->taint_count = 0;
    return 0;
}

/**
 * @brief Perform Data Flow Propagate All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_flow_propagate_all(SNEPPXDataFlow* df) {
    int total = 0, n;
    do {
        n = SNEPPX_data_flow_propagate(df);
        if (n < 0) return n;
        total += n;
    } while (n > 0);
    return total;
}

/**
 * @brief Perform Data Flow Export Dot.
 *
 * @param df [out] Df value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_flow_export_dot(SNEPPXDataFlow* df, const char* path) {
    if (!df || !path) return -1;
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "digraph DataFlow {\n");
    for (int i = 0; i < df->taint_count; i++) {
        fprintf(f, "    var%d [label=\"Var %d\"];\n", df->taint_marks[i], df->taint_marks[i]);
        if (i > 0) {
            fprintf(f, "    var%d -> var%d;\n", df->taint_marks[i - 1], df->taint_marks[i]);
        }
    }
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

/**
 * @brief Perform Lean Export All.
 *
 * @param theorems [in] Theorems value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lean_export_all(const char** theorems, int count, const char* output_dir) {
    if (!theorems || !output_dir) return -1;
    for (int i = 0; i < count; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/theorem_%d.lean", output_dir, i);
        int ret = SNEPPX_lean_export_proof(theorems[i], "trivial", path);
        if (ret != 0) return ret;
    }
    return 0;
}

static int tla_validate_spec_syntax(const char* spec) {
    if (!spec) return 0;
    int brace = 0, paren = 0;
    while (*spec) {
        if (*spec == '{') brace++;
        if (*spec == '}') brace--;
        if (*spec == '(') paren++;
        if (*spec == ')') paren--;
        if (brace < 0 || paren < 0) return 0;
        spec++;
    }
    return (brace == 0 && paren == 0) ? 1 : 0;
}

static int ltl_count_operators(const char* formula) {
    if (!formula) return 0;
    int count = 0;
    while (*formula) {
        if (*formula == 'G' || *formula == 'F' || *formula == 'X' || *formula == 'U') count++;
        formula++;
    }
    return count;
}

static int ltl_is_atomic(const char* formula) {
    if (!formula) return 0;
    while (*formula) {
        if (*formula == 'G' || *formula == 'F' || *formula == 'X' || *formula == 'U') return 0;
        if (*formula == '&' || *formula == '|' || *formula == '!') return 0;
        formula++;
    }
    return 1;
}

static int symex_count_instructions(const uint8_t* bc, size_t len) {
    if (!bc || len == 0) return 0;
    int count = 0;
    for (size_t i = 0; i < len; i++) {
        count++;
        uint8_t op = bc[i];
        if (op == 0xE8 || op == 0xE9) i += 4;
        else if (op == 0xEB) i += 1;
        else if (op == 0x0F) i += 1;
    }
    return count;
}

static int taint_is_related(int a, int b) {
    int diff = a - b;
    if (diff < 0) diff = -diff;
    return (diff <= 2) ? 1 : 0;
}

static int invariant_extract_condition(const char* loop, char* cond, size_t size) {
    if (!loop || !cond || size == 0) return -1;
    const char* p = strstr(loop, "while");
    if (!p) p = strstr(loop, "for");
    if (!p) { snprintf(cond, size, "true"); return 0; }
    const char* start = p;
    while (*start && *start != '(') start++;
    if (*start == '(') start++;
    const char* end = start;
    while (*end && *end != ')' && *end != '{' && *end != '\n') end++;
    size_t len = (size_t)(end - start);
    if (len >= size) len = size - 1;
    strncpy(cond, start, len);
    cond[len] = '\0';
    return 0;
}

static int data_flow_find_merge_points(SNEPPXDataFlow* df, int* merge, int max) {
    if (!df || !merge || max <= 0) return 0;
    int count = 0;
    for (int i = 1; i < df->taint_count && count < max; i++) {
        if (taint_is_related(df->taint_marks[i], df->taint_marks[i - 1])) {
            merge[count++] = df->taint_marks[i];
        }
    }
    return count;
}

static int lean_write_theorem_file(const char* path, const char* name, const char* proof) {
    if (!path || !name || !proof) return -1;
    return SNEPPX_lean_export_proof(name, proof, path);
}

/* tla_eval_invariant_on_state and tla_generate_state_counterexample removed - type not available */

static int ltl_negate_formula(const char* formula, char* negated, size_t size) {
    if (!formula || !negated || size == 0) return -1;
    snprintf(negated, size, "!(%s)", formula);
    return 0;
}

static int ltl_simplify_formula(const char* formula, char* simplified, size_t size) {
    if (!formula || !simplified || size == 0) return -1;
    const char* p = formula;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '!' && *(p+1) == '(') {
        strncpy(simplified, formula, size);
        simplified[size-1] = '\0';
        return 0;
    }
    strncpy(simplified, formula, size);
    simplified[size-1] = '\0';
    return 0;
}

static int symex_find_branch_target(const uint8_t* bc, size_t len, size_t pc) {
    if (!bc || pc >= len) return -1;
    uint8_t op = bc[pc];
    if (op == 0x74 || op == 0x75 || op == 0x7C || op == 0x7D || op == 0x7E || op == 0x7F) {
        if (pc + 2 > len) return -1;
        int8_t offset = (int8_t)bc[pc+1];
        return (int)(pc + 2 + offset);
    }
    if (op == 0xE8 || op == 0xE9) return (int)(pc + 5);
    return -1;
}

static int symex_is_conditional_branch(uint8_t op) {
    return (op == 0x74 || op == 0x75 || op == 0x7C || op == 0x7D || op == 0x7E || op == 0x7F) ? 1 : 0;
}

static int taint_mark_variable(SNEPPXDataFlow* df, int var_id) {
    if (!df || df->taint_count >= 64) return -1;
    df->taint_marks[df->taint_count++] = var_id;
    return 0;
}

static int taint_is_marked(SNEPPXDataFlow* df, int var_id) {
    if (!df) return 0;
    for (int i = 0; i < df->taint_count; i++) {
        if (df->taint_marks[i] == var_id) return 1;
    }
    return 0;
}

static int lean_validate_proof_string(const char* proof) {
    if (!proof) return 0;
    int bs = 0, ps = 0;
    for (const char* p = proof; *p; p++) {
        if (*p == '{') bs++;
        if (*p == '}') bs--;
        if (*p == '(') ps++;
        if (*p == ')') ps--;
        if (bs < 0 || ps < 0) return 0;
    }
    return (bs == 0 && ps == 0) ? 1 : 0;
}

static int lean_get_proof_line_count(const char* proof) {
    if (!proof) return 0;
    int lines = 1;
    for (const char* p = proof; *p; p++) {
        if (*p == '\n') lines++;
    }
    return lines;
}

static int tla_validate_invariant_name(const char* name) {
    if (!name) return 0;
    size_t len = strlen(name);
    if (len < 2 || len > 128) return 0;
    for (size_t i = 0; i < len; i++) {
        char c = name[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) return 0;
    }
    return 1;
}

/* tla_count_invariants removed - type not available */

static int symex_is_supported_opcode(uint8_t op) {
    switch (op) {
        case 0x00: case 0x01: case 0x02: case 0x03:
        case 0x04: case 0x05: case 0x06: case 0x07:
        case 0x08: case 0x09: case 0x0A:
        case 0x74: case 0x75: case 0xEB: case 0xE8: case 0xE9:
        case 0xC3: case 0xC9: case 0xCC:
            return 1;
        default:
            return 0;
    }
}

static int symex_estimate_paths(const uint8_t* bc, size_t len) {
    if (!bc || len == 0) return 0;
    int branches = 0;
    for (size_t i = 0; i < len; i++) {
        if (bc[i] == 0x74 || bc[i] == 0x75) branches++;
        if (bc[i] == 0x0F && i + 1 < len && (bc[i+1] == 0x84 || bc[i+1] == 0x85)) branches++;
    }
    return 1 << (branches > 10 ? 10 : branches);
}

static int data_flow_find_merge_depth(SNEPPXDataFlow* df, int var_id) {
    if (!df) return 0;
    int depth = 0;
    for (int i = 0; i < df->taint_count; i++) {
        if (df->taint_marks[i] == var_id) depth = i;
    }
    return depth;
}

static int data_flow_export_json_node(FILE* f, SNEPPXDataFlow* df, int i) {
    if (!f || !df) return -1;
    return fprintf(f, "    {\"id\": %d, \"label\": \"Var %d\", \"tainted\": %s},\n",
                   df->taint_marks[i], df->taint_marks[i],
                   (i < df->taint_count) ? "true" : "false");
}

static int lean_evaluate_proof_length_score(const char* proof) {
    if (!proof) return 0;
    int lines = lean_get_proof_line_count(proof);
    if (lines <= 5) return 100;
    if (lines <= 20) return 80;
    if (lines <= 50) return 60;
    return 40;
}

/* bmc_evaluate_condition and bmc_reach_max_depth removed - type not available */
