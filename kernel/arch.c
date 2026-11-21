#include "system_architecture_definitions.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
/*
 * SNEPPX - Arch
 *
 * WHAT
 *   Arch.
 *
 * CONCEPT
 *   Provides neural architecture.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


#elif defined(__linux__)
#include <unistd.h>
#include <stdio.h>
#endif

/**
 * @brief Perform Arch Has Avx.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arch_has_avx(void) {
#if defined(__AVX__)
    return 1;
#elif defined(__x86_64__) || defined(_M_AMD64)
    return 1;
#else
    return 0;
#endif
}

/**
 * @brief Perform Arch Has Avx2.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arch_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#elif defined(__x86_64__) || defined(_M_AMD64)
    return 1;
#else
    return 0;
#endif
}

/**
 * @brief Perform Arch Has Neon.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arch_has_neon(void) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
    return 1;
#else
    return 0;
#endif
}

/**
 * @brief Perform Arch Num Cores.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arch_num_cores(void) {
#if defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (int)sysinfo.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_ONLN)
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#else
    return 1;
#endif
/*
 * SNEPPX - Architecture Variants
 *
 * WHAT
 *   Architecture Variants.
 *
 * CONCEPT
 *   Architecture Variants implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


}

/**
 * @brief Perform Arch Cache Line Size.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arch_cache_line_size(void) {
    return 64;
}

/**
 * @brief Perform Arch Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXArchConfig SNEPPX_arch_config_default(void) {
    SNEPPXArchConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.hss_config = SNEPPX_hss_config_default();
    cfg.ser_config = SNEPPX_ser_config_default();
    cfg.arc_config = SNEPPX_arc_config_default();
    cfg.npe_config = SNEPPX_npe_config_default();
    cfg.fm_config = SNEPPX_fm_config_default();
    cfg.attention_config = SNEPPX_attn_config_default();
    cfg.input_dim = 512;
    cfg.output_dim = 512;
    cfg.vocab_size = 256;
    cfg.seed = 42;
    cfg.enable_attention = 1;
    cfg.enable_hss = 1;
    cfg.enable_ser = 0;
    cfg.enable_arc = 0;
    cfg.enable_npe = 0;
    cfg.enable_fm = 0;
    return cfg;
}

/**
 * @brief Create Model.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXModel* SNEPPX_model_create(const SNEPPXArchConfig* config) {
    if (!config) return NULL;
    SNEPPXModel* model = (SNEPPXModel*)SNEPPX_malloc(sizeof(SNEPPXModel), 64);
    if (!model) return NULL;
    memset(model, 0, sizeof(SNEPPXModel));

    model->config = *config;
    unsigned int seed = config->seed ? config->seed : 42;
    size_t in_dim = config->input_dim ? config->input_dim : 16;
    size_t vocab = config->vocab_size ? config->vocab_size : 256;

    if (config->enable_hss) {
        SNEPPXHSSConfig hss_cfg = config->hss_config;
        if (hss_cfg.input_dim == 0) { hss_cfg = SNEPPX_hss_config_default(); hss_cfg.input_dim = in_dim; }
        if (hss_cfg.output_dim == 0) hss_cfg.output_dim = in_dim;
        model->hss_model = SNEPPX_hss_model_create(&hss_cfg, seed);
        if (!model->hss_model) { SNEPPX_model_destroy(model); return NULL; }
    }

    if (config->enable_ser) {
        SNEPPXSERConfig ser_cfg = config->ser_config;
        if (ser_cfg.num_experts == 0) { ser_cfg = SNEPPX_ser_config_default(); ser_cfg.input_dim = in_dim; }
        if (ser_cfg.output_dim == 0) ser_cfg.output_dim = in_dim;
        model->ser_model = SNEPPX_ser_model_create(&ser_cfg, seed + 1, 1);
        if (!model->ser_model) { SNEPPX_model_destroy(model); return NULL; }
    }

    if (config->enable_arc) {
        SNEPPXARCConfig arc_cfg = config->arc_config;
        if (arc_cfg.input_guard_strength == 0.0f) arc_cfg = SNEPPX_arc_config_default();
        model->arc_layer = SNEPPX_arc_layer_create(&arc_cfg, in_dim, in_dim, seed + 2);
        if (!model->arc_layer) { SNEPPX_model_destroy(model); return NULL; }
    }

    if (config->enable_npe) {
        SNEPPXNPEConfig npe_cfg = config->npe_config;
        if (npe_cfg.max_program_length == 0) npe_cfg = SNEPPX_npe_config_default();
        model->npe_program = SNEPPX_npe_compile_mlp(in_dim, in_dim * 2);
        if (!model->npe_program) { SNEPPX_model_destroy(model); return NULL; }
        unsigned long s = seed;
        size_t total_w = in_dim * (in_dim * 2) + (in_dim * 2) + (in_dim * 2) * in_dim + in_dim;
        size_t limit = total_w < model->npe_program->memory->size ? total_w : model->npe_program->memory->size;
        for (size_t i = 0; i < limit; i++) {
            s = s * 1103515245UL + 12345UL;
            ((float*)model->npe_program->memory->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
        }
        model->npe_vm = SNEPPX_npe_vm_create(&npe_cfg);
        if (!model->npe_vm) { SNEPPX_model_destroy(model); return NULL; }
        SNEPPX_npe_vm_load(model->npe_vm, model->npe_program);
    }

    if (config->enable_fm) {
        SNEPPXFMConfig fm_cfg = config->fm_config;
        if (fm_cfg.num_nodes == 0) fm_cfg = SNEPPX_fm_config_default();
        model->fm_controller = SNEPPX_fm_controller_create(&fm_cfg);
        if (!model->fm_controller) { SNEPPX_model_destroy(model); return NULL; }
    }

    if (config->enable_attention) {
        SNEPPXAttentionConfig attn_cfg = config->attention_config;
        if (attn_cfg.d_model == 0) { attn_cfg = SNEPPX_attn_config_default(); }
        attn_cfg.d_model = in_dim;
        if (attn_cfg.head_dim == 0) attn_cfg.head_dim = in_dim / (attn_cfg.num_heads > 0 ? attn_cfg.num_heads : 1);
        model->attention = SNEPPX_attn_weights_create(attn_cfg, seed + 3);
        if (!model->attention) { SNEPPX_model_destroy(model); return NULL; }

        size_t esh[] = {vocab, in_dim};
        model->embed_weight = SNEPPX_tensor_create(esh, 2, SNEPPX_FLOAT32);
        if (model->embed_weight) {
            unsigned long s = seed + 4;
            for (size_t i = 0; i < vocab * in_dim; i++) {
                s = s * 1103515245UL + 12345UL;
                ((float*)model->embed_weight->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
            }
        }
        size_t ush[] = {in_dim, vocab};
        model->unembed_weight = SNEPPX_tensor_create(ush, 2, SNEPPX_FLOAT32);
        if (model->unembed_weight) {
            unsigned long s = seed + 5;
            for (size_t i = 0; i < in_dim * vocab; i++) {
                s = s * 1103515245UL + 12345UL;
                ((float*)model->unembed_weight->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
            }
        }
    }

    return model;
}

/**
 * @brief Destroy Model.
 */
void SNEPPX_model_destroy(SNEPPXModel* model) {
    if (!model) return;
    if (model->hss_model) SNEPPX_hss_model_destroy(model->hss_model);
    if (model->ser_model) SNEPPX_ser_model_destroy(model->ser_model);
    if (model->arc_layer) SNEPPX_arc_layer_destroy(model->arc_layer);
    if (model->npe_vm) SNEPPX_npe_vm_destroy(model->npe_vm);
    if (model->npe_program) SNEPPX_npe_program_destroy(model->npe_program);
    if (model->fm_controller) SNEPPX_fm_controller_destroy(model->fm_controller);
    if (model->attention) SNEPPX_attn_weights_destroy(model->attention);
    if (model->embed_weight) SNEPPX_tensor_destroy(model->embed_weight);
    if (model->unembed_weight) SNEPPX_tensor_destroy(model->unembed_weight);
    SNEPPX_free(model, sizeof(SNEPPXModel));
}

/**
 * @brief Run the forward pass for Model.
 *
 * @param model [out] Model value.
 * @param input [in] Input value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_forward(SNEPPXModel* model, const SNEPPXTensor* input, SNEPPXTensor** output) {
    if (!model || !input || !output) return 1;

    SNEPPXTensor* current = NULL;

    /* For attention models with embedding, do embedding lookup on token indices */
    if (model->attention && model->embed_weight && input->ndim == 2) {
        /* Input: [B, S] token IDs stored as floats */
        size_t B = input->shape[0], S = input->shape[1];
        size_t D = model->attention->config.d_model;
        (void)D;
        SNEPPXTensor* idx_t = SNEPPX_tensor_create(input->shape, input->ndim, SNEPPX_INT32);
        if (!idx_t) return 1;
        size_t n = B * S;
        int* id = (int*)idx_t->data;
        float* fd = (float*)input->data;
        for (size_t i = 0; i < n; i++) id[i] = (int)fd[i];
        current = SNEPPX_tensor_embedding(model->embed_weight, idx_t);
        SNEPPX_tensor_destroy(idx_t);
        if (current) {
            size_t sh3[] = {B, S, D};
            SNEPPXTensor* r = SNEPPX_tensor_reshape(current, sh3, 3);
            if (r) { SNEPPX_tensor_destroy(current); current = r; }
        }
    } else if (input->ndim == 3) {
        size_t b = input->shape[0], s = input->shape[1], d = input->shape[2];
        size_t sh[] = {b * s, d};
        current = SNEPPX_tensor_create(sh, 2, SNEPPX_FLOAT32);
        if (current) memcpy(current->data, input->data, b * s * d * sizeof(float));
    } else if (input->ndim == 2) {
        current = SNEPPX_tensor_create(input->shape, input->ndim, SNEPPX_FLOAT32);
        if (current) memcpy(current->data, input->data, input->size * sizeof(float));
    }
    if (!current) return 1;

    if (model->attention && model->embed_weight) {
        size_t S = current->ndim >= 3 ? current->shape[1] : current->shape[0];
        SNEPPXTensor* cos_full = SNEPPX_rope_precompute(S, model->attention->config.head_dim, model->attention->config.rope_base);
        if (cos_full) {
            SNEPPXTensor* attn_out = SNEPPX_attn_forward(model->attention, current, cos_full, cos_full);
            SNEPPX_tensor_destroy(cos_full);
            if (attn_out) {
                SNEPPX_tensor_destroy(current);
                current = attn_out;
            }
        }
    }

    if (model->hss_model) {
        SNEPPXTensor* hss_out = NULL;
        if (SNEPPX_hss_forward(model->hss_model, current, &hss_out) == 0 && hss_out) {
            SNEPPX_tensor_destroy(current);
            current = hss_out;
        }
    }

    if (model->ser_model) {
        size_t seq = current->ndim == 2 ? current->shape[0] : current->shape[1];
        size_t dim = current->shape[current->ndim - 1];
        SNEPPXTensor flat;
        size_t fsh[] = {seq, dim};
        flat.data = current->data; flat.shape = fsh; flat.ndim = 2;
        flat.size = seq * dim; flat.item_size = sizeof(float);
        flat.dtype = SNEPPX_FLOAT32; flat.strides = NULL;
        SNEPPXTensor* ser_out = NULL;
        SNEPPX_ser_forward(model->ser_model->layers[0], &flat, &ser_out);
        if (ser_out) {
            SNEPPX_tensor_destroy(current);
            current = ser_out;
        }
    }

    if (model->arc_layer) {
        float metrics[4];
        SNEPPXTensor* arc_out = NULL;
        SNEPPX_arc_forward(model->arc_layer, current, &arc_out, metrics);
        if (arc_out) {
            SNEPPX_tensor_destroy(current);
            current = arc_out;
        }
    }

    if (model->npe_vm && model->npe_program) {
        SNEPPXTensor* npe_out = NULL;
        SNEPPX_npe_vm_run(model->npe_vm, current, &npe_out);
        if (npe_out) {
            SNEPPX_tensor_destroy(current);
            current = npe_out;
        }
    }

    if (model->fm_controller) {
        SNEPPXTensor* fm_out = NULL;
        int ret = SNEPPX_fm_forward(model->fm_controller, 0, current, &fm_out);
        if (ret == 0 && fm_out) {
            SNEPPX_tensor_destroy(current);
            current = fm_out;
        }
    }

    *output = current;
    return current ? 0 : 1;
}

/**
 * @brief Perform Model Get Params.
 *
 * @param model [in] Model value.
 * @param out [out] Out value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_model_get_params(const SNEPPXModel* model, SNEPPXTensor** out, size_t max_out) {
    if (!model) return 0;
    size_t total = 0;

    if (model->attention) {
        total += 8; /* 8 weight tensors: w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o */
    }
    if (model->hss_model) {
        total += SNEPPX_hss_get_params(model->hss_model, NULL, 0);
    }
    if (model->ser_model) {
        total += SNEPPX_ser_get_params(model->ser_model, NULL, 0);
    }
    if (model->arc_layer) {
        total += SNEPPX_arc_get_params(model->arc_layer, NULL, 0);
    }
    if (model->npe_program) {
        total += SNEPPX_npe_get_params(model->npe_program, NULL, 0);
    }
    if (model->fm_controller) {
        total += SNEPPX_fm_get_params(model->fm_controller, NULL, 0);
    }
    if (model->embed_weight) total++;
    if (model->unembed_weight) total++;

    size_t idx = 0;
    if (out) {
        if (model->attention) {
            SNEPPXTensor* attn_params[8];
            int n = SNEPPX_attn_get_params(model->attention, attn_params, 8);
            for (int i = 0; i < n && idx < max_out; i++, idx++) out[idx] = attn_params[i];
        }
        if (model->hss_model) {
            size_t n = SNEPPX_hss_get_params(model->hss_model, out + idx, max_out > idx ? max_out - idx : 0);
            idx += n;
        }
        if (model->ser_model) {
            size_t n = SNEPPX_ser_get_params(model->ser_model, out + idx, max_out > idx ? max_out - idx : 0);
            idx += n;
        }
        if (model->arc_layer) {
            size_t n = SNEPPX_arc_get_params(model->arc_layer, out + idx, max_out > idx ? max_out - idx : 0);
            idx += n;
        }
        if (model->npe_program) {
            size_t n = SNEPPX_npe_get_params(model->npe_program, out + idx, max_out > idx ? max_out - idx : 0);
            idx += n;
        }
        if (model->fm_controller) {
            size_t n = SNEPPX_fm_get_params(model->fm_controller, out + idx, max_out > idx ? max_out - idx : 0);
            idx += n;
        }
        if (model->embed_weight && idx < max_out) out[idx++] = model->embed_weight;
        if (model->unembed_weight && idx < max_out) out[idx++] = model->unembed_weight;
    }
    return total;
}

/**
 * @brief Perform Model Build Train Graph.
 *
 * @param model [out] Model value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_build_train_graph(SNEPPXModel* model, SNEPPXTape* tape,
                                  SNEPPXVariable* input_var,
                                  SNEPPXVariable** weight_vars, size_t num_weights,
                                  SNEPPXVariable** output_var) {
    if (!model || !tape || !input_var || !output_var) return -1;
    (void)num_weights;
    SNEPPXVariable* current = input_var;
    size_t woff = 0;

    if (model->attention) {
        size_t S = current->data->ndim >= 2 ? current->data->shape[1] : current->data->shape[0];
        SNEPPXTensor* cos_t = SNEPPX_rope_precompute(S, model->attention->config.head_dim, model->attention->config.rope_base);
        SNEPPXTensor* sin_t = cos_t ? SNEPPX_tensor_copy(cos_t) : NULL;
        SNEPPXVariable* attn_out = NULL;
        int ret = SNEPPX_attn_build_train_graph(model->attention, tape, current,
                                               weight_vars + woff, 8, &attn_out,
                                               cos_t, sin_t);
        if (cos_t) SNEPPX_tensor_destroy(cos_t);
        if (sin_t) SNEPPX_tensor_destroy(sin_t);
        if (ret != 0 || !attn_out) return -1;
        current = attn_out;
        woff += 8;
    }

    /* Flatten 3D -> 2D for HSS/SER/ARC */
    if ((model->hss_model || model->ser_model || model->arc_layer) && current->data->ndim == 3) {
        size_t flat_sh[] = {current->data->shape[0] * current->data->shape[1], current->data->shape[2]};
        current = SNEPPX_reshape(tape, current, flat_sh, 2);
    }

    if (model->hss_model) {
        size_t nhss = SNEPPX_hss_get_params(model->hss_model, NULL, 0);
        SNEPPXVariable* hss_out = NULL;
        int ret = SNEPPX_hss_build_train_graph(model->hss_model, tape, current,
                                              weight_vars + woff, nhss, &hss_out);
        if (ret != 0 || !hss_out) return -1;
        current = hss_out;
        woff += nhss;
    }

    if (model->ser_model) {
        size_t nser = SNEPPX_ser_get_params(model->ser_model, NULL, 0);
        SNEPPXVariable* ser_out = NULL;
        int ret = SNEPPX_ser_build_train_graph(model->ser_model, tape, current,
                                              weight_vars + woff, nser, &ser_out);
        if (ret != 0 || !ser_out) return -1;
        current = ser_out;
        woff += nser;
    }

    if (model->arc_layer) {
        size_t narc = SNEPPX_arc_get_params(model->arc_layer, NULL, 0);
        SNEPPXVariable* arc_out = NULL;
        int ret = SNEPPX_arc_build_train_graph(model->arc_layer, tape, current,
                                              weight_vars + woff, narc, &arc_out);
        if (ret != 0 || !arc_out) return -1;
        current = arc_out;
        woff += narc;
    }

    if (model->npe_program) {
        if (current->data->ndim == 3) {
            size_t flat_sh[] = {current->data->shape[0] * current->data->shape[1], current->data->shape[2]};
            current = SNEPPX_reshape(tape, current, flat_sh, 2);
        }
        size_t nnpe = SNEPPX_npe_get_params(model->npe_program, NULL, 0);
        SNEPPXVariable* npe_out = NULL;
        int ret = SNEPPX_npe_build_train_graph(model->npe_program, tape, current,
                                              weight_vars + woff, nnpe, &npe_out);
        if (ret != 0 || !npe_out) return -1;
        current = npe_out;
        woff += nnpe;
    }

    if (model->fm_controller) {
        size_t nfm = SNEPPX_fm_get_params(model->fm_controller, NULL, 0);
        SNEPPXVariable* fm_out = NULL;
        int ret = SNEPPX_fm_build_train_graph(model->fm_controller, tape, current,
                                             weight_vars + woff, nfm, &fm_out);
        if (ret != 0 || !fm_out) return -1;
        current = fm_out;
        woff += nfm;
    }

    *output_var = current;
    return 0;
}
