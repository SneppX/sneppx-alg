#include "differentiable_training_pipeline.h"
#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include "system_architecture_definitions.h"
#include "multi_head_attention_module.h"
#include "trainer_cuda.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
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
 * @brief Perform Train Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXTrainConfig SNEPPX_train_config_default(void) {
    SNEPPXTrainConfig cfg;
    cfg.num_epochs = 10;
    cfg.batch_size = 32;
    cfg.learning_rate = 0.01f;
    cfg.log_interval = 10;
    cfg.save_interval = 100;
    cfg.device = SNEPPX_DEVICE_CPU;
    cfg.use_cuda_optimizer = 0;
    return cfg;
}

/**
 * @brief Create Trainer.
 *
 * @param model [out] Model value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTrainer* SNEPPX_trainer_create(SNEPPXModel* model, const SNEPPXTrainConfig* config) {
    if (!model || !config) return NULL;
    SNEPPXTrainer* trainer = (SNEPPXTrainer*)SNEPPX_malloc(sizeof(SNEPPXTrainer), 64);
    if (!trainer) return NULL;
    memset(trainer, 0, sizeof(SNEPPXTrainer));

    trainer->model = model;

    SNEPPXOptimizerConfig opt_cfg = SNEPPX_optimizer_config_default();
    opt_cfg.learning_rate = config->learning_rate;
    trainer->optimizer = SNEPPX_optimizer_create(&opt_cfg);
    if (!trainer->optimizer) { SNEPPX_free(trainer, sizeof(SNEPPXTrainer)); return NULL; }

    size_t hist_shape[] = {1024};
    trainer->loss_history = SNEPPX_tensor_zeros(hist_shape, 1, SNEPPX_FLOAT32);
    if (!trainer->loss_history) {
        SNEPPX_optimizer_destroy(trainer->optimizer);
        SNEPPX_free(trainer, sizeof(SNEPPXTrainer));
        return NULL;
    }

    trainer->step_count = 0;

    if (config->use_cuda_optimizer && SNEPPX_trainer_cuda_available()) {
        size_t nw = SNEPPX_model_get_params(trainer->model, NULL, 0);
        if (nw > 0) {
            SNEPPXTensor** params = (SNEPPXTensor**)SNEPPX_malloc(nw * sizeof(SNEPPXTensor*), 64);
            if (params) {
                SNEPPX_model_get_params(trainer->model, params, nw);
                if (SNEPPX_trainer_cuda_init(params, nw) != 0) {
                    printf("Warning: CUDA optimizer init failed, falling back to CPU\n");
                }
                SNEPPX_free(params, nw * sizeof(SNEPPXTensor*));
            }
        }
    }

    return trainer;
}

/**
 * @brief Destroy Trainer.
 */
void SNEPPX_trainer_destroy(SNEPPXTrainer* trainer) {
    if (!trainer) return;
    if (trainer->optimizer) SNEPPX_optimizer_destroy(trainer->optimizer);
    if (trainer->loss_history) SNEPPX_tensor_destroy(trainer->loss_history);
    SNEPPX_trainer_cuda_shutdown();
    SNEPPX_free(trainer, sizeof(SNEPPXTrainer));
}

/**
 * @brief Perform Trainer Train Step.
 *
 * @param trainer [out] Trainer value.
 * @param batch_input [in] Batch Input value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_trainer_train_step(SNEPPXTrainer* trainer, const SNEPPXTensor* batch_input, const SNEPPXTensor* batch_target) {
    if (!trainer || !batch_input || !batch_target) return -1.0f;

    size_t nw = SNEPPX_model_get_params(trainer->model, NULL, 0);

    SNEPPXTensor** params = (SNEPPXTensor**)SNEPPX_malloc(nw * sizeof(SNEPPXTensor*), 64);
    if (!params) return -1.0f;
    SNEPPX_model_get_params(trainer->model, params, nw);

    SNEPPXTape* tape = SNEPPX_tape_create();

    SNEPPXVariable** wv = (SNEPPXVariable**)SNEPPX_malloc(nw * sizeof(SNEPPXVariable*), 64);
    if (!wv) { SNEPPX_free(params, nw * sizeof(SNEPPXTensor*)); SNEPPX_tape_destroy(tape); return -1.0f; }
    for (size_t i = 0; i < nw; i++) {
        wv[i] = SNEPPX_variable_create(params[i], 1);
    }

    /* Prepare input variable: for attention models, do embedding lookup */
    SNEPPXVariable* inv = NULL;
    if (trainer->model->attention && trainer->model->embed_weight && batch_input->ndim >= 2) {
        /* Input is token indices [B, S] — do embedding lookup */
        float* fdata = (float*)batch_input->data;
        size_t n = batch_input->shape[0] * batch_input->shape[1];
        SNEPPXTensor* idx_t = SNEPPX_tensor_create(batch_input->shape, batch_input->ndim, SNEPPX_INT32);
        if (!idx_t) return -1.0f;
        int* id = (int*)idx_t->data;
        for (size_t i = 0; i < n; i++) id[i] = (int)fdata[i];
        SNEPPXVariable* idx_v = SNEPPX_variable_create(idx_t, 0);
        SNEPPX_tape_record(tape, idx_v);
        inv = SNEPPX_embedding(tape, wv[nw - 2], idx_v);
        /* Embedding returns 2D [B*S, D]; reshape to 3D [B, S, D] for attention */
        if (inv && inv->data && inv->data->ndim == 2) {
            size_t b = batch_input->shape[0], s = batch_input->shape[1];
            size_t d = inv->data->shape[1];
            size_t sh3[] = {b, s, d};
            inv = SNEPPX_reshape(tape, inv, sh3, 3);
        }
    } else {
        /* Input is raw float features */
        size_t in_size = batch_input->size;
        SNEPPXTensor* inv_t = SNEPPX_tensor_create(batch_input->shape, batch_input->ndim, SNEPPX_FLOAT32);
        if (!inv_t) return -1.0f;
        memcpy(inv_t->data, batch_input->data, in_size * sizeof(float));
        inv = SNEPPX_variable_create(inv_t, 0);
    }

    /* Build the unified train graph */
    SNEPPXVariable* outv = NULL;
    int ret = SNEPPX_model_build_train_graph(trainer->model, tape, inv, wv, nw, &outv);

    /* For decoder models (attention), unembed to vocab logits */
    if (ret == 0 && outv && trainer->model->embed_weight && trainer->model->unembed_weight && nw >= 2) {
        if (outv->data->ndim == 3) {
            size_t outf_sh[] = {outv->data->shape[0] * outv->data->shape[1], outv->data->shape[2]};
            SNEPPXVariable* outf = SNEPPX_reshape(tape, outv, outf_sh, 2);
            outv = SNEPPX_matmul(tape, outf, wv[nw - 1]);
        } else if (outv->data->ndim == 2) {
            outv = SNEPPX_matmul(tape, outv, wv[nw - 1]);
        } else if (outv->data->ndim == 1) {
            size_t unsq_sh[] = {1, outv->data->shape[0]};
            SNEPPXVariable* unsq = SNEPPX_reshape(tape, outv, unsq_sh, 2);
            outv = SNEPPX_matmul(tape, unsq, wv[nw - 1]);
        }
    }

    float loss_val = -1.0f;
    if (ret == 0 && outv) {
        size_t out_sz = outv->data->size;
        size_t tgt_sz = batch_target->size;
        size_t copy_sz = out_sz < tgt_sz ? out_sz : tgt_sz;
        SNEPPXTensor* tgt_flat = SNEPPX_tensor_zeros(&out_sz, 1, SNEPPX_FLOAT32);
        if (tgt_flat) {
            memcpy((float*)tgt_flat->data, (float*)batch_target->data, copy_sz * sizeof(float));
        }
        SNEPPXVariable* tgv = SNEPPX_variable_create(tgt_flat, 0);
        SNEPPXVariable* loss = SNEPPX_mse_loss(tape, outv, tgv);

        if (loss) {
            loss_val = SNEPPX_variable_item(loss);
            SNEPPX_tape_backward(tape, loss);

            SNEPPXTensor** grads = (SNEPPXTensor**)SNEPPX_malloc(nw * sizeof(SNEPPXTensor*), 64);
            if (grads) {
                for (size_t i = 0; i < nw; i++) {
                    grads[i] = wv[i]->grad;
                }
                if (trainer->config.use_cuda_optimizer) {
                    SNEPPX_trainer_cuda_optimizer_step(trainer->optimizer, params, grads, nw);
                } else {
                    SNEPPX_optimizer_step(trainer->optimizer, params, grads, nw);
                }
                SNEPPX_free(grads, nw * sizeof(SNEPPXTensor*));
            }
        }

        tgv->data = NULL; SNEPPX_variable_destroy(tgv);
    }

    size_t idx = trainer->step_count;
    if (idx < trainer->loss_history->size) {
        ((float*)trainer->loss_history->data)[idx] = loss_val;
    }

    trainer->step_count++;

    for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; SNEPPX_variable_destroy(wv[i]); }
    /* inv was recorded on the tape (has gradient), so tape handles its destruction without explicit destroy */
    SNEPPX_free(wv, nw * sizeof(SNEPPXVariable*));
    SNEPPX_free(params, nw * sizeof(SNEPPXTensor*));
    SNEPPX_tape_destroy(tape);

    if (trainer->config.log_interval > 0 && trainer->step_count % trainer->config.log_interval == 0) {
        printf("[Step %zu] loss = %.6f\n", trainer->step_count, (double)loss_val);
    }

    return loss_val;
}

/**
 * @brief Perform Trainer Evaluate.
 *
 * @param trainer [out] Trainer value.
 * @param val_input [in] Val Input value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_trainer_evaluate(SNEPPXTrainer* trainer, const SNEPPXTensor* val_input, const SNEPPXTensor* val_target) {
    if (!trainer || !val_input || !val_target) return -1.0f;

    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_model_forward(trainer->model, val_input, &output);
    if (ret != 0 || !output) return -1.0f;

    float* od = (float*)output->data;
    float* td = (float*)val_target->data;
    size_t sz = output->size < val_target->size ? output->size : val_target->size;

    float sum_sq = 0.0f;
    for (size_t i = 0; i < sz; i++) {
        float d = od[i] - td[i];
        sum_sq += d * d;
    }
    float loss = sum_sq / (float)(sz > 0 ? sz : 1);

    SNEPPX_tensor_destroy(output);
    return loss;
}

/**
 * @brief Perform Trainer Save Checkpoint.
 *
 * @param trainer [out] Trainer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_save_checkpoint(SNEPPXTrainer* trainer, const char* path) {
    if (!trainer || !path) return 1;
    FILE* f = fopen(path, "wb");
    if (!f) return 1;

    unsigned int magic = 0x41524958;
    unsigned int version = 2;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&trainer->step_count, sizeof(trainer->step_count), 1, f);

    size_t nw = SNEPPX_model_get_params(trainer->model, NULL, 0);
    fwrite(&nw, sizeof(nw), 1, f);

    if (nw > 0) {
        SNEPPXTensor** params = (SNEPPXTensor**)SNEPPX_malloc(nw * sizeof(SNEPPXTensor*), 64);
        if (params) {
            SNEPPX_model_get_params(trainer->model, params, nw);
            for (size_t i = 0; i < nw; i++) {
                if (!params[i]) continue;
                unsigned int ndim = (unsigned int)params[i]->ndim;
                fwrite(&ndim, sizeof(ndim), 1, f);
                fwrite(params[i]->shape, sizeof(size_t), ndim, f);
                fwrite(&params[i]->size, sizeof(params[i]->size), 1, f);
                fwrite(params[i]->data, sizeof(float), params[i]->size, f);
            }
            SNEPPX_free(params, nw * sizeof(SNEPPXTensor*));
        }
    }

    fclose(f);
    return 0;
}

/**
 * @brief Perform Trainer Load Checkpoint.
 *
 * @param trainer [out] Trainer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_load_checkpoint(SNEPPXTrainer* trainer, const char* path) {
    if (!trainer || !path) return 1;
    FILE* f = fopen(path, "rb");
    if (!f) return 1;

    unsigned int magic, version;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x41524958) {
        fclose(f); return 1;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version < 1 || version > 2) {
        fclose(f); return 1;
    }
    if (fread(&trainer->step_count, sizeof(trainer->step_count), 1, f) != 1) {
        fclose(f); return 1;
    }

    if (version >= 2) {
        size_t nw = 0;
        if (fread(&nw, sizeof(nw), 1, f) != 1) { fclose(f); return 1; }
        if (nw > 0) {
            SNEPPXTensor** params = (SNEPPXTensor**)SNEPPX_malloc(nw * sizeof(SNEPPXTensor*), 64);
            if (params) {
                size_t actual = SNEPPX_model_get_params(trainer->model, params, nw);
                for (size_t i = 0; i < actual && i < nw; i++) {
                    if (!params[i]) continue;
                    unsigned int ndim;
                    size_t stored_shape[8];
                    size_t stored_size;
                    if (fread(&ndim, sizeof(ndim), 1, f) != 1) break;
                    if (ndim > 8) { for (size_t j = 0; j < ndim; j++) { size_t tmp; fread(&tmp, sizeof(tmp), 1, f); } break; }
                    if (fread(stored_shape, sizeof(size_t), ndim, f) != ndim) break;
                    if (fread(&stored_size, sizeof(stored_size), 1, f) != 1) break;
                    size_t copy_sz = stored_size < params[i]->size ? stored_size : params[i]->size;
                    if (fread(params[i]->data, sizeof(float), copy_sz, f) != copy_sz) break;
                    if (stored_size > copy_sz) { float tmp; for (size_t j = copy_sz; j < stored_size; j++) fread(&tmp, sizeof(float), 1, f); }
                }
                SNEPPX_free(params, nw * sizeof(SNEPPXTensor*));
            }
        }
    }

    fclose(f);
    return 0;
}
