#ifndef SNEPPX_TPU_DRIVER_H
#define SNEPPX_TPU_DRIVER_H
/*
 * SNEPPX - Tpu Driver
 *
 * WHAT
 *   Tpu Driver.
 *
 * CONCEPT
 *   Provides the Tpu Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * TPU Driver Interface — v2.0 (Google TPU / custom ASIC acceleration)
 *
 * PURPOSE: Abstract TPU and NPU devices behind the kernel's device ops.
 * TPUs use a different programming model (XLA / PjRt) than GPUs; this
 * interface wraps the PjRt client, executed programs, and device memory.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h, multidimensional_tensor_engine.h
 * VERSION: v2.0 — production-scale distributed training
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXTensor SNEPPXTensor;

/* ---------- TPU device capabilities ---------- */
typedef struct {
    char     name[256];
    size_t   device_memory_bytes;
    int      num_cores;
    int      num_chips;
    int      topology_ring_size;       /* for all-reduce ring */
    int      supports_bfloat16;
    int      supports_int8;
    int      supports_sparse_core;
} SNEPPXTPUDeviceProps;

/* ---------- PjRt-style client / executable handles (opaque) ---------- */
typedef struct SNEPPXTPUClient {
    void*    pjrt_client;             /* PJRT_Client* */
    int      device_id;
} SNEPPXTPUClient;

typedef struct SNEPPXTPUExecutable {
    void*    pjrt_executable;         /* PJRT_Executable* */
    size_t   input_count;
    size_t   output_count;
} SNEPPXTPUExecutable;

typedef struct {
    int                device_id;
    SNEPPXTPUDeviceProps props;
    SNEPPXTPUClient*     client;
    size_t             alloc_bytes;
    int                error_state;
} SNEPPXTPUContext;

/* ---------- Driver lifecycle ---------- */
/**
 * @brief Perform Tpu Register Driver.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tpu_register_driver(void);
/**
 * @brief Perform Tpu Get Device Count.
 *
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tpu_get_device_count(int* count);
/**
 * @brief Perform Tpu Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param props [out] Props value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tpu_get_device_props(int dev_id, SNEPPXTPUDeviceProps* props);
/**
 * @brief Perform Tpu Create Context.
 *
 * @param device_id [in] Device Id value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTPUContext* SNEPPX_tpu_create_context(int device_id);
/**
 * @brief Perform Tpu Destroy Context.
 *
 * @param ctx [out] Ctx value.
 */
void            SNEPPX_tpu_destroy_context(SNEPPXTPUContext* ctx);

/* ---------- Memory ---------- */
/**
 * @brief Perform Tpu Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpu_mem_alloc(void** dev_ptr, size_t bytes, SNEPPXTPUContext* ctx);
/**
 * @brief Free Tpu Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpu_mem_free(void* dev_ptr, SNEPPXTPUContext* ctx);
/**
 * @brief Perform Tpu Mem Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpu_mem_htod(void* dev_dst, const void* host_src, size_t bytes, SNEPPXTPUContext* ctx);
/**
 * @brief Perform Tpu Mem Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpu_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes, SNEPPXTPUContext* ctx);

/* ---------- Compilation and execution (XLA) ---------- */
/**
 * @brief Perform Tpu Compile.
 *
 * @param hlo_module [in] Hlo Module value.
 * @param hlo_len [in] Hlo Len value.
 * @param ctx [out] Ctx value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTPUExecutable* SNEPPX_tpu_compile(const char* hlo_module, size_t hlo_len, SNEPPXTPUContext* ctx);
/**
 * @brief Destroy Tpu Executable.
 *
 * @param exec [out] Exec value.
 */
void               SNEPPX_tpu_executable_destroy(SNEPPXTPUExecutable* exec);
/**
 * @brief Perform Tpu Execute.
 *
 * @param exec [out] Exec value.
 * @param inputs [out] Inputs value.
 * @param num_inputs [in] Num Inputs value.
 * @param outputs [out] Outputs value.
 * @param num_outputs [in] Num Outputs value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int                SNEPPX_tpu_execute(SNEPPXTPUExecutable* exec,
                                    SNEPPXTensor** inputs, size_t num_inputs,
                                    SNEPPXTensor** outputs, size_t num_outputs,
                                    SNEPPXTPUContext* ctx);

/* ---------- Collective operations (v2.0) ---------- */
/**
 * @brief Perform Tpu All Reduce.
 *
 * @param send_buf [out] Send Buf value.
 * @param recv_buf [out] Recv Buf value.
 * @param count [in] Count value.
 * @param dtype [in] Dtype value.
 * @param reduce_op [in] Reduce Op value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpu_all_reduce(void* send_buf, void* recv_buf, size_t count,
                        int dtype, int reduce_op, SNEPPXTPUContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_TPU_DRIVER_H */
