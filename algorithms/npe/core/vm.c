#include "neural_programming_engine.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
/*
 * SNEPPX - NPE 16-Register Neural VM
 *
 * WHAT
 *   NPE 16-Register Neural VM.
 *
 * CONCEPT
 *   16-register neural virtual machine with 32 opcodes and dispatch loop.
 *
 * ROLE
 *   The core VM executes NPE programs: a 16-register file, 32 opcodes covering tensor ops, control flow, and neural primitives, with a dispatch loop that fetches, decodes, and executes instructions.
 *
 * REFERENCES
 *   None (internal algorithm).
 */




static SNEPPXTensor* tensor_add(const SNEPPXTensor* a, const SNEPPXTensor* b) {
    if (a->size == b->size) {
        SNEPPXTensor* c = SNEPPX_tensor_create(a->shape, a->ndim, SNEPPX_FLOAT32);
        if (!c) return NULL;
        float* ad = (float*)a->data, *bd = (float*)b->data, *cd = (float*)c->data;
        for (size_t i = 0; i < a->size; i++) cd[i] = ad[i] + bd[i];
        return c;
    }
    if (b->ndim == 1 && a->ndim == 2 && a->shape[1] == b->shape[0]) {
        SNEPPXTensor* c = SNEPPX_tensor_create(a->shape, a->ndim, SNEPPX_FLOAT32);
        if (!c) return NULL;
        float* ad = (float*)a->data, *bd = (float*)b->data, *cd = (float*)c->data;
        for (size_t i = 0; i < a->shape[0]; i++)
            for (size_t j = 0; j < a->shape[1]; j++)
                cd[i * a->shape[1] + j] = ad[i * a->shape[1] + j] + bd[j];
        return c;
    }
    return NULL;
}

static SNEPPXTensor* tensor_mul(const SNEPPXTensor* a, const SNEPPXTensor* b) {
    if (a->size == b->size) {
        SNEPPXTensor* c = SNEPPX_tensor_create(a->shape, a->ndim, SNEPPX_FLOAT32);
        if (!c) return NULL;
        float* ad = (float*)a->data, *bd = (float*)b->data, *cd = (float*)c->data;
        for (size_t i = 0; i < a->size; i++) cd[i] = ad[i] * bd[i];
        return c;
    }
    if (b->ndim == 1 && a->ndim == 2 && a->shape[1] == b->shape[0]) {
        SNEPPXTensor* c = SNEPPX_tensor_create(a->shape, a->ndim, SNEPPX_FLOAT32);
        if (!c) return NULL;
        float* ad = (float*)a->data, *bd = (float*)b->data, *cd = (float*)c->data;
        for (size_t i = 0; i < a->shape[0]; i++)
            for (size_t j = 0; j < a->shape[1]; j++)
                cd[i * a->shape[1] + j] = ad[i * a->shape[1] + j] * bd[j];
        return c;
    }
    return NULL;
}

static SNEPPXTensor* tensor_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b) {
    size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
    size_t shape_c[] = {m, n};
    SNEPPXTensor* c = SNEPPX_tensor_zeros(shape_c, 2, SNEPPX_FLOAT32);
    if (!c) return NULL;
    float* ad = (float*)a->data, *bd = (float*)b->data, *cd = (float*)c->data;
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            for (size_t l = 0; l < k; l++)
                cd[i * n + j] += ad[i * k + l] * bd[l * n + j];
    return c;
}

static SNEPPXTensor* tensor_relu(const SNEPPXTensor* a) {
    SNEPPXTensor* c = SNEPPX_tensor_create(a->shape, a->ndim, SNEPPX_FLOAT32);
    if (!c) return NULL;
    float* ad = (float*)a->data, *cd = (float*)c->data;
    for (size_t i = 0; i < a->size; i++) cd[i] = ad[i] > 0.0f ? ad[i] : 0.0f;
    return c;
}

static SNEPPXTensor* tensor_softmax(const SNEPPXTensor* a) {
    size_t last_dim = a->shape[a->ndim - 1];
    size_t outer = a->size / last_dim;
    SNEPPXTensor* c = SNEPPX_tensor_create(a->shape, a->ndim, SNEPPX_FLOAT32);
    if (!c) return NULL;
    float* ad = (float*)a->data, *cd = (float*)c->data;
    for (size_t o = 0; o < outer; o++) {
        float* row = ad + o * last_dim;
        float* crow = cd + o * last_dim;
        float maxv = row[0];
        for (size_t i = 1; i < last_dim; i++) if (row[i] > maxv) maxv = row[i];
        float sum = 0.0f;
        for (size_t i = 0; i < last_dim; i++) { crow[i] = expf(row[i] - maxv); sum += crow[i]; }
        for (size_t i = 0; i < last_dim; i++) crow[i] /= sum;
    }
    return c;
}

static SNEPPXTensor* tensor_layernorm(const SNEPPXTensor* a) {
    size_t last_dim = a->shape[a->ndim - 1];
    size_t outer = a->size / last_dim;
    SNEPPXTensor* c = SNEPPX_tensor_create(a->shape, a->ndim, SNEPPX_FLOAT32);
    if (!c) return NULL;
    float* ad = (float*)a->data, *cd = (float*)c->data;
    for (size_t o = 0; o < outer; o++) {
        float* row = ad + o * last_dim;
        float* crow = cd + o * last_dim;
        float mean = 0.0f;
        for (size_t i = 0; i < last_dim; i++) mean += row[i];
        mean /= (float)last_dim;
        float var = 0.0f;
        for (size_t i = 0; i < last_dim; i++) { float d = row[i] - mean; var += d * d; }
        var /= (float)last_dim;
        float std = sqrtf(var + 1e-5f);
        for (size_t i = 0; i < last_dim; i++) crow[i] = (row[i] - mean) / std;
    }
    return c;
}

static SNEPPXTensor* tensor_attention(const SNEPPXTensor* q, const SNEPPXTensor* k, const SNEPPXTensor* v) {
    size_t seq = q->shape[0], dim = q->shape[1];
    size_t shape_k[] = {dim, seq};
    SNEPPXTensor kt;
    kt.data = k->data; kt.shape = shape_k; kt.ndim = 2;
    kt.size = dim * seq; kt.item_size = sizeof(float);
    kt.dtype = SNEPPX_FLOAT32; kt.strides = NULL;

    SNEPPXTensor* scores = tensor_matmul(q, &kt);
    if (!scores) return NULL;
    float* sd = (float*)scores->data;
    float scale = sqrtf((float)dim);
    for (size_t i = 0; i < scores->size; i++) sd[i] /= scale;

    SNEPPXTensor* attn = tensor_softmax(scores);
    SNEPPX_tensor_destroy(scores);
    if (!attn) return NULL;

    SNEPPXTensor* result = tensor_matmul(attn, v);
    SNEPPX_tensor_destroy(attn);
    return result;
}

static SNEPPXTensor* tensor_sub(const SNEPPXTensor* a, const SNEPPXTensor* b) {
    if (!a || !b) return NULL;
    if (a->size == b->size) {
        SNEPPXTensor* c = SNEPPX_tensor_create(a->shape, a->ndim, SNEPPX_FLOAT32);
        if (!c) return NULL; float* ad=(float*)a->data,*bd=(float*)b->data,*cd=(float*)c->data;
        for (size_t i=0;i<a->size;i++) cd[i]=ad[i]-bd[i]; return c;
    }
    if (b->ndim==1&&a->ndim==2&&a->shape[1]==b->shape[0]) {
        SNEPPXTensor* c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);
        if (!c)return NULL;float*ad=(float*)a->data,*bd=(float*)b->data,*cd=(float*)c->data;
        for(size_t i=0;i<a->shape[0];i++)for(size_t j=0;j<a->shape[1];j++)
            cd[i*a->shape[1]+j]=ad[i*a->shape[1]+j]-bd[j];return c;
    } return NULL;
}

static SNEPPXTensor* tensor_div(const SNEPPXTensor* a, const SNEPPXTensor* b) {
    if (!a||!b) return NULL;
    if (a->size==b->size) {
        SNEPPXTensor* c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);
        if(!c)return NULL;float*ad=(float*)a->data,*bd=(float*)b->data,*cd=(float*)c->data;
        for(size_t i=0;i<a->size;i++)cd[i]=bd[i]!=0?ad[i]/bd[i]:0;return c;
    }
    if(b->ndim==1&&a->ndim==2&&a->shape[1]==b->shape[0]){
        SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);
        if(!c)return NULL;float*ad=(float*)a->data,*bd=(float*)b->data,*cd=(float*)c->data;
        for(size_t i=0;i<a->shape[0];i++)for(size_t j=0;j<a->shape[1];j++)
            cd[i*a->shape[1]+j]=bd[j]!=0?ad[i*a->shape[1]+j]/bd[j]:0;return c;
    }return NULL;
}

static SNEPPXTensor* tensor_exp(const SNEPPXTensor* a) {
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    float*ad=(float*)a->data,*cd=(float*)c->data;
    for(size_t i=0;i<a->size;i++)cd[i]=expf(ad[i]);return c;
}

static SNEPPXTensor* tensor_log(const SNEPPXTensor* a) {
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    float*ad=(float*)a->data,*cd=(float*)c->data;
    for(size_t i=0;i<a->size;i++)cd[i]=ad[i]>0?logf(ad[i]):-1e10f;return c;
}

static SNEPPXTensor* tensor_sqrt(const SNEPPXTensor* a) {
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    float*ad=(float*)a->data,*cd=(float*)c->data;
    for(size_t i=0;i<a->size;i++)cd[i]=ad[i]>=0?sqrtf(ad[i]):0;return c;
}

static SNEPPXTensor* tensor_pow(const SNEPPXTensor* a, float exp) {
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    float*ad=(float*)a->data,*cd=(float*)c->data;
    for(size_t i=0;i<a->size;i++)cd[i]=powf(ad[i],exp);return c;
}

static SNEPPXTensor* tensor_sin(const SNEPPXTensor* a) {
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    float*ad=(float*)a->data,*cd=(float*)c->data;
    for(size_t i=0;i<a->size;i++)cd[i]=sinf(ad[i]);return c;
}

static SNEPPXTensor* tensor_cos(const SNEPPXTensor* a) {
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    float*ad=(float*)a->data,*cd=(float*)c->data;
    for(size_t i=0;i<a->size;i++)cd[i]=cosf(ad[i]);return c;
}

static SNEPPXTensor* tensor_tanh(const SNEPPXTensor* a) {
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    float*ad=(float*)a->data,*cd=(float*)c->data;
    for(size_t i=0;i<a->size;i++)cd[i]=tanhf(ad[i]);return c;
}

static SNEPPXTensor* tensor_sigmoid(const SNEPPXTensor* a) {
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    float*ad=(float*)a->data,*cd=(float*)c->data;
    for(size_t i=0;i<a->size;i++)cd[i]=1.0f/(1.0f+expf(-ad[i]));return c;
}

static SNEPPXTensor* tensor_gelu(const SNEPPXTensor* a) {
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    float*ad=(float*)a->data,*cd=(float*)c->data;
    for(size_t i=0;i<a->size;i++){
        float x=ad[i];cd[i]=0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x)));}return c;
}

static SNEPPXTensor* tensor_silu(const SNEPPXTensor* a) {
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    float*ad=(float*)a->data,*cd=(float*)c->data;
    for(size_t i=0;i<a->size;i++){float x=ad[i];cd[i]=x/(1.0f+expf(-x));}return c;
}

static SNEPPXTensor* tensor_dropout(const SNEPPXTensor* a, float rate) {
    (void)rate;
    SNEPPXTensor*c=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!c)return NULL;
    memcpy(c->data,a->data,a->size*a->item_size);return c;
}

static SNEPPXTensor* tensor_conv2d(const SNEPPXTensor* input, const SNEPPXTensor* kernel) {
    size_t ic=input->shape[0],ih=input->shape[1],iw=input->shape[2];
    size_t oc=kernel->shape[0],kh=kernel->shape[1],kw=kernel->shape[2];
    size_t oh=ih-kh+1,ow=iw-kw+1;size_t sh[]={oc,oh,ow};
    SNEPPXTensor*c=SNEPPX_tensor_create(sh,3,SNEPPX_FLOAT32);if(!c)return NULL;
    float*id=(float*)input->data,*kd=(float*)kernel->data,*cd=(float*)c->data;
    memset(cd,0,c->size*sizeof(float));
    for(size_t o=0;o<oc;o++)for(size_t i=0;i<ic;i++)for(size_t h=0;h<oh;h++)
        for(size_t w=0;w<ow;w++){float s=0;for(size_t khh=0;khh<kh;khh++)
            for(size_t kww=0;kww<kw;kww++)
                s+=id[i*ih*iw+(h+khh)*iw+(w+kww)]*kd[o*ic*kh*kw+i*kh*kw+khh*kw+kww];
            cd[o*oh*ow+h*ow+w]+=s;}return c;
}

static SNEPPXTensor* tensor_pool2d(const SNEPPXTensor* a, size_t pool_h, size_t pool_w) {
    size_t c=a->shape[0],h=a->shape[1],w=a->shape[2];
    size_t oh=h/pool_h,ow=w/pool_w;size_t sh[]={c,oh,ow};
    SNEPPXTensor* result=SNEPPX_tensor_create(sh,3,SNEPPX_FLOAT32);if(!result)return NULL;
    float*ad=(float*)a->data,*rd=(float*)result->data;
    for(size_t ch=0;ch<c;ch++)for(size_t ph=0;ph<oh;ph++)
        for(size_t pw=0;pw<ow;pw++){float mx=-1e10f;
            for(size_t kh=0;kh<pool_h;kh++)for(size_t kw=0;kw<pool_w;kw++){
                float v=ad[ch*h*w+(ph*pool_h+kh)*w+(pw*pool_w+kw)];
                if(v>mx)mx=v;}rd[ch*oh*ow+ph*ow+pw]=mx;}return result;
}

static SNEPPXTensor* tensor_batchnorm(const SNEPPXTensor* a, const SNEPPXTensor* gamma, const SNEPPXTensor* beta) {
    size_t c=a->shape[1];
    SNEPPXTensor* r=SNEPPX_tensor_create(a->shape,a->ndim,SNEPPX_FLOAT32);if(!r)return NULL;
    float*ad=(float*)a->data,*rd=(float*)r->data;
    float*gd=(float*)gamma->data,*bd=(float*)beta->data;
    for(size_t ch=0;ch<c;ch++){float sum=0,sq=0;size_t n=a->size/c;
        for(size_t i=0;i<n;i++){float v=ad[i*c+ch];sum+=v;sq+=v*v;}
        float mean=sum/(float)n,var=sq/(float)n-mean*mean;
        float std=sqrtf(var+1e-5f);
        for(size_t i=0;i<n;i++)rd[i*c+ch]=gd[ch]*(ad[i*c+ch]-mean)/std+bd[ch];}return r;
}

static SNEPPXTensor* tensor_embedding(const SNEPPXTensor* indices, const SNEPPXTensor* weights) {
    size_t n=indices->size,d=weights->shape[1];size_t sh[]={n,d};
    SNEPPXTensor*r=SNEPPX_tensor_create(sh,2,SNEPPX_FLOAT32);if(!r)return NULL;
    float*id=(float*)indices->data,*wd=(float*)weights->data,*rd=(float*)r->data;
    for(size_t i=0;i<n;i++){size_t idx=(size_t)id[i];
        memcpy(rd+i*d,wd+idx*d,d*sizeof(float));}return r;
}

static SNEPPXTensor* tensor_crossentropy(const SNEPPXTensor* pred, const SNEPPXTensor* target) {
    size_t n=pred->shape[0],c=pred->shape[1];
    SNEPPXTensor* loss=SNEPPX_tensor_zeros(NULL,0,SNEPPX_FLOAT32);if(!loss)return NULL;
    float*pd=(float*)pred->data,*td=(float*)target->data;
    float*ld=(float*)loss->data;*ld=0;
    for(size_t i=0;i<n;i++){size_t t=(size_t)td[i];
        float p=pd[i*c+t];if(p<1e-7f)p=1e-7f;*ld-=logf(p);}*ld/=(float)n;return loss;
}

static SNEPPXTensor* tensor_mse(const SNEPPXTensor* pred, const SNEPPXTensor* target) {
    if(!pred||!target)return NULL;
    SNEPPXTensor*l=SNEPPX_tensor_zeros(NULL,0,SNEPPX_FLOAT32);if(!l)return NULL;
    float*pd=(float*)pred->data,*td=(float*)target->data,*ld=(float*)l->data;*ld=0;
    for(size_t i=0;i<pred->size;i++){float d=pd[i]-td[i];*ld+=d*d;}*ld/=(float)pred->size;return l;
}

static SNEPPXTensor* tensor_concat(const SNEPPXTensor* a, const SNEPPXTensor* b, int axis) {
    (void)axis;
    size_t n=a->size+b->size;
    size_t sh[]={a->shape[0]+b->shape[0]};
    if(a->ndim>1)for(size_t i=1;i<a->ndim;i++)sh[i]=a->shape[i];
    SNEPPXTensor*r=SNEPPX_tensor_create(sh,a->ndim,SNEPPX_FLOAT32);if(!r)return NULL;
    memcpy(r->data,a->data,a->size*a->item_size);
    memcpy((float*)r->data+a->size,b->data,b->size*b->item_size);return r;
}

static SNEPPXTensor* tensor_split_half(const SNEPPXTensor* a) {
    size_t h=a->shape[0]/2;size_t sh[]={h};if(a->ndim>1)for(size_t i=1;i<a->ndim;i++)sh[i]=a->shape[i];
    SNEPPXTensor*r=SNEPPX_tensor_create(sh,a->ndim,SNEPPX_FLOAT32);if(!r)return NULL;
    memcpy(r->data,a->data,h*a->item_size);return r;
}

/**
 * @brief Create Npe Vm.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEVM* SNEPPX_npe_vm_create(const SNEPPXNPEConfig* config) {
    if (!config) return NULL;
    SNEPPXNPEVM* vm = (SNEPPXNPEVM*)SNEPPX_malloc(sizeof(SNEPPXNPEVM), 64);
    if (!vm) return NULL;
    vm->step_limit = config->step_limit;
    vm->max_trace = config->max_program_length * 2;
    vm->execution_trace = (SNEPPXNPEInstruction*)SNEPPX_malloc(vm->max_trace * sizeof(SNEPPXNPEInstruction), 64);
    if (!vm->execution_trace) { SNEPPX_free(vm, sizeof(SNEPPXNPEVM)); return NULL; }
    memset(vm->execution_trace, 0, vm->max_trace * sizeof(SNEPPXNPEInstruction));
    vm->program = NULL;
    vm->trace_length = 0;
    vm->jit_profile = NULL;
    if (config->jit_enabled) {
        vm->jit_profile = SNEPPX_npe_jit_profile_create(config->jit_hot_threshold);
    }
    return vm;
}

/**
 * @brief Destroy Npe Vm.
 */
void SNEPPX_npe_vm_destroy(SNEPPXNPEVM* vm) {
    if (!vm) return;
    SNEPPX_free(vm->execution_trace, vm->max_trace * sizeof(SNEPPXNPEInstruction));
    if (vm->jit_profile) SNEPPX_npe_jit_profile_destroy(vm->jit_profile);
    SNEPPX_free(vm, sizeof(SNEPPXNPEVM));
}

/**
 * @brief Load Npe Vm.
 *
 * @param vm [out] Vm value.
 */
void SNEPPX_npe_vm_load(SNEPPXNPEVM* vm, SNEPPXNPEProgram* prog) {
    vm->program = prog;
    vm->trace_length = 0;
    if (prog) prog->pc = 0;
}

/**
 * @brief Perform Npe Vm Step.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npe_vm_step(SNEPPXNPEVM* vm) {
    if (!vm->program) return 1;
    SNEPPXNPEProgram* prog = vm->program;
    if (prog->pc >= prog->num_instructions) return 1;

    SNEPPXNPEInstruction inst = prog->instructions[prog->pc];
    if (vm->trace_length < vm->max_trace) {
        vm->execution_trace[vm->trace_length++] = inst;
    }

    if (vm->jit_profile) {
        SNEPPX_npe_jit_record(vm->jit_profile, inst.opcode, 0.0f);
    }

    int d = inst.dest_reg;
    int sa = inst.src_reg_a;
    int sb = inst.src_reg_b;
    int imm = inst.immediate;
    int ret = 0;

    switch (inst.opcode) {
        case SNEPPX_NOP:
            prog->pc++;
            break;

        case SNEPPX_LOAD: {
            if (d < 0 || d >= 16) { ret = 1; break; }
            if (prog->registers[d]) SNEPPX_tensor_destroy(prog->registers[d]);
            size_t size_k = 1;
            if (inst.shape_a[0] > 0) size_k = (size_t)inst.shape_a[0];
            if (inst.shape_a[1] > 0) size_k *= (size_t)inst.shape_a[1];
            size_t addr = (size_t)imm;
            if (addr + size_k > prog->memory->size) { ret = 1; break; }
            if (inst.shape_a[1] > 0) {
                size_t sh[] = {(size_t)inst.shape_a[0], (size_t)inst.shape_a[1]};
                prog->registers[d] = SNEPPX_tensor_create(sh, 2, SNEPPX_FLOAT32);
            } else {
                size_t sh[] = {size_k};
                prog->registers[d] = SNEPPX_tensor_create(sh, 1, SNEPPX_FLOAT32);
            }
            if (prog->registers[d]) {
                memcpy((float*)prog->registers[d]->data,
                       (float*)prog->memory->data + addr,
                       size_k * sizeof(float));
            }
            prog->pc++;
            break;
        }

        case SNEPPX_STORE: {
            if (sa < 0 || sa >= 16 || !prog->registers[sa]) { ret = 1; break; }
            size_t addr = (size_t)imm;
            size_t sz = prog->registers[sa]->size;
            if (addr + sz > prog->memory->size) { ret = 1; break; }
            memcpy((float*)prog->memory->data + addr,
                   (float*)prog->registers[sa]->data,
                   sz * sizeof(float));
            prog->pc++;
            break;
        }

        case SNEPPX_ADD:
            if (d >= 0 && d < 16 && sa >= 0 && sa < 16 && sb >= 0 && sb < 16 &&
                prog->registers[sa] && prog->registers[sb]) {
                if (prog->registers[d]) SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d] = tensor_add(prog->registers[sa], prog->registers[sb]);
            }
            prog->pc++;
            break;

        case SNEPPX_MUL:
            if (d >= 0 && d < 16 && sa >= 0 && sa < 16 && sb >= 0 && sb < 16 &&
                prog->registers[sa] && prog->registers[sb]) {
                if (prog->registers[d]) SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d] = tensor_mul(prog->registers[sa], prog->registers[sb]);
            }
            prog->pc++;
            break;

        case SNEPPX_MATMUL: {
            SNEPPXTensor* mm = NULL;
            if (d >= 0 && d < 16 && sa >= 0 && sa < 16 && sb >= 0 && sb < 16 &&
                prog->registers[sa] && prog->registers[sb]) {
                mm = tensor_matmul(prog->registers[sa], prog->registers[sb]);
                if ((inst.immediate & 0x20000000) && mm) {
                    int bias_reg = inst.immediate & 0xFF;
                    if (bias_reg >= 0 && bias_reg < 16 && prog->registers[bias_reg]) {
                        SNEPPXTensor* biased = tensor_add(mm, prog->registers[bias_reg]);
                        SNEPPX_tensor_destroy(mm); mm = biased;
                    }
                    if (mm) {
                        SNEPPXTensor* activ = tensor_relu(mm);
                        SNEPPX_tensor_destroy(mm); mm = activ;
                    }
                } else if ((inst.immediate & 0x80000000) && mm) {
                    SNEPPXTensor* activ = tensor_relu(mm);
                    SNEPPX_tensor_destroy(mm); mm = activ;
                } else if ((inst.immediate & 0x40000000) && mm) {
                    int bias_reg = inst.immediate & 0xFF;
                    if (bias_reg >= 0 && bias_reg < 16 && prog->registers[bias_reg]) {
                        SNEPPXTensor* biased = tensor_add(mm, prog->registers[bias_reg]);
                        SNEPPX_tensor_destroy(mm); mm = biased;
                    }
                }
                if (prog->registers[d]) SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d] = mm;
            }
            prog->pc++;
            break;
        }

        case SNEPPX_RELU:
            if (d >= 0 && d < 16 && sa >= 0 && sa < 16 && prog->registers[sa]) {
                if (prog->registers[d]) SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d] = tensor_relu(prog->registers[sa]);
            }
            prog->pc++;
            break;

        case SNEPPX_SOFTMAX:
            if (d >= 0 && d < 16 && sa >= 0 && sa < 16 && prog->registers[sa]) {
                if (prog->registers[d]) SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d] = tensor_softmax(prog->registers[sa]);
            }
            prog->pc++;
            break;

        case SNEPPX_LAYERNORM:
            if (d >= 0 && d < 16 && sa >= 0 && sa < 16 && prog->registers[sa]) {
                if (prog->registers[d]) SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d] = tensor_layernorm(prog->registers[sa]);
            }
            prog->pc++;
            break;

        case SNEPPX_ATTENTION:
            if (d >= 0 && d < 16 && sa >= 0 && sa < 16 && sb >= 0 && sb < 16 &&
                prog->registers[sa] && prog->registers[sb]) {
                if (prog->registers[d]) SNEPPX_tensor_destroy(prog->registers[d]);
                SNEPPXTensor* k = inst.immediate >= 0 && inst.immediate < 16 ? prog->registers[inst.immediate] : NULL;
                if (k) {
                    prog->registers[d] = tensor_attention(prog->registers[sa], prog->registers[sb], k);
                }
            }
            prog->pc++;
            break;

        case SNEPPX_BRANCH:
            if (sa >= 0 && sa < 16 && prog->registers[sa]) {
                float* sd = (float*)prog->registers[sa]->data;
                if (sd[0] > 0.0f) {
                    prog->pc = (size_t)imm;
                } else {
                    prog->pc++;
                }
            } else {
                prog->pc++;
            }
            break;

        case SNEPPX_SUB:
            if (d>=0&&d<16&&sa>=0&&sa<16&&sb>=0&&sb<16&&prog->registers[sa]&&prog->registers[sb]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_sub(prog->registers[sa],prog->registers[sb]);}
            prog->pc++;break;

        case SNEPPX_DIV:
            if (d>=0&&d<16&&sa>=0&&sa<16&&sb>=0&&sb<16&&prog->registers[sa]&&prog->registers[sb]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_div(prog->registers[sa],prog->registers[sb]);}
            prog->pc++;break;

        case SNEPPX_EXP:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_exp(prog->registers[sa]);}
            prog->pc++;break;

        case SNEPPX_LOG:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_log(prog->registers[sa]);}
            prog->pc++;break;

        case SNEPPX_SQRT:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_sqrt(prog->registers[sa]);}
            prog->pc++;break;

        case SNEPPX_POW:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                float exp_val=*(float*)&imm;
                prog->registers[d]=tensor_pow(prog->registers[sa],exp_val);}
            prog->pc++;break;

        case SNEPPX_SIN:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_sin(prog->registers[sa]);}
            prog->pc++;break;

        case SNEPPX_COS:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_cos(prog->registers[sa]);}
            prog->pc++;break;

        case SNEPPX_TANH:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_tanh(prog->registers[sa]);}
            prog->pc++;break;

        case SNEPPX_SIGMOID:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_sigmoid(prog->registers[sa]);}
            prog->pc++;break;

        case SNEPPX_GELU:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_gelu(prog->registers[sa]);}
            prog->pc++;break;

        case SNEPPX_SILU:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_silu(prog->registers[sa]);}
            prog->pc++;break;

        case SNEPPX_DROPOUT:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                float rate=*(float*)&imm;
                prog->registers[d]=tensor_dropout(prog->registers[sa],rate);}
            prog->pc++;break;

        case SNEPPX_CONV2D:
            if(d>=0&&d<16&&sa>=0&&sa<16&&sb>=0&&sb<16&&prog->registers[sa]&&prog->registers[sb]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_conv2d(prog->registers[sa],prog->registers[sb]);}
            prog->pc++;break;

        case SNEPPX_POOL2D:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                size_t ph=((size_t)imm)>>16&0xFFFF,pw=(size_t)imm&0xFFFF;
                if(ph==0)ph=2;if(pw==0)pw=2;
                prog->registers[d]=tensor_pool2d(prog->registers[sa],ph,pw);}
            prog->pc++;break;

        case SNEPPX_BATCHNORM:
            if(d>=0&&d<16&&sa>=0&&sa<16&&sb>=0&&sb<16&&prog->registers[sa]&&prog->registers[sb]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                SNEPPXTensor*beta=inst.immediate>=0&&inst.immediate<16?prog->registers[inst.immediate]:NULL;
                if(beta)prog->registers[d]=tensor_batchnorm(prog->registers[sa],prog->registers[sb],beta);}
            prog->pc++;break;

        case SNEPPX_EMBEDDING:
            if(d>=0&&d<16&&sa>=0&&sa<16&&sb>=0&&sb<16&&prog->registers[sa]&&prog->registers[sb]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_embedding(prog->registers[sa],prog->registers[sb]);}
            prog->pc++;break;

        case SNEPPX_CROSSENTROPY:
            if(d>=0&&d<16&&sa>=0&&sa<16&&sb>=0&&sb<16&&prog->registers[sa]&&prog->registers[sb]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_crossentropy(prog->registers[sa],prog->registers[sb]);}
            prog->pc++;break;

        case SNEPPX_MSE:
            if(d>=0&&d<16&&sa>=0&&sa<16&&sb>=0&&sb<16&&prog->registers[sa]&&prog->registers[sb]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_mse(prog->registers[sa],prog->registers[sb]);}
            prog->pc++;break;

        case SNEPPX_CONCAT:
            if(d>=0&&d<16&&sa>=0&&sa<16&&sb>=0&&sb<16&&prog->registers[sa]&&prog->registers[sb]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                int axis=imm;
                prog->registers[d]=tensor_concat(prog->registers[sa],prog->registers[sb],axis);}
            prog->pc++;break;

        case SNEPPX_SPLIT:
            if(d>=0&&d<16&&sa>=0&&sa<16&&prog->registers[sa]){
                if(prog->registers[d])SNEPPX_tensor_destroy(prog->registers[d]);
                prog->registers[d]=tensor_split_half(prog->registers[sa]);}
            prog->pc++;break;

        case SNEPPX_HALT:
            return 1;

        default:
            prog->pc++;
            break;
    }

    if (prog->pc >= prog->num_instructions) ret = 1;
    return ret;
}

/**
 * @brief Run Npe Vm.
 *
 * @param vm [out] Vm value.
 * @param input [out] Input value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npe_vm_run(SNEPPXNPEVM* vm, SNEPPXTensor* input, SNEPPXTensor** output) {
    if (!vm->program) return 1;
    SNEPPXNPEProgram* prog = vm->program;
    prog->pc = 0;
    vm->trace_length = 0;

    if (prog->registers[0]) SNEPPX_tensor_destroy(prog->registers[0]);
    prog->registers[0] = SNEPPX_tensor_create(input->shape, input->ndim, SNEPPX_FLOAT32);
    if (prog->registers[0]) {
        memcpy((float*)prog->registers[0]->data, (float*)input->data, input->size * sizeof(float));
    }

    size_t steps = 0;
    while (steps < vm->step_limit) {
        int r = SNEPPX_npe_vm_step(vm);
        if (r != 0) break;
        steps++;
    }

    int last_reg = -1;
    for (int i = 15; i >= 0; i--) {
        if (prog->registers[i]) { last_reg = i; break; }
    }
    if (last_reg < 0) { *output = NULL; return 1; }

    *output = SNEPPX_tensor_create(prog->registers[last_reg]->shape,
                                 prog->registers[last_reg]->ndim, SNEPPX_FLOAT32);
    if (*output) {
        memcpy((float*)(*output)->data, (float*)prog->registers[last_reg]->data,
               prog->registers[last_reg]->size * sizeof(float));
    }

    if (vm->jit_profile && vm->jit_profile->total_instructions >= vm->jit_profile->hot_threshold) {
        SNEPPX_npe_vm_optimize(vm);
    }

    return steps >= vm->step_limit ? 2 : 0;
}

/**
 * @brief Perform Npe Vm Optimize.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npe_vm_optimize(SNEPPXNPEVM* vm) {
    if (!vm || !vm->program) return -1;
    if (!vm->jit_profile) return -1;

    SNEPPXNPEProgram* optimized = SNEPPX_npe_jit_optimize(vm->jit_profile, vm->program, vm->program->memory);
    if (!optimized) return -1;

    SNEPPX_npe_program_destroy(vm->program);
    vm->program = optimized;
    vm->program->pc = 0;
    vm->trace_length = 0;

    vm->jit_profile->is_profiling = 0;
    return 0;
}
