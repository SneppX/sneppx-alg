#include "recurrent_neural_network.h"
#include "multidimensional_tensor_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Rnn
 *
 * WHAT
 *   Rnn.
 *
 * CONCEPT
 *   Provides the Rnn.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum { RNN_VANILLA, RNN_LSTM, RNN_GRU } RNNT;

struct SNEPPXRNN {
    size_t input_size, hidden_size, num_layers;
    int bidirectional;
    RNNT type;
    size_t* in_dim;
    SNEPPXTensor** Wi_f; SNEPPXTensor** Wh_f; SNEPPXTensor** bi_f; SNEPPXTensor** bh_f;
    SNEPPXTensor** Wi_b; SNEPPXTensor** Wh_b; SNEPPXTensor** bi_b; SNEPPXTensor** bh_b;
};

static void linear_fwd(const float* x, size_t m, const float* w, size_t k, size_t out, float* y) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < out; j++) {
        float s = 0; for (size_t p = 0; p < k; p++) s += x[i*k+p]*w[j*k+p]; y[i*out+j]=s;
    }
}
static float sigmoidf_x(float x) { return 1.0f/(1.0f+expf(-x)); }

/**
 * @brief Create Rnn.
 *
 * @param input_size [in] Input Size value.
 * @param hidden_size [in] Hidden Size value.
 * @param num_layers [in] Num Layers value.
 * @param bidirectional [in] Bidirectional value.
 * @param dropout [in] Dropout value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXRNN* SNEPPX_rnn_create(size_t input_size, size_t hidden_size, size_t num_layers,
        int bidirectional, float dropout, const char* rnn_type) {
    (void)dropout;
    if (num_layers == 0 || hidden_size == 0) return NULL;
    RNNT t = RNN_LSTM;
    if (rnn_type) {
        if (strstr(rnn_type,"gru")) t = RNN_GRU;
        else if (strstr(rnn_type,"vanilla")||strstr(rnn_type,"tanh")||strstr(rnn_type,"relu")) t = RNN_VANILLA;
    }
    SNEPPXRNN* m = (SNEPPXRNN*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->input_size=input_size; m->hidden_size=hidden_size; m->num_layers=num_layers; m->bidirectional=bidirectional; m->type=t;
    m->in_dim=(size_t*)calloc(num_layers,sizeof(size_t));
    m->Wi_f=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->Wh_f=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->bi_f=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->bh_f=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->Wi_b=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->Wh_b=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->bi_b=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->bh_b=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    size_t H = hidden_size;
    size_t gates = (t==RNN_VANILLA)?1:4;
    if (t==RNN_GRU) gates = 3;
    for (size_t l=0;l<num_layers;l++){
        size_t dim = (l==0)?input_size:(H*(bidirectional?2:1));
        m->in_dim[l]=dim;
        m->Wi_f[l]=SNEPPX_tensor_randn((size_t[]){gates*H,dim},2,SNEPPX_FLOAT32);
        m->Wh_f[l]=SNEPPX_tensor_randn((size_t[]){gates*H,H},2,SNEPPX_FLOAT32);
        m->bi_f[l]=SNEPPX_tensor_zeros((size_t[]){gates*H},1,SNEPPX_FLOAT32);
        m->bh_f[l]=SNEPPX_tensor_zeros((size_t[]){gates*H},1,SNEPPX_FLOAT32);
        if (bidirectional){
            m->Wi_b[l]=SNEPPX_tensor_randn((size_t[]){gates*H,dim},2,SNEPPX_FLOAT32);
            m->Wh_b[l]=SNEPPX_tensor_randn((size_t[]){gates*H,H},2,SNEPPX_FLOAT32);
            m->bi_b[l]=SNEPPX_tensor_zeros((size_t[]){gates*H},1,SNEPPX_FLOAT32);
            m->bh_b[l]=SNEPPX_tensor_zeros((size_t[]){gates*H},1,SNEPPX_FLOAT32);
        }
    }
    return m;
}

/**
 * @brief Destroy Rnn.
 */
void SNEPPX_rnn_destroy(void* rnn) {
    SNEPPXRNN* m=(SNEPPXRNN*)rnn; if(!m) return;
    for (size_t l=0;l<m->num_layers;l++){
        SNEPPX_tensor_destroy(m->Wi_f[l]); SNEPPX_tensor_destroy(m->Wh_f[l]); SNEPPX_tensor_destroy(m->bi_f[l]); SNEPPX_tensor_destroy(m->bh_f[l]);
        if (m->bidirectional){ SNEPPX_tensor_destroy(m->Wi_b[l]); SNEPPX_tensor_destroy(m->Wh_b[l]); SNEPPX_tensor_destroy(m->bi_b[l]); SNEPPX_tensor_destroy(m->bh_b[l]); }
    }
    free(m->in_dim); free(m->Wi_f); free(m->Wh_f); free(m->bi_f); free(m->bh_f);
    free(m->Wi_b); free(m->Wh_b); free(m->bi_b); free(m->bh_b);
    free(m);
}

static void rnn_dir(SNEPPXRNN* m, size_t l, const float* x, size_t seq, size_t batch,
        float* out, int reverse) {
    size_t H=m->hidden_size, dim=m->in_dim[l];
    size_t gates=(m->type==RNN_VANILLA)?1:(m->type==RNN_GRU?3:4);
    const float* Wi = (const float*)((reverse?m->Wi_b[l]:m->Wi_f[l])->data);
    const float* Wh = (const float*)((reverse?m->Wh_b[l]:m->Wh_f[l])->data);
    const float* bi = (const float*)((reverse?m->bi_b[l]:m->bi_f[l])->data);
    const float* bh = (const float*)((reverse?m->bh_b[l]:m->bh_f[l])->data);
    float* hprev=(float*)calloc(batch*H,sizeof(float));
    float* cprev=(float*)calloc(batch*H,sizeof(float));
    float* xi=(float*)malloc(gates*H*sizeof(float));
    float* xh=(float*)malloc(gates*H*sizeof(float));
    if(!hprev||!cprev||!xi||!xh){free(hprev);free(cprev);free(xi);free(xh);return;}
    for (size_t s=0;s<seq;s++){
        size_t t = reverse ? (seq-1-s) : s;
        for (size_t b=0;b<batch;b++){
            const float* xt = x + (t*batch+b)*dim;
            float* ht = hprev + b*H;
            float* ct = cprev + b*H;
            linear_fwd(xt,1,Wi,dim,gates*H,xi);
            linear_fwd(ht,1,Wh,H,gates*H,xh);
            if (m->type==RNN_VANILLA){
                for (size_t j=0;j<H;j++) ht[j]=tanhf(xi[j]+xh[j]+bi[j]+bh[j]);
            } else if (m->type==RNN_LSTM){
                for (size_t j=0;j<H;j++){
                    float ii=sigmoidf_x(xi[j]+xh[j]+bi[j]+bh[j]);
                    float ff=sigmoidf_x(xi[H+j]+xh[H+j]+bi[H+j]+bh[H+j]);
                    float gg=tanhf(xi[2*H+j]+xh[2*H+j]+bi[2*H+j]+bh[2*H+j]);
                    float oo=sigmoidf_x(xi[3*H+j]+xh[3*H+j]+bi[3*H+j]+bh[3*H+j]);
                    ct[j]=ff*ct[j]+ii*gg;
                    ht[j]=oo*tanhf(ct[j]);
                }
            } else { /* GRU */
                for (size_t j=0;j<H;j++){
                    float z=sigmoidf_x(xi[j]+xh[j]+bi[j]+bh[j]);
                    float r=sigmoidf_x(xi[H+j]+xh[H+j]+bi[H+j]+bh[H+j]);
                    float n=tanhf(xi[2*H+j]+r*xh[2*H+j]+bi[2*H+j]+bh[2*H+j]);
                    ht[j]=(1.0f-z)*n + z*ht[j];
                }
            }
            memcpy(out+(s*batch+b)*H, ht, H*sizeof(float));
        }
    }
    free(hprev); free(cprev); free(xi); free(xh);
}

/**
 * @brief Run the forward pass for Rnn.
 *
 * @param rnn [out] Rnn value.
 * @param input [in] Input value.
 * @param seq_len [in] Seq Len value.
 * @param batch_size [in] Batch Size value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rnn_forward(void* rnn, const float* input, size_t seq_len, size_t batch_size, float* output, float* hidden) {
    SNEPPXRNN* m=(SNEPPXRNN*)rnn;
    if(!m||!input||!output) return -1;
    size_t H=m->hidden_size, dim0=m->input_size;
    float* cur=(float*)malloc(seq_len*batch_size*dim0*sizeof(float));
    if(!cur) return -1;
    memcpy(cur,input,seq_len*batch_size*dim0*sizeof(float));
    float* next=NULL;
    for (size_t l=0;l<m->num_layers;l++){
        size_t out_dim = H*(m->bidirectional?2:1);
        float* fwd=(float*)malloc(seq_len*batch_size*H*sizeof(float));
        rnn_dir(m,l,cur,seq_len,batch_size,fwd,0);
        if (m->bidirectional){
            float* bwd=(float*)malloc(seq_len*batch_size*H*sizeof(float));
            rnn_dir(m,l,cur,seq_len,batch_size,bwd,1);
            float* cat=(float*)malloc(seq_len*batch_size*out_dim*sizeof(float));
            for (size_t i=0;i<seq_len*batch_size;i++){ memcpy(cat+i*out_dim, fwd+i*H, H*sizeof(float)); memcpy(cat+i*out_dim+H, bwd+i*H, H*sizeof(float)); }
            free(fwd); free(bwd); next=cat;
        } else next=fwd;
        free(cur); cur=next; dim0=out_dim;
    }
    memcpy(output,cur,seq_len*batch_size*dim0*sizeof(float));
    if (hidden) memcpy(hidden, cur+(seq_len-1)*batch_size*dim0, dim0*batch_size*sizeof(float));
    free(cur);
    return 0;
}
