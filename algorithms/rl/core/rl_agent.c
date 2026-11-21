#include "reinforcement_learning.h"
#include "multidimensional_tensor_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Rl Agent
 *
 * WHAT
 *   Rl Agent.
 *
 * CONCEPT
 *   Provides the Rl Agent.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


struct SNEPPXRLAgent {
    size_t state_dim, action_dim, hidden;
    float lr, gamma;
    SNEPPXTensor* W0; SNEPPXTensor* b0;
    SNEPPXTensor* W1; SNEPPXTensor* b1;
    SNEPPXTensor* W2; SNEPPXTensor* b2;
};

static void gemm1(const float* x, size_t m, const float* w, size_t k, size_t o, float* y) {
    for (size_t i=0;i<m;i++) for (size_t j=0;j<o;j++){float s=0;for(size_t p=0;p<k;p++)s+=x[i*k+p]*w[j*k+p];y[i*o+j]=s;}
}

/**
 * @brief Create Rl.
 *
 * @param state_dim [in] State Dim value.
 * @param action_dim [in] Action Dim value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param lr [in] Lr value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXRLAgent* SNEPPX_rl_create(size_t state_dim, size_t action_dim, size_t hidden_dim, float lr, float gamma) {
    if (state_dim==0||action_dim==0||hidden_dim==0) return NULL;
    SNEPPXRLAgent* m=(SNEPPXRLAgent*)calloc(1,sizeof(*m));
    if(!m) return NULL;
    m->state_dim=state_dim; m->action_dim=action_dim; m->hidden=hidden_dim; m->lr=lr; m->gamma=gamma;
    m->W0=SNEPPX_tensor_randn((size_t[]){hidden_dim,state_dim},2,SNEPPX_FLOAT32);
    m->b0=SNEPPX_tensor_zeros((size_t[]){hidden_dim},1,SNEPPX_FLOAT32);
    m->W1=SNEPPX_tensor_randn((size_t[]){hidden_dim,hidden_dim},2,SNEPPX_FLOAT32);
    m->b1=SNEPPX_tensor_zeros((size_t[]){hidden_dim},1,SNEPPX_FLOAT32);
    m->W2=SNEPPX_tensor_randn((size_t[]){action_dim,hidden_dim},2,SNEPPX_FLOAT32);
    m->b2=SNEPPX_tensor_zeros((size_t[]){action_dim},1,SNEPPX_FLOAT32);
    return m;
}
/**
 * @brief Destroy Rl.
 *
 * @param agent [out] Agent value.
 */
void SNEPPX_rl_destroy(void* agent){
    SNEPPXRLAgent* m=(SNEPPXRLAgent*)agent; if(!m)return;
    SNEPPX_tensor_destroy(m->W0);SNEPPX_tensor_destroy(m->b0);
    SNEPPX_tensor_destroy(m->W1);SNEPPX_tensor_destroy(m->b1);
    SNEPPX_tensor_destroy(m->W2);SNEPPX_tensor_destroy(m->b2);
    free(m);
}

static void forward_q(SNEPPXRLAgent* m, const float* s, float* q, float* h0, float* h1) {
    float* a0=(float*)malloc(m->hidden*sizeof(float));
    float* a1=(float*)malloc(m->hidden*sizeof(float));
    if(!a0||!a1) return;
    gemm1(s,1,(float*)m->W0->data,m->state_dim,m->hidden,a0);
    for(size_t i=0;i<m->hidden;i++){a0[i]+=((float*)m->b0->data)[i]; h0[i]=a0[i]>0?a0[i]:0;}
    gemm1(h0,1,(float*)m->W1->data,m->hidden,m->hidden,a1);
    for(size_t i=0;i<m->hidden;i++){a1[i]+=((float*)m->b1->data)[i]; h1[i]=a1[i]>0?a1[i]:0;}
    gemm1(h1,1,(float*)m->W2->data,m->hidden,m->action_dim,q);
    for(size_t i=0;i<m->action_dim;i++) q[i]+=((float*)m->b2->data)[i];
    free(a0); free(a1);
}

/**
 * @brief Perform Rl Select Action.
 *
 * @param agent [out] Agent value.
 * @param state [in] State value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rl_select_action(void* agent, const float* state, float* action) {
    SNEPPXRLAgent* m=(SNEPPXRLAgent*)agent;
    if(!m||!state||!action) return -1;
    float* q=(float*)malloc(m->action_dim*sizeof(float));
    float* h0=(float*)malloc(m->hidden*sizeof(float));
    float* h1=(float*)malloc(m->hidden*sizeof(float));
    if(!q||!h0||!h1){free(q);free(h0);free(h1);return -1;}
    forward_q(m,state,q,h0,h1);
    int best=0; float bv=q[0];
    for(size_t a=1;a<m->action_dim;a++) if(q[a]>bv){bv=q[a];best=(int)a;}
    action[0]=(float)best;
    free(q); free(h0); free(h1);
    return 0;
}

/**
 * @brief Update Rl.
 *
 * @param agent [out] Agent value.
 * @param state [in] State value.
 * @param action [in] Action value.
 * @param reward [in] Reward value.
 * @param next_state [in] Next State value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rl_update(void* agent, const float* state, const float* action, float reward,
        const float* next_state, int done) {
    SNEPPXRLAgent* m=(SNEPPXRLAgent*)agent;
    if(!m||!state||!action||!next_state) return -1;
    float* q=(float*)malloc(m->action_dim*sizeof(float));
    float* h0=(float*)malloc(m->hidden*sizeof(float));
    float* h1=(float*)malloc(m->hidden*sizeof(float));
    if(!q||!h0||!h1){free(q);free(h0);free(h1);return -1;}
    forward_q(m,state,q,h0,h1);
    size_t a_t=(size_t)action[0];
    if (a_t>=m->action_dim) a_t=m->action_dim-1;
    float target;
    if (done) target=reward;
    else {
        float* qn=(float*)malloc(m->action_dim*sizeof(float));
        float* h0n=(float*)malloc(m->hidden*sizeof(float));
        float* h1n=(float*)malloc(m->hidden*sizeof(float));
        forward_q(m,next_state,qn,h0n,h1n);
        float mx=qn[0]; for(size_t a=1;a<m->action_dim;a++) if(qn[a]>mx)mx=qn[a];
        target=reward+m->gamma*mx;
        free(qn);free(h0n);free(h1n);
    }
    float td = q[a_t]-target;
    float* g2=(float*)calloc(m->action_dim,sizeof(float));
    g2[a_t]=td;
    float* g1=(float*)malloc(m->hidden*sizeof(float));
    float* g0=(float*)malloc(m->hidden*sizeof(float));
    for(size_t p=0;p<m->hidden;p++){float s=0;for(size_t j=0;j<m->action_dim;j++)s+=g2[j]*((float*)m->W2->data)[j*m->hidden+p]; g1[p]=s*(h1[p]>0?1.0f:0.0f);}
    for(size_t p=0;p<m->hidden;p++){float s=0;for(size_t j=0;j<m->hidden;j++)s+=g1[j]*((float*)m->W1->data)[j*m->hidden+p]; g0[p]=s*(h0[p]>0?1.0f:0.0f);}
    /* SGD updates */
    for(size_t j=0;j<m->action_dim;j++) for(size_t p=0;p<m->hidden;p++)
        ((float*)m->W2->data)[j*m->hidden+p]-=m->lr*g2[j]*h1[p];
    for(size_t j=0;j<m->action_dim;j++) ((float*)m->b2->data)[j]-=m->lr*g2[j];
    for(size_t j=0;j<m->hidden;j++) for(size_t p=0;p<m->hidden;p++)
        ((float*)m->W1->data)[j*m->hidden+p]-=m->lr*g1[j]*h0[p];
    for(size_t j=0;j<m->hidden;j++) ((float*)m->b1->data)[j]-=m->lr*g1[j];
    for(size_t j=0;j<m->hidden;j++) for(size_t p=0;p<m->state_dim;p++)
        ((float*)m->W0->data)[j*m->state_dim+p]-=m->lr*g0[j]*state[p];
    for(size_t j=0;j<m->hidden;j++) ((float*)m->b0->data)[j]-=m->lr*g0[j];
    free(q); free(h0); free(h1); free(g2); free(g1); free(g0);
    return 0;
}
