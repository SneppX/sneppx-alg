#include "graph_convolutional_network.h"
#include "multidimensional_tensor_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Gcn
 *
 * WHAT
 *   Gcn.
 *
 * CONCEPT
 *   Provides the Gcn.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


struct SNEPPXGCN {
    size_t in_features, out_features, hidden_features, num_layers;
    float dropout;
    SNEPPXTensor** W;   /* num_layers weight matrices */
};

static void gemm(const float* x, size_t m, const float* w, size_t k, size_t o, float* y) {
    for (size_t i=0;i<m;i++) for (size_t j=0;j<o;j++){float s=0;for(size_t p=0;p<k;p++)s+=x[i*k+p]*w[j*k+p];y[i*o+j]=s;}
}

/**
 * @brief Create Gcn.
 *
 * @param in_features [in] In Features value.
 * @param out_features [in] Out Features value.
 * @param hidden_features [in] Hidden Features value.
 * @param num_layers [in] Num Layers value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXGCN* SNEPPX_gcn_create(size_t in_features, size_t out_features, size_t hidden_features, int num_layers, float dropout) {
    if (num_layers==0) return NULL;
    SNEPPXGCN* m=(SNEPPXGCN*)calloc(1,sizeof(*m));
    if(!m) return NULL;
    m->in_features=in_features; m->out_features=out_features; m->hidden_features=hidden_features;
    m->num_layers=(size_t)num_layers; m->dropout=dropout;
    m->W=(SNEPPXTensor**)calloc(m->num_layers,sizeof(SNEPPXTensor*));
    for (size_t l=0;l<m->num_layers;l++){
        if (l==0) m->W[l]=SNEPPX_tensor_randn((size_t[]){hidden_features,in_features},2,SNEPPX_FLOAT32);
        else if (l+1==m->num_layers) m->W[l]=SNEPPX_tensor_randn((size_t[]){out_features,hidden_features},2,SNEPPX_FLOAT32);
        else m->W[l]=SNEPPX_tensor_randn((size_t[]){hidden_features,hidden_features},2,SNEPPX_FLOAT32);
    }
    return m;
}
/**
 * @brief Destroy Gcn.
 *
 * @param gcn [out] Gcn value.
 */
void SNEPPX_gcn_destroy(void* gcn){
    SNEPPXGCN* m=(SNEPPXGCN*)gcn; if(!m)return;
    for(size_t l=0;l<m->num_layers;l++) SNEPPX_tensor_destroy(m->W[l]);
    free(m->W); free(m);
}

/**
 * @brief Run the forward pass for Gcn.
 *
 * @param gcn [out] Gcn value.
 * @param adj [in] Adj value.
 * @param features [in] Features value.
 * @param num_nodes [in] Num Nodes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gcn_forward(void* gcn, const float* adj, const float* features, size_t num_nodes, float* output) {
    SNEPPXGCN* m=(SNEPPXGCN*)gcn;
    if(!m||!adj||!features||!output) return -1;
    size_t N=num_nodes;
    /* Build symmetric normalized adjacency A_hat = D^{-1/2}(A+I)D^{-1/2} */
    float* A=(float*)malloc(N*N*sizeof(float));
    float* deg=(float*)malloc(N*sizeof(float));
    if(!A||!deg){free(A);free(deg);return -1;}
    for(size_t i=0;i<N*N;i++) A[i]=adj[i];
    for(size_t i=0;i<N;i++) A[i*N+i]+=1.0f; /* self loops */
    for(size_t i=0;i<N;i++){float d=0;for(size_t j=0;j<N;j++)d+=A[i*N+j];deg[i]= (float)(1.0/sqrt(d>0?d:1.0));}
    for(size_t i=0;i<N;i++) for(size_t j=0;j<N;j++) A[i*N+j]*=deg[i]*deg[j];

    size_t cur_dim=m->in_features;
    float* h=(float*)malloc(N*cur_dim*sizeof(float));
    float* hn=(float*)malloc(N*(m->num_layers>1?m->hidden_features:m->out_features)*sizeof(float));
    if(!h||!hn){free(A);free(deg);free(h);free(hn);return -1;}
    memcpy(h,features,N*cur_dim*sizeof(float));
    for (size_t l=0;l<m->num_layers;l++){
        size_t out_dim = (l+1==m->num_layers)? m->out_features : m->hidden_features;
        float* agg=(float*)malloc(N*out_dim*sizeof(float));
        float* tmp=(float*)malloc(N*cur_dim*sizeof(float));
        if(!agg||!tmp){free(A);free(deg);free(h);free(hn);free(agg);free(tmp);return -1;}
        gemm(A,N,h,N,cur_dim,tmp);            /* A_hat @ h */
        gemm(tmp,N,(float*)m->W[l]->data,cur_dim,out_dim,agg); /* (A_hat h) W */
        int last = (l+1==m->num_layers);
        for(size_t i=0;i<N*out_dim;i++) agg[i]=(last)?agg[i]:(agg[i]>0?agg[i]:0.0f);
        memcpy(hn,agg,N*out_dim*sizeof(float));
        cur_dim=out_dim;
        free(h); h=hn; hn=(float*)malloc(N*(m->num_layers>1?m->hidden_features:m->out_features)*sizeof(float));
        free(agg); free(tmp);
    }
    memcpy(output,h,N*cur_dim*sizeof(float));
    free(A); free(deg); free(h);
    return 0;
}
