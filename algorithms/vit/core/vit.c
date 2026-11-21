#include "vision_transformer.h"
#include "multidimensional_tensor_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

/*
 * SNEPPX - Vit
 *
 * WHAT
 *   Vit.
 *
 * CONCEPT
 *   Provides the Vit.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


struct SNEPPXVisionTransformer {
    size_t img_size, patch_size, in_channels, num_classes, hidden, heads, layers;
    float dropout;
    size_t n_patches;
    SNEPPXTensor* Wproj;     /* [hidden, in_channels*patch*patch] */
    SNEPPXTensor* cls_token; /* [1, hidden] */
    SNEPPXTensor* pos_embed; /* [n_patches+1, hidden] */
    SNEPPXTensor** Wq; SNEPPXTensor** Wk; SNEPPXTensor** Wv; SNEPPXTensor** Wo;
    SNEPPXTensor** W1; SNEPPXTensor** W2;
    SNEPPXTensor** ln1_g; SNEPPXTensor** ln1_b; SNEPPXTensor** ln2_g; SNEPPXTensor** ln2_b;
    SNEPPXTensor* ln_f_g; SNEPPXTensor* ln_f_b;
    SNEPPXTensor* Wcls;      /* [num_classes, hidden] */
};

static void linear_fwd(const float* x, size_t m, const float* w, size_t k, size_t out, float* y) {
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < out; j++) {
        float s = 0; for (size_t p = 0; p < k; p++) s += x[i*k+p]*w[j*k+p]; y[i*out+j]=s;
    }
}
static void layernorm_fwd(const float* x, const float* g, const float* b, size_t m, size_t h, float eps, float* y) {
    for (size_t i = 0; i < m; i++) {
        const float* r = x + i*h; float mean=0; for (size_t p=0;p<h;p++) mean+=r[p]; mean/=(float)h;
        float var=0; for (size_t p=0;p<h;p++){float d=r[p]-mean;var+=d*d;} var/=(float)h;
        float inv=1.0f/sqrtf(var+eps);
        for (size_t p=0;p<h;p++){float v=(r[p]-mean)*inv; if(g)v*=g[p]; if(b)v+=b[p]; y[i*h+p]=v;}
    }
}

/**
 * @brief Create Vit.
 *
 * @param img_size [in] Img Size value.
 * @param patch_size [in] Patch Size value.
 * @param in_channels [in] In Channels value.
 * @param num_classes [in] Num Classes value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param num_heads [in] Num Heads value.
 * @param num_layers [in] Num Layers value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVisionTransformer* SNEPPX_vit_create(size_t img_size, size_t patch_size, size_t in_channels,
        size_t num_classes, size_t hidden_dim, size_t num_heads, size_t num_layers, float dropout) {
    if (img_size % patch_size != 0 || hidden_dim % num_heads != 0 || num_layers == 0) return NULL;
    SNEPPXVisionTransformer* m = (SNEPPXVisionTransformer*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->img_size=img_size; m->patch_size=patch_size; m->in_channels=in_channels; m->num_classes=num_classes;
    m->hidden=hidden_dim; m->heads=num_heads; m->layers=num_layers; m->dropout=dropout;
    size_t P = patch_size, Cin = in_channels;
    size_t patch_dim = Cin*P*P;
    m->n_patches = (img_size/patch_size)*(img_size/patch_size);
    m->Wproj = SNEPPX_tensor_randn((size_t[]){hidden_dim, patch_dim}, 2, SNEPPX_FLOAT32);
    m->cls_token = SNEPPX_tensor_randn((size_t[]){1, hidden_dim}, 2, SNEPPX_FLOAT32);
    m->pos_embed = SNEPPX_tensor_randn((size_t[]){m->n_patches+1, hidden_dim}, 2, SNEPPX_FLOAT32);
    m->Wq=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->Wk=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->Wv=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->Wo=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->W1=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->W2=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->ln1_g=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->ln1_b=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->ln2_g=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    m->ln2_b=(SNEPPXTensor**)calloc(num_layers,sizeof(SNEPPXTensor*));
    size_t shp[]={hidden_dim};
    for (size_t l=0;l<num_layers;l++){
        m->Wq[l]=SNEPPX_tensor_randn((size_t[]){hidden_dim,hidden_dim},2,SNEPPX_FLOAT32);
        m->Wk[l]=SNEPPX_tensor_randn((size_t[]){hidden_dim,hidden_dim},2,SNEPPX_FLOAT32);
        m->Wv[l]=SNEPPX_tensor_randn((size_t[]){hidden_dim,hidden_dim},2,SNEPPX_FLOAT32);
        m->Wo[l]=SNEPPX_tensor_randn((size_t[]){hidden_dim,hidden_dim},2,SNEPPX_FLOAT32);
        m->W1[l]=SNEPPX_tensor_randn((size_t[]){hidden_dim*4,hidden_dim},2,SNEPPX_FLOAT32);
        m->W2[l]=SNEPPX_tensor_randn((size_t[]){hidden_dim,hidden_dim*4},2,SNEPPX_FLOAT32);
        m->ln1_g[l]=SNEPPX_tensor_ones(shp,1,SNEPPX_FLOAT32);
        m->ln1_b[l]=SNEPPX_tensor_zeros(shp,1,SNEPPX_FLOAT32);
        m->ln2_g[l]=SNEPPX_tensor_ones(shp,1,SNEPPX_FLOAT32);
        m->ln2_b[l]=SNEPPX_tensor_zeros(shp,1,SNEPPX_FLOAT32);
    }
    m->ln_f_g=SNEPPX_tensor_ones(shp,1,SNEPPX_FLOAT32);
    m->ln_f_b=SNEPPX_tensor_zeros(shp,1,SNEPPX_FLOAT32);
    m->Wcls=SNEPPX_tensor_randn((size_t[]){num_classes,hidden_dim},2,SNEPPX_FLOAT32);
    if (!m->Wproj || !m->cls_token || !m->pos_embed || !m->Wcls) { SNEPPX_vit_destroy(m); return NULL; }
    return m;
}

/**
 * @brief Destroy Vit.
 */
void SNEPPX_vit_destroy(void* vit) {
    SNEPPXVisionTransformer* m=(SNEPPXVisionTransformer*)vit;
    if(!m) return;
    SNEPPX_tensor_destroy(m->Wproj); SNEPPX_tensor_destroy(m->cls_token); SNEPPX_tensor_destroy(m->pos_embed);
    SNEPPX_tensor_destroy(m->ln_f_g); SNEPPX_tensor_destroy(m->ln_f_b); SNEPPX_tensor_destroy(m->Wcls);
    for (size_t l=0;l<m->layers;l++){
        SNEPPX_tensor_destroy(m->Wq[l]); SNEPPX_tensor_destroy(m->Wk[l]); SNEPPX_tensor_destroy(m->Wv[l]);
        SNEPPX_tensor_destroy(m->Wo[l]); SNEPPX_tensor_destroy(m->W1[l]); SNEPPX_tensor_destroy(m->W2[l]);
        SNEPPX_tensor_destroy(m->ln1_g[l]); SNEPPX_tensor_destroy(m->ln1_b[l]);
        SNEPPX_tensor_destroy(m->ln2_g[l]); SNEPPX_tensor_destroy(m->ln2_b[l]);
    }
    free(m->Wq); free(m->Wk); free(m->Wv); free(m->Wo); free(m->W1); free(m->W2);
    free(m->ln1_g); free(m->ln1_b); free(m->ln2_g); free(m->ln2_b);
    free(m);
}

static void vit_encoder_block(SNEPPXVisionTransformer* m, size_t l, float* x, size_t seq) {
    size_t H = m->hidden, Nh = m->heads, hd = H/Nh;
    float* xn = (float*)malloc(seq*H*sizeof(float));
    float* q = (float*)malloc(seq*H*sizeof(float));
    float* k = (float*)malloc(seq*H*sizeof(float));
    float* v = (float*)malloc(seq*H*sizeof(float));
    float* attn = (float*)malloc(seq*H*sizeof(float));
    float* proj = (float*)malloc(seq*H*sizeof(float));
    float* ffn = (float*)malloc(seq*H*4*sizeof(float));
    float* ffn2 = (float*)malloc(seq*H*sizeof(float));
    if(!xn||!q||!k||!v||!attn||!proj||!ffn||!ffn2){goto done;}
    layernorm_fwd(x,(float*)m->ln1_g[l]->data,(float*)m->ln1_b[l]->data,seq,H,1e-5f,xn);
    linear_fwd(xn,seq,(float*)m->Wq[l]->data,H,H,q);
    linear_fwd(xn,seq,(float*)m->Wk[l]->data,H,H,k);
    linear_fwd(xn,seq,(float*)m->Wv[l]->data,H,H,v);
    memset(attn,0,seq*H*sizeof(float));
    for (size_t hh=0;hh<Nh;hh++) for (size_t i=0;i<seq;i++){
        float scores[4096]; float mx=-1e30f;
        for (size_t j=0;j<seq;j++){float s=0;for(size_t p=0;p<hd;p++)s+=q[(i*Nh+hh)*hd+p]*k[(j*Nh+hh)*hd+p]; s/=sqrtf((float)hd); scores[j]=s; if(s>mx)mx=s;}
        float sum=0; for(size_t j=0;j<seq;j++){scores[j]=expf(scores[j]-mx);sum+=scores[j];}
        float inv=sum>0?1.0f/sum:0; for(size_t j=0;j<seq;j++){float a=scores[j]*inv;for(size_t p=0;p<hd;p++)attn[(i*Nh+hh)*hd+p]+=a*v[(j*Nh+hh)*hd+p];}
    }
    linear_fwd(attn,seq,(float*)m->Wo[l]->data,H,H,proj);
    for(size_t i=0;i<seq*H;i++) x[i]+=proj[i];
    layernorm_fwd(x,(float*)m->ln2_g[l]->data,(float*)m->ln2_b[l]->data,seq,H,1e-5f,xn);
    linear_fwd(xn,seq,(float*)m->W1[l]->data,H,H*4,ffn);
    for(size_t i=0;i<seq*H*4;i++) ffn[i]=ffn[i]>0?ffn[i]:0;
    linear_fwd(ffn,seq,(float*)m->W2[l]->data,H*4,H,ffn2);
    for(size_t i=0;i<seq*H;i++) x[i]+=ffn2[i];
done:
    free(xn);free(q);free(k);free(v);free(attn);free(proj);free(ffn);free(ffn2);
}

/**
 * @brief Run the forward pass for Vit.
 *
 * @param vit [out] Vit value.
 * @param images [in] Images value.
 * @param batch_size [in] Batch Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vit_forward(void* vit, const float* images, size_t batch_size, float* logits) {
    SNEPPXVisionTransformer* m=(SNEPPXVisionTransformer*)vit;
    if(!m||!images||!logits) return -1;
    size_t H=m->hidden, P=m->patch_size, Cin=m->in_channels;
    size_t G=m->img_size/P, N=G*G, seq=N+1, patch_dim=Cin*P*P;
    float* x=(float*)malloc(seq*H*sizeof(float));
    if(!x) return -1;
    for (size_t b=0;b<batch_size;b++){
        const float* img = images + b*Cin*m->img_size*m->img_size;
        for (size_t pj=0;pj<G;pj++) for (size_t pi=0;pi<G;pi++){
            size_t patch_idx = pj*G+pi;
            float* flat = (float*)malloc(patch_dim*sizeof(float));
            if (!flat) continue;
            for (size_t c=0;c<Cin;c++) for (size_t yy=0;yy<P;yy++) for (size_t xx=0;xx<P;xx++){
                size_t si = ((c*m->img_size + (pj*P+yy))*m->img_size + (pi*P+xx));
                flat[(c*P+yy)*P+xx] = img[si];
            }
            float* pe = (float*)malloc(H*sizeof(float));
            linear_fwd(flat,1,(float*)m->Wproj->data,patch_dim,H,pe);
            memcpy(x + (patch_idx+1)*H, pe, H*sizeof(float));
            free(pe);
            free(flat);
        }
        memcpy(x, m->cls_token->data, H*sizeof(float));
        for (size_t i=0;i<seq;i++) for (size_t p=0;p<H;p++) x[i*H+p]+=((float*)m->pos_embed->data)[i*H+p];
        for (size_t l=0;l<m->layers;l++) vit_encoder_block(m,l,x,seq);
        layernorm_fwd(x,(float*)m->ln_f_g->data,(float*)m->ln_f_b->data,1,H,1e-5f,x);
        linear_fwd(x,1,(float*)m->Wcls->data,H,m->num_classes,logits+b*m->num_classes);
    }
    free(x);
    return 0;
}

/**
 * @brief Perform Vit Extract Features.
 *
 * @param vit [out] Vit value.
 * @param images [in] Images value.
 * @param batch_size [in] Batch Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vit_extract_features(void* vit, const float* images, size_t batch_size, float* features) {
    SNEPPXVisionTransformer* m=(SNEPPXVisionTransformer*)vit;
    if(!m||!images||!features) return -1;
    size_t H=m->hidden, P=m->patch_size, Cin=m->in_channels;
    size_t G=m->img_size/P, N=G*G, seq=N+1, patch_dim=Cin*P*P;
    float* x=(float*)malloc(seq*H*sizeof(float));
    if(!x) return -1;
    for (size_t b=0;b<batch_size;b++){
        const float* img = images + b*Cin*m->img_size*m->img_size;
        for (size_t pj=0;pj<G;pj++) for (size_t pi=0;pi<G;pi++){
            size_t patch_idx=pj*G+pi;
            float* flat=(float*)malloc(patch_dim*sizeof(float));
            if (!flat) continue;
            for (size_t c=0;c<Cin;c++) for (size_t yy=0;yy<P;yy++) for (size_t xx=0;xx<P;xx++){
                size_t si=((c*m->img_size+(pj*P+yy))*m->img_size+(pi*P+xx));
                flat[(c*P+yy)*P+xx]=img[si];
            }
            float* pe=(float*)malloc(H*sizeof(float));
            linear_fwd(flat,1,(float*)m->Wproj->data,patch_dim,H,pe);
            memcpy(x+(patch_idx+1)*H,pe,H*sizeof(float));
            free(pe);
            free(flat);
        }
        memcpy(x,m->cls_token->data,H*sizeof(float));
        for (size_t i=0;i<seq;i++) for (size_t p=0;p<H;p++) x[i*H+p]+=((float*)m->pos_embed->data)[i*H+p];
        for (size_t l=0;l<m->layers;l++) vit_encoder_block(m,l,x,seq);
        memcpy(features+b*H, x, H*sizeof(float)); /* CLS token features */
    }
    free(x);
    return 0;
}
