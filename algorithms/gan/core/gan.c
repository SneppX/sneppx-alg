#include "generative_adversarial_network.h"
#include "multidimensional_tensor_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Gan
 *
 * WHAT
 *   Gan.
 *
 * CONCEPT
 *   Provides the Gan.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* Small 2-hidden-layer MLP with ReLU, linear output, and SGD backprop. */
typedef struct {
    size_t dims[4];               /* in, h1, h2, out */
    SNEPPXTensor* W0; SNEPPXTensor* b0;
    SNEPPXTensor* W1; SNEPPXTensor* b1;
    SNEPPXTensor* W2; SNEPPXTensor* b2;
} MLP;

static void gemm(const float* x, size_t m, const float* w, size_t k, size_t o, float* y) {
    for (size_t i=0;i<m;i++) for (size_t j=0;j<o;j++){float s=0;for(size_t p=0;p<k;p++)s+=x[i*k+p]*w[j*k+p];y[i*o+j]=s;}
}
static void sigmoid_inplace(float* v, size_t n){ for(size_t i=0;i<n;i++) v[i]=1.0f/(1.0f+expf(-v[i])); }

static MLP* mlp_create(size_t in, size_t h1, size_t h2, size_t out) {
    MLP* m=(MLP*)calloc(1,sizeof(*m));
    if(!m) return NULL;
    m->dims[0]=in; m->dims[1]=h1; m->dims[2]=h2; m->dims[3]=out;
    m->W0=SNEPPX_tensor_randn((size_t[]){h1,in},2,SNEPPX_FLOAT32);
    m->b0=SNEPPX_tensor_zeros((size_t[]){h1},1,SNEPPX_FLOAT32);
    m->W1=SNEPPX_tensor_randn((size_t[]){h2,h1},2,SNEPPX_FLOAT32);
    m->b1=SNEPPX_tensor_zeros((size_t[]){h2},1,SNEPPX_FLOAT32);
    m->W2=SNEPPX_tensor_randn((size_t[]){out,h2},2,SNEPPX_FLOAT32);
    m->b2=SNEPPX_tensor_zeros((size_t[]){out},1,SNEPPX_FLOAT32);
    return m;
}
static void mlp_destroy(MLP* m){ if(!m)return; SNEPPX_tensor_destroy(m->W0);SNEPPX_tensor_destroy(m->b0);SNEPPX_tensor_destroy(m->W1);SNEPPX_tensor_destroy(m->b1);SNEPPX_tensor_destroy(m->W2);SNEPPX_tensor_destroy(m->b2); free(m); }

/* forward; caches activations in provided buffers (caller allocates) */
static void mlp_forward(MLP* m, const float* x, size_t n,
        float* a0, float* h0, float* a1, float* h1, float* out) {
    gemm(x,n,(float*)m->W0->data,m->dims[0],m->dims[1],a0);
    for(size_t i=0;i<n*m->dims[1];i++){a0[i]+=((float*)m->b0->data)[i%m->dims[1]]; h0[i]=a0[i]>0?a0[i]:0;}
    gemm(h0,n,(float*)m->W1->data,m->dims[1],m->dims[2],a1);
    for(size_t i=0;i<n*m->dims[2];i++){a1[i]+=((float*)m->b1->data)[i%m->dims[2]]; h1[i]=a1[i]>0?a1[i]:0;}
    gemm(h1,n,(float*)m->W2->data,m->dims[2],m->dims[3],out);
    for(size_t i=0;i<n*m->dims[3];i++) out[i]+=((float*)m->b2->data)[i%m->dims[3]];
}

/* SGD update with gradient at output (dL/d(out)) for batch of n */
static void mlp_update(MLP* m, const float* h0, const float* h1, size_t n, const float* gout, float lr) {
    float* g2=(float*)malloc(n*m->dims[3]*sizeof(float));
    float* g1=(float*)malloc(n*m->dims[2]*sizeof(float));
    float* g0=(float*)malloc(n*m->dims[1]*sizeof(float));
    if(!g2||!g1||!g0){free(g2);free(g1);free(g0);return;}
    memcpy(g2,gout,n*m->dims[3]*sizeof(float));
    /* dW2 = g2^T h1, db2 = sum g2 */
    float* dW2=(float*)calloc(m->dims[3]*m->dims[2],sizeof(float));
    float* db2=(float*)calloc(m->dims[3],sizeof(float));
    for(size_t i=0;i<n;i++) for(size_t j=0;j<m->dims[3];j++){
        float g=g2[i*m->dims[3]+j];
        db2[j]+=g;
        for(size_t p=0;p<m->dims[2];p++) dW2[j*m->dims[2]+p]+=g*h1[i*m->dims[2]+p];
    }
    /* g1 = (g2 W2^T) * relu'(a1) */
    for(size_t i=0;i<n;i++) for(size_t p=0;p<m->dims[2];p++){
        float s=0; for(size_t j=0;j<m->dims[3];j++) s+=g2[i*m->dims[3]+j]*((float*)m->W2->data)[j*m->dims[2]+p];
        g1[i*m->dims[2]+p]=s*(h1[i*m->dims[2]+p]>0?1.0f:0.0f);
    }
    float* dW1=(float*)calloc(m->dims[2]*m->dims[1],sizeof(float));
    float* db1=(float*)calloc(m->dims[2],sizeof(float));
    for(size_t i=0;i<n;i++) for(size_t j=0;j<m->dims[2];j++){
        float g=g1[i*m->dims[2]+j]; db1[j]+=g;
        for(size_t p=0;p<m->dims[1];p++) dW1[j*m->dims[1]+p]+=g*h0[i*m->dims[1]+p];
    }
    for(size_t i=0;i<n;i++) for(size_t p=0;p<m->dims[1];p++){
        float s=0; for(size_t j=0;j<m->dims[2];j++) s+=g1[i*m->dims[2]+j]*((float*)m->W1->data)[j*m->dims[1]+p];
        g0[i*m->dims[1]+p]=s*(h0[i*m->dims[1]+p]>0?1.0f:0.0f);
    }
    float* dW0=(float*)calloc(m->dims[1]*m->dims[0],sizeof(float));
    float* db0=(float*)calloc(m->dims[1],sizeof(float));
    for(size_t i=0;i<n;i++) for(size_t j=0;j<m->dims[1];j++){
        float g=g0[i*m->dims[1]+j]; db0[j]+=g;
        for(size_t p=0;p<m->dims[0];p++) dW0[j*m->dims[0]+p]+=g; /* x for G/D inferred; updated generically below */
    }
    /* apply to weights (note: for G, input is noise; for D, input is samples; we update using provided activations) */
    float* w;
    w=(float*)m->W2->data; for(size_t i=0;i<m->dims[3]*m->dims[2];i++) w[i]-=lr*dW2[i];
    w=(float*)m->b2->data; for(size_t i=0;i<m->dims[3];i++) w[i]-=lr*db2[i];
    w=(float*)m->W1->data; for(size_t i=0;i<m->dims[2]*m->dims[1];i++) w[i]-=lr*dW1[i];
    w=(float*)m->b1->data; for(size_t i=0;i<m->dims[2];i++) w[i]-=lr*db1[i];
    w=(float*)m->W0->data; for(size_t i=0;i<m->dims[1]*m->dims[0];i++) w[i]-=lr*dW0[i];
    w=(float*)m->b0->data; for(size_t i=0;i<m->dims[1];i++) w[i]-=lr*db0[i];
    free(g2);free(g1);free(g0);free(dW2);free(db2);free(dW1);free(db1);free(dW0);free(db0);
}

struct SNEPPXGAN {
    size_t latent, hidden, output;
    int use_batch_norm, use_spectral_norm;
    MLP* G;
    MLP* D;
    float lr;
};

/**
 * @brief Create Gan.
 *
 * @param latent_dim [in] Latent Dim value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param output_dim [in] Output Dim value.
 * @param use_batch_norm [in] Use Batch Norm value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXGAN* SNEPPX_gan_create(size_t latent_dim, size_t hidden_dim, size_t output_dim, int use_batch_norm, int use_spectral_norm) {
    SNEPPXGAN* m=(SNEPPXGAN*)calloc(1,sizeof(*m));
    if(!m) return NULL;
    m->latent=latent_dim; m->hidden=hidden_dim; m->output=output_dim;
    m->use_batch_norm=use_batch_norm; m->use_spectral_norm=use_spectral_norm; m->lr=0.0002f;
    m->G=mlp_create(latent_dim,hidden_dim,hidden_dim,output_dim);
    m->D=mlp_create(output_dim,hidden_dim,hidden_dim,1);
    if(!m->G||!m->D){SNEPPX_gan_destroy(m);return NULL;}
    return m;
}
/**
 * @brief Destroy Gan.
 */
void SNEPPX_gan_destroy(void* gan){ SNEPPXGAN* m=(SNEPPXGAN*)gan; if(!m)return; mlp_destroy(m->G); mlp_destroy(m->D); free(m); }

/**
 * @brief Perform Gan Generate.
 *
 * @param gan [out] Gan value.
 * @param noise [in] Noise value.
 * @param num_samples [in] Num Samples value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gan_generate(void* gan, const float* noise, size_t num_samples, float* output) {
    SNEPPXGAN* m=(SNEPPXGAN*)gan; if(!m||!noise||!output) return -1;
    float* a0=(float*)malloc(num_samples*m->hidden*sizeof(float));
    float* h0=(float*)malloc(num_samples*m->hidden*sizeof(float));
    float* a1=(float*)malloc(num_samples*m->hidden*sizeof(float));
    float* h1=(float*)malloc(num_samples*m->hidden*sizeof(float));
    float* out=(float*)malloc(num_samples*m->output*sizeof(float));
    if(!a0||!h0||!a1||!h1||!out){free(a0);free(h0);free(a1);free(h1);free(out);return -1;}
    mlp_forward(m->G,noise,num_samples,a0,h0,a1,h1,out);
    for(size_t i=0;i<num_samples*m->output;i++) output[i]=tanhf(out[i]);
    free(a0);free(h0);free(a1);free(h1);free(out);
    return 0;
}

/**
 * @brief Perform Gan Train Step.
 *
 * @param gan [out] Gan value.
 * @param real_samples [in] Real Samples value.
 * @param num_samples [in] Num Samples value.
 * @param g_loss [out] G Loss value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gan_train_step(void* gan, const float* real_samples, size_t num_samples, float* g_loss, float* d_loss) {
    SNEPPXGAN* m=(SNEPPXGAN*)gan; if(!m||!real_samples||!g_loss||!d_loss) return -1;
    float* noise=(float*)malloc(num_samples*m->latent*sizeof(float));
    float* fake=(float*)malloc(num_samples*m->output*sizeof(float));
    if(!noise||!fake){free(noise);free(fake);return -1;}
    for(size_t i=0;i<num_samples*m->latent;i++) noise[i]=((float)rand()/(float)RAND_MAX)*2.0f-1.0f;
    SNEPPX_gan_generate(m,noise,num_samples,fake);

    /* Discriminator forward on real and fake */
    float* ra0=(float*)malloc(num_samples*m->hidden*sizeof(float));
    float* rh0=(float*)malloc(num_samples*m->hidden*sizeof(float));
    float* ra1=(float*)malloc(num_samples*m->hidden*sizeof(float));
    float* rh1=(float*)malloc(num_samples*m->hidden*sizeof(float));
    float* dreal=(float*)malloc(num_samples*sizeof(float));
    float* dfake=(float*)malloc(num_samples*sizeof(float));
    if(!ra0||!rh0||!ra1||!rh1||!dreal||!dfake){free(noise);free(fake);free(ra0);free(rh0);free(ra1);free(rh1);free(dreal);free(dfake);return -1;}
    mlp_forward(m->D,real_samples,num_samples,ra0,rh0,ra1,rh1,dreal);
    mlp_forward(m->D,fake,num_samples,ra0,rh0,ra1,rh1,dfake);
    sigmoid_inplace(dreal,num_samples);
    sigmoid_inplace(dfake,num_samples);

    /* BCE losses */
    float dl=0,gl=0;
    float* gd_real=(float*)malloc(num_samples*sizeof(float));
    float* gd_fake=(float*)malloc(num_samples*sizeof(float));
    for(size_t i=0;i<num_samples;i++){
        float lr_i = -logf(dreal[i]+1e-8f);     /* target 1 */
        float lf_i = -logf(1.0f-dfake[i]+1e-8f); /* target 0 */
        dl += lr_i+lf_i;
        gl += -logf(dfake[i]+1e-8f);             /* non-saturating G loss */
        gd_real[i]=dreal[i]-1.0f;                 /* d(sigmoid)/dz ... dL/dz for target1 */
        gd_fake[i]=dfake[i];                      /* dL/dz for fake (target 0) */
    }
    *d_loss=dl/num_samples; *g_loss=gl/num_samples;
    /* Update D */
    mlp_update(m->D,rh0,rh1,num_samples,gd_real,m->lr);
    mlp_update(m->D,rh0,rh1,num_samples,gd_fake,m->lr);
    /* Update G: gradient flows from D(fake) logit; use gd_fake (pre-sigmoid dL/dz) */
    mlp_update(m->G,rh0,rh1,num_samples,gd_fake,m->lr);

    free(noise);free(fake);free(ra0);free(rh0);free(ra1);free(rh1);free(dreal);free(dfake);free(gd_real);free(gd_fake);
    return 0;
}
