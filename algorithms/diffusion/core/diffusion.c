#include "diffusion_model.h"
#include "multidimensional_tensor_engine.h"
#include <stdlib.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <math.h>

#define SNEPPX_DIF_MAXT 10000

/*
 * SNEPPX - Diffusion
 *
 * WHAT
 *   Diffusion.
 *
 * CONCEPT
 *   Provides the Diffusion.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


struct SNEPPXDiffusion {
    size_t img_channels, img_size, hidden, T;
    int schedule_type;
    size_t feat_dim;                 /* img_channels * img_size * img_size */
    SNEPPXTensor* Wemb;              /* [hidden, hidden] timestep projection */
    SNEPPXTensor* W1;                /* [hidden, feat+hidden] */
    SNEPPXTensor* W2;                /* [hidden, hidden] */
    SNEPPXTensor* Wout;              /* [feat, hidden] */
    float* sqrt_ab;                  /* sqrt(alpha_bar_t), length T */
    float* sqrt_1mab;                /* sqrt(1-alpha_bar_t) */
    float* betas;                    /* length T */
};

static void linear_fwd(const float* x, size_t m, const float* w, size_t k, size_t out, float* y) {
    for (size_t i=0;i<m;i++) for (size_t j=0;j<out;j++){float s=0;for(size_t p=0;p<k;p++)s+=x[i*k+p]*w[j*k+p];y[i*out+j]=s;}
}

/**
 * @brief Create Diffusion.
 *
 * @param img_channels [in] Img Channels value.
 * @param img_size [in] Img Size value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param num_timesteps [in] Num Timesteps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXDiffusion* SNEPPX_diffusion_create(size_t img_channels, size_t img_size, size_t hidden_dim, int num_timesteps, int noise_schedule_type) {
    if (img_channels==0||img_size==0||num_timesteps<=0) return NULL;
    SNEPPXDiffusion* m=(SNEPPXDiffusion*)calloc(1,sizeof(*m));
    if(!m) return NULL;
    m->img_channels=img_channels; m->img_size=img_size; m->hidden=hidden_dim; m->T=num_timesteps; m->schedule_type=noise_schedule_type;
    size_t feat=img_channels*img_size*img_size; m->feat_dim=feat;
    m->Wemb=SNEPPX_tensor_randn((size_t[]){hidden_dim,hidden_dim},2,SNEPPX_FLOAT32);
    m->W1=SNEPPX_tensor_randn((size_t[]){hidden_dim,feat+hidden_dim},2,SNEPPX_FLOAT32);
    m->W2=SNEPPX_tensor_randn((size_t[]){hidden_dim,hidden_dim},2,SNEPPX_FLOAT32);
    m->Wout=SNEPPX_tensor_randn((size_t[]){feat,hidden_dim},2,SNEPPX_FLOAT32);
    /* cosine or linear beta schedule */
    m->betas=(float*)malloc(sizeof(float)*num_timesteps);
    m->sqrt_ab=(float*)malloc(sizeof(float)*num_timesteps);
    m->sqrt_1mab=(float*)malloc(sizeof(float)*num_timesteps);
    float abar=1.0f;
    for (int t=0;t<num_timesteps;t++){
        float beta;
        if (noise_schedule_type==1) { /* cosine */
            float f=(float)(t+1)/(float)num_timesteps;
            float fc=(float)t/(float)num_timesteps;
            float ad=0.008f;
            float num=(float)cosf((fc+ad)/(1.0f+ad)*(float)M_PI*0.5);
            float den=(float)cosf((f+ad)/(1.0f+ad)*(float)M_PI*0.5);
            beta=(float)fmin(1.0- num/den, 0.999);
        } else { /* linear */
            beta=0.0001f+ (0.02f-0.0001f)*(float)t/(float)(num_timesteps-1);
        }
        m->betas[t]=beta;
        abar*=(1.0f-beta);
        m->sqrt_ab[t]=sqrtf(abar);
        m->sqrt_1mab[t]=sqrtf(1.0f-abar);
    }
    return m;
}

/**
 * @brief Destroy Diffusion.
 */
void SNEPPX_diffusion_destroy(void* model) {
    SNEPPXDiffusion* m=(SNEPPXDiffusion*)model; if(!m) return;
    SNEPPX_tensor_destroy(m->Wemb); SNEPPX_tensor_destroy(m->W1); SNEPPX_tensor_destroy(m->W2); SNEPPX_tensor_destroy(m->Wout);
    free(m->betas); free(m->sqrt_ab); free(m->sqrt_1mab); free(m);
}

/* Predict noise eps_theta(x_t, t) -> out [n, feat] */
static void predict_noise(SNEPPXDiffusion* m, const float* xt, const float* tt_emb, size_t n, float* out) {
    size_t H=m->hidden, feat=m->feat_dim;
    float* cat=(float*)malloc((feat+H)*sizeof(float));
    float* h1=(float*)malloc(n*H*sizeof(float));
    float* h2=(float*)malloc(n*H*sizeof(float));
    if(!cat||!h1||!h2){free(cat);free(h1);free(h2);return;}
    for (size_t i=0;i<n;i++){
        memcpy(cat, xt+i*feat, feat*sizeof(float));
        memcpy(cat+feat, tt_emb, H*sizeof(float));
        linear_fwd(cat+i*0,1,(float*)m->W1->data,feat+H,H,h1+i*H);
        for (size_t j=0;j<H;j++) h1[i*H+j]=(h1[i*H+j]>0?h1[i*H+j]:0.0f);
    }
    /* add timestep embedding to h1 before W2 (broadcast) */
    for (size_t i=0;i<n;i++) for (size_t j=0;j<H;j++) h1[i*H+j]+=tt_emb[j];
    linear_fwd(h1,n,(float*)m->W2->data,H,H,h2);
    for (size_t i=0;i<n;i++) for (size_t j=0;j<H;j++) h2[i*H+j]=(h2[i*H+j]>0?h2[i*H+j]:0.0f);
    linear_fwd(h2,n,(float*)m->Wout->data,H,feat,out);
    free(cat); free(h1); free(h2);
}

/* sinusoidal timestep embedding */
static void timestep_embed(int t, size_t H, float* out) {
    for (size_t j=0;j<H;j++){
        float f=(float)t/(powf(10000.0f,(float)(j)/(float)H));
        out[j]= (j%2==0)? (float)sinf(f) : (float)cosf(f);
    }
}

/**
 * @brief Perform Diffusion Sample.
 *
 * @param model [out] Model value.
 * @param output [out] Output value.
 * @param num_samples [in] Num Samples value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_diffusion_sample(void* model, float* output, size_t num_samples, const float* cond) {
    (void)cond;
    SNEPPXDiffusion* m=(SNEPPXDiffusion*)model;
    if(!m||!output) return -1;
    size_t feat=m->feat_dim, H=m->hidden;
    float* x=(float*)malloc(num_samples*feat*sizeof(float));
    float* eps=(float*)malloc(num_samples*feat*sizeof(float));
    float* temb=(float*)malloc(H*sizeof(float));
    if(!x||!eps||!temb){free(x);free(eps);free(temb);return -1;}
    for (size_t i=0;i<num_samples*feat;i++) x[i]=((float)rand()/(float)RAND_MAX)*2.0f-1.0f;
    for (int t=m->T-1;t>=0;t--){
        timestep_embed(t,H,temb);
        predict_noise(m,x,temb,num_samples,eps);
        float ab=m->sqrt_ab[t], om=m->sqrt_1mab[t];
        float beta = (t>0)? m->betas[t] : 0.0f;
        float sigma = (t>0)? sqrtf(beta) : 0.0f;
        for (size_t i=0;i<num_samples*feat;i++){
            float z=((float)rand()/(float)RAND_MAX)*2.0f-1.0f;
            x[i]=(x[i]-om*eps[i])/ (ab+1e-8f) + sigma*z;
        }
    }
    memcpy(output,x,num_samples*feat*sizeof(float));
    free(x); free(eps); free(temb);
    return 0;
}

/**
 * @brief Perform Diffusion Train Step.
 *
 * @param model [out] Model value.
 * @param images [in] Images value.
 * @param num_samples [in] Num Samples value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_diffusion_train_step(void* model, const float* images, size_t num_samples, float* loss) {
    SNEPPXDiffusion* m=(SNEPPXDiffusion*)model;
    if(!m||!images||!loss) return -1;
    size_t feat=m->feat_dim, H=m->hidden;
    float* x0=(float*)malloc(num_samples*feat*sizeof(float));
    float* noise=(float*)malloc(num_samples*feat*sizeof(float));
    float* xt=(float*)malloc(num_samples*feat*sizeof(float));
    float* eps=(float*)malloc(num_samples*feat*sizeof(float));
    float* temb=(float*)malloc(H*sizeof(float));
    if(!x0||!noise||!xt||!eps||!temb){free(x0);free(noise);free(xt);free(eps);free(temb);return -1;}
    memcpy(x0,images,num_samples*feat*sizeof(float));
    for (size_t i=0;i<num_samples*feat;i++) noise[i]=((float)rand()/(float)RAND_MAX)*2.0f-1.0f;
    int t=(int)((float)rand()/(float)RAND_MAX*(float)m->T);
    float ab=m->sqrt_ab[t], om=m->sqrt_1mab[t];
    for (size_t i=0;i<num_samples*feat;i++) xt[i]=ab*x0[i]+om*noise[i];
    timestep_embed(t,H,temb);
    predict_noise(m,xt,temb,num_samples,eps);
    float total=0;
    for (size_t i=0;i<num_samples*feat;i++){ float d=eps[i]-noise[i]; total+=d*d; }
    *loss=total/(num_samples*feat);
    free(x0); free(noise); free(xt); free(eps); free(temb);
    return 0;
}
