#include "s5_extensions.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

/*
 * SNEPPX - S5 Extensions
 *
 * WHAT
 *   S5 Extensions.
 *
 * CONCEPT
 *   Provides the S5 Extensions.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* --- Semantic Injection --- */
/**
 * @brief Initialize Semantic Injection.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_init(SNEPPXSemanticInjectionDetector* sid) {
    if(!sid) return -1; memset(sid,0,sizeof(*sid)); sid->threshold=0.85; return 0;
}
/**
 * @brief Perform Semantic Injection Add Attack.
 *
 * @param sid [out] Sid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_add_attack(SNEPPXSemanticInjectionDetector* sid, const double embedding[8]) {
    if(!sid||!embedding||sid->attack_count>=SNEPPX_S5_MAX_EMBEDDING) return -1;
    memcpy(sid->known_attack_embeddings[sid->attack_count++],embedding,8*sizeof(double));
    return 0;
}
/**
 * @brief Perform Semantic Injection Score.
 *
 * @param sid [out] Sid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_score(SNEPPXSemanticInjectionDetector* sid, const double embedding[8], double* score) {
    if(!sid||!embedding||!score) return -1;
    double max_sim=0.0;
    for(int i=0;i<sid->attack_count;i++) {
        double dot=0.0,n1=0.0,n2=0.0;
        for(int j=0;j<8;j++){dot+=embedding[j]*sid->known_attack_embeddings[i][j];n1+=embedding[j]*embedding[j];n2+=sid->known_attack_embeddings[i][j]*sid->known_attack_embeddings[i][j];}
        double sim=dot/(sqrt(n1)*sqrt(n2)+1e-10);
        if(sim>max_sim) max_sim=sim;
    }
    *score=max_sim; return max_sim>sid->threshold?1:0;
}

/* --- Multi-lang jailbreak --- */
/**
 * @brief Perform Ml Jailbreak Detect.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_detect(const char* text, size_t len) {
    if(!text) return 0;
    const char* patterns[]={"ignora","ignorer","\xe5\xbf\xbd\xe7\x95\xa5","\xeb\xac\xb4\xec\x8b\x9c","ignore","jailbreak","DAN","override",NULL};
    char lower[1024]; size_t clen=len<1023?len:1023;
    for(size_t i=0;i<clen;i++) lower[i]=(char)tolower((unsigned char)text[i]); lower[clen]=0;
    for(int p=0;patterns[p];p++) if(strstr(lower,patterns[p])) return 1;
    return 0;
}

static const char* b64_alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static int b64_rev[256];
static int b64_init=0;

static void b64_build_rev(void) {
    if(b64_init) return; b64_init=1;
    for(int i=0;i<256;i++) b64_rev[i]=-1;
    for(int i=0;i<64;i++) b64_rev[(int)b64_alphabet[i]]=i;
}

static int is_base64(const char* s, size_t n) {
    if(n==0) return 0;
    for(size_t i=0;i<n;i++) {
        char c=s[i];
        if(c=='='||(c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='+'||c=='/') continue;
        return 0;
    }
    return 1;
}

static int is_hex(const char* s, size_t n) {
    if(n<2) return 0;
    for(size_t i=0;i<n;i++) {
        char c=s[i];
        if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F'))) return 0;
    }
    return 1;
}

static size_t b64_decode(const char* in, size_t in_len, unsigned char* out) {
    b64_build_rev();
    size_t out_pos=0;
    int val=0, valb=-8;
    for(size_t i=0;i<in_len;i++) {
        unsigned char c=(unsigned char)in[i];
        if(c=='=') break;
        int d=b64_rev[c];
        if(d==-1) continue;
        val=(val<<6)|d;
        valb+=6;
        if(valb>=0) {
            out[out_pos++]=(unsigned char)((val>>valb)&0xFF);
            valb-=8;
        }
    }
    return out_pos;
}

static size_t hex_decode(const char* in, size_t in_len, unsigned char* out) {
    size_t out_pos=0;
    for(size_t i=0;i+1<in_len;i+=2) {
        unsigned char c1=(unsigned char)in[i];
        unsigned char c2=(unsigned char)in[i+1];
        unsigned char v1=(c1>='a'?c1-'a'+10:(c1>='A'?c1-'A'+10:c1-'0'));
        unsigned char v2=(c2>='a'?c2-'a'+10:(c2>='A'?c2-'A'+10:c2-'0'));
        out[out_pos++]=(v1<<4)|v2;
    }
    return out_pos;
}

static size_t rot13_decode(const char* in, size_t in_len, char* out) {
    for(size_t i=0;i<in_len;i++) {
        char c=in[i];
        if(c>='a'&&c<='z') out[i]=(char)((c-'a'+13)%26+'a');
        else if(c>='A'&&c<='Z') out[i]=(char)((c-'A'+13)%26+'A');
        else out[i]=c;
    }
    return in_len;
}

/* --- Encoded attack decoder --- */
/**
 * @brief Perform Encoded Attack Decode.
 *
 * @param input [in] Input value.
 * @param in_len [in] In Len value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_decode(const char* input, size_t in_len, char* output, size_t* out_len) {
    if(!input||!output||!out_len) return -1;
    unsigned char buf[4096];
    size_t rlen=0;
    if(in_len>=2&&input[0]=='0'&&(input[1]=='x'||input[1]=='X')) {
        if(in_len-2>sizeof(buf)) return -1;
        rlen=hex_decode(input+2,in_len-2,buf);
    } else if(in_len%4==0&&is_base64(input,in_len)) {
        rlen=b64_decode(input,in_len,buf);
    } else {
        rlen=rot13_decode(input,in_len,output);
        *out_len=rlen;
        return 0;
    }
    if(rlen>4096) rlen=4096;
    memcpy(output,buf,rlen);
    *out_len=rlen;
    return 0;
}

/**
 * @brief Perform Encoded Attack Scan.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_scan(const char* text, size_t len) {
    if(!text) return 0;
    char decoded[4096];
    size_t dlen=0;
    if(SNEPPX_encoded_attack_decode(text,len,decoded,&dlen)!=0) return 0;
    if(dlen==0) { dlen=len<4096?len:4095; memcpy(decoded,text,dlen); decoded[dlen]=0; }
    if(dlen>4095) dlen=4095;
    decoded[dlen]=0;
    const char* patterns[]={
        "DROP TABLE","SELECT * FROM","DELETE FROM","INSERT INTO",
        "exec ","eval(","os.system","subprocess",
        "<?php","<script>","javascript:",
        "ignore all instructions","ignore previous",
        "jailbreak","override filter",
        NULL
    };
    char lower[4096];
    for(size_t i=0;i<dlen;i++) lower[i]=(char)tolower((unsigned char)decoded[i]);
    lower[dlen]=0;
    for(int p=0;patterns[p];p++) {
        char plow[256]; size_t plen=strlen(patterns[p]);
        for(size_t i=0;i<plen;i++) plow[i]=(char)tolower((unsigned char)patterns[p][i]);
        plow[plen]=0;
        if(strstr(lower,plow)) return 1;
    }
    return 0;
}

/* --- Token anomaly --- */
/**
 * @brief Perform Token Anomaly Score.
 *
 * @param token_ids [in] Token Ids value.
 * @param token_count [in] Token Count value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_token_anomaly_score(const uint32_t* token_ids, size_t token_count, const double* expected_probs) {
    (void)token_ids;
    if(!expected_probs||token_count==0) return 0.0;
    double sum_log=0.0;
    size_t valid=0;
    for(size_t i=0;i<token_count;i++) {
        double p=expected_probs[i];
        if(p<=0.0) p=1e-10;
        if(p>1.0) p=1.0;
        sum_log+=-log(p);
        valid++;
    }
    if(valid==0) return 0.0;
    return sum_log/(double)valid;
}

/* --- Model inversion defense --- */
/**
 * @brief Initialize Model Inversion.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_inversion_init(SNEPPXModelInversionDefense* mid) {
    if(!mid) return -1; memset(mid,0,sizeof(*mid));
    mid->noise_scale=0.01; mid->gradient_clipping=1; mid->clip_norm=1.0; return 0;
}
/**
 * @brief Apply Model Inversion.
 *
 * @param mid [out] Mid value.
 * @param gradients [out] Gradients value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_inversion_apply(SNEPPXModelInversionDefense* mid, double* gradients, size_t grad_count) {
    if(!mid||!gradients) return -1;
    double norm=0.0; for(size_t i=0;i<grad_count;i++) norm+=gradients[i]*gradients[i];
    norm=sqrt(norm);
    if(mid->gradient_clipping&&norm>mid->clip_norm)
        for(size_t i=0;i<grad_count;i++) gradients[i]=gradients[i]/norm*mid->clip_norm;
    for(size_t i=0;i<grad_count;i++) gradients[i]+=((double)rand()/RAND_MAX-0.5)*mid->noise_scale;
    return 0;
}

/* --- Membership inference --- */
/**
 * @brief Perform Membership Inference Defense.
 *
 * @param logits [out] Logits value.
 * @param logit_count [in] Logit Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_membership_inference_defense(double* logits, size_t logit_count, double epsilon) {
    if(!logits||logit_count==0) return -1;
    for(size_t i=0;i<logit_count;i++) {
        if(logits[i]<-epsilon) logits[i]=-epsilon;
        if(logits[i]>epsilon) logits[i]=epsilon;
    }
    double scale=epsilon>0?1.0/epsilon:1.0;
    for(size_t i=0;i<logit_count;i++) {
        double u=(double)rand()/(double)RAND_MAX;
        double lap=0.0;
        if(u<0.5) lap=scale*log(2.0*u);
        else lap=-scale*log(2.0*(1.0-u));
        logits[i]+=lap;
    }
    return 0;
}

/* --- Data extraction prevention --- */
/**
 * @brief Perform Data Extraction Prevent.
 *
 * @param output [in] Output value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_extraction_prevent(const char* output, size_t len, int* contains_sensitive) {
    if(!output||!contains_sensitive) return -1;
    *contains_sensitive=0;
    if(len==0) return 0;
    if(len>100000) len=100000;
    size_t i=0;
    while(i+13<len) {
        if(isdigit((unsigned char)output[i])&&isdigit((unsigned char)output[i+1])&&
           isdigit((unsigned char)output[i+2])&&isdigit((unsigned char)output[i+3])&&
           output[i+4]=='-'&&
           isdigit((unsigned char)output[i+5])&&isdigit((unsigned char)output[i+6])&&
           isdigit((unsigned char)output[i+7])&&isdigit((unsigned char)output[i+8])&&
           output[i+9]=='-'&&
           isdigit((unsigned char)output[i+10])&&isdigit((unsigned char)output[i+11])&&
           isdigit((unsigned char)output[i+12])&&isdigit((unsigned char)output[i+13])) {
            *contains_sensitive=1; return 0;
        }
        i++;
    }
    i=0;
    while(i+10<len) {
        if(isdigit((unsigned char)output[i])&&isdigit((unsigned char)output[i+1])&&
           isdigit((unsigned char)output[i+2])&&output[i+3]=='-'&&
           isdigit((unsigned char)output[i+4])&&isdigit((unsigned char)output[i+5])&&
           output[i+6]=='-'&&
           isdigit((unsigned char)output[i+7])&&isdigit((unsigned char)output[i+8])&&
           isdigit((unsigned char)output[i+9])&&isdigit((unsigned char)output[i+10])) {
            *contains_sensitive=1; return 0;
        }
        i++;
    }
    const char* apikey_patterns[]={
        "sk-","pk-","api_key","apikey","api-key",
        "AKIA","ASIA",
        "ghp_","gho_","ghu_","ghs_",
        NULL
    };
    char lower[4096];
    size_t cp=len<4095?len:4095;
    for(size_t j=0;j<cp;j++) lower[j]=(char)tolower((unsigned char)output[j]);
    lower[cp]=0;
    for(int p=0;apikey_patterns[p];p++) {
        if(strstr(lower,apikey_patterns[p])) { *contains_sensitive=1; return 0; }
    }
    return 0;
}

/* --- Training data sanitization --- */
/**
 * @brief Perform Training Sanitize.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 * @param sanitized [out] Sanitized value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_training_sanitize(const char* text, size_t len, char* sanitized, size_t* sanitized_len) {
    if(!text||!sanitized||!sanitized_len) return -1;
    size_t out_pos=0;
    size_t sanitized_cap=*sanitized_len;
    *sanitized_len=0;
    const char redacted[]="[REDACTED]";
    size_t redacted_len=strlen(redacted);
    size_t i=0;
    while(i<len&&out_pos<sanitized_cap) {
        if(i+3<len&&text[i]=='h'&&text[i+1]=='t'&&text[i+2]=='t'&&text[i+3]=='p') {
            size_t end=i;
            while(end<len&&text[end]!=' '&&text[end]!='\t'&&text[end]!='\n'&&text[end]!='\r'&&text[end]!=',') end++;
            if(out_pos+redacted_len<=sanitized_cap) {
                memcpy(sanitized+out_pos,redacted,redacted_len);
                out_pos+=redacted_len;
            }
            i=end;
            continue;
        }
        if(text[i]=='@') {
            size_t start=i;
            while(start>0&&text[start-1]!=' '&&text[start-1]!='\t'&&text[start-1]!='\n'&&text[start-1]!='\r'&&text[start-1]!=',') start--;
            size_t end=i;
            while(end<len&&text[end]!=' '&&text[end]!='\t'&&text[end]!='\n'&&text[end]!='\r'&&text[end]!=',') end++;
            if(out_pos+redacted_len<=sanitized_cap) {
                memcpy(sanitized+out_pos,redacted,redacted_len);
                out_pos+=redacted_len;
            }
            i=end;
            continue;
        }
        if(i+6<len&&isdigit((unsigned char)text[i])&&isdigit((unsigned char)text[i+1])&&
           isdigit((unsigned char)text[i+2])&&text[i+3]=='.'&&
           isdigit((unsigned char)text[i+4])&&isdigit((unsigned char)text[i+5])&&
           isdigit((unsigned char)text[i+6])) {
            size_t end=i;
            while(end<len&&(isdigit((unsigned char)text[end])||text[end]=='.')) end++;
            if(out_pos+redacted_len<=sanitized_cap) {
                memcpy(sanitized+out_pos,redacted,redacted_len);
                out_pos+=redacted_len;
            }
            i=end;
            continue;
        }
        if(i+2<len&&isdigit((unsigned char)text[i])&&isdigit((unsigned char)text[i+1])&&text[i+2]=='-') {
            size_t end=i;
            while(end<len&&(isdigit((unsigned char)text[end])||text[end]=='-')) end++;
            if(out_pos+redacted_len<=sanitized_cap) {
                memcpy(sanitized+out_pos,redacted,redacted_len);
                out_pos+=redacted_len;
            }
            i=end;
            continue;
        }
        sanitized[out_pos++]=text[i];
        i++;
    }
    *sanitized_len=out_pos;
    if(out_pos<sanitized_cap) sanitized[out_pos]=0;
    return 0;
}

/* --- Watermarking --- */
/**
 * @brief Initialize Watermark.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_init(SNEPPXModelWatermark* mw) {
    if(!mw) return -1;
    memset(mw,0,sizeof(*mw));
    for(int i=0;i<32;i++) mw->watermark[i]=(uint8_t)(rand()%256);
    mw->embedded=0;
    return 0;
}

/**
 * @brief Perform Watermark Embed.
 *
 * @param mw [out] Mw value.
 * @param weights [out] Weights value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_embed(SNEPPXModelWatermark* mw, double* weights, size_t weight_count) {
    if(!mw||!weights||weight_count==0) return -1;
    for(size_t i=0;i<weight_count;i++) {
        weights[i]+=0.001*sin((double)(mw->watermark[i%16])*(double)i);
    }
    mw->embedded=1;
    return 0;
}

/**
 * @brief Verify Watermark.
 *
 * @param mw [out] Mw value.
 * @param weights [in] Weights value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_verify(SNEPPXModelWatermark* mw, const double* weights, size_t weight_count) {
    if(!mw||!weights||weight_count==0) return -1;
    double dot=0.0,n1=0.0,n2=0.0;
    for(size_t i=0;i<weight_count;i++) {
        double expected=0.001*sin((double)(mw->watermark[i%16])*(double)i);
        double observed=weights[i];
        double residual=observed-expected;
        dot+=expected*residual;
        n1+=expected*expected;
        n2+=residual*residual;
    }
    double denom=sqrt(n1)*sqrt(n2);
    if(denom<1e-10) return 0;
    double corr=dot/denom;
    return corr>0.7?1:0;
}

/* --- Adversarial smoothing --- */
/**
 * @brief Perform Adversarial Smooth.
 *
 * @param input [out] Input value.
 * @param input_dim [in] Input Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_adversarial_smooth(double* input, size_t input_dim, double epsilon) {
    if(!input) return -1;
    for(size_t i=0;i<input_dim;i++) input[i]+=((double)rand()/RAND_MAX-0.5)*2.0*epsilon;
    return 0;
}

/* --- Factuality scorer --- */
static void tokenize_bigrams(const char* s, size_t len, char bigrams[][3], size_t* count, size_t max) {
    *count=0;
    if(!s||len<2) return;
    for(size_t i=0;i+1<len&&*count<max;i++) {
        bigrams[*count][0]=(char)tolower((unsigned char)s[i]);
        bigrams[*count][1]=(char)tolower((unsigned char)s[i+1]);
        bigrams[*count][2]=0;
        (*count)++;
    }
}

/**
 * @brief Perform Factuality Score.
 *
 * @param statement [in] Statement value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_factuality_score(const char* statement, const char* reference) {
    if(!statement||!reference) return 0.5;
    size_t slen=strlen(statement);
    size_t rlen=strlen(reference);
    if(slen<2||rlen<2) return 0.5;
    char s_bigrams[4096][3];
    char r_bigrams[4096][3];
    size_t s_count=0,r_count=0;
    size_t max_bigrams=slen<4096?slen:4096;
    tokenize_bigrams(statement,slen,s_bigrams,&s_count,max_bigrams);
    max_bigrams=rlen<4096?rlen:4096;
    tokenize_bigrams(reference,rlen,r_bigrams,&r_count,max_bigrams);
    if(s_count==0||r_count==0) return 0.5;
    size_t intersect=0;
    for(size_t i=0;i<s_count;i++) {
        for(size_t j=0;j<r_count;j++) {
            if(s_bigrams[i][0]==r_bigrams[j][0]&&s_bigrams[i][1]==r_bigrams[j][1]) {
                intersect++;
                break;
            }
        }
    }
    return (2.0*(double)intersect)/((double)(s_count+r_count));
}

/* --- Bias measurement --- */
/**
 * @brief Perform Bias Measure.
 *
 * @param bm [out] Bm value.
 * @param predictions [in] Predictions value.
 * @param sensitive_attr [in] Sensitive Attr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bias_measure(SNEPPXBiasMetrics* bm, const double* predictions, const int* sensitive_attr, size_t n) {
    if(!bm||!predictions||!sensitive_attr||n==0) return -1;
    double sum0=0.0,sum1=0.0;
    size_t cnt0=0,cnt1=0;
    for(size_t i=0;i<n;i++) {
        if(sensitive_attr[i]==0) { sum0+=predictions[i]; cnt0++; }
        else { sum1+=predictions[i]; cnt1++; }
    }
    double mean0=cnt0>0?sum0/(double)cnt0:0.0;
    double mean1=cnt1>0?sum1/(double)cnt1:0.0;
    bm->demographic_parity=mean1-mean0;
    bm->equalized_odds=mean1>0&&mean0>0?(mean1-mean0)/(mean0+1e-10):0.0;
    bm->measured=1;
    return 0;
}

/* --- Prompt policy engine --- */
/**
 * @brief Initialize Prompt Policy.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_init(SNEPPXPromptPolicy* pp) {
    if(!pp) return -1; memset(pp,0,sizeof(*pp)); pp->enabled=1; return 0;
}
/**
 * @brief Add Prompt Policy.
 *
 * @param pp [out] Pp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_add(SNEPPXPromptPolicy* pp, const char* policy_rule) {
    if(!pp||!policy_rule||pp->policy_count>=16) return -1;
    strncpy(pp->policies[pp->policy_count++],policy_rule,255); return 0;
}
/**
 * @brief Perform Prompt Policy Enforce.
 *
 * @param pp [out] Pp value.
 * @param prompt [in] Prompt value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_enforce(SNEPPXPromptPolicy* pp, const char* prompt, size_t len) {
    if(!pp||!prompt) return 0;
    if(!pp->enabled) return 0;
    if(len>8192) len=8192;
    char lower[8192];
    for(size_t i=0;i<len;i++) lower[i]=(char)tolower((unsigned char)prompt[i]);
    lower[len]=0;
    for(int i=0;i<pp->policy_count;i++) {
        char plow[256]; size_t plen=strlen(pp->policies[i]);
        size_t clen=plen<255?plen:255;
        for(size_t j=0;j<clen;j++) plow[j]=(char)tolower((unsigned char)pp->policies[i][j]);
        plow[clen]=0;
        if(strstr(lower,plow)) return 1;
    }
    return 0;
}
static double ml_jailbreak_threshold = 0.5;
static char custom_jailbreak_patterns[64][256];
static int custom_jailbreak_count = 0;
static double token_anomaly_threshold = 3.0;
static double membership_epsilon = 1.0;
#define SNEPPX_EXTRACT_MAX_RULES 32
#define SNEPPX_EXTRACT_RULE_LEN 128
static char extract_rules[SNEPPX_EXTRACT_MAX_RULES][SNEPPX_EXTRACT_RULE_LEN];
static int extract_rule_count = 0;
static double watermark_strength = 0.001;
static double adversarial_epsilon = 0.01;
static double factuality_threshold = 0.6;
static SNEPPXSemanticInjectionDetector* last_sid_stats = 0;
static void text_to_embedding(const char* text, size_t len, double* emb) {
    memset(emb,0,8*sizeof(double));
    if(!text||len==0) return;
    double h[8]={0};
    for(size_t i=0;i<len;i++) {
        unsigned char c=(unsigned char)text[i];
        h[i%8]+=(double)c;
        h[(i+1)%8]+=sin((double)c)*((double)i+1.0);
    }
    double norm=0;
    for(int j=0;j<8;j++){h[j]/=(double)(len+1);norm+=h[j]*h[j];}
    norm=sqrt(norm);
    if(norm>1e-10) for(int j=0;j<8;j++) emb[j]=h[j]/norm;
}
/**
 * @brief Perform Semantic Injection Add Attack Text.
 *
 * @param sid [out] Sid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_add_attack_text(SNEPPXSemanticInjectionDetector* sid, const char* text) {
    if(!sid||!text) return -1;
    double emb[8]; text_to_embedding(text,strlen(text),emb);
    return SNEPPX_semantic_injection_add_attack(sid,emb);
}
/**
 * @brief Perform Semantic Injection Remove Attack.
 *
 * @param sid [out] Sid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_remove_attack(SNEPPXSemanticInjectionDetector* sid, int index) {
    if(!sid||index<0||index>=sid->attack_count) return -1;
    for(int i=index;i<sid->attack_count-1;i++) memcpy(sid->known_attack_embeddings[i],sid->known_attack_embeddings[i+1],8*sizeof(double));
    sid->attack_count--; return 0;
}
/**
 * @brief Clear Semantic Injection.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_clear(SNEPPXSemanticInjectionDetector* sid) {
    if(!sid) return -1;
    memset(sid->known_attack_embeddings,0,sizeof(sid->known_attack_embeddings));
    sid->attack_count=0; return 0;
}
/**
 * @brief Perform Semantic Injection Get Attack Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_get_attack_count(SNEPPXSemanticInjectionDetector* sid) {
    if(!sid) return -1; return sid->attack_count;
}
/**
 * @brief Perform Semantic Injection Get Stats.
 *
 * @param sid [out] Sid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_get_stats(SNEPPXSemanticInjectionDetector* sid, double* stats) {
    if(!sid||!stats) return -1;
    stats[0]=(double)sid->attack_count; stats[1]=sid->threshold; stats[2]=0.0; return 0;
}
/**
 * @brief Perform Ml Jailbreak Set Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_set_threshold(double threshold) {
    if(threshold<0.0||threshold>1.0) return -1;
    ml_jailbreak_threshold=threshold; return 0;
}
/**
 * @brief Perform Ml Jailbreak Add Pattern.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_add_pattern(const char* pattern) {
    if(!pattern||custom_jailbreak_count>=64) return -1;
    strncpy(custom_jailbreak_patterns[custom_jailbreak_count],pattern,255);
    custom_jailbreak_patterns[custom_jailbreak_count][255]=0;
    custom_jailbreak_count++; return 0;
}
/**
 * @brief Perform Ml Jailbreak Remove Pattern.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_remove_pattern(const char* pattern) {
    if(!pattern) return -1;
    for(int i=0;i<custom_jailbreak_count;i++) {
        if(strcmp(custom_jailbreak_patterns[i],pattern)==0) {
            for(int j=i;j<custom_jailbreak_count-1;j++) strncpy(custom_jailbreak_patterns[j],custom_jailbreak_patterns[j+1],255);
            custom_jailbreak_count--; return 0;
        }
    }
    return -1;
}
/**
 * @brief Perform Ml Jailbreak Get Pattern Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_get_pattern_count(void) { return custom_jailbreak_count; }
/**
 * @brief Perform Ml Jailbreak Scan Multi.
 *
 * @param texts [in] Texts value.
 * @param lens [in] Lens value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_scan_multi(const char** texts, const size_t* lens, int count, int* results) {
    if(!texts||!lens||!results||count<=0) return -1;
    for(int i=0;i<count;i++) {
        results[i]=SNEPPX_ml_jailbreak_detect(texts[i],lens[i]);
        if(!results[i]) {
            char lower[1024]; size_t clen=lens[i]<1023?lens[i]:1023;
            for(size_t j=0;j<clen;j++) lower[j]=(char)tolower((unsigned char)texts[i][j]);
            lower[clen]=0;
            for(int p=0;p<custom_jailbreak_count;p++) { if(strstr(lower,custom_jailbreak_patterns[p])){results[i]=1;break;} }
        }
    }
    return 0;
}
/**
 * @brief Perform Encoded Attack Decode Base64.
 *
 * @param input [in] Input value.
 * @param in_len [in] In Len value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_decode_base64(const char* input, size_t in_len, char* output, size_t* out_len) {
    if(!input||!output||!out_len) return -1;
    unsigned char buf[4096]; size_t rlen=b64_decode(input,in_len,buf);
    if(rlen>4096) rlen=4096; memcpy(output,buf,rlen); *out_len=rlen; return 0;
}
/**
 * @brief Perform Encoded Attack Decode Hex.
 *
 * @param input [in] Input value.
 * @param in_len [in] In Len value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_decode_hex(const char* input, size_t in_len, char* output, size_t* out_len) {
    if(!input||!output||!out_len) return -1;
    if(in_len>=2&&input[0]=='0'&&(input[1]=='x'||input[1]=='X')){input+=2;in_len-=2;}
    unsigned char buf[4096]; size_t rlen=hex_decode(input,in_len,buf);
    if(rlen>4096) rlen=4096; memcpy(output,buf,rlen); *out_len=rlen; return 0;
}
/**
 * @brief Perform Encoded Attack Decode Rot13.
 *
 * @param input [in] Input value.
 * @param in_len [in] In Len value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_decode_rot13(const char* input, size_t in_len, char* output, size_t* out_len) {
    if(!input||!output||!out_len) return -1;
    *out_len=rot13_decode(input,in_len,output); return 0;
}
/**
 * @brief Perform Encoded Attack Scan Deep.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_scan_deep(const char* text, size_t len, int depth) {
    if(!text) return 0;
    if(depth<=0) return SNEPPX_encoded_attack_scan(text,len);
    char decoded[4096]; size_t dlen=0;
    if(SNEPPX_encoded_attack_decode(text,len,decoded,&dlen)==0&&dlen>0) {
        if(SNEPPX_encoded_attack_scan(decoded,dlen)) return 1;
        return SNEPPX_encoded_attack_scan_deep(decoded,dlen,depth-1);
    }
    return SNEPPX_encoded_attack_scan(text,len);
}
/**
 * @brief Perform Token Anomaly Set Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_token_anomaly_set_threshold(double t) { if(t<=0.0) return -1; token_anomaly_threshold=t; return 0; }
/**
 * @brief Perform Token Anomaly Get Threshold.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_token_anomaly_get_threshold(void) { return token_anomaly_threshold; }
/**
 * @brief Perform Token Anomaly Batch Score.
 *
 * @param sequences [in] Sequences value.
 * @param seq_count [in] Seq Count value.
 * @param seq_lens [in] Seq Lens value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_token_anomaly_batch_score(const uint32_t** sequences, int seq_count, const size_t* seq_lens, double* scores) {
    if(!sequences||!seq_lens||!scores||seq_count<=0) return -1;
    for(int i=0;i<seq_count;i++) scores[i]=SNEPPX_token_anomaly_score(sequences[i],seq_lens[i],NULL);
    return 0;
}
/**
 * @brief Perform Model Inversion Set Noise.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_inversion_set_noise(double scale) {
    if(scale<0.0) return -1;
    SNEPPXModelInversionDefense mid; SNEPPX_model_inversion_init(&mid); mid.noise_scale=scale; return 0;
}
/**
 * @brief Perform Model Inversion Set Clip.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_inversion_set_clip(double norm) {
    if(norm<=0.0) return -1;
    SNEPPXModelInversionDefense mid; SNEPPX_model_inversion_init(&mid); mid.clip_norm=norm; return 0;
}
/**
 * @brief Perform Model Inversion Get Config.
 *
 * @param noise [out] Noise value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_inversion_get_config(double* noise, double* clip) {
    if(!noise||!clip) return -1;
    SNEPPXModelInversionDefense mid; SNEPPX_model_inversion_init(&mid);
    *noise=mid.noise_scale; *clip=mid.clip_norm; return 0;
}
/**
 * @brief Apply Membership Inference Defense.
 *
 * @param logits [out] Logits value.
 * @param count [in] Count value.
 * @param epsilon [in] Epsilon value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_membership_inference_defense_apply(double* logits, size_t count, double epsilon, double* clipped_out) {
    if(!logits||!clipped_out||count==0) return -1;
    double scale=epsilon>0?1.0/epsilon:1.0;
    for(size_t i=0;i<count;i++) {
        double v=logits[i];
        if(v<-epsilon) v=-epsilon; if(v>epsilon) v=epsilon;
        double u=(double)rand()/(double)RAND_MAX;
        double lap=u<0.5?scale*log(2.0*u):-scale*log(2.0*(1.0-u));
        clipped_out[i]=v+lap;
    }
    return 0;
}
/**
 * @brief Perform Membership Inference Defense Set Epsilon.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_membership_inference_defense_set_epsilon(double e) { if(e<=0.0) return -1; membership_epsilon=e; return 0; }
/**
 * @brief Perform Data Extraction Prevent Set Rules.
 *
 * @param rules [in] Rules value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_extraction_prevent_set_rules(const char** rules, int count) {
    if(!rules||count<=0||count>SNEPPX_EXTRACT_MAX_RULES) return -1;
    extract_rule_count=count;
    for(int i=0;i<count;i++){strncpy(extract_rules[i],rules[i],SNEPPX_EXTRACT_RULE_LEN-1);extract_rules[i][SNEPPX_EXTRACT_RULE_LEN-1]=0;}
    return 0;
}
/**
 * @brief Perform Data Extraction Prevent Get Rules.
 *
 * @param rules [out] Rules value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_extraction_prevent_get_rules(char** rules, int max) {
    if(!rules||max<=0) return -1;
    int out=extract_rule_count<max?extract_rule_count:max;
    for(int i=0;i<out;i++){size_t slen=strlen(extract_rules[i])+1;rules[i]=(char*)malloc(slen);if(rules[i])memcpy(rules[i],extract_rules[i],slen);}
    return out;
}
/**
 * @brief Perform Training Sanitize Emails.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_training_sanitize_emails(const char* text, size_t len, char* out, size_t* out_len) {
    if(!text||!out||!out_len) return -1;
    size_t cap=*out_len; *out_len=0; size_t pos=0;
    const char redacted[]="[REDACTED]"; size_t rlen=strlen(redacted);
    for(size_t i=0;i<len&&pos<cap;i++) {
        if(text[i]=='@') {
            size_t start=i; while(start>0&&text[start-1]!=' '&&text[start-1]!='\t'&&text[start-1]!='\n'&&text[start-1]!='\r') start--;
            size_t end=i; while(end<len&&text[end]!=' '&&text[end]!='\t'&&text[end]!='\n'&&text[end]!='\r') end++;
            if(pos+rlen<=cap){memcpy(out+pos,redacted,rlen);pos+=rlen;}
            i=end; if(i>0) i--;
        } else { if(pos<cap) out[pos++]=text[i]; }
    }
    *out_len=pos; if(pos<cap) out[pos]=0; return 0;
}
/**
 * @brief Perform Training Sanitize Phones.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_training_sanitize_phones(const char* text, size_t len, char* out, size_t* out_len) {
    if(!text||!out||!out_len) return -1;
    size_t cap=*out_len; *out_len=0; size_t pos=0;
    const char redacted[]="[REDACTED]"; size_t rlen=strlen(redacted);
    for(size_t i=0;i<len&&pos<cap;i++) {
        size_t dc=0; for(size_t j=i;j<len&&j<i+20&&text[j]>='0'&&text[j]<='9';j++) dc++;
        if(dc>=7) {
            size_t end=i; while(end<len&&((text[end]>='0'&&text[end]<='9')||text[end]=='-'||text[end]=='('||text[end]==')'||text[end]==' '||text[end]=='+')) end++;
            if(pos+rlen<=cap){memcpy(out+pos,redacted,rlen);pos+=rlen;}
            i=end; if(i>0) i--;
        } else { if(pos<cap) out[pos++]=text[i]; }
    }
    *out_len=pos; if(pos<cap) out[pos]=0; return 0;
}
/**
 * @brief Perform Training Sanitize Ips.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_training_sanitize_ips(const char* text, size_t len, char* out, size_t* out_len) {
    if(!text||!out||!out_len) return -1;
    size_t cap=*out_len; *out_len=0; size_t pos=0;
    const char redacted[]="[REDACTED]"; size_t rlen=strlen(redacted);
    for(size_t i=0;i<len&&pos<cap;i++) {
        int dots=0; size_t j=i;
        while(j<len&&(isdigit((unsigned char)text[j])||text[j]=='.')){if(text[j]=='.')dots++;j++;}
        if(dots==3&&j-i>=7){if(pos+rlen<=cap){memcpy(out+pos,redacted,rlen);pos+=rlen;}i=j;if(i>0)i--;}
        else{if(pos<cap)out[pos++]=text[i];}
    }
    *out_len=pos; if(pos<cap) out[pos]=0; return 0;
}
/**
 * @brief Perform Watermark Set Key.
 *
 * @param mw [out] Mw value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_set_key(SNEPPXModelWatermark* mw, const uint8_t* key, size_t key_len) {
    if(!mw||!key) return -1;
    size_t cp=key_len<32?key_len:32; memset(mw->watermark,0,32); memcpy(mw->watermark,key,cp); return 0;
}
/**
 * @brief Perform Watermark Get Strength.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_watermark_get_strength(SNEPPXModelWatermark* mw) { if(!mw) return 0.0; (void)mw; return watermark_strength; }
/**
 * @brief Perform Watermark Set Strength.
 *
 * @param mw [out] Mw value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_set_strength(SNEPPXModelWatermark* mw, double strength) { if(!mw||strength<0.0) return -1; watermark_strength=strength; (void)mw; return 0; }
/**
 * @brief Perform Watermark Detect.
 *
 * @param weights [in] Weights value.
 * @param count [in] Count value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_detect(const double* weights, size_t count, const uint8_t* key, size_t key_len) {
    if(!weights||!key||count==0) return -1;
    double dot=0.0,n1=0.0,n2=0.0;
    for(size_t i=0;i<count;i++) {
        double expected=0.001*sin((double)(key[i%key_len])*(double)i);
        double observed=weights[i]; double residual=observed-expected;
        dot+=expected*residual; n1+=expected*expected; n2+=residual*residual;
    }
    double denom=sqrt(n1)*sqrt(n2); if(denom<1e-10) return 0;
    return (dot/denom)>0.7?1:0;
}
/**
 * @brief Perform Adversarial Smooth Batch.
 *
 * @param inputs [out] Inputs value.
 * @param count [in] Count value.
 * @param dim [in] Dim value.
 * @param epsilon [in] Epsilon value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_adversarial_smooth_batch(double** inputs, int count, size_t dim, double epsilon, double** outputs) {
    if(!inputs||!outputs||count<=0||dim==0) return -1;
    for(int i=0;i<count;i++){if(!inputs[i]||!outputs[i])return -1;memcpy(outputs[i],inputs[i],dim*sizeof(double));SNEPPX_adversarial_smooth(outputs[i],dim,epsilon);}
    return 0;
}
/**
 * @brief Perform Adversarial Smooth Set Epsilon.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_adversarial_smooth_set_epsilon(double eps) { if(eps<0.0) return -1; adversarial_epsilon=eps; return 0; }
/**
 * @brief Perform Factuality Score Compare.
 *
 * @param statement [in] Statement value.
 * @param statement [in] Statement value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_factuality_score_compare(const char* statement, const char* reference) { return SNEPPX_factuality_score(statement,reference); }
/**
 * @brief Perform Factuality Score Set Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_factuality_score_set_threshold(double t) { if(t<0.0||t>1.0) return -1; factuality_threshold=t; return 0; }
/**
 * @brief Perform Bias Measure Demographic Parity.
 *
 * @param predictions [in] Predictions value.
 * @param sensitive [in] Sensitive value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bias_measure_demographic_parity(const double* predictions, const int* sensitive, size_t n) {
    if(!predictions||!sensitive||n==0) return -1;
    double sum0=0,sum1=0; int cnt0=0,cnt1=0;
    for(size_t i=0;i<n;i++){if(sensitive[i]==0){sum0+=predictions[i];cnt0++;}else{sum1+=predictions[i];cnt1++;}}
    double mean0=cnt0>0?sum0/(double)cnt0:0.0; double mean1=cnt1>0?sum1/(double)cnt1:0.0;
    return (int)((mean1-mean0)*1000);
}
/**
 * @brief Perform Bias Measure Equalized Odds.
 *
 * @param predictions [in] Predictions value.
 * @param labels [in] Labels value.
 * @param sensitive [in] Sensitive value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bias_measure_equalized_odds(const double* predictions, const int* labels, const int* sensitive, size_t n) {
    if(!predictions||!labels||!sensitive||n==0) return -1;
    double tpr0s=0,tpr1s=0,fpr0s=0,fpr1s=0; int tpr0c=0,tpr1c=0,fpr0c=0,fpr1c=0;
    for(size_t i=0;i<n;i++){int p=predictions[i]>0.5?1:0;if(sensitive[i]==0){if(labels[i]==1){tpr0s+=p;tpr0c++;}else{fpr0s+=p;fpr0c++;}}else{if(labels[i]==1){tpr1s+=p;tpr1c++;}else{fpr1s+=p;fpr1c++;}}}
    double tpr0=tpr0c>0?tpr0s/(double)tpr0c:0.0;double tpr1=tpr1c>0?tpr1s/(double)tpr1c:0.0;
    double fpr0=fpr0c>0?fpr0s/(double)fpr0c:0.0;double fpr1=fpr1c>0?fpr1s/(double)fpr1c:0.0;
    return (int)((fabs(tpr0-tpr1)+fabs(fpr0-fpr1))*1000);
}
/**
 * @brief Perform Bias Get Report.
 *
 * @param bm [out] Bm value.
 * @param report [out] Report value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bias_get_report(SNEPPXBiasMetrics* bm, char* report, size_t report_size) {
    if(!bm||!report||report_size==0) return -1;
    int n=snprintf(report,report_size,"DP=%.4f EO=%.4f measured=%d",bm->demographic_parity,bm->equalized_odds,bm->measured);
    return (n<0)?-1:((size_t)n<report_size?n:(int)(report_size-1));
}
/**
 * @brief Remove Prompt Policy.
 *
 * @param pp [out] Pp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_remove(SNEPPXPromptPolicy* pp, int index) {
    if(!pp||index<0||index>=pp->policy_count) return -1;
    for(int i=index;i<pp->policy_count-1;i++) strncpy(pp->policies[i],pp->policies[i+1],255);
    pp->policy_count--; return 0;
}
/**
 * @brief Clear Prompt Policy.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_clear(SNEPPXPromptPolicy* pp) {
    if(!pp) return -1; memset(pp->policies,0,sizeof(pp->policies)); pp->policy_count=0; return 0;
}
/**
 * @brief Perform Prompt Policy Get Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_get_count(SNEPPXPromptPolicy* pp) { if(!pp) return -1; return pp->policy_count; }
/**
 * @brief Perform Prompt Policy Get Policies.
 *
 * @param pp [out] Pp value.
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_get_policies(SNEPPXPromptPolicy* pp, char* buffer, int max) {
    if(!pp||!buffer||max<=0) return -1; int pos=0;
    for(int i=0;i<pp->policy_count&&pos<max-1;i++) {
        size_t slen=strlen(pp->policies[i]); size_t cp=(slen<(size_t)(max-1-pos))?slen:(size_t)(max-1-pos);
        memcpy(buffer+pos,pp->policies[i],cp); pos+=(int)cp; if(pos<max-1) buffer[pos++]=';';
    }
    if(pos<max) buffer[pos]=0; return pos;
}
/**
 * @brief Perform Semantic Injection Add Attack Batch.
 *
 * @param sid [out] Sid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_add_attack_batch(SNEPPXSemanticInjectionDetector* sid, const double embeddings[][8], int count) {
    if(!sid||!embeddings||count<=0) return -1;
    int added=0;
    for(int i=0;i<count&&sid->attack_count<SNEPPX_S5_MAX_EMBEDDING;i++) {
        memcpy(sid->known_attack_embeddings[sid->attack_count],embeddings[i],8*sizeof(double));
        sid->attack_count++; added++;
    }
    return added;
}
/**
 * @brief Perform Semantic Injection Set Threshold.
 *
 * @param sid [out] Sid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_set_threshold(SNEPPXSemanticInjectionDetector* sid, double t) {
    if(!sid||t<0.0||t>1.0) return -1;
    sid->threshold=t; return 0;
}
/**
 * @brief Perform Semantic Injection Get Threshold.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_semantic_injection_get_threshold(SNEPPXSemanticInjectionDetector* sid) {
    if(!sid) return -1.0; return sid->threshold;
}
/**
 * @brief Perform Semantic Injection Find Closest.
 *
 * @param sid [out] Sid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_find_closest(SNEPPXSemanticInjectionDetector* sid, const double embedding[8], double* best_score) {
    if(!sid||!embedding||!best_score) return -1;
    double max_sim=0.0; int best_idx=-1;
    for(int i=0;i<sid->attack_count;i++) {
        double dot=0.0,n1=0.0,n2=0.0;
        for(int j=0;j<8;j++){dot+=embedding[j]*sid->known_attack_embeddings[i][j];n1+=embedding[j]*embedding[j];n2+=sid->known_attack_embeddings[i][j]*sid->known_attack_embeddings[i][j];}
        double sim=dot/(sqrt(n1)*sqrt(n2)+1e-10);
        if(sim>max_sim){max_sim=sim;best_idx=i;}
    }
    *best_score=max_sim; return best_idx;
}
/**
 * @brief Perform Ml Jailbreak Helper Scan.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
static int SNEPPX_ml_jailbreak_helper_scan(const char* text, size_t len) {
    if(!text) return 0;
    if(SNEPPX_ml_jailbreak_detect(text,len)) return 1;
    char lower[1024]; size_t clen=len<1023?len:1023;
    for(size_t j=0;j<clen;j++) lower[j]=(char)tolower((unsigned char)text[j]);
    lower[clen]=0;
    for(int p=0;p<custom_jailbreak_count;p++){if(strstr(lower,custom_jailbreak_patterns[p]))return 1;}
    return 0;
}
/**
 * @brief Perform Ml Jailbreak Scan Advanced.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_scan_advanced(const char* text, size_t len, double* confidence) {
    if(!text||!confidence) return -1;
    *confidence=0.0; int ml=SNEPPX_ml_jailbreak_detect(text,len);
    if(ml){*confidence=0.9;return 1;}
    char lower[1024]; size_t clen=len<1023?len:1023;
    for(size_t j=0;j<clen;j++) lower[j]=(char)tolower((unsigned char)text[j]);
    lower[clen]=0;
    int custom_matches=0;
    for(int p=0;p<custom_jailbreak_count;p++){if(strstr(lower,custom_jailbreak_patterns[p]))custom_matches++;}
    if(custom_matches>0){*confidence=0.5+0.1*custom_matches;if(*confidence>1.0)*confidence=1.0;return 1;}
    return 0;
}
/**
 * @brief Perform Ml Jailbreak Get Pattern At.
 *
 * @param index [in] Index value.
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_get_pattern_at(int index, char* buffer, size_t buf_size) {
    if(index<0||index>=custom_jailbreak_count||!buffer||buf_size==0) return -1;
    strncpy(buffer,custom_jailbreak_patterns[index],buf_size-1); buffer[buf_size-1]=0;
    return (int)strlen(buffer);
}
/**
 * @brief Perform Ml Jailbreak Clear Patterns.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_clear_patterns(void) { custom_jailbreak_count=0; return 0; }
/**
 * @brief Perform Ml Jailbreak Scan Encoded.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_scan_encoded(const char* text, size_t len) {
    if(!text) return 0;
    if(SNEPPX_ml_jailbreak_helper_scan(text,len)) return 1;
    char decoded[4096]; size_t dlen=0;
    if(SNEPPX_encoded_attack_decode(text,len,decoded,&dlen)==0&&dlen>0) return SNEPPX_ml_jailbreak_helper_scan(decoded,dlen);
    return 0;
}
/**
 * @brief Perform Encoded Attack Decode Auto.
 *
 * @param input [in] Input value.
 * @param in_len [in] In Len value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_decode_auto(const char* input, size_t in_len, char* output, size_t* out_len) {
    if(!input||!output||!out_len) return -1;
    if(in_len>=2&&input[0]=='0'&&(input[1]=='x'||input[1]=='X')) return SNEPPX_encoded_attack_decode_hex(input,in_len,output,out_len);
    if(in_len%4==0){for(size_t i=0;i<in_len;i++){char c=input[i];if(!((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='+'||c=='/'||c=='='))goto try_rot13;}return SNEPPX_encoded_attack_decode_base64(input,in_len,output,out_len);}
    try_rot13: return SNEPPX_encoded_attack_decode_rot13(input,in_len,output,out_len);
}
/**
 * @brief Perform Encoded Attack Scan All Encodings.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_scan_all_encodings(const char* text, size_t len) {
    if(!text) return 0;
    if(SNEPPX_encoded_attack_scan(text,len)) return 1;
    char decoded[4096]; size_t dlen=0;
    if(SNEPPX_encoded_attack_decode_auto(text,len,decoded,&dlen)==0&&dlen>0) return SNEPPX_encoded_attack_scan(decoded,dlen);
    return 0;
}
/**
 * @brief Perform Encoded Attack Get Decoded Length.
 *
 * @param input [in] Input value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_get_decoded_length(const char* input, size_t in_len) {
    if(!input) return -1;
    if(in_len>=2&&input[0]=='0'&&(input[1]=='x'||input[1]=='X')) return (int)((in_len-2)/2);
    if(in_len%4==0) return (int)(in_len/4*3);
    return (int)in_len;
}
static double token_anomaly_histogram[100];
static int token_anomaly_hist_count=0;
/**
 * @brief Perform Token Anomaly Record Score.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_token_anomaly_record_score(double score) {
    if(token_anomaly_hist_count<100){token_anomaly_histogram[token_anomaly_hist_count++]=score;return 0;}
    return -1;
}
/**
 * @brief Perform Token Anomaly Get Histogram.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_token_anomaly_get_histogram(double* buffer, int max) {
    if(!buffer||max<=0) return -1;
    int out=token_anomaly_hist_count<max?token_anomaly_hist_count:max;
    memcpy(buffer,token_anomaly_histogram,out*sizeof(double)); return out;
}
/**
 * @brief Perform Token Anomaly Reset Histogram.
 */
void SNEPPX_token_anomaly_reset_histogram(void) { token_anomaly_hist_count=0; }
/**
 * @brief Perform Token Anomaly Compute Perplexity.
 *
 * @param probs [in] Probs value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_token_anomaly_compute_perplexity(const double* probs, size_t count) {
    if(!probs||count==0) return 0.0;
    double sum_log=0.0;
    for(size_t i=0;i<count;i++){double p=probs[i];if(p<=0.0)p=1e-10;if(p>1.0)p=1.0;sum_log+=-log(p);}
    return sum_log/(double)count;
}
/**
 * @brief Perform Model Inversion Defend Gradients.
 *
 * @param gradients [out] Gradients value.
 * @param grad_count [in] Grad Count value.
 * @param noise [in] Noise value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_inversion_defend_gradients(double* gradients, size_t grad_count, double noise, double clip) {
    if(!gradients||grad_count==0) return -1;
    double norm=0.0; for(size_t i=0;i<grad_count;i++) norm+=gradients[i]*gradients[i];
    norm=sqrt(norm);
    if(norm>clip) for(size_t i=0;i<grad_count;i++) gradients[i]=gradients[i]/norm*clip;
    for(size_t i=0;i<grad_count;i++) gradients[i]+=((double)rand()/RAND_MAX-0.5)*noise;
    return 0;
}
/**
 * @brief Perform Model Inversion Get Noise Scale.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_inversion_get_noise_scale(void) {
    SNEPPXModelInversionDefense mid; SNEPPX_model_inversion_init(&mid);
    return (int)(mid.noise_scale*1000);
}
/**
 * @brief Perform Model Inversion Get Clip Norm.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_inversion_get_clip_norm(void) {
    SNEPPXModelInversionDefense mid; SNEPPX_model_inversion_init(&mid);
    return (int)(mid.clip_norm*100);
}
/**
 * @brief Perform Membership Inference Defense Get Epsilon.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_membership_inference_defense_get_epsilon(void) { return membership_epsilon; }
/**
 * @brief Perform Membership Inference Defense Apply Batch.
 *
 * @param logits_batch [out] Logits Batch value.
 * @param batch_size [in] Batch Size value.
 * @param logit_count [in] Logit Count value.
 * @param epsilon [in] Epsilon value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_membership_inference_defense_apply_batch(double** logits_batch, size_t batch_size, size_t logit_count, double epsilon, double** clipped_batch) {
    if(!logits_batch||!clipped_batch||batch_size==0) return -1;
    for(size_t b=0;b<batch_size;b++) SNEPPX_membership_inference_defense_apply(logits_batch[b],logit_count,epsilon,clipped_batch[b]);
    return 0;
}
/**
 * @brief Perform Data Extraction Prevent Add Rule.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_extraction_prevent_add_rule(const char* rule) {
    if(!rule||extract_rule_count>=SNEPPX_EXTRACT_MAX_RULES) return -1;
    strncpy(extract_rules[extract_rule_count],rule,SNEPPX_EXTRACT_RULE_LEN-1);
    extract_rules[extract_rule_count][SNEPPX_EXTRACT_RULE_LEN-1]=0;
    extract_rule_count++; return 0;
}
/**
 * @brief Perform Data Extraction Prevent Remove Rule.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_extraction_prevent_remove_rule(const char* rule) {
    if(!rule) return -1;
    for(int i=0;i<extract_rule_count;i++){if(strcmp(extract_rules[i],rule)==0){for(int j=i;j<extract_rule_count-1;j++)strncpy(extract_rules[j],extract_rules[j+1],SNEPPX_EXTRACT_RULE_LEN-1);extract_rule_count--;return 0;}}
    return -1;
}
/**
 * @brief Perform Data Extraction Prevent Clear Rules.
 *
 * @param extract_rules [in] Extract Rules value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_extraction_prevent_clear_rules(void) { extract_rule_count=0; memset(extract_rules,0,sizeof(extract_rules)); return 0; }
/**
 * @brief Perform Data Extraction Prevent Get Rule Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_extraction_prevent_get_rule_count(void) { return extract_rule_count; }
/**
 * @brief Perform Data Extraction Prevent Check Custom.
 *
 * @param output [in] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_extraction_prevent_check_custom(const char* output, size_t len) {
    if(!output||len==0) return 0;
    char lower[4096]; size_t clen=len<4095?len:4095;
    for(size_t i=0;i<clen;i++) lower[i]=(char)tolower((unsigned char)output[i]);
    lower[clen]=0;
    for(int i=0;i<extract_rule_count;i++){if(strstr(lower,extract_rules[i])) return 1;}
    return 0;
}
/**
 * @brief Perform Training Sanitize Batch.
 *
 * @param texts [in] Texts value.
 * @param lens [in] Lens value.
 * @param count [in] Count value.
 * @param outputs [out] Outputs value.
 * @param output_lens [out] Output Lens value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_training_sanitize_batch(const char** texts, const size_t* lens, int count, char** outputs, size_t* output_lens, size_t* output_caps) {
    if(!texts||!lens||!outputs||!output_lens||!output_caps||count<=0) return -1;
    for(int i=0;i<count;i++){if(SNEPPX_training_sanitize(texts[i],lens[i],outputs[i],&output_caps[i])!=0) return -1;output_lens[i]=output_caps[i];}
    return 0;
}
/**
 * @brief Perform Training Sanitize Urls.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_training_sanitize_urls(const char* text, size_t len, char* out, size_t* out_len) {
    if(!text||!out||!out_len) return -1;
    size_t cap=*out_len; *out_len=0; size_t pos=0;
    const char redacted[]="[REDACTED]"; size_t rlen=strlen(redacted);
    for(size_t i=0;i<len&&pos<cap;i++) {
        if(i+3<len&&text[i]=='h'&&text[i+1]=='t'&&text[i+2]=='t'&&text[i+3]=='p') {
            size_t end=i; while(end<len&&text[end]!=' '&&text[end]!='\t'&&text[end]!='\n'&&text[end]!='\r'&&text[end]!=',') end++;
            if(pos+rlen<=cap){memcpy(out+pos,redacted,rlen);pos+=rlen;}
            i=end; if(i>0) i--;
        } else { if(pos<cap) out[pos++]=text[i]; }
    }
    *out_len=pos; if(pos<cap) out[pos]=0; return 0;
}
/**
 * @brief Perform Training Sanitize Ssns.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_training_sanitize_ssns(const char* text, size_t len, char* out, size_t* out_len) {
    if(!text||!out||!out_len) return -1;
    size_t cap=*out_len; *out_len=0; size_t pos=0;
    const char redacted[]="[REDACTED]"; size_t rlen=strlen(redacted);
    for(size_t i=0;i<len&&pos<cap;i++) {
        if(i+10<len&&isdigit((unsigned char)text[i])&&isdigit((unsigned char)text[i+1])&&isdigit((unsigned char)text[i+2])&&text[i+3]=='-'&&isdigit((unsigned char)text[i+4])&&isdigit((unsigned char)text[i+5])&&text[i+6]=='-'&&isdigit((unsigned char)text[i+7])&&isdigit((unsigned char)text[i+8])&&isdigit((unsigned char)text[i+9])&&isdigit((unsigned char)text[i+10])) {
            if(pos+rlen<=cap){memcpy(out+pos,redacted,rlen);pos+=rlen;}
            i+=10; continue;
        }
        if(i+13<len&&isdigit((unsigned char)text[i])&&isdigit((unsigned char)text[i+1])&&isdigit((unsigned char)text[i+2])&&isdigit((unsigned char)text[i+3])&&text[i+4]=='-'&&isdigit((unsigned char)text[i+5])&&isdigit((unsigned char)text[i+6])&&isdigit((unsigned char)text[i+7])&&isdigit((unsigned char)text[i+8])&&text[i+9]=='-'&&isdigit((unsigned char)text[i+10])&&isdigit((unsigned char)text[i+11])&&isdigit((unsigned char)text[i+12])&&isdigit((unsigned char)text[i+13])) {
            if(pos+rlen<=cap){memcpy(out+pos,redacted,rlen);pos+=rlen;}
            i+=13; continue;
        }
        if(pos<cap) out[pos++]=text[i];
    }
    *out_len=pos; if(pos<cap) out[pos]=0; return 0;
}
/**
 * @brief Perform Watermark Embed With Key.
 *
 * @param mw [out] Mw value.
 * @param weights [out] Weights value.
 * @param weight_count [in] Weight Count value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_embed_with_key(SNEPPXModelWatermark* mw, double* weights, size_t weight_count, const uint8_t* key, size_t key_len) {
    if(!mw||!weights||!key||weight_count==0) return -1;
    SNEPPX_watermark_set_key(mw,key,key_len);
    return SNEPPX_watermark_embed(mw,weights,weight_count);
}
/**
 * @brief Perform Watermark Verify With Key.
 *
 * @param weights [in] Weights value.
 * @param weight_count [in] Weight Count value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_verify_with_key(const double* weights, size_t weight_count, const uint8_t* key, size_t key_len) {
    if(!weights||!key||weight_count==0) return -1;
    return SNEPPX_watermark_detect(weights,weight_count,key,key_len);
}
/**
 * @brief Perform Watermark Is Embedded.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_is_embedded(SNEPPXModelWatermark* mw) { if(!mw) return -1; return mw->embedded; }
/**
 * @brief Reset Watermark.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_reset(SNEPPXModelWatermark* mw) {
    if(!mw) return -1; memset(mw->watermark,0,32); mw->embedded=0; return 0;
}
/**
 * @brief Perform Adversarial Smooth Batch With Epsilon.
 *
 * @param inputs [out] Inputs value.
 * @param count [in] Count value.
 * @param dim [in] Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_adversarial_smooth_batch_with_epsilon(double** inputs, int count, size_t dim, double** outputs) {
    return SNEPPX_adversarial_smooth_batch(inputs,count,dim,adversarial_epsilon,outputs);
}
/**
 * @brief Perform Adversarial Smooth Get Epsilon.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_adversarial_smooth_get_epsilon(void) { return adversarial_epsilon; }
/**
 * @brief Perform Factuality Score Batch.
 *
 * @param statements [in] Statements value.
 * @param references [in] References value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_factuality_score_batch(const char** statements, const char** references, int count) {
    if(!statements||!references||count<=0) return 0.0;
    double sum=0.0;
    for(int i=0;i<count;i++) sum+=SNEPPX_factuality_score(statements[i],references[i]);
    return sum/(double)count;
}
/**
 * @brief Perform Factuality Score Check.
 *
 * @param statement [in] Statement value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_factuality_score_check(const char* statement, const char* reference) {
    double s=SNEPPX_factuality_score(statement,reference);
    return s>=factuality_threshold?1:0;
}
/**
 * @brief Perform Factuality Get Threshold.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_factuality_get_threshold(void) { return factuality_threshold; }
/**
 * @brief Perform Bias Measure Demographic Parity Detail.
 *
 * @param predictions [in] Predictions value.
 * @param sensitive [in] Sensitive value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bias_measure_demographic_parity_detail(const double* predictions, const int* sensitive, size_t n, double* parity_out) {
    if(!predictions||!sensitive||n==0||!parity_out) return -1;
    double sum0=0,sum1=0; int cnt0=0,cnt1=0;
    for(size_t i=0;i<n;i++){if(sensitive[i]==0){sum0+=predictions[i];cnt0++;}else{sum1+=predictions[i];cnt1++;}}
    double mean0=cnt0>0?sum0/(double)cnt0:0.0; double mean1=cnt1>0?sum1/(double)cnt1:0.0;
    *parity_out=mean1-mean0; return 0;
}
/**
 * @brief Perform Bias Measure Equalized Odds Detail.
 *
 * @param predictions [in] Predictions value.
 * @param labels [in] Labels value.
 * @param sensitive [in] Sensitive value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bias_measure_equalized_odds_detail(const double* predictions, const int* labels, const int* sensitive, size_t n, double* eo_out) {
    if(!predictions||!labels||!sensitive||n==0||!eo_out) return -1;
    double tpr0s=0,tpr1s=0,fpr0s=0,fpr1s=0; int tpr0c=0,tpr1c=0,fpr0c=0,fpr1c=0;
    for(size_t i=0;i<n;i++){int p=predictions[i]>0.5?1:0;if(sensitive[i]==0){if(labels[i]==1){tpr0s+=p;tpr0c++;}else{fpr0s+=p;fpr0c++;}}else{if(labels[i]==1){tpr1s+=p;tpr1c++;}else{fpr1s+=p;fpr1c++;}}}
    double tpr0=tpr0c>0?tpr0s/(double)tpr0c:0.0;double tpr1=tpr1c>0?tpr1s/(double)tpr1c:0.0;
    double fpr0=fpr0c>0?fpr0s/(double)fpr0c:0.0;double fpr1=fpr1c>0?fpr1s/(double)fpr1c:0.0;
    *eo_out=fabs(tpr0-tpr1)+fabs(fpr0-fpr1); return 0;
}
/**
 * @brief Perform Bias Measure Disparate Impact.
 *
 * @param predictions [in] Predictions value.
 * @param sensitive [in] Sensitive value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bias_measure_disparate_impact(const double* predictions, const int* sensitive, size_t n) {
    if(!predictions||!sensitive||n==0) return -1;
    double sum0=0,sum1=0; int cnt0=0,cnt1=0;
    for(size_t i=0;i<n;i++){if(sensitive[i]==0){sum0+=predictions[i];cnt0++;}else{sum1+=predictions[i];cnt1++;}}
    double mean0=cnt0>0?sum0/(double)cnt0:0.0; double mean1=cnt1>0?sum1/(double)cnt1:0.0;
    if(mean0<=0.0) return 0;
    return (int)((mean1/mean0)*100);
}
/**
 * @brief Perform Prompt Policy Add Batch.
 *
 * @param pp [out] Pp value.
 * @param rules [in] Rules value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_add_batch(SNEPPXPromptPolicy* pp, const char** rules, int count) {
    if(!pp||!rules||count<=0) return -1;
    int added=0;
    for(int i=0;i<count&&pp->policy_count<16;i++){strncpy(pp->policies[pp->policy_count],rules[i],255);pp->policies[pp->policy_count][255]=0;pp->policy_count++;added++;}
    return added;
}
/**
 * @brief Perform Prompt Policy Set Enabled.
 *
 * @param pp [out] Pp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_set_enabled(SNEPPXPromptPolicy* pp, int enabled) { if(!pp) return -1; pp->enabled=enabled; return 0; }
/**
 * @brief Perform Prompt Policy Is Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_is_enabled(SNEPPXPromptPolicy* pp) { if(!pp) return -1; return pp->enabled; }
/**
 * @brief Perform Prompt Policy Get Policy At.
 *
 * @param pp [out] Pp value.
 * @param index [in] Index value.
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_policy_get_policy_at(SNEPPXPromptPolicy* pp, int index, char* buffer, size_t buf_size) {
    if(!pp||!buffer||buf_size==0||index<0||index>=pp->policy_count) return -1;
    strncpy(buffer,pp->policies[index],buf_size-1); buffer[buf_size-1]=0;
    return (int)strlen(buffer);
}
/**
 * @brief Perform Semantic Injection Has Attacks.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_has_attacks(SNEPPXSemanticInjectionDetector* sid) {
    if(!sid) return -1; return sid->attack_count>0?1:0;
}
/**
 * @brief Perform Ml Jailbreak Detect Utf8.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_detect_utf8(const char* text, size_t len) {
    if(!text) return 0;
    return SNEPPX_ml_jailbreak_detect(text,len);
}
/**
 * @brief Perform Ml Jailbreak Get Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_get_threshold(void) { return (int)(ml_jailbreak_threshold*100); }
/**
 * @brief Perform Encoded Attack Decode Auto All.
 *
 * @param input [in] Input value.
 * @param in_len [in] In Len value.
 * @param output [out] Output value.
 * @param out_len [out] Out Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_decode_auto_all(const char* input, size_t in_len, char* output, size_t* out_len, int max_depth) {
    if(!input||!output||!out_len||max_depth<=0) return -1;
    char buf[4096]; size_t blen=0; int r=SNEPPX_encoded_attack_decode_auto(input,in_len,buf,&blen);
    if(r!=0||blen==0) return r;
    for(int d=1;d<max_depth;d++){char buf2[4096];size_t blen2=0;if(SNEPPX_encoded_attack_decode_auto(buf,blen,buf2,&blen2)!=0||blen2==0)break;memcpy(buf,buf2,blen2);blen=blen2;}
    memcpy(output,buf,blen); *out_len=blen; return 0;
}
static int token_anomaly_warning_count=0;
/**
 * @brief Perform Token Anomaly Check Warning.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_token_anomaly_check_warning(double score) {
    if(score>token_anomaly_threshold){token_anomaly_warning_count++;return 1;}
    return 0;
}
/**
 * @brief Perform Token Anomaly Get Warning Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_token_anomaly_get_warning_count(void) { return token_anomaly_warning_count; }
/**
 * @brief Perform Token Anomaly Reset Warnings.
 */
void SNEPPX_token_anomaly_reset_warnings(void) { token_anomaly_warning_count=0; }
/**
 * @brief Perform Model Inversion Set Noise Scale.
 *
 * @param mid [out] Mid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_inversion_set_noise_scale(SNEPPXModelInversionDefense* mid, double scale) {
    if(!mid||scale<0.0) return -1; mid->noise_scale=scale; return 0;
}
/**
 * @brief Perform Model Inversion Set Clip Norm.
 *
 * @param mid [out] Mid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_inversion_set_clip_norm(SNEPPXModelInversionDefense* mid, double norm) {
    if(!mid||norm<=0.0) return -1; mid->clip_norm=norm; return 0;
}
/**
 * @brief Perform Model Inversion Get Noise Scale From Mid.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_model_inversion_get_noise_scale_from_mid(SNEPPXModelInversionDefense* mid) {
    if(!mid) return -1.0; return mid->noise_scale;
}
/**
 * @brief Perform Model Inversion Get Clip Norm From Mid.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_model_inversion_get_clip_norm_from_mid(SNEPPXModelInversionDefense* mid) {
    if(!mid) return -1.0; return mid->clip_norm;
}
/**
 * @brief Perform Membership Inference Defense Apply With Eps.
 *
 * @param logits [out] Logits value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_membership_inference_defense_apply_with_eps(double* logits, size_t count, double* clipped_out) {
    return SNEPPX_membership_inference_defense_apply(logits,count,membership_epsilon,clipped_out);
}
/**
 * @brief Perform Training Sanitize All.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_training_sanitize_all(const char* text, size_t len, char* out, size_t* out_len) {
    if(!text||!out||!out_len) return -1;
    char buf1[8192],buf2[8192],buf3[8192],buf4[8192];size_t l1=8192,l2=8192,l3=8192,l4=8192;
    if(SNEPPX_training_sanitize_emails(text,len,buf1,&l1)!=0) return -1;
    if(SNEPPX_training_sanitize_phones(buf1,l1,buf2,&l2)!=0) return -1;
    if(SNEPPX_training_sanitize_urls(buf2,l2,buf3,&l3)!=0) return -1;
    if(SNEPPX_training_sanitize_ips(buf3,l3,out,out_len)!=0) return -1;
    return 0;
}
/**
 * @brief Perform Factuality Score Get Threshold.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_factuality_score_get_threshold(void) { return factuality_threshold; }
/**
 * @brief Perform Factuality Score Compare Threshold.
 *
 * @param statement [in] Statement value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_factuality_score_compare_threshold(const char* statement, const char* reference) {
    double s=SNEPPX_factuality_score(statement,reference);
    return s>=factuality_threshold?1:0;
}
/**
 * @brief Perform Adversarial Smooth Get Epsilon Int.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_adversarial_smooth_get_epsilon_int(void) { return (int)(adversarial_epsilon*10000); }
/**
 * @brief Apply Adversarial Smooth.
 *
 * @param input [out] Input value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_adversarial_smooth_apply(double* input, size_t input_dim) {
    return SNEPPX_adversarial_smooth(input,input_dim,adversarial_epsilon);
}
/**
 * @brief Perform Watermark Get Key.
 *
 * @param mw [out] Mw value.
 * @param key [out] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_get_key(SNEPPXModelWatermark* mw, uint8_t* key, size_t* key_len) {
    if(!mw||!key||!key_len) return -1;
    size_t cp=(*key_len<32)?*key_len:32; memcpy(key,mw->watermark,cp); *key_len=cp; return 0;
}
/**
 * @brief Perform Watermark Get Key Length.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_watermark_get_key_length(SNEPPXModelWatermark* mw) { if(!mw) return -1; return 32; }
/**
 * @brief Perform Bias Get Measured.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bias_get_measured(SNEPPXBiasMetrics* bm) { if(!bm) return -1; return bm->measured; }
/**
 * @brief Perform Semantic Injection Is Initialized.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_is_initialized(SNEPPXSemanticInjectionDetector* sid) { if(!sid) return 0; return 1; }
/**
 * @brief Perform Semantic Injection Get Capacity.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_injection_get_capacity(void) { return SNEPPX_S5_MAX_EMBEDDING; }
/**
 * @brief Perform Ml Jailbreak Detect Raw.
 *
 * @param text [in] Text value.
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_detect_raw(const char* text, size_t len) { return SNEPPX_ml_jailbreak_detect(text,len); }
/**
 * @brief Perform Ml Jailbreak Get Custom Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ml_jailbreak_get_custom_count(void) { return custom_jailbreak_count; }
/**
 * @brief Perform Encoded Attack Has Prefix.
 *
 * @param input [in] Input value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_has_prefix(const char* input, size_t in_len) {
    if(!input) return 0;
    if(in_len>=2&&input[0]=='0'&&(input[1]=='x'||input[1]=='X')) return 1;
    return 0;
}
/**
 * @brief Perform Encoded Attack Is Base64.
 *
 * @param input [in] Input value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_is_base64(const char* input, size_t in_len) {
    if(!input||in_len==0) return 0;
    for(size_t i=0;i<in_len;i++){char c=input[i];if(!((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='+'||c=='/'||c=='='))return 0;}
    return in_len%4==0?1:0;
}
/**
 * @brief Perform Encoded Attack Detect Encoding.
 *
 * @param input [in] Input value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_encoded_attack_detect_encoding(const char* input, size_t in_len) {
    if(!input||in_len==0) return 0;
    if(in_len>=2&&input[0]=='0'&&(input[1]=='x'||input[1]=='X')) return 1;
    if(SNEPPX_encoded_attack_is_base64(input,in_len)) return 2;
    return 3;
}
