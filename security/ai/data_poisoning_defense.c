#include "data_poisoning_defense.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define SNEPPX_POISON_ENSEMBLE_K 5
#define SNEPPX_POISON_MAX_SAMPLES 4096

/*
 * SNEPPX - Data Poisoning Defense
 *
 * WHAT
 *   Data Poisoning Defense.
 *
 * CONCEPT
 *   Provides the Data Poisoning Defense.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static double ensemble_means[SNEPPX_POISON_ENSEMBLE_K][SNEPPX_POISON_MAX_FEATURES];
static double ensemble_stds[SNEPPX_POISON_ENSEMBLE_K][SNEPPX_POISON_MAX_FEATURES];
static int ensemble_trained[SNEPPX_POISON_ENSEMBLE_K];
static int ensemble_count = 0;
static int poison_sample_count = 0;

/**
 * @brief Initialize Poison Detector.
 *
 * @param pd [out] Pd value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_init(SNEPPXPoisonDetector* pd, int feature_count) {
    if (!pd || feature_count <= 0 || feature_count > SNEPPX_POISON_MAX_FEATURES) return -1;
    pd->feature_means = (double*)calloc(feature_count, sizeof(double));
    pd->feature_stds = (double*)calloc(feature_count, sizeof(double));
    if (!pd->feature_means || !pd->feature_stds) {
        free(pd->feature_means); free(pd->feature_stds);
        return -1;
    }
    pd->feature_count = feature_count;
    pd->outlier_threshold = 3.0;
    pd->trained = 0;
    return 0;
}

/**
 * @brief Destroy Poison Detector.
 */
void SNEPPX_poison_detector_destroy(SNEPPXPoisonDetector* pd) {
    if (pd) {
        free(pd->feature_means);
        free(pd->feature_stds);
        memset(pd, 0, sizeof(*pd));
    }
}

/**
 * @brief Train Poison Detector.
 *
 * @param pd [out] Pd value.
 * @param samples [in] Samples value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_train(SNEPPXPoisonDetector* pd,
                                 const double* samples, int sample_count) {
    if (!pd || !samples || sample_count < 2) return -1;
    int f = pd->feature_count;
    for (int j = 0; j < f; j++) {
        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < sample_count; i++) {
            double v = samples[i * f + j];
            sum += v;
            sum_sq += v * v;
        }
        pd->feature_means[j] = sum / sample_count;
        double var = (sum_sq / sample_count) - (pd->feature_means[j] * pd->feature_means[j]);
        pd->feature_stds[j] = (var > 0) ? sqrt(var) : 1.0;
    }
    pd->trained = 1;
    poison_sample_count = sample_count;
    if (ensemble_count < SNEPPX_POISON_ENSEMBLE_K) {
        for (int j = 0; j < f; j++) {
            ensemble_means[ensemble_count][j] = pd->feature_means[j];
            ensemble_stds[ensemble_count][j] = pd->feature_stds[j];
        }
        ensemble_trained[ensemble_count] = 1;
        ensemble_count++;
    }
    return 0;
}

/**
 * @brief Perform Poison Detector Score.
 *
 * @param pd [out] Pd value.
 * @param sample [in] Sample value.
 * @param feature_count [in] Feature Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_score(SNEPPXPoisonDetector* pd,
                                 const double* sample, int feature_count,
                                 double* outlier_score) {
    if (!pd || !pd->trained || !sample || !outlier_score) return -1;
    if (feature_count != pd->feature_count) return -1;
    double max_z = 0.0;
    for (int j = 0; j < feature_count; j++) {
        double z = fabs(sample[j] - pd->feature_means[j]) / pd->feature_stds[j];
        if (z > max_z) max_z = z;
    }
    *outlier_score = max_z;
    return 0;
}

/**
 * @brief Perform Poison Detector Is Outlier.
 *
 * @param pd [out] Pd value.
 * @param sample [in] Sample value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_is_outlier(SNEPPXPoisonDetector* pd,
                                      const double* sample, int feature_count) {
    double score;
    if (SNEPPX_poison_detector_score(pd, sample, feature_count, &score) != 0) return 0;
    return (score > pd->outlier_threshold) ? 1 : 0;
}

/**
 * @brief Perform Poison Detector Update Incremental.
 *
 * @param pd [out] Pd value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_update_incremental(SNEPPXPoisonDetector* pd,
                                              const double* sample) {
    if (!pd || !sample || !pd->trained) return -1;
    int f = pd->feature_count;
    double n = (double)poison_sample_count;
    for (int j = 0; j < f; j++) {
        double old_mean = pd->feature_means[j];
        pd->feature_means[j] = (old_mean * n + sample[j]) / (n + 1.0);
        double old_var = pd->feature_stds[j] * pd->feature_stds[j];
        double new_var = (n * (old_var + old_mean * old_mean) + sample[j] * sample[j]) / (n + 1.0) - pd->feature_means[j] * pd->feature_means[j];
        pd->feature_stds[j] = (new_var > 0) ? sqrt(new_var) : 1.0;
    }
    poison_sample_count++;
    return 0;
}

/**
 * @brief Perform Poison Detector Get Stats.
 *
 * @param pd [out] Pd value.
 * @param sample_count [out] Sample Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_get_stats(SNEPPXPoisonDetector* pd, int* sample_count, double* outlier_rate) {
    if (!pd || !sample_count || !outlier_rate) return -1;
    *sample_count = poison_sample_count;
    *outlier_rate = 0.0;
    return 0;
}

/**
 * @brief Reset Poison Detector.
 */
void SNEPPX_poison_detector_reset(SNEPPXPoisonDetector* pd) {
    if (!pd) return;
    memset(pd->feature_means, 0, sizeof(double) * pd->feature_count);
    memset(pd->feature_stds, 0, sizeof(double) * pd->feature_count);
    pd->trained = 0;
    poison_sample_count = 0;
}

/**
 * @brief Perform Poison Detector Ensemble Score.
 *
 * @param pd [out] Pd value.
 * @param sample [in] Sample value.
 * @param feature_count [in] Feature Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_ensemble_score(SNEPPXPoisonDetector* pd,
                                          const double* sample, int feature_count,
                                          double* ensemble_outlier_score) {
    if (!pd || !sample || !ensemble_outlier_score) return -1;
    if (feature_count != pd->feature_count) return -1;
    if (ensemble_count == 0) return SNEPPX_poison_detector_score(pd, sample, feature_count, ensemble_outlier_score);
    double max_z = 0.0;
    for (int k = 0; k < ensemble_count; k++) {
        if (!ensemble_trained[k]) continue;
        double max_per_model = 0.0;
        for (int j = 0; j < feature_count; j++) {
            double z = fabs(sample[j] - ensemble_means[k][j]) / (ensemble_stds[k][j] > 0 ? ensemble_stds[k][j] : 1.0);
            if (z > max_per_model) max_per_model = z;
        }
        if (max_per_model > max_z) max_z = max_per_model;
    }
    *ensemble_outlier_score = max_z;
    return 0;
}
/**
 * @brief Update Poison Detector Incremental.
 *
 * @param pd [out] Pd value.
 * @param sample [in] Sample value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_incremental_update(SNEPPXPoisonDetector* pd, const double* sample, double learning_rate) {
    if (!pd || !sample || !pd->trained) return -1;
    if (learning_rate <= 0.0 || learning_rate > 1.0) learning_rate = 0.1;
    int f = pd->feature_count;
    for (int j = 0; j < f; j++) {
        double diff = sample[j] - pd->feature_means[j];
        pd->feature_means[j] += learning_rate * diff;
        double var = pd->feature_stds[j] * pd->feature_stds[j];
        var += learning_rate * (diff * diff - var);
        pd->feature_stds[j] = (var > 0) ? sqrt(var) : 1.0;
    }
    poison_sample_count++;
    return 0;
}
/**
 * @brief Perform Poison Detector Set Threshold.
 *
 * @param pd [out] Pd value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_set_threshold(SNEPPXPoisonDetector* pd, double t) {
    if (!pd || t <= 0.0) return -1;
    pd->outlier_threshold = t;
    return 0;
}
/**
 * @brief Perform Poison Detector Get Threshold.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_poison_detector_get_threshold(SNEPPXPoisonDetector* pd) {
    if (!pd) return -1.0;
    return pd->outlier_threshold;
}
/**
 * @brief Perform Poison Detector Get Feature Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_get_feature_count(SNEPPXPoisonDetector* pd) {
    if (!pd) return -1;
    return pd->feature_count;
}
/**
 * @brief Perform Poison Detector Batch Check.
 *
 * @param pd [out] Pd value.
 * @param samples [in] Samples value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_batch_check(SNEPPXPoisonDetector* pd, const double* samples, int count, int* results) {
    if (!pd || !samples || !results || count <= 0) return -1;
    int f = pd->feature_count;
    for (int i = 0; i < count; i++) {
        double max_z = 0.0;
        for (int j = 0; j < f; j++) {
            double z = fabs(samples[i * f + j] - pd->feature_means[j]) / pd->feature_stds[j];
            if (z > max_z) max_z = z;
        }
        results[i] = (max_z > pd->outlier_threshold) ? 1 : 0;
    }
    return 0;
}
/**
 * @brief Perform Poison Detector Ensemble Check.
 *
 * @param count [in] Count value.
 * @param sample [in] Sample value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_ensemble_check(SNEPPXPoisonDetector* detectors[], int count, const double* sample, int feature_count) {
    if (!detectors || count <= 0 || !sample || feature_count <= 0) return -1;
    int poison_votes = 0;
    for (int d = 0; d < count; d++) {
        if (!detectors[d] || !detectors[d]->trained) continue;
        if (feature_count != detectors[d]->feature_count) continue;
        double max_z = 0.0;
        for (int j = 0; j < feature_count; j++) {
            double z = fabs(sample[j] - detectors[d]->feature_means[j]) / detectors[d]->feature_stds[j];
            if (z > max_z) max_z = z;
        }
        if (max_z > detectors[d]->outlier_threshold) poison_votes++;
    }
    return (poison_votes > count / 2) ? 1 : 0;
}
static int poison_total_checks=0;
static int poison_total_outliers=0;
static double poison_aggregate_scores[SNEPPX_POISON_MAX_SAMPLES];
static int poison_aggregate_count=0;
/**
 * @brief Perform Poison Detector Get Aggregate Stats.
 *
 * @param mean [out] Mean value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_get_aggregate_stats(double* mean, double* std) {
    if(!mean||!std||poison_aggregate_count==0) return -1;
    double sum=0.0; for(int i=0;i<poison_aggregate_count;i++) sum+=poison_aggregate_scores[i];
    *mean=sum/(double)poison_aggregate_count;
    double var=0.0; for(int i=0;i<poison_aggregate_count;i++){double d=poison_aggregate_scores[i]-*mean;var+=d*d;}
    *std=sqrt(var/(double)poison_aggregate_count); return 0;
}
/**
 * @brief Perform Poison Detector Record Score.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_record_score(double score) {
    if(poison_aggregate_count>=SNEPPX_POISON_MAX_SAMPLES) return -1;
    poison_aggregate_scores[poison_aggregate_count++]=score; return 0;
}
/**
 * @brief Perform Poison Detector Clear Aggregate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_clear_aggregate(void) { poison_aggregate_count=0; return 0; }
/**
 * @brief Perform Poison Detector Get Aggregate Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_get_aggregate_count(void) { return poison_aggregate_count; }
/**
 * @brief Perform Poison Detector Ensemble Get Model Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_ensemble_get_model_count(void) { return ensemble_count; }
/**
 * @brief Perform Poison Detector Ensemble Get Model Stats.
 *
 * @param model_index [in] Model Index value.
 * @param mean [out] Mean value.
 * @param std [out] Std value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_ensemble_get_model_stats(int model_index, double* mean, double* std, int* trained) {
    if(model_index<0||model_index>=ensemble_count||!mean||!std||!trained) return -1;
    *trained=ensemble_trained[model_index];
    double m_sum=0.0,v_sum=0.0;
    for(int j=0;j<SNEPPX_POISON_MAX_FEATURES;j++){m_sum+=ensemble_means[model_index][j];v_sum+=ensemble_stds[model_index][j];}
    *mean=m_sum/(double)SNEPPX_POISON_MAX_FEATURES;
    *std=v_sum/(double)SNEPPX_POISON_MAX_FEATURES;
    return 0;
}
/**
 * @brief Reset Poison Detector Ensemble.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_ensemble_reset(void) {
    memset(ensemble_means,0,sizeof(ensemble_means));
    memset(ensemble_stds,0,sizeof(ensemble_stds));
    memset(ensemble_trained,0,sizeof(ensemble_trained));
    ensemble_count=0; return 0;
}
/**
 * @brief Perform Poison Detector Ensemble Add Model.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_ensemble_add_model(SNEPPXPoisonDetector* pd) {
    if(!pd||!pd->trained||ensemble_count>=SNEPPX_POISON_ENSEMBLE_K) return -1;
    int f=pd->feature_count;
    for(int j=0;j<f;j++){ensemble_means[ensemble_count][j]=pd->feature_means[j];ensemble_stds[ensemble_count][j]=pd->feature_stds[j];}
    ensemble_trained[ensemble_count]=1; ensemble_count++; return 0;
}
/**
 * @brief Perform Poison Detector Cross Validate.
 *
 * @param samples [in] Samples value.
 * @param sample_count [in] Sample Count value.
 * @param folds [in] Folds value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_cross_validate(const double* samples, int sample_count, int folds, double* mean_accuracy) {
    if(!samples||sample_count<2||folds<2||!mean_accuracy) return -1;
    double total_acc=0.0; int fold_size=sample_count/folds;
    for(int f=0;f<folds;f++) {
        int test_start=f*fold_size; int test_end=(f+1)*fold_size;
        if(f==folds-1) test_end=sample_count;
        int test_count=test_end-test_start;
        int train_count=sample_count-test_count;
        SNEPPXPoisonDetector pd; SNEPPX_poison_detector_init(&pd,1);
        double* train=(double*)malloc(train_count*sizeof(double));
        if(!train){SNEPPX_poison_detector_destroy(&pd);return -1;}
        int ti=0;
        for(int i=0;i<sample_count;i++){if(i<test_start||i>=test_end){train[ti]=samples[i];ti++;}}
        SNEPPX_poison_detector_train(&pd,train,train_count);
        int correct=0;
        for(int i=test_start;i<test_end;i++){double s;SNEPPX_poison_detector_score(&pd,&samples[i],1,&s);if(s<=pd.outlier_threshold)correct++;}
        total_acc+=(double)correct/(double)test_count;
        free(train); SNEPPX_poison_detector_destroy(&pd);
    }
    *mean_accuracy=total_acc/(double)folds; return 0;
}
/**
 * @brief Perform Poison Detector Feature Importance.
 *
 * @param samples [in] Samples value.
 * @param sample_count [in] Sample Count value.
 * @param importance [out] Importance value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_feature_importance(const double* samples, int sample_count, double* importance, int feature_count) {
    if(!samples||!importance||sample_count<2||feature_count<=0) return -1;
    SNEPPXPoisonDetector pd; SNEPPX_poison_detector_init(&pd,feature_count);
    SNEPPX_poison_detector_train(&pd,samples,sample_count);
    for(int j=0;j<feature_count;j++) importance[j]=pd.feature_stds[j];
    SNEPPX_poison_detector_destroy(&pd); return 0;
}
/**
 * @brief Perform Poison Detector Find Outliers.
 *
 * @param samples [in] Samples value.
 * @param sample_count [in] Sample Count value.
 * @param outlier_indices [out] Outlier Indices value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_find_outliers(const double* samples, int sample_count, int* outlier_indices, int* outlier_count) {
    if(!samples||!outlier_indices||!outlier_count||sample_count<2) return -1;
    SNEPPXPoisonDetector pd; SNEPPX_poison_detector_init(&pd,1);
    SNEPPX_poison_detector_train(&pd,samples,sample_count);
    int oc=0;
    for(int i=0;i<sample_count;i++){double s;SNEPPX_poison_detector_score(&pd,&samples[i],1,&s);if(s>pd.outlier_threshold&&oc<sample_count)outlier_indices[oc++]=i;}
    *outlier_count=oc;
    SNEPPX_poison_detector_destroy(&pd); return 0;
}
/**
 * @brief Perform Poison Detector Adaptive Threshold.
 *
 * @param samples [in] Samples value.
 * @param sample_count [in] Sample Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_adaptive_threshold(const double* samples, int sample_count, double sensitivity) {
    if(!samples||sample_count<2) return -1;
    SNEPPXPoisonDetector pd; SNEPPX_poison_detector_init(&pd,1);
    SNEPPX_poison_detector_train(&pd,samples,sample_count);
    double mean_score=0.0; int valid=0;
    for(int i=0;i<sample_count;i++){double s;SNEPPX_poison_detector_score(&pd,&samples[i],1,&s);mean_score+=s;valid++;}
    if(valid==0){SNEPPX_poison_detector_destroy(&pd);return -1;}
    mean_score/=(double)valid;
    pd.outlier_threshold=mean_score*sensitivity;
    SNEPPX_poison_detector_destroy(&pd); return 0;
}
/**
 * @brief Perform Poison Detector Export Model.
 *
 * @param pd [out] Pd value.
 * @param means [out] Means value.
 * @param stds [out] Stds value.
 * @param feature_count [out] Feature Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_export_model(SNEPPXPoisonDetector* pd, double* means, double* stds, int* feature_count, double* threshold) {
    if(!pd||!means||!stds||!feature_count||!threshold) return -1;
    *feature_count=pd->feature_count;
    *threshold=pd->outlier_threshold;
    memcpy(means,pd->feature_means,pd->feature_count*sizeof(double));
    memcpy(stds,pd->feature_stds,pd->feature_count*sizeof(double));
    return 0;
}
/**
 * @brief Perform Poison Detector Import Model.
 *
 * @param pd [out] Pd value.
 * @param means [in] Means value.
 * @param stds [in] Stds value.
 * @param feature_count [in] Feature Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_import_model(SNEPPXPoisonDetector* pd, const double* means, const double* stds, int feature_count, double threshold) {
    if(!pd||!means||!stds||feature_count<=0) return -1;
    if(pd->feature_means) free(pd->feature_means);
    if(pd->feature_stds) free(pd->feature_stds);
    pd->feature_means=(double*)malloc(feature_count*sizeof(double));
    pd->feature_stds=(double*)malloc(feature_count*sizeof(double));
    if(!pd->feature_means||!pd->feature_stds){free(pd->feature_means);free(pd->feature_stds);return -1;}
    pd->feature_count=feature_count;
    pd->outlier_threshold=threshold;
    memcpy(pd->feature_means,means,feature_count*sizeof(double));
    memcpy(pd->feature_stds,stds,feature_count*sizeof(double));
    pd->trained=1; return 0;
}
/**
 * @brief Perform Poison Detector Is Trained.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_is_trained(SNEPPXPoisonDetector* pd) { if(!pd) return -1; return pd->trained?1:0; }
/**
 * @brief Perform Poison Detector Get Sample Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_get_sample_count(void) { return poison_sample_count; }
/**
 * @brief Perform Poison Detector Get Total Checks.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_get_total_checks(void) { return poison_total_checks; }
/**
 * @brief Perform Poison Detector Get Total Outliers.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_get_total_outliers(void) { return poison_total_outliers; }
/**
 * @brief Perform Poison Detector Score With Details.
 *
 * @param pd [out] Pd value.
 * @param sample [in] Sample value.
 * @param feature_count [in] Feature Count value.
 * @param outlier_score [out] Outlier Score value.
 * @param max_feature_z [out] Max Feature Z value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_score_with_details(SNEPPXPoisonDetector* pd, const double* sample, int feature_count, double* outlier_score, double* max_feature_z, int* max_feature_idx) {
    if(!pd||!sample||!outlier_score||!max_feature_z||!max_feature_idx) return -1;
    if(feature_count!=pd->feature_count) return -1;
    double max_z=0.0; int max_idx=-1;
    for(int j=0;j<feature_count;j++){double z=fabs(sample[j]-pd->feature_means[j])/pd->feature_stds[j];if(z>max_z){max_z=z;max_idx=j;}}
    *outlier_score=max_z; *max_feature_z=max_z; *max_feature_idx=max_idx;
    poison_total_checks++;
    if(max_z>pd->outlier_threshold) poison_total_outliers++;
    return 0;
}
/**
 * @brief Perform Poison Detector Ensemble Score Weighted.
 *
 * @param count [in] Count value.
 * @param sample [in] Sample value.
 * @param feature_count [in] Feature Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_ensemble_score_weighted(SNEPPXPoisonDetector* detectors[], int count, const double* sample, int feature_count, double* ensemble_outlier_score) {
    if(!detectors||count<=0||!sample||!ensemble_outlier_score) return -1;
    double weighted_sum=0.0; double weight_total=0.0;
    for(int d=0;d<count;d++){if(!detectors[d]||!detectors[d]->trained||feature_count!=detectors[d]->feature_count) continue;double max_z=0.0;for(int j=0;j<feature_count;j++){double z=fabs(sample[j]-detectors[d]->feature_means[j])/detectors[d]->feature_stds[j];if(z>max_z)max_z=z;}weighted_sum+=max_z*detectors[d]->outlier_threshold;weight_total+=detectors[d]->outlier_threshold;}
    if(weight_total<=0.0) return -1;
    *ensemble_outlier_score=weighted_sum/weight_total; return 0;
}
/**
 * @brief Perform Poison Detector Batch Score With Stats.
 *
 * @param pd [out] Pd value.
 * @param samples [in] Samples value.
 * @param count [in] Count value.
 * @param scores [out] Scores value.
 * @param outlier_flags [out] Outlier Flags value.
 * @param mean_score [out] Mean Score value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_batch_score_with_stats(SNEPPXPoisonDetector* pd, const double* samples, int count, double* scores, int* outlier_flags, double* mean_score, double* max_score) {
    if(!pd||!samples||count<=0||!scores||!outlier_flags||!mean_score||!max_score) return -1;
    int f=pd->feature_count; double sum=0.0; double mx=0.0;
    for(int i=0;i<count;i++){double max_z=0.0;for(int j=0;j<f;j++){double z=fabs(samples[i*f+j]-pd->feature_means[j])/pd->feature_stds[j];if(z>max_z)max_z=z;}scores[i]=max_z;outlier_flags[i]=(max_z>pd->outlier_threshold)?1:0;sum+=max_z;if(max_z>mx)mx=max_z;}
    *mean_score=sum/(double)count; *max_score=mx; return 0;
}
/**
 * @brief Perform Poison Detector Train Incremental Batch.
 *
 * @param pd [out] Pd value.
 * @param samples [in] Samples value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_train_incremental_batch(SNEPPXPoisonDetector* pd, const double* samples, int count) {
    if(!pd||!samples||count<=0) return -1;
    int f=pd->feature_count;
    for(int i=0;i<count;i++){SNEPPX_poison_detector_incremental_update(pd,&samples[i*f],0.1);}
    return 0;
}
/**
 * @brief Perform Poison Detector Compare Models.
 *
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_compare_models(SNEPPXPoisonDetector* a, SNEPPXPoisonDetector* b, double* divergence) {
    if(!a||!b||!a->trained||!b->trained||divergence) return -1;
    if(a->feature_count!=b->feature_count) return -1;
    double d=0.0;
    for(int j=0;j<a->feature_count;j++){double dm=a->feature_means[j]-b->feature_means[j];double ds=a->feature_stds[j]-b->feature_stds[j];d+=dm*dm+ds*ds;}
    *divergence=sqrt(d)/(double)a->feature_count; return 0;
}
/**
 * @brief Perform Poison Detector Detect Backdoor.
 *
 * @param samples [in] Samples value.
 * @param sample_count [in] Sample Count value.
 * @param labels [in] Labels value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_detect_backdoor(const double* samples, int sample_count, const int* labels, double* backdoor_score) {
    if(!samples||sample_count<2||!labels||!backdoor_score) return -1;
    SNEPPXPoisonDetector pd; SNEPPX_poison_detector_init(&pd,1);
    SNEPPX_poison_detector_train(&pd,samples,sample_count);
    double poison_scores[1024]; int poison_count=0;
    for(int i=0;i<sample_count;i++){double s;SNEPPX_poison_detector_score(&pd,&samples[i],1,&s);if(labels[i]==1&&s>pd.outlier_threshold)poison_scores[poison_count++]=s;}
    if(poison_count==0){*backdoor_score=0.0;SNEPPX_poison_detector_destroy(&pd);return 0;}
    double sum=0.0; for(int i=0;i<poison_count;i++) sum+=poison_scores[i];
    *backdoor_score=sum/(double)poison_count;
    SNEPPX_poison_detector_destroy(&pd); return 0;
}
/**
 * @brief Perform Poison Detector Compute Statistics.
 *
 * @param samples [in] Samples value.
 * @param sample_count [in] Sample Count value.
 * @param mean [out] Mean value.
 * @param variance [out] Variance value.
 * @param min_val [out] Min Val value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_compute_statistics(const double* samples, int sample_count, double* mean, double* variance, double* min_val, double* max_val) {
    if(!samples||sample_count<1||!mean||!variance||!min_val||!max_val) return -1;
    double sum=0.0; *min_val=samples[0]; *max_val=samples[0];
    for(int i=0;i<sample_count;i++){sum+=samples[i];if(samples[i]<*min_val)*min_val=samples[i];if(samples[i]>*max_val)*max_val=samples[i];}
    *mean=sum/(double)sample_count;
    double var=0.0; for(int i=0;i<sample_count;i++){double d=samples[i]-*mean;var+=d*d;}
    *variance=var/(double)sample_count; return 0;
}
/**
 * @brief Perform Poison Detector Normalize Samples.
 *
 * @param samples [out] Samples value.
 * @param sample_count [in] Sample Count value.
 * @param new_min [in] New Min value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_normalize_samples(double* samples, int sample_count, double new_min, double new_max) {
    if(!samples||sample_count<1) return -1;
    double mn=samples[0],mx=samples[0];
    for(int i=0;i<sample_count;i++){if(samples[i]<mn)mn=samples[i];if(samples[i]>mx)mx=samples[i];}
    double range=mx-mn; if(range<1e-10) range=1.0;
    for(int i=0;i<sample_count;i++) samples[i]=new_min+(samples[i]-mn)/range*(new_max-new_min);
    return 0;
}
/**
 * @brief Perform Poison Detector Zscore Transform.
 *
 * @param samples [out] Samples value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_zscore_transform(double* samples, int sample_count) {
    if(!samples||sample_count<2) return -1;
    double sum=0.0; for(int i=0;i<sample_count;i++) sum+=samples[i];
    double mean=sum/(double)sample_count;
    double var=0.0; for(int i=0;i<sample_count;i++){double d=samples[i]-mean;var+=d*d;}
    double std=sqrt(var/(double)sample_count); if(std<1e-10) std=1.0;
    for(int i=0;i<sample_count;i++) samples[i]=(samples[i]-mean)/std;
    return 0;
}
/**
 * @brief Perform Poison Detector Ensemble Predict Proba.
 *
 * @param count [in] Count value.
 * @param sample [in] Sample value.
 * @param feature_count [in] Feature Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_ensemble_predict_proba(SNEPPXPoisonDetector* detectors[], int count, const double* sample, int feature_count, double* probability) {
    if(!detectors||count<=0||!sample||!probability) return -1;
    double sum_proba=0.0; int valid=0;
    for(int d=0;d<count;d++){if(!detectors[d]||!detectors[d]->trained||feature_count!=detectors[d]->feature_count) continue;double max_z=0.0;for(int j=0;j<feature_count;j++){double z=fabs(sample[j]-detectors[d]->feature_means[j])/detectors[d]->feature_stds[j];if(z>max_z)max_z=z;}double p=max_z/(max_z+detectors[d]->outlier_threshold);sum_proba+=p;valid++;}
    if(valid==0) return -1;
    *probability=sum_proba/(double)valid; return 0;
}
/**
 * @brief Perform Poison Detector Train With Labels.
 *
 * @param pd [out] Pd value.
 * @param samples [in] Samples value.
 * @param sample_count [in] Sample Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_train_with_labels(SNEPPXPoisonDetector* pd, const double* samples, int sample_count, const int* labels) {
    if(!pd||!samples||sample_count<2||!labels) return -1;
    int clean_count=0; for(int i=0;i<sample_count;i++) if(labels[i]==0) clean_count++;
    if(clean_count<2) return -1;
    double* clean=(double*)malloc(clean_count*sizeof(double)); if(!clean) return -1;
    int ci=0; for(int i=0;i<sample_count;i++){if(labels[i]==0)clean[ci++]=samples[i];}
    int r=SNEPPX_poison_detector_train(pd,clean,clean_count);
    free(clean); return r;
}
/**
 * @brief Perform Poison Detector Roc Auc.
 *
 * @param scores [in] Scores value.
 * @param labels [in] Labels value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_roc_auc(const double* scores, const int* labels, int count, double* auc) {
    if(!scores||!labels||count<2||!auc) return -1;
    int pos_count=0,neg_count=0;
    for(int i=0;i<count;i++){if(labels[i]==1)pos_count++;else neg_count++;}
    if(pos_count==0||neg_count==0) return -1;
    double* pos_scores=(double*)malloc(pos_count*sizeof(double));
    double* neg_scores=(double*)malloc(neg_count*sizeof(double));
    if(!pos_scores||!neg_scores){free(pos_scores);free(neg_scores);return -1;}
    int pi=0,ni=0;
    for(int i=0;i<count;i++){if(labels[i]==1)pos_scores[pi++]=scores[i];else neg_scores[ni++]=scores[i];}
    double concordant=0.0;
    for(int i=0;i<pos_count;i++){for(int j=0;j<neg_count;j++){if(pos_scores[i]>neg_scores[j])concordant+=1.0;else if(pos_scores[i]==neg_scores[j])concordant+=0.5;}}
    *auc=concordant/(double)(pos_count*neg_count);
    free(pos_scores); free(neg_scores); return 0;
}
/**
 * @brief Perform Poison Detector Precision Recall.
 *
 * @param scores [in] Scores value.
 * @param labels [in] Labels value.
 * @param count [in] Count value.
 * @param threshold [in] Threshold value.
 * @param precision [out] Precision value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_precision_recall(const double* scores, const int* labels, int count, double threshold, double* precision, double* recall) {
    if(!scores||!labels||count<2||!precision||!recall) return -1;
    int tp=0,fp=0,fn=0;
    for(int i=0;i<count;i++){int pred=scores[i]>=threshold?1:0;if(pred==1&&labels[i]==1)tp++;else if(pred==1&&labels[i]==0)fp++;else if(pred==0&&labels[i]==1)fn++;}
    *precision=(tp+fp)>0?(double)tp/(double)(tp+fp):0.0;
    *recall=(tp+fn)>0?(double)tp/(double)(tp+fn):0.0;
    return 0;
}
/**
 * @brief Perform Poison Detector F1 Score.
 *
 * @param scores [in] Scores value.
 * @param labels [in] Labels value.
 * @param count [in] Count value.
 * @param threshold [in] Threshold value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_f1_score(const double* scores, const int* labels, int count, double threshold, double* f1) {
    double p,r; if(SNEPPX_poison_detector_precision_recall(scores,labels,count,threshold,&p,&r)!=0) return -1;
    *f1=(p+r)>0?2.0*p*r/(p+r):0.0; return 0;
}
/**
 * @brief Perform Poison Detector Optimal Threshold.
 *
 * @param scores [in] Scores value.
 * @param labels [in] Labels value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_optimal_threshold(const double* scores, const int* labels, int count, double* optimal_threshold) {
    if(!scores||!labels||count<2||!optimal_threshold) return -1;
    double best_f1=0.0; double best_t=0.0;
    for(int i=0;i<count;i++){double t=scores[i];double f1;if(SNEPPX_poison_detector_f1_score(scores,labels,count,t,&f1)==0&&f1>best_f1){best_f1=f1;best_t=t;}}
    *optimal_threshold=best_t; return 0;
}
/**
 * @brief Perform Poison Detector Confusion Matrix.
 *
 * @param scores [in] Scores value.
 * @param labels [in] Labels value.
 * @param count [in] Count value.
 * @param threshold [in] Threshold value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_confusion_matrix(const double* scores, const int* labels, int count, double threshold, int matrix[4]) {
    if(!scores||!labels||count<2||!matrix) return -1;
    matrix[0]=0;matrix[1]=0;matrix[2]=0;matrix[3]=0;
    for(int i=0;i<count;i++){int pred=scores[i]>=threshold?1:0;if(pred==1&&labels[i]==1)matrix[0]++;else if(pred==1&&labels[i]==0)matrix[1]++;else if(pred==0&&labels[i]==1)matrix[2]++;else matrix[3]++;}
    return 0;
}
/**
 * @brief Perform Poison Detector Mcc.
 *
 * @param scores [in] Scores value.
 * @param labels [in] Labels value.
 * @param count [in] Count value.
 * @param threshold [in] Threshold value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_mcc(const double* scores, const int* labels, int count, double threshold, double* mcc) {
    int cm[4]; if(SNEPPX_poison_detector_confusion_matrix(scores,labels,count,threshold,cm)!=0) return -1;
    int tp=cm[0],fp=cm[1],fn=cm[2],tn=cm[3];
    double d=(double)(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn);
    *mcc=d>0?((double)(tp*tn-fp*fn))/sqrt(d):0.0;
    return 0;
}
/**
 * @brief Perform Poison Detector Expected Calibration Error.
 *
 * @param scores [in] Scores value.
 * @param labels [in] Labels value.
 * @param count [in] Count value.
 * @param bins [in] Bins value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_expected_calibration_error(const double* scores, const int* labels, int count, int bins, double* ece) {
    if(!scores||!labels||count<2||bins<2||!ece) return -1;
    double bin_size=1.0/(double)bins;
    double* bin_acc=(double*)calloc(bins,sizeof(double));
    double* bin_conf=(double*)calloc(bins,sizeof(double));
    int* bin_count=(int*)calloc(bins,sizeof(int));
    if(!bin_acc||!bin_conf||!bin_count){free(bin_acc);free(bin_conf);free(bin_count);return -1;}
    for(int i=0;i<count;i++){int b=(int)(scores[i]/bin_size);if(b>=bins)b=bins-1;bin_conf[b]+=scores[i];int pred=scores[i]>=0.5?1:0;if(pred==labels[i])bin_acc[b]+=1.0;bin_count[b]++;}
    double total=0.0;int total_count=0;
    for(int b=0;b<bins;b++){if(bin_count[b]>0){double acc=bin_acc[b]/(double)bin_count[b];double conf=bin_conf[b]/(double)bin_count[b];total+=fabs(acc-conf)*(double)bin_count[b];total_count+=bin_count[b];}}
    *ece=total_count>0?total/(double)total_count:0.0;
    free(bin_acc);free(bin_conf);free(bin_count);return 0;
}
/**
 * @brief Perform Poison Detector Detect Poisoned Fraction.
 *
 * @param samples [in] Samples value.
 * @param sample_count [in] Sample Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_detect_poisoned_fraction(const double* samples, int sample_count, double* fraction) {
    if(!samples||sample_count<2||!fraction) return -1;
    SNEPPXPoisonDetector pd; SNEPPX_poison_detector_init(&pd,1);
    SNEPPX_poison_detector_train(&pd,samples,sample_count);
    int outliers=0;
    for(int i=0;i<sample_count;i++){double s;SNEPPX_poison_detector_score(&pd,&samples[i],1,&s);if(s>pd.outlier_threshold)outliers++;}
    *fraction=(double)outliers/(double)sample_count;
    SNEPPX_poison_detector_destroy(&pd); return 0;
}
/**
 * @brief Perform Poison Detector Summary Stats.
 *
 * @param samples [in] Samples value.
 * @param sample_count [in] Sample Count value.
 * @param mean [out] Mean value.
 * @param median [out] Median value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_summary_stats(const double* samples, int sample_count, double* mean, double* median, double* std) {
    if(!samples||sample_count<1||!mean||!median||!std) return -1;
    double* sorted=(double*)malloc(sample_count*sizeof(double));
    if(!sorted) return -1;
    memcpy(sorted,samples,sample_count*sizeof(double));
    for(int i=0;i<sample_count-1;i++){for(int j=0;j<sample_count-1-i;j++){if(sorted[j]>sorted[j+1]){double t=sorted[j];sorted[j]=sorted[j+1];sorted[j+1]=t;}}}
    *median=sample_count%2==1?sorted[sample_count/2]:(sorted[sample_count/2-1]+sorted[sample_count/2])/2.0;
    double sum=0.0; for(int i=0;i<sample_count;i++) sum+=sorted[i];
    *mean=sum/(double)sample_count;
    double var=0.0; for(int i=0;i<sample_count;i++){double d=sorted[i]-*mean;var+=d*d;}
    *std=sqrt(var/(double)sample_count);
    free(sorted); return 0;
}
/**
 * @brief Perform Poison Detector Is Clean.
 *
 * @param pd [out] Pd value.
 * @param sample [in] Sample value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_poison_detector_is_clean(SNEPPXPoisonDetector* pd, const double* sample, int feature_count) {
    if(!pd||!sample) return 0;
    double s; if(SNEPPX_poison_detector_score(pd,sample,feature_count,&s)!=0) return 1;
    return s<=pd->outlier_threshold?1:0;
}
