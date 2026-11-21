#ifndef SNEPPX_DATA_POISONING_DEFENSE_H
#define SNEPPX_DATA_POISONING_DEFENSE_H
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


/*
 * S5 AI Sanitizer — Data Poisoning Defense
 * Detects poisoned training samples, outlier data points, and
 * backdoor triggers in datasets.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_POISON_MAX_FEATURES 1024

typedef struct {
    double* feature_means;
    double* feature_stds;
    int feature_count;
    double outlier_threshold;
    int trained;
} SNEPPXPoisonDetector;

/**
 * @brief Initialize Poison Detector.
 *
 * @param pd [out] Pd value.
 * @param feature_count [in] Feature Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_poison_detector_init(SNEPPXPoisonDetector* pd, int feature_count);
/**
 * @brief Destroy Poison Detector.
 *
 * @param pd [out] Pd value.
 */
void SNEPPX_poison_detector_destroy(SNEPPXPoisonDetector* pd);
/**
 * @brief Train Poison Detector.
 *
 * @param pd [out] Pd value.
 * @param samples [in] Samples value.
 * @param sample_count [in] Sample Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_poison_detector_train(SNEPPXPoisonDetector* pd,
                                 const double* samples, int sample_count);
/**
 * @brief Perform Poison Detector Score.
 *
 * @param pd [out] Pd value.
 * @param sample [in] Sample value.
 * @param feature_count [in] Feature Count value.
 * @param outlier_score [out] Outlier Score value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_poison_detector_score(SNEPPXPoisonDetector* pd,
                                 const double* sample, int feature_count,
                                 double* outlier_score);
/**
 * @brief Perform Poison Detector Is Outlier.
 *
 * @param pd [out] Pd value.
 * @param sample [in] Sample value.
 * @param feature_count [in] Feature Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_poison_detector_is_outlier(SNEPPXPoisonDetector* pd,
                                      const double* sample, int feature_count);

#ifdef __cplusplus
}
#endif
#endif
