#ifndef SNEPPX_SERVICE_DISCOVERY_H
#define SNEPPX_SERVICE_DISCOVERY_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
/*
 * SNEPPX - Service Discovery
 *
 * WHAT
 *   Service Discovery.
 *
 * CONCEPT
 *   Provides the Service Discovery.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
typedef enum { SNEPPX_SD_DNS, SNEPPX_SD_CONSUL, SNEPPX_SD_ETCD, SNEPPX_SD_K8S } SNEPPXSDProvider;
typedef struct { char name[128]; char host[256]; int port; char** tags; size_t num_tags; uint64_t last_seen; } SNEPPXSDInstance;
/**
 * @brief Initialize Sd.
 *
 * @param provider [in] Provider value.
 * @param service_name [in] Service Name value.
 * @param port [in] Port value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_sd_init(SNEPPXSDProvider provider, const char* service_name, int port);
/**
 * @brief Destroy Sd.
 *
 * @param sd [out] Sd value.
 */
void SNEPPX_sd_destroy(void* sd);
/**
 * @brief Perform Sd Register.
 *
 * @param sd [out] Sd value.
 * @param ttl_secs [in] Ttl Secs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sd_register(void* sd, int ttl_secs);
/**
 * @brief Perform Sd Unregister.
 *
 * @param sd [out] Sd value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sd_unregister(void* sd);
/**
 * @brief Perform Sd Discover.
 *
 * @param sd [out] Sd value.
 * @param instances [out] Instances value.
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sd_discover(void* sd, SNEPPXSDInstance** instances, size_t* count);
/**
 * @brief Free Sd Instances.
 *
 * @param instances [out] Instances value.
 * @param count [in] Count value.
 */
void SNEPPX_sd_instances_free(SNEPPXSDInstance* instances, size_t count);
#ifdef __cplusplus
}
#endif
#endif
