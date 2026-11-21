#include "mqtt_client.h"
#include <stdlib.h>
#include <string.h>

/*
 * SNEPPX - Mqtt Client
 *
 * WHAT
 *   Mqtt Client.
 *
 * CONCEPT
 *   Provides the Mqtt Client.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Mqtt Connect.
 *
 * @param broker [in] Broker value.
 * @param port [in] Port value.
 * @param client_id [in] Client Id value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_mqtt_connect(const char* broker, int port, const char* client_id, int keepalive) { (void)broker; (void)port; (void)client_id; (void)keepalive; return NULL; }
/**
 * @brief Perform Mqtt Disconnect.
 */
void SNEPPX_mqtt_disconnect(void* client) { free(client); }
/**
 * @brief Perform Mqtt Subscribe.
 *
 * @param client [out] Client value.
 * @param topic [in] Topic value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_subscribe(void* client, const char* topic, int qos) { (void)client; (void)topic; (void)qos; return 0; }
/**
 * @brief Perform Mqtt Unsubscribe.
 *
 * @param client [out] Client value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_unsubscribe(void* client, const char* topic) { (void)client; (void)topic; return 0; }
/**
 * @brief Perform Mqtt Publish.
 *
 * @param client [out] Client value.
 * @param topic [in] Topic value.
 * @param payload [in] Payload value.
 * @param len [in] Len value.
 * @param qos [in] Qos value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_publish(void* client, const char* topic, const unsigned char* payload, size_t len, int qos, int retain) { (void)client; (void)topic; (void)payload; (void)len; (void)qos; (void)retain; return 0; }
int SNEPPX_mqtt_set_on_message(void* client, void (*cb)(void* client, const char* topic, const unsigned char* payload, size_t len)) { (void)client; (void)cb; return 0; }
/**
 * @brief Start Mqtt Loop.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_loop_start(void* client) { (void)client; return 0; }
/**
 * @brief Stop Mqtt Loop.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_loop_stop(void* client) { (void)client; return 0; }
/**
 * @brief Perform Mqtt Is Connected.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_is_connected(void* client) { (void)client; return 0; }
