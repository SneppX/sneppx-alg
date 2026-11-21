#ifndef SNEPPX_MQTT_CLIENT_H
#define SNEPPX_MQTT_CLIENT_H
#include <stddef.h>
#ifdef __cplusplus
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


extern "C" {
#endif
/**
 * @brief Perform Mqtt Connect.
 *
 * @param broker [in] Broker value.
 * @param port [in] Port value.
 * @param client_id [in] Client Id value.
 * @param keepalive [in] Keepalive value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_mqtt_connect(const char* broker, int port, const char* client_id, int keepalive);
/**
 * @brief Perform Mqtt Disconnect.
 *
 * @param client [out] Client value.
 */
void SNEPPX_mqtt_disconnect(void* client);
/**
 * @brief Perform Mqtt Subscribe.
 *
 * @param client [out] Client value.
 * @param topic [in] Topic value.
 * @param qos [in] Qos value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_subscribe(void* client, const char* topic, int qos);
/**
 * @brief Perform Mqtt Unsubscribe.
 *
 * @param client [out] Client value.
 * @param topic [in] Topic value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_unsubscribe(void* client, const char* topic);
/**
 * @brief Perform Mqtt Publish.
 *
 * @param client [out] Client value.
 * @param topic [in] Topic value.
 * @param payload [in] Payload value.
 * @param len [in] Len value.
 * @param qos [in] Qos value.
 * @param retain [in] Retain value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_publish(void* client, const char* topic, const unsigned char* payload, size_t len, int qos, int retain);
int SNEPPX_mqtt_set_on_message(void* client, void (*cb)(void* client, const char* topic, const unsigned char* payload, size_t len));
/**
 * @brief Start Mqtt Loop.
 *
 * @param client [out] Client value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_loop_start(void* client);
/**
 * @brief Stop Mqtt Loop.
 *
 * @param client [out] Client value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_loop_stop(void* client);
/**
 * @brief Perform Mqtt Is Connected.
 *
 * @param client [out] Client value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mqtt_is_connected(void* client);
#ifdef __cplusplus
}
#endif
#endif
