#include "heartbeat.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifndef _WIN32
#include <fcntl.h>
#endif

#ifdef _WIN32
#pragma comment(lib, "ws2_32.lib")
/*
 * SNEPPX - Heartbeat
 *
 * WHAT
 *   Heartbeat.
 *
 * CONCEPT
 *   Provides heartbeat monitoring.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int winsock_init_count = 0;
#endif

static int64_t snepx_ns_now(void) {
#ifdef _WIN32
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    return ((int64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
}

/**
 * @brief Initialize Heartbeat.
 *
 * @param hb [out] Hb value.
 * @param world_size [in] World Size value.
 * @param rank [in] Rank value.
 * @param interval_ms [in] Interval Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heartbeat_init(SNEPPXHeartbeat** hb, int world_size, int rank,
                           int interval_ms, int timeout_ms) {
    if (!hb || world_size <= 0 || rank < 0 || rank >= world_size) return -1;
    SNEPPXHeartbeat* h = (SNEPPXHeartbeat*)calloc(1, sizeof(SNEPPXHeartbeat));
    if (!h) return -1;
    h->world_size = world_size;
    h->rank = rank;
    h->heartbeat_interval_ms = interval_ms > 0 ? interval_ms : 1000;
    h->timeout_ms = timeout_ms > 0 ? timeout_ms : 5000;
    h->socket = SNEPPX_INVALID_SOCKET;
    h->num_peers = 0;
    h->peer_addrs = (struct sockaddr_in*)calloc(world_size, sizeof(struct sockaddr_in));
    h->peer_status = (SNEPPXHeartbeatStatus*)calloc(world_size, sizeof(SNEPPXHeartbeatStatus));
    h->last_heartbeat_ns = (int64_t*)calloc(world_size, sizeof(int64_t));
    h->missed_count = (uint32_t*)calloc(world_size, sizeof(uint32_t));
    if (!h->peer_addrs || !h->peer_status || !h->last_heartbeat_ns || !h->missed_count) {
        free(h->peer_addrs); free(h->peer_status);
        free(h->last_heartbeat_ns); free(h->missed_count); free(h);
        return -1;
    }
    for (int i = 0; i < world_size; i++) {
        h->peer_status[i] = SNEPPX_HB_ALIVE;
        h->last_heartbeat_ns[i] = snepx_ns_now();
    }
    h->running = 1;
    *hb = h;
    return 0;
}

/**
 * @brief Perform Heartbeat Listen.
 *
 * @param hb [out] Hb value.
 * @param addr [in] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heartbeat_listen(SNEPPXHeartbeat* hb, const char* addr, int port) {
    if (!hb) return -1;
#ifdef _WIN32
    WSADATA wsa;
    if (winsock_init_count == 0) WSAStartup(MAKEWORD(2, 2), &wsa);
    winsock_init_count++;
#endif
    hb->socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (hb->socket == SNEPPX_INVALID_SOCKET) return -1;
    if (addr) strncpy(hb->listen_addr, addr, sizeof(hb->listen_addr) - 1);
    hb->listen_port = port > 0 ? port : SNEPPX_HB_PORT_BASE + hb->rank;
    struct sockaddr_in local;
    memset(&local, 0, sizeof(local));
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = addr ? inet_addr(addr) : INADDR_ANY;
    local.sin_port = htons((uint16_t)hb->listen_port);
    if (bind(hb->socket, (struct sockaddr*)&local, sizeof(local)) < 0) return -1;
#ifdef _WIN32
    unsigned long nonblock = 1;
    ioctlsocket(hb->socket, FIONBIO, &nonblock);
#else
    int flags = fcntl(hb->socket, F_GETFL, 0);
    fcntl(hb->socket, F_SETFL, flags | O_NONBLOCK);
#endif
    return 0;
}

/**
 * @brief Perform Heartbeat Connect.
 *
 * @param hb [out] Hb value.
 * @param peer_rank [in] Peer Rank value.
 * @param addr [in] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heartbeat_connect(SNEPPXHeartbeat* hb, int peer_rank,
                              const char* addr, int port) {
    if (!hb || !addr || peer_rank < 0 || peer_rank >= hb->world_size) return -1;
    memset(&hb->peer_addrs[peer_rank], 0, sizeof(struct sockaddr_in));
    hb->peer_addrs[peer_rank].sin_family = AF_INET;
    hb->peer_addrs[peer_rank].sin_addr.s_addr = inet_addr(addr);
    hb->peer_addrs[peer_rank].sin_port = htons((uint16_t)(port > 0 ? port : SNEPPX_HB_PORT_BASE + peer_rank));
    hb->num_peers++;
    return 0;
}

/**
 * @brief Perform Heartbeat Send.
 *
 * @param hb [out] Hb value.
 * @param peer_rank [in] Peer Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heartbeat_send(SNEPPXHeartbeat* hb, int peer_rank,
                           SNEPPXHeartbeatStatus status) {
    if (!hb || peer_rank < 0 || peer_rank >= hb->world_size) return -1;
    SNEPPXHeartbeatMessage msg;
    memset(&msg, 0, sizeof(msg));
    msg.magic = SNEPPX_HB_MAGIC;
    msg.rank = hb->rank;
    msg.world_size = hb->world_size;
    msg.status = status;
    msg.timestamp_ns = snepx_ns_now();
    int ret = sendto(hb->socket, (const char*)&msg, sizeof(msg), 0,
                     (struct sockaddr*)&hb->peer_addrs[peer_rank],
                     sizeof(struct sockaddr_in));
    return ret > 0 ? 0 : -1;
}

/**
 * @brief Perform Heartbeat Recv.
 *
 * @param hb [out] Hb value.
 * @param msg [out] Msg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heartbeat_recv(SNEPPXHeartbeat* hb, SNEPPXHeartbeatMessage* msg,
                           int timeout_ms) {
    if (!hb || !msg) return -1;
    (void)timeout_ms;
    struct sockaddr_in from;
    socklen_t from_len = sizeof(from);
    int ret = recvfrom(hb->socket, (char*)msg, sizeof(*msg), 0,
                       (struct sockaddr*)&from, &from_len);
    if (ret <= 0) return -1;
    if (msg->magic != SNEPPX_HB_MAGIC) return -1;
    if (msg->rank >= 0 && msg->rank < hb->world_size) {
        hb->peer_status[msg->rank] = msg->status;
        hb->last_heartbeat_ns[msg->rank] = snepx_ns_now();
        hb->missed_count[msg->rank] = 0;
    }
    return 0;
}

/**
 * @brief Perform Heartbeat Check Alive.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heartbeat_check_alive(SNEPPXHeartbeat* hb) {
    if (!hb) return 0;
    int64_t now = snepx_ns_now();
    int alive_count = 0;
    for (int i = 0; i < hb->world_size; i++) {
        if (i == hb->rank) { alive_count++; continue; }
        int64_t elapsed = now - hb->last_heartbeat_ns[i];
        int64_t elapsed_ms = elapsed / 1000000LL;
        if (elapsed_ms > hb->timeout_ms) {
            hb->missed_count[i]++;
            if (hb->missed_count[i] >= 3) {
                hb->peer_status[i] = SNEPPX_HB_DEAD;
            } else {
                hb->peer_status[i] = SNEPPX_HB_SUSPECT;
            }
        } else {
            hb->peer_status[i] = SNEPPX_HB_ALIVE;
            alive_count++;
        }
    }
    return alive_count;
}

/**
 * @brief Perform Heartbeat Get Alive Ranks.
 *
 * @param hb [out] Hb value.
 * @param alive_ranks [out] Alive Ranks value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_heartbeat_get_alive_ranks(SNEPPXHeartbeat* hb, int* alive_ranks,
                                      int max_count) {
    if (!hb || !alive_ranks) return 0;
    int count = 0;
    for (int i = 0; i < hb->world_size && count < max_count; i++) {
        if (hb->peer_status[i] == SNEPPX_HB_ALIVE) {
            alive_ranks[count++] = i;
        }
    }
    return count;
}

/**
 * @brief Destroy Heartbeat.
 */
void SNEPPX_heartbeat_destroy(SNEPPXHeartbeat* hb) {
    if (!hb) return;
    hb->running = 0;
    if (hb->socket != SNEPPX_INVALID_SOCKET) {
#ifdef _WIN32
        closesocket(hb->socket);
        winsock_init_count--;
        if (winsock_init_count == 0) WSACleanup();
#else
        close(hb->socket);
#endif
/*
 * SNEPPX - Distributed Heartbeat
 *
 * WHAT
 *   Distributed Heartbeat.
 *
 * CONCEPT
 *   Distributed Heartbeat implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


    }
    free(hb->peer_addrs);
    free(hb->peer_status);
    free(hb->last_heartbeat_ns);
    free(hb->missed_count);
    free(hb);
}
