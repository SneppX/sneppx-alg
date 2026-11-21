#include "grpc_service.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
/*
 * SNEPPX - Grpc Service
 *
 * WHAT
 *   Grpc Service.
 *
 * CONCEPT
 *   Provides the Grpc Service.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


  typedef int socklen_t;
  #define CLOSE_SOCKET closesocket
  #define ISVALIDSOCK(s) ((s) != INVALID_SOCKET)
  #define SOCKET_ERRNO WSAGetLastError()
  #define SHUT_RDWR SD_BOTH
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <errno.h>
  #define CLOSE_SOCKET close
  #define ISVALIDSOCK(s) ((s) >= 0)
  #define SOCKET_ERRNO errno
  typedef int SOCKET;
#endif

static int gRPC_initialized = 0;

#ifdef _WIN32
static int init_winsock(void) {
    if (gRPC_initialized) return 0;
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return -1;
    gRPC_initialized = 1;
    return 0;
}
#else
static int init_winsock(void) { gRPC_initialized = 1; return 0; }
#endif

static int send_all(SOCKET sock, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    while (len > 0) {
#ifdef _WIN32
        int sent = send(sock, p, (int)(len > 65536 ? 65536 : len), 0);
#else
        ssize_t sent = send(sock, p, len > 65536 ? 65536 : len, MSG_NOSIGNAL);
#endif
        if (sent <= 0) return -1;
        p += sent;
        len -= (size_t)sent;
    }
    return 0;
}

static int recv_all(SOCKET sock, void* buf, size_t len) {
    char* p = (char*)buf;
    while (len > 0) {
#ifdef _WIN32
        int n = recv(sock, p, (int)(len > 65536 ? 65536 : len), 0);
#else
        ssize_t n = recv(sock, p, len > 65536 ? 65536 : len, MSG_WAITALL);
#endif
        if (n <= 0) return -1;
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

typedef enum {
    MSG_REGISTER_NODE = 1,
    MSG_GET_WORLD_SIZE,
    MSG_GET_RANK,
    MSG_BARRIER,
    MSG_ALL_GATHER,
    MSG_SEND_TENSOR,
    MSG_RECV_TENSOR,
    MSG_SET_AUTH,
    MSG_ACK = 0x80
} MessageType;

typedef struct {
    uint32_t type;
    uint32_t length;
} MessageHeader;

static int send_message(SOCKET sock, uint32_t type, const void* payload, uint32_t length) {
    MessageHeader hdr;
    hdr.type = type;
    hdr.length = length;
    if (send_all(sock, &hdr, sizeof(hdr)) != 0) return -1;
    if (length > 0 && send_all(sock, payload, length) != 0) return -1;
    return 0;
}

static int recv_message(SOCKET sock, uint32_t* type, void** payload, uint32_t* length) {
    MessageHeader hdr;
    if (recv_all(sock, &hdr, sizeof(hdr)) != 0) return -1;
    *type = hdr.type;
    *length = hdr.length;
    *payload = NULL;
    if (hdr.length > 0) {
        *payload = malloc(hdr.length);
        if (!*payload) return -1;
        if (recv_all(sock, *payload, hdr.length) != 0) { free(*payload); return -1; }
    }
    return 0;
}

static SOCKET gRPC_listen_sock = -1;

/**
 * @brief Start Grpc Server.
 *
 * @param server [out] Server value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_server_start(SNEPPXGRPCServer** server, int port) {
    if (!server) return -1;
    if (init_winsock() != 0) return -1;
    *server = (SNEPPXGRPCServer*)calloc(1, sizeof(SNEPPXGRPCServer));
    if (!*server) return -1;
    (*server)->port = port;
    (*server)->is_running = 1;
    (*server)->server = NULL;
    (*server)->coordination_svc = NULL;
    (*server)->transfer_svc = NULL;

    gRPC_listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (!ISVALIDSOCK(gRPC_listen_sock)) { free(*server); *server = NULL; return -1; }
    int opt = 1;
    setsockopt(gRPC_listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((unsigned short)port);
    if (bind(gRPC_listen_sock, (struct sockaddr*)&addr, sizeof(addr)) != 0 ||
        listen(gRPC_listen_sock, 5) != 0) {
        CLOSE_SOCKET(gRPC_listen_sock);
        gRPC_listen_sock = -1;
        free(*server); *server = NULL;
        return -1;
    }
    return 0;
}

/**
 * @brief Stop Grpc Server.
 */
void SNEPPX_grpc_server_stop(SNEPPXGRPCServer* server) {
    if (!server) return;
    server->is_running = 0;
    if (ISVALIDSOCK(gRPC_listen_sock)) {
        CLOSE_SOCKET(gRPC_listen_sock);
        gRPC_listen_sock = -1;
    }
    free(server);
}

/**
 * @brief Perform Grpc Server Wait.
 */
void SNEPPX_grpc_server_wait(SNEPPXGRPCServer* server) {
    if (!server || !ISVALIDSOCK(gRPC_listen_sock)) return;
    while (server->is_running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        SOCKET client = accept(gRPC_listen_sock, (struct sockaddr*)&client_addr, &addr_len);
        if (!ISVALIDSOCK(client)) break;
        uint32_t type, length;
        void* payload = NULL;
        if (recv_message(client, &type, &payload, &length) == 0) {
            uint32_t ack_type = type | MSG_ACK;
            send_message(client, ack_type, NULL, 0);
            free(payload);
        }
        CLOSE_SOCKET(client);
    }
}

/**
 * @brief Create Grpc Stub.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXGRPCStub* SNEPPX_grpc_stub_create(const char* target) {
    if (!target) return NULL;
    if (init_winsock() != 0) return NULL;
    SNEPPXGRPCStub* stub = (SNEPPXGRPCStub*)calloc(1, sizeof(SNEPPXGRPCStub));
    if (!stub) return NULL;
    snprintf(stub->target, sizeof(stub->target), "%s", target);
    stub->channel = NULL;
    stub->coordination_stub = NULL;
    stub->transfer_stub = NULL;
    stub->connected = 1;
    return stub;
}

/**
 * @brief Destroy Grpc Stub.
 */
void SNEPPX_grpc_stub_destroy(SNEPPXGRPCStub* stub) {
    if (!stub) return;
    if (stub->channel) CLOSE_SOCKET((SOCKET)(size_t)stub->channel);
    free(stub);
}

static int stub_connect(SNEPPXGRPCStub* stub) {
    if (!stub) return -1;
    if (stub->channel) return 0;
    char host[256]; int port = 8080;
    snprintf(host, sizeof(host), "%s", stub->target);
    char* colon = strchr(host, ':');
    if (colon) { *colon = '\0'; port = atoi(colon + 1); if (port <= 0) port = 8080; }

    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
    if (!ISVALIDSOCK(sock)) return -1;
    struct hostent* he = gethostbyname(host);
    if (!he) { CLOSE_SOCKET(sock); return -1; }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    memcpy(&addr.sin_addr, he->h_addr_list[0], (size_t)he->h_length);
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        CLOSE_SOCKET(sock);
        return -1;
    }
    stub->channel = (void*)(size_t)sock;
    stub->connected = 1;
    return 0;
}

/**
 * @brief Perform Grpc Register Node.
 *
 * @param stub [out] Stub value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_register_node(SNEPPXGRPCStub* stub, const SNEPPXGRPCNodeInfo* info) {
    if (!stub || !info) return -1;
    if (stub_connect(stub) != 0) { stub->connected = 0; return -1; }
    SOCKET sock = (SOCKET)(size_t)stub->channel;
    if (send_message(sock, MSG_REGISTER_NODE, info, sizeof(*info)) != 0) return -1;
    uint32_t type, length; void* payload = NULL;
    if (recv_message(sock, &type, &payload, &length) != 0) return -1;
    int ok = (type == (MSG_REGISTER_NODE | MSG_ACK)) ? 0 : -1;
    free(payload);
    return ok;
}

/**
 * @brief Perform Grpc Get World Size.
 *
 * @param stub [out] Stub value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_get_world_size(SNEPPXGRPCStub* stub, int* size) {
    if (!stub || !size) return -1;
    if (!stub->connected) { *size = 1; return 0; }
    SOCKET sock = (SOCKET)(size_t)stub->channel;
    if (send_message(sock, MSG_GET_WORLD_SIZE, NULL, 0) != 0) { *size = 1; return 0; }
    uint32_t type, length; void* payload = NULL;
    if (recv_message(sock, &type, &payload, &length) != 0) { *size = 1; return 0; }
    if (payload && length >= sizeof(int)) memcpy(size, payload, sizeof(int));
    else *size = 1;
    free(payload);
    return 0;
}

/**
 * @brief Perform Grpc Get Rank.
 *
 * @param stub [out] Stub value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_get_rank(SNEPPXGRPCStub* stub, int* rank) {
    if (!stub || !rank) return -1;
    if (!stub->connected) { *rank = 0; return 0; }
    SOCKET sock = (SOCKET)(size_t)stub->channel;
    if (send_message(sock, MSG_GET_RANK, NULL, 0) != 0) { *rank = 0; return 0; }
    uint32_t type, length; void* payload = NULL;
    if (recv_message(sock, &type, &payload, &length) != 0) { *rank = 0; return 0; }
    if (payload && length >= sizeof(int)) memcpy(rank, payload, sizeof(int));
    else *rank = 0;
    free(payload);
    return 0;
}

/**
 * @brief Perform Grpc Barrier.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_barrier(SNEPPXGRPCStub* stub) {
    if (!stub) return -1;
    if (!stub->connected) return 0;
    SOCKET sock = (SOCKET)(size_t)stub->channel;
    if (send_message(sock, MSG_BARRIER, NULL, 0) != 0) return -1;
    uint32_t type, length; void* payload = NULL;
    if (recv_message(sock, &type, &payload, &length) != 0) return -1;
    free(payload);
    return (type == (MSG_BARRIER | MSG_ACK)) ? 0 : -1;
}

/**
 * @brief Perform Grpc All Gather.
 *
 * @param stub [out] Stub value.
 * @param send_buf [in] Send Buf value.
 * @param recv_buf [out] Recv Buf value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_all_gather(SNEPPXGRPCStub* stub, const void* send_buf, void** recv_buf, size_t elem_size) {
    if (!stub || !recv_buf) return -1;
    if (!stub->connected) {
        *recv_buf = malloc(elem_size);
        if (*recv_buf && send_buf) memcpy(*recv_buf, send_buf, elem_size);
        return 0;
    }
    int world_size = 1;
    SNEPPX_grpc_get_world_size(stub, &world_size);
    size_t total = elem_size * (size_t)world_size;
    *recv_buf = malloc(total);
    if (!*recv_buf) return -1;
    SOCKET sock = (SOCKET)(size_t)stub->channel;
    if (send_message(sock, MSG_ALL_GATHER, send_buf, (uint32_t)elem_size) != 0) {
        free(*recv_buf); *recv_buf = NULL; return -1;
    }
    uint32_t type, length; void* payload = NULL;
    if (recv_message(sock, &type, &payload, &length) != 0) {
        free(*recv_buf); *recv_buf = NULL; return -1;
    }
    if (payload && length > 0) {
        memcpy(*recv_buf, payload, length < total ? (size_t)length : total);
        free(payload);
    }
    return 0;
}

/**
 * @brief Perform Grpc Send Tensor.
 *
 * @param stub [out] Stub value.
 * @param tensor [in] Tensor value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_send_tensor(SNEPPXGRPCStub* stub, const void* tensor, int dest_rank) {
    if (!stub || !tensor) return -1;
    (void)dest_rank;
    if (!stub->connected) return 0;
    SOCKET sock = (SOCKET)(size_t)stub->channel;
    if (send_message(sock, MSG_SEND_TENSOR, tensor, sizeof(SNEPPXTensor)) != 0) return -1;
    uint32_t type, length; void* payload = NULL;
    if (recv_message(sock, &type, &payload, &length) != 0) return -1;
    free(payload);
    return (type == (MSG_SEND_TENSOR | MSG_ACK)) ? 0 : -1;
}

/**
 * @brief Perform Grpc Recv Tensor.
 *
 * @param stub [out] Stub value.
 * @param tensor [out] Tensor value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_recv_tensor(SNEPPXGRPCStub* stub, void** tensor, int src_rank) {
    if (!stub || !tensor) return -1;
    (void)src_rank;
    if (!stub->connected) { *tensor = NULL; return 0; }
    SOCKET sock = (SOCKET)(size_t)stub->channel;
    if (send_message(sock, MSG_RECV_TENSOR, NULL, 0) != 0) { *tensor = NULL; return -1; }
    uint32_t type, length; void* payload = NULL;
    if (recv_message(sock, &type, &payload, &length) != 0) { *tensor = NULL; return -1; }
    if (payload && length > 0) {
        *tensor = malloc(length);
        if (*tensor) memcpy(*tensor, payload, length);
        free(payload);
    } else {
        *tensor = NULL;
    }
    return 0;
}

/**
 * @brief Perform Grpc Set Auth Token.
 *
 * @param stub [out] Stub value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_set_auth_token(SNEPPXGRPCStub* stub, const char* token) {
    if (!stub || !token) return -1;
    if (!stub->connected) return 0;
    SOCKET sock = (SOCKET)(size_t)stub->channel;
    uint32_t len = (uint32_t)(strlen(token) + 1);
    if (send_message(sock, MSG_SET_AUTH, token, len) != 0) return -1;
    uint32_t type, length; void* payload = NULL;
    if (recv_message(sock, &type, &payload, &length) != 0) return -1;
    free(payload);
    return (type == (MSG_SET_AUTH | MSG_ACK)) ? 0 : -1;
}

/**
 * @brief Perform Grpc Status String.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_grpc_status_string(SNEPPXGRPCStatus status) {
    switch (status) {
        case SNEPPX_GRPC_OK: return "OK";
        case SNEPPX_GRPC_UNAVAILABLE: return "UNAVAILABLE";
        case SNEPPX_GRPC_DEADLINE_EXCEEDED: return "DEADLINE_EXCEEDED";
        case SNEPPX_GRPC_INTERNAL: return "INTERNAL";
        case SNEPPX_GRPC_UNAUTHENTICATED: return "UNAUTHENTICATED";
        default: return "UNKNOWN";
    }
}
