#include "http_server.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <windows.h>
  #pragma comment(lib, "ws2_32.lib")
/*
 * SNEPPX - Http Server
 *
 * WHAT
 *   Http Server.
 *
 * CONCEPT
 *   Provides the Http Server.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


  typedef SOCKET sock_t;
  #define SNEPPX_INVALID_SOCKET INVALID_SOCKET
  #define snepx_close closesocket
  #define snepx_sleep(ms) Sleep(ms)
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #include <pthread.h>
  #include <sys/time.h>
  #include <dirent.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <limits.h>
  typedef int sock_t;
  #define SNEPPX_INVALID_SOCKET (-1)
  #define snepx_close close
  #define snepx_sleep(ms) usleep((ms) * 1000)
#endif

#define MAX_ROUTES 128
#define MAX_MIDDLEWARE 32
#define MAX_THREADS 16
#define BUF_SIZE 65536
#define MAX_HEADERS 64
#define MAX_REQUEST_HEADER_SIZE 16384U
#define MAX_REQUEST_BODY_SIZE 32768U
#define MAX_ROUTE_PARAMS 8

/* ---- Request / Response ---- */

struct SNEPPX_HttpRequest {
    char method[16];
    char path[1024];
    char query[1024];
    char headers[MAX_HEADERS][256];
    int num_headers;
    char body[BUF_SIZE];
    size_t body_len;
    void* userdata;
    sock_t client_fd;
    SNEPPX_HttpServer* server;
    /* Path parameters captured from routes like /v1/models/{id} */
    char params[MAX_ROUTE_PARAMS][2][256];
    int num_params;
};

struct SNEPPX_HttpResponse {
    int status_code;
    char headers[MAX_HEADERS][256];
    int num_headers;
    char body[BUF_SIZE];
    size_t body_len;
};

/* ---- Route / Middleware ---- */

typedef struct {
    char method[16];
    char path_pattern[256];
    SNEPPX_http_handler_fn handler;
    void* userdata;
} RouteEntry;

typedef struct {
    SNEPPX_http_middleware_fn fn;
    void* userdata;
} MiddlewareEntry;

struct SNEPPX_HttpServer {
    int port;
    int num_threads;
    volatile int running;
    sock_t listen_fd;
    RouteEntry routes[MAX_ROUTES];
    int num_routes;
    MiddlewareEntry middleware[MAX_MIDDLEWARE];
    int num_middleware;
    char static_dirs[MAX_ROUTES][2][256];
    int num_static_dirs;
    int active_workers;
    int accept_loop_active;
    int stop_requested;
#ifdef _WIN32
    int winsock_started;
    CRITICAL_SECTION worker_lock;
    CONDITION_VARIABLE worker_cond;
#else
    pthread_mutex_t worker_lock;
    pthread_cond_t worker_cond;
#endif
};


/* ---- Bounded worker lifecycle ---- */
static void server_lock(SNEPPX_HttpServer* srv) {
#ifdef _WIN32
    EnterCriticalSection(&srv->worker_lock);
#else
    pthread_mutex_lock(&srv->worker_lock);
#endif
}
static void server_unlock(SNEPPX_HttpServer* srv) {
#ifdef _WIN32
    LeaveCriticalSection(&srv->worker_lock);
#else
    pthread_mutex_unlock(&srv->worker_lock);
#endif
}
static void server_wake_waiters(SNEPPX_HttpServer* srv) {
#ifdef _WIN32
    WakeAllConditionVariable(&srv->worker_cond);
#else
    pthread_cond_broadcast(&srv->worker_cond);
#endif
}
static int server_lock_init(SNEPPX_HttpServer* srv) {
#ifdef _WIN32
    InitializeCriticalSection(&srv->worker_lock);
    InitializeConditionVariable(&srv->worker_cond);
    return 0;
#else
    if (pthread_mutex_init(&srv->worker_lock, NULL) != 0) return -1;
    if (pthread_cond_init(&srv->worker_cond, NULL) != 0) {
        pthread_mutex_destroy(&srv->worker_lock);
        return -1;
    }
    return 0;
#endif
}
static void server_lock_destroy(SNEPPX_HttpServer* srv) {
#ifdef _WIN32
    DeleteCriticalSection(&srv->worker_lock);
#else
    pthread_cond_destroy(&srv->worker_cond);
    pthread_mutex_destroy(&srv->worker_lock);
#endif
}
static int server_is_running(SNEPPX_HttpServer* srv) {
    server_lock(srv);
    int running = srv->running && !srv->stop_requested;
    server_unlock(srv);
    return running;
}
static int server_reserve_worker(SNEPPX_HttpServer* srv) {
    server_lock(srv);
    int allowed = srv->running && !srv->stop_requested &&
                  srv->active_workers < srv->num_threads;
    if (allowed) srv->active_workers++;
    server_unlock(srv);
    return allowed;
}
static void server_release_worker(SNEPPX_HttpServer* srv) {
    server_lock(srv);
    if (srv->active_workers > 0) srv->active_workers--;
    server_wake_waiters(srv);
    server_unlock(srv);
}
static void server_wait_idle(SNEPPX_HttpServer* srv) {
    server_lock(srv);
    while (srv->active_workers > 0 || srv->accept_loop_active) {
#ifdef _WIN32
        SleepConditionVariableCS(&srv->worker_cond, &srv->worker_lock, INFINITE);
#else
        pthread_cond_wait(&srv->worker_cond, &srv->worker_lock);
#endif
    }
    server_unlock(srv);
}
static void server_finish_accept_loop(SNEPPX_HttpServer* srv, sock_t listen_fd) {
    sock_t close_fd = SNEPPX_INVALID_SOCKET;
    server_lock(srv);
    srv->running = 0;
    srv->accept_loop_active = 0;
    if (listen_fd != SNEPPX_INVALID_SOCKET && srv->listen_fd == listen_fd) {
        close_fd = srv->listen_fd;
        srv->listen_fd = SNEPPX_INVALID_SOCKET;
    }
    server_wake_waiters(srv);
    server_unlock(srv);
    if (close_fd != SNEPPX_INVALID_SOCKET) {
#ifdef _WIN32
        shutdown(close_fd, SD_BOTH);
#else
        shutdown(close_fd, SHUT_RDWR);
#endif
        snepx_close(close_fd);
    }
}
static void server_set_client_timeouts(sock_t client_fd) {
#ifdef _WIN32
    DWORD timeout_ms = 5000;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout_ms, sizeof(timeout_ms));
#else
    struct timeval timeout = {5, 0};
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif
}
static void server_send_busy_response(sock_t client_fd) {
    static const char response[] =
        "HTTP/1.1 503 Service Unavailable\r\n"
        "Connection: close\r\n"
        "Content-Length: 0\r\n\r\n";
    size_t sent = 0;
    while (sent < sizeof(response) - 1) {
        int n = send(client_fd, response + sent, (int)(sizeof(response) - 1 - sent), 0);
        if (n <= 0) break;
        sent += (size_t)n;
    }
}

/* ---- Utility ---- */

static int str_starts_with(const char* s, const char* prefix) {
    while (*prefix) { if (*s++ != *prefix++) return 0; }
    return 1;
}

/* Resolve both paths before comparing them so traversal and symlink escapes
 * cannot expose files outside a configured static directory. */
static int path_is_within_directory(const char* root, const char* candidate) {
#ifdef _WIN32
    char resolved_root[MAX_PATH * 4];
    char resolved_candidate[MAX_PATH * 4];
    HANDLE root_handle = CreateFileA(root, FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (root_handle == INVALID_HANDLE_VALUE) return 0;
    HANDLE candidate_handle = CreateFileA(candidate, FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (candidate_handle == INVALID_HANDLE_VALUE) {
        CloseHandle(root_handle);
        return 0;
    }
    DWORD root_len = GetFinalPathNameByHandleA(root_handle, resolved_root,
        (DWORD)sizeof(resolved_root), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    DWORD candidate_len = GetFinalPathNameByHandleA(candidate_handle, resolved_candidate,
        (DWORD)sizeof(resolved_candidate), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    CloseHandle(candidate_handle);
    CloseHandle(root_handle);
    if (root_len == 0 || candidate_len == 0 ||
        root_len >= sizeof(resolved_root) || candidate_len >= sizeof(resolved_candidate) ||
        candidate_len <= root_len) return 0;
    if (_strnicmp(resolved_root, resolved_candidate, root_len) != 0) return 0;
    return resolved_root[root_len - 1] == '\\' || resolved_root[root_len - 1] == '/' ||
           resolved_candidate[root_len] == '\\' || resolved_candidate[root_len] == '/';
#else
    char resolved_root[PATH_MAX];
    char resolved_candidate[PATH_MAX];
    if (!realpath(root, resolved_root) || !realpath(candidate, resolved_candidate)) return 0;
    size_t root_len = strlen(resolved_root);
    size_t candidate_len = strlen(resolved_candidate);
    if (candidate_len <= root_len || strncmp(resolved_root, resolved_candidate, root_len) != 0) return 0;
    if (root_len == 1 && resolved_root[0] == '/') return resolved_candidate[0] == '/';
    return resolved_candidate[root_len] == '/';
#endif
}

static void split_query(const char* path, char* out_path, char* out_query) {
    const char* q = strchr(path, '?');
    if (q) {
        size_t plen = (size_t)(q - path);
        if (plen > 1023) plen = 1023;
        memcpy(out_path, path, plen);
        out_path[plen] = '\0';
        strncpy(out_query, q + 1, 1023);
        out_query[1023] = '\0';
    } else {
        strncpy(out_path, path, 1023);
        out_path[1023] = '\0';
        out_query[0] = '\0';
    }
}

static int match_route(const char* pattern, const char* path, SNEPPX_HttpRequest* req) {
    /* Support {param} segments in the pattern (e.g. /v1/models/{id}).
     * Captured values are stored in req->params. */
    req->num_params = 0;
    while (*pattern && *path) {
        if (*pattern == '{') {
            const char* close = strchr(pattern, '}');
            if (!close) return 0;
            size_t name_len = (size_t)(close - pattern) - 1;
            const char* seg_end = strchr(path, '/');
            size_t seg_len = seg_end ? (size_t)(seg_end - path) : strlen(path);
            if (seg_len == 0) return 0;
            if (req->num_params < MAX_ROUTE_PARAMS) {
                size_t n = name_len < 255 ? name_len : 255;
                memcpy(req->params[req->num_params][0], pattern + 1, n);
                req->params[req->num_params][0][n] = '\0';
                n = seg_len < 255 ? seg_len : 255;
                memcpy(req->params[req->num_params][1], path, n);
                req->params[req->num_params][1][n] = '\0';
                req->num_params++;
            }
            pattern = close + 1;
            path = seg_end ? seg_end : path + seg_len;
        } else if (*pattern == *path) {
            pattern++;
            path++;
        } else {
            return 0;
        }
    }
    return *pattern == '\0' && *path == '\0';
}

static int ascii_lower(int ch) {
    return (ch >= 'A' && ch <= 'Z') ? ch + ('a' - 'A') : ch;
}
static int ascii_equal_n(const char* a, const char* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (ascii_lower((unsigned char)a[i]) != ascii_lower((unsigned char)b[i])) return 0;
    }
    return 1;
}
static const char* find_header(SNEPPX_HttpRequest* req, const char* name) {
    if (!req || !name) return NULL;
    size_t wanted_len = strlen(name);
    for (int i = 0; i < req->num_headers; i++) {
        const char* colon = strchr(req->headers[i], ':');
        if (!colon) continue;
        size_t name_len = (size_t)(colon - req->headers[i]);
        if (name_len == wanted_len && ascii_equal_n(req->headers[i], name, name_len)) {
            const char* value = colon + 1;
            while (*value == ' ' || *value == '\t') value++;
            return value;
        }
    }
    return NULL;
}

static int is_http_token_char(unsigned char ch);
static void add_header(SNEPPX_HttpResponse* resp, const char* name, const char* value) {
    if (!resp || !name || !value || resp->num_headers >= MAX_HEADERS) return;
    size_t name_len = strlen(name);
    size_t value_len = strlen(value);
    if (name_len == 0 || name_len > sizeof(resp->headers[0]) - 3 ||
        value_len > sizeof(resp->headers[0]) - name_len - 3) return;
    for (size_t i = 0; i < name_len; i++)
        if (!is_http_token_char((unsigned char)name[i])) return;
    for (size_t i = 0; i < value_len; i++) {
        unsigned char ch = (unsigned char)value[i];
        if ((ch < 0x20 && ch != '\t') || ch == 0x7f) return;
    }
    if ((name_len == 14 && ascii_equal_n(name, "Content-Length", 14)) ||
        (name_len == 17 && ascii_equal_n(name, "Transfer-Encoding", 17))) return;
    char* dest = resp->headers[resp->num_headers];
    memcpy(dest, name, name_len);
    dest[name_len] = ':';
    dest[name_len + 1] = ' ';
    memcpy(dest + name_len + 2, value, value_len);
    dest[name_len + 2 + value_len] = '\0';
    resp->num_headers++;
}

/* ---- Parse HTTP request ---- */

static int is_http_token_char(unsigned char ch) {
    if ((ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'Z') ||
        (ch >= 'a' && ch <= 'z')) return 1;
    return strchr("!#$%&'*+-.^_`|~", ch) != NULL;
}
static const char* find_crlf(const char* start, const char* end) {
    for (const char* p = start; p + 1 < end; p++) {
        if (p[0] == '\r' && p[1] == '\n') return p;
    }
    return NULL;
}
static int parse_request_framing(const char* raw, size_t raw_len,
                                 size_t* header_len_out, size_t* body_len_out) {
    if (!raw || !header_len_out || !body_len_out) return -1;
    size_t header_len = 0;
    for (size_t i = 0; i + 3 < raw_len; i++) {
        if (raw[i] == '\r' && raw[i+1] == '\n' && raw[i+2] == '\r' && raw[i+3] == '\n') {
            header_len = i + 4;
            break;
        }
    }
    if (!header_len) return 0;
    if (header_len > MAX_REQUEST_HEADER_SIZE) return -3;
    const char* header_end = raw + header_len - 2;
    const char* request_line_end = find_crlf(raw, header_end + 2);
    if (!request_line_end || request_line_end == raw) return -1;
    char seen[MAX_HEADERS][256];
    int seen_count = 0, has_content_length = 0;
    size_t body_len = 0;
    const char* p = request_line_end + 2;
    while (p < header_end) {
        const char* line_end = find_crlf(p, header_end + 2);
        if (!line_end || line_end == p || line_end > header_end || p[0] == ' ' || p[0] == '\t') return -1;
        const char* colon = (const char*)memchr(p, ':', (size_t)(line_end - p));
        if (!colon || colon == p) return -1;
        size_t name_len = (size_t)(colon - p);
        if (name_len >= sizeof(seen[0])) return -1;
        for (size_t i = 0; i < name_len; i++) if (!is_http_token_char((unsigned char)p[i])) return -1;
        const char* value = colon + 1;
        const char* value_end = line_end;
        while (value < value_end && (*value == ' ' || *value == '\t')) value++;
        while (value_end > value && (value_end[-1] == ' ' || value_end[-1] == '\t')) value_end--;
        for (const char* q = value; q < value_end; q++) {
            unsigned char ch = (unsigned char)*q;
            if ((ch < 0x20 && ch != '\t') || ch == 0x7f) return -1;
        }
        for (int i = 0; i < seen_count; i++) {
            if (strlen(seen[i]) == name_len && ascii_equal_n(seen[i], p, name_len)) return -1;
        }
        if (seen_count >= MAX_HEADERS) return -1;
        memcpy(seen[seen_count], p, name_len);
        seen[seen_count][name_len] = '\0';
        seen_count++;
        if (name_len == 17 && ascii_equal_n(p, "Transfer-Encoding", 17)) return -1;
        if (name_len == 6 && ascii_equal_n(p, "Expect", 6)) return -1;
        if (name_len == 14 && ascii_equal_n(p, "Content-Length", 14)) {
            if (has_content_length || value == value_end) return -1;
            has_content_length = 1;
            size_t parsed = 0;
            for (const char* q = value; q < value_end; q++) {
                if (*q < '0' || *q > '9') return -1;
                unsigned digit = (unsigned)(*q - '0');
                if (parsed > (SIZE_MAX - digit) / 10) return -1;
                parsed = parsed * 10 + digit;
                if (parsed > MAX_REQUEST_BODY_SIZE) return -2;
            }
            body_len = parsed;
        }
        p = line_end + 2;
    }
    *header_len_out = header_len;
    *body_len_out = body_len;
    return 1;
}

static int parse_http_request(const char* raw, size_t raw_len, SNEPPX_HttpRequest* req) {
    if (!raw || !req) return -1;
    memset(req, 0, sizeof(*req));
    size_t header_len = 0, body_len = 0;
    if (parse_request_framing(raw, raw_len, &header_len, &body_len) != 1 ||
        header_len > raw_len || body_len != raw_len - header_len) return -1;
    const char* header_end = raw + header_len - 2;
    const char* line_end = find_crlf(raw, header_end + 2);
    if (!line_end || line_end == raw) return -1;
    const char* first_space = (const char*)memchr(raw, ' ', (size_t)(line_end - raw));
    if (!first_space || first_space == raw) return -1;
    const char* second_space = (const char*)memchr(first_space + 1, ' ', (size_t)(line_end - first_space - 1));
    if (!second_space || second_space == first_space + 1 ||
        memchr(second_space + 1, ' ', (size_t)(line_end - second_space - 1))) return -1;
    size_t method_len = (size_t)(first_space - raw);
    size_t target_len = (size_t)(second_space - first_space - 1);
    size_t version_len = (size_t)(line_end - second_space - 1);
    if (method_len >= sizeof(req->method) || target_len == 0 || target_len >= 1024) return -1;
    const char* target = first_space + 1;
    const char* version = second_space + 1;
    if (!((version_len == 8 && memcmp(version, "HTTP/1.0", 8) == 0) ||
          (version_len == 8 && memcmp(version, "HTTP/1.1", 8) == 0)) || target[0] != '/') return -1;
    for (size_t i = 0; i < method_len; i++) {
        if (!is_http_token_char((unsigned char)raw[i])) return -1;
        req->method[i] = raw[i];
    }
    req->method[method_len] = '\0';
    size_t path_len = target_len, query_len = 0;
    const char* query_mark = (const char*)memchr(target, '?', target_len);
    if (query_mark) {
        path_len = (size_t)(query_mark - target);
        query_len = target_len - path_len - 1;
        if (query_len >= sizeof(req->query)) return -1;
        memcpy(req->query, query_mark + 1, query_len);
        req->query[query_len] = '\0';
    }
    if (path_len == 0 || path_len >= sizeof(req->path)) return -1;
    for (size_t i = 0; i < path_len; i++) {
        unsigned char ch = (unsigned char)target[i];
        if (ch < 0x21 || ch == 0x7f || ch == '#' || ch == '\\') return -1;
        req->path[i] = (char)ch;
    }
    req->path[path_len] = '\0';
    const char* p = line_end + 2;
    while (p < header_end) {
        const char* h_end = find_crlf(p, header_end + 2);
        if (!h_end || h_end == p || h_end > header_end || req->num_headers >= MAX_HEADERS) return -1;
        const char* colon = (const char*)memchr(p, ':', (size_t)(h_end - p));
        if (!colon || colon == p) return -1;
        size_t name_len = (size_t)(colon - p);
        const char* value = colon + 1;
        const char* value_end = h_end;
        while (value < value_end && (*value == ' ' || *value == '\t')) value++;
        while (value_end > value && (value_end[-1] == ' ' || value_end[-1] == '\t')) value_end--;
        size_t value_len = (size_t)(value_end - value);
        if (name_len + 2 + value_len >= sizeof(req->headers[0])) return -1;
        char* dest = req->headers[req->num_headers];
        memcpy(dest, p, name_len);
        dest[name_len] = ':';
        dest[name_len + 1] = ' ';
        memcpy(dest + name_len + 2, value, value_len);
        dest[name_len + 2 + value_len] = '\0';
        req->num_headers++;
        p = h_end + 2;
    }
    if (body_len >= sizeof(req->body)) return -1;
    if (body_len) memcpy(req->body, raw + header_len, body_len);
    req->body[body_len] = '\0';
    req->body_len = body_len;
    return 0;
}

/* ---- Build HTTP response ---- */

static int build_response(SNEPPX_HttpResponse* resp, char* out, size_t out_max) {
    if (!resp || !out || out_max == 0 || resp->num_headers < 0 ||
        resp->num_headers > MAX_HEADERS || resp->body_len >= sizeof(resp->body)) return -1;
    int sc = resp->status_code;
    if (sc < 100 || sc > 599) return -1;
    const char* status_text = "Unknown";
    switch (sc) {
        case 200: status_text = "OK"; break;
        case 201: status_text = "Created"; break;
        case 202: status_text = "Accepted"; break;
        case 204: status_text = "No Content"; break;
        case 400: status_text = "Bad Request"; break;
        case 401: status_text = "Unauthorized"; break;
        case 403: status_text = "Forbidden"; break;
        case 404: status_text = "Not Found"; break;
        case 405: status_text = "Method Not Allowed"; break;
        case 408: status_text = "Request Timeout"; break;
        case 413: status_text = "Payload Too Large"; break;
        case 415: status_text = "Unsupported Media Type"; break;
        case 422: status_text = "Unprocessable Content"; break;
        case 429: status_text = "Too Many Requests"; break;
        case 431: status_text = "Request Header Fields Too Large"; break;
        case 500: status_text = "Internal Server Error"; break;
        case 503: status_text = "Service Unavailable"; break;
        default: break;
    }
    size_t pos = 0;
    int n = snprintf(out, out_max, "HTTP/1.1 %d %s\r\n", sc, status_text);
    if (n < 0 || (size_t)n >= out_max) return -1;
    pos = (size_t)n;
    int has_content_type = 0;
    for (int i = 0; i < resp->num_headers; i++) {
        const char* colon = strchr(resp->headers[i], ':');
        if (!colon) return -1;
        size_t name_len = (size_t)(colon - resp->headers[i]);
        if (name_len == 12 && ascii_equal_n(resp->headers[i], "Content-Type", 12))
            has_content_type = 1;
        size_t remaining = out_max - pos;
        n = snprintf(out + pos, remaining, "%s\r\n", resp->headers[i]);
        if (n < 0 || (size_t)n >= remaining) return -1;
        pos += (size_t)n;
    }
    if (!has_content_type) {
        size_t remaining = out_max - pos;
        n = snprintf(out + pos, remaining, "Content-Type: text/plain\r\n");
        if (n < 0 || (size_t)n >= remaining) return -1;
        pos += (size_t)n;
    }
    size_t remaining = out_max - pos;
    n = snprintf(out + pos, remaining, "Content-Length: %u\r\n\r\n", (unsigned)resp->body_len);
    if (n < 0 || (size_t)n >= remaining) return -1;
    pos += (size_t)n;
    if (pos >= out_max || resp->body_len > out_max - pos - 1) return -1;
    if (resp->body_len) memcpy(out + pos, resp->body, resp->body_len);
    pos += resp->body_len;
    out[pos] = '\0';
    return pos > (size_t)INT_MAX ? -1 : (int)pos;
}

/* ---- Handle single connection ---- */

static int send_all(sock_t fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        int n = send(fd, data + sent, (int)(len - sent), 0);
        if (n <= 0) return -1;
        sent += (size_t)n;
    }
    return 0;
}
static void send_simple_error(sock_t fd, int status, const char* reason, const char* body) {
    char response[512];
    size_t body_len = strlen(body);
    int n = snprintf(response, sizeof(response),
        "HTTP/1.1 %d %s\r\nContent-Type: text/plain\r\nContent-Length: %u\r\nConnection: close\r\n\r\n%s",
        status, reason, (unsigned)body_len, body);
    if (n > 0 && (size_t)n < sizeof(response)) send_all(fd, response, (size_t)n);
}

static void handle_connection(sock_t client_fd, SNEPPX_HttpServer* srv) {
    char raw[BUF_SIZE];
    size_t raw_len = 0, header_len = 0, body_len = 0, expected_len = 0;
    int framing_ready = 0;
    while (raw_len < sizeof(raw) - 1) {
        int n = recv(client_fd, raw + raw_len, (int)(sizeof(raw) - 1 - raw_len), 0);
        if (n <= 0) {
            if (n < 0) send_simple_error(client_fd, 408, "Request Timeout", "Request Timeout\n");
            else send_simple_error(client_fd, 400, "Bad Request", "Bad Request\n");
            snepx_close(client_fd);
            return;
        }
        raw_len += (size_t)n;
        raw[raw_len] = '\0';
        if (!framing_ready) {
            int framing = parse_request_framing(raw, raw_len, &header_len, &body_len);
            if (framing == -1) {
                send_simple_error(client_fd, 400, "Bad Request", "Bad Request\n");
                snepx_close(client_fd); return;
            }
            if (framing == -2) {
                send_simple_error(client_fd, 413, "Payload Too Large", "Payload Too Large\n");
                snepx_close(client_fd); return;
            }
            if (framing == -3 || (framing == 0 && raw_len >= MAX_REQUEST_HEADER_SIZE)) {
                send_simple_error(client_fd, 431, "Request Header Fields Too Large", "Request Headers Too Large\n");
                snepx_close(client_fd); return;
            }
            if (framing == 1) {
                if (header_len > SIZE_MAX - body_len || header_len + body_len > sizeof(raw) - 1) {
                    send_simple_error(client_fd, 413, "Payload Too Large", "Payload Too Large\n");
                    snepx_close(client_fd); return;
                }
                expected_len = header_len + body_len;
                framing_ready = 1;
            }
        }
        if (framing_ready) {
            if (raw_len > expected_len) {
                send_simple_error(client_fd, 400, "Bad Request", "Unexpected bytes after request body\n");
                snepx_close(client_fd); return;
            }
            if (raw_len == expected_len) break;
        }
    }
    if (!framing_ready || raw_len != expected_len) {
        send_simple_error(client_fd, 413, "Payload Too Large", "Request Too Large\n");
        snepx_close(client_fd); return;
    }

    SNEPPX_HttpRequest req;
    SNEPPX_HttpResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.status_code = 200;
    if (parse_http_request(raw, raw_len, &req) != 0) {
        send_simple_error(client_fd, 400, "Bad Request", "Bad Request\n");
        snepx_close(client_fd); return;
    }
    req.client_fd = client_fd;
    req.server = srv;

    int abort = 0;
    for (int i = 0; i < srv->num_middleware; i++) {
        resp.status_code = 200;
        resp.body_len = 0;
        resp.num_headers = 0;
        if (srv->middleware[i].fn(&req, &resp, srv->middleware[i].userdata) != 0) {
            abort = 1;
            break;
        }
    }

    if (!abort) {
        int handled = 0;
        for (int i = 0; i < srv->num_static_dirs; i++) {
            const char* prefix = srv->static_dirs[i][0];
            size_t prefix_len = strlen(prefix);
            if (strncmp(req.path, prefix, prefix_len) != 0 ||
                (req.path[prefix_len] != '\0' && req.path[prefix_len] != '/')) continue;
            handled = 1;
            if (strcmp(req.method, "GET") != 0) {
                resp.status_code = 405;
                add_header(&resp, "Allow", "GET");
                SNEPPX_http_response_set_body_str(&resp, "Method Not Allowed");
                break;
            }
            const char* rel = req.path + prefix_len;
            if (*rel == '/') rel++;
            if (!*rel) {
                resp.status_code = 404;
                SNEPPX_http_response_set_body_str(&resp, "Not Found");
                break;
            }
            char filepath[512];
            int path_len = snprintf(filepath, sizeof(filepath), "%s/%s", srv->static_dirs[i][1], rel);
            if (path_len < 0 || (size_t)path_len >= sizeof(filepath) ||
                !path_is_within_directory(srv->static_dirs[i][1], filepath)) {
                resp.status_code = 404;
                SNEPPX_http_response_set_body_str(&resp, "Not Found");
                break;
            }
            FILE* f = fopen(filepath, "rb");
            if (!f) {
                resp.status_code = 404;
                SNEPPX_http_response_set_body_str(&resp, "Not Found");
                break;
            }
            if (fseek(f, 0, SEEK_END) != 0) {
                fclose(f);
                resp.status_code = 500;
                SNEPPX_http_response_set_body_str(&resp, "Static file read failed");
                break;
            }
            long flen = ftell(f);
            if (flen < 0 || (unsigned long)flen >= sizeof(resp.body) || fseek(f, 0, SEEK_SET) != 0) {
                fclose(f);
                resp.status_code = 413;
                SNEPPX_http_response_set_body_str(&resp, "Static file is too large");
                break;
            }
            size_t file_size = (size_t)flen;
            size_t bytes_read = file_size ? fread(resp.body, 1, file_size, f) : 0;
            int read_failed = bytes_read != file_size || ferror(f);
            fclose(f);
            if (read_failed) {
                resp.status_code = 500;
                SNEPPX_http_response_set_body_str(&resp, "Static file read failed");
                break;
            }
            resp.body_len = file_size;
            resp.body[file_size] = '\0';
            const char* content_type = "application/octet-stream";
            const char* ext = strrchr(filepath, '.');
            if (ext) {
                if (strcmp(ext, ".html") == 0) content_type = "text/html";
                else if (strcmp(ext, ".css") == 0) content_type = "text/css";
                else if (strcmp(ext, ".js") == 0) content_type = "application/javascript";
                else if (strcmp(ext, ".json") == 0) content_type = "application/json";
                else if (strcmp(ext, ".png") == 0) content_type = "image/png";
                else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0) content_type = "image/jpeg";
            }
            add_header(&resp, "Content-Type", content_type);
            break;
        }
        if (!handled) {
            for (int i = 0; i < srv->num_routes; i++) {
                if (strcmp(req.method, srv->routes[i].method) == 0 &&
                    match_route(srv->routes[i].path_pattern, req.path, &req)) {
                    resp.status_code = 200;
                    resp.body_len = 0;
                    resp.num_headers = 0;
                    if (srv->routes[i].handler(&req, &resp, srv->routes[i].userdata) != 0) {
                        if (resp.status_code == 200) resp.status_code = 500;
                    }
                    handled = 1;
                    break;
                }
            }
        }
        if (!handled) {
            resp.status_code = 404;
            SNEPPX_http_response_set_body_str(&resp, "Not Found");
        }
    }

    char out[BUF_SIZE];
    int response_len = build_response(&resp, out, sizeof(out));
    if (response_len <= 0 || (size_t)response_len >= sizeof(out)) {
        send_simple_error(client_fd, 500, "Internal Server Error", "Internal Server Error\n");
    } else {
        send_all(client_fd, out, (size_t)response_len);
    }
    snepx_close(client_fd);
}

/* ---- Thread worker ---- */

#ifdef _WIN32
DWORD WINAPI worker_thread(LPVOID arg) {
    void** argv = (void**)arg;
    sock_t client_fd = *(sock_t*)argv[0];
    SNEPPX_HttpServer* srv = (SNEPPX_HttpServer*)argv[1];
    handle_connection(client_fd, srv);
    server_release_worker(srv);
    free(argv[0]);
    free(argv);
    return 0;
}
#else
static void* worker_thread(void* arg) {
    void** argv = (void**)arg;
    sock_t client_fd = *(sock_t*)argv[0];
    SNEPPX_HttpServer* srv = (SNEPPX_HttpServer*)argv[1];
    handle_connection(client_fd, srv);
    server_release_worker(srv);
    free(argv[0]);
    free(argv);
    return NULL;
}
#endif

/* ---- API Implementation ---- */

/**
 * @brief Create Http Server.
 *
 * @param port [in] Port value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPX_HttpServer* SNEPPX_http_server_create(int port, int num_threads) {
    if (port < 1 || port > 65535) return NULL;
    SNEPPX_HttpServer* srv = (SNEPPX_HttpServer*)calloc(1, sizeof(*srv));
    if (!srv) return NULL;
    if (server_lock_init(srv) != 0) { free(srv); return NULL; }
    srv->port = port;
    srv->num_threads = num_threads < 1 ? 1 : (num_threads > MAX_THREADS ? MAX_THREADS : num_threads);
    srv->running = 0;
    srv->listen_fd = SNEPPX_INVALID_SOCKET;
    return srv;
}

/**
 * @brief Start Http Server.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_server_start(SNEPPX_HttpServer* srv) {
    if (!srv) return -1;
    server_lock(srv);
    if (srv->accept_loop_active || srv->active_workers > 0) {
        server_unlock(srv);
        return -1;
    }
    srv->accept_loop_active = 1;
    srv->stop_requested = 0;
    srv->running = 0;
    server_unlock(srv);

#ifdef _WIN32
    if (!srv->winsock_started) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            server_finish_accept_loop(srv, SNEPPX_INVALID_SOCKET);
            return -1;
        }
        srv->winsock_started = 1;
    }
#endif

    sock_t listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == SNEPPX_INVALID_SOCKET) {
        server_finish_accept_loop(srv, SNEPPX_INVALID_SOCKET);
        return -1;
    }
    server_lock(srv);
    if (srv->stop_requested) {
        server_unlock(srv);
        snepx_close(listen_fd);
        server_finish_accept_loop(srv, SNEPPX_INVALID_SOCKET);
        return -1;
    }
    srv->listen_fd = listen_fd;
    server_unlock(srv);

    int opt = 1;
#ifdef _WIN32
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((unsigned short)srv->port);
    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        server_finish_accept_loop(srv, listen_fd);
        return -1;
    }
    if (listen(listen_fd, srv->num_threads) != 0) {
        server_finish_accept_loop(srv, listen_fd);
        return -1;
    }

    server_lock(srv);
    if (srv->stop_requested) {
        server_unlock(srv);
        server_finish_accept_loop(srv, listen_fd);
        return -1;
    }
    srv->running = 1;
    server_unlock(srv);

    while (server_is_running(srv)) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        sock_t client_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (client_fd == SNEPPX_INVALID_SOCKET) {
            if (!server_is_running(srv)) break;
            snepx_sleep(10);
            continue;
        }

        server_set_client_timeouts(client_fd);
        if (!server_reserve_worker(srv)) {
            server_send_busy_response(client_fd);
            snepx_close(client_fd);
            continue;
        }

        void** args = (void**)malloc(2 * sizeof(void*));
        if (!args) {
            snepx_close(client_fd);
            server_release_worker(srv);
            continue;
        }
        sock_t* pfd = (sock_t*)malloc(sizeof(sock_t));
        if (!pfd) {
            free(args);
            snepx_close(client_fd);
            server_release_worker(srv);
            continue;
        }
        *pfd = client_fd;
        args[0] = (void*)pfd;
        args[1] = (void*)srv;
#ifdef _WIN32
        HANDLE th = CreateThread(NULL, 0, worker_thread, args, 0, NULL);
        if (th) CloseHandle(th);
        else {
            free(pfd); free(args); snepx_close(client_fd);
            server_release_worker(srv);
        }
#else
        pthread_t tid;
        if (pthread_create(&tid, NULL, worker_thread, args) == 0) pthread_detach(tid);
        else {
            free(pfd); free(args); snepx_close(client_fd);
            server_release_worker(srv);
        }
#endif
    }

    server_finish_accept_loop(srv, listen_fd);
    return 0;
}

/**
 * @brief Stop Http Server.
 */
void SNEPPX_http_server_stop(SNEPPX_HttpServer* srv) {
    if (!srv) return;
    server_lock(srv);
    srv->stop_requested = 1;
    srv->running = 0;
    sock_t listen_fd = srv->listen_fd;
#ifdef _WIN32
    if (listen_fd != SNEPPX_INVALID_SOCKET) srv->listen_fd = SNEPPX_INVALID_SOCKET;
#endif
    server_wake_waiters(srv);
    server_unlock(srv);
    if (listen_fd != SNEPPX_INVALID_SOCKET) {
#ifdef _WIN32
        shutdown(listen_fd, SD_BOTH);
        snepx_close(listen_fd); /* closesocket is required to wake a blocking Winsock accept */
#else
        shutdown(listen_fd, SHUT_RDWR); /* accept loop closes the descriptor after it wakes */
#endif
    }
}

/**
 * @brief Destroy Http Server.
 */
void SNEPPX_http_server_destroy(SNEPPX_HttpServer* srv) {
    if (!srv) return;
    SNEPPX_http_server_stop(srv);
    server_wait_idle(srv);
#ifdef _WIN32
    if (srv->winsock_started) WSACleanup();
#endif
    server_lock_destroy(srv);
    free(srv);
}

/**
 * @brief Perform Http Server Add Route.
 *
 * @param srv [out] Srv value.
 * @param method [in] Method value.
 * @param path [in] Path value.
 * @param handler [in] Handler value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_server_add_route(SNEPPX_HttpServer* srv, const char* method, const char* path,
                                  SNEPPX_http_handler_fn handler, void* userdata) {
    if (!srv || !method || !path || !handler || srv->num_routes >= MAX_ROUTES) return -1;
    size_t method_len = strlen(method);
    size_t path_len = strlen(path);
    if (method_len == 0 || method_len >= sizeof(srv->routes[0].method) ||
        path_len == 0 || path_len >= sizeof(srv->routes[0].path_pattern) ||
        path[0] != '/' || strchr(path, '?') || strchr(path, '#') || strchr(path, '\\')) return -1;
    for (size_t i = 0; i < method_len; i++)
        if (!is_http_token_char((unsigned char)method[i])) return -1;
    for (size_t i = 0; i < path_len; i++) {
        unsigned char ch = (unsigned char)path[i];
        if (ch < 0x21 || ch == 0x7f) return -1;
    }
    for (int i = 0; i < srv->num_routes; i++) {
        if (strcmp(srv->routes[i].method, method) == 0 &&
            strcmp(srv->routes[i].path_pattern, path) == 0) return -1;
    }
    int i = srv->num_routes;
    memcpy(srv->routes[i].method, method, method_len + 1);
    memcpy(srv->routes[i].path_pattern, path, path_len + 1);
    srv->routes[i].handler = handler;
    srv->routes[i].userdata = userdata;
    srv->num_routes++;
    return 0;
}

/**
 * @brief Perform Http Server Add Static Dir.
 *
 * @param srv [out] Srv value.
 * @param url_prefix [in] Url Prefix value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_server_add_static_dir(SNEPPX_HttpServer* srv, const char* url_prefix, const char* dir_path) {
    if (!srv || !url_prefix || !dir_path || !url_prefix[0] || !dir_path[0] ||
        srv->num_static_dirs >= MAX_ROUTES) return -1;
    size_t prefix_len = strlen(url_prefix);
    size_t dir_len = strlen(dir_path);
    if (prefix_len < 2 || prefix_len >= sizeof(srv->static_dirs[0][0]) ||
        dir_len >= sizeof(srv->static_dirs[0][1]) || url_prefix[0] != '/' ||
        url_prefix[prefix_len - 1] == '/' || strchr(url_prefix, '?') ||
        strchr(url_prefix, '#') || strchr(url_prefix, '\\')) return -1;
    for (int i = 0; i < srv->num_static_dirs; i++)
        if (strcmp(srv->static_dirs[i][0], url_prefix) == 0) return -1;
    int i = srv->num_static_dirs;
    memcpy(srv->static_dirs[i][0], url_prefix, prefix_len + 1);
    memcpy(srv->static_dirs[i][1], dir_path, dir_len + 1);
    srv->num_static_dirs++;
    return 0;
}

/**
 * @brief Perform Http Server Add Middleware.
 *
 * @param srv [out] Srv value.
 * @param mw [in] Mw value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_server_add_middleware(SNEPPX_HttpServer* srv, SNEPPX_http_middleware_fn mw, void* userdata) {
    if (!srv || !mw || srv->num_middleware >= MAX_MIDDLEWARE) return -1;
    int i = srv->num_middleware;
    srv->middleware[i].fn = mw;
    srv->middleware[i].userdata = userdata;
    srv->num_middleware++;
    return 0;
}

/* ---- Request accessors ---- */

/**
 * @brief Perform Http Request Method.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_http_request_method(SNEPPX_HttpRequest* req) { return req ? req->method : NULL; }
/**
 * @brief Perform Http Request Path.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_http_request_path(SNEPPX_HttpRequest* req) { return req ? req->path : NULL; }
/**
 * @brief Perform Http Request Header.
 *
 * @param req [out] Req value.
 * @param req [in] Req value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_http_request_header(SNEPPX_HttpRequest* req, const char* name) { return find_header(req, name); }
/**
 * @brief Perform Http Request Body.
 *
 * @param req [out] Req value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_http_request_body(SNEPPX_HttpRequest* req, size_t* len) { if (!req) { if (len) *len = 0; return NULL; } if (len) *len = req->body_len; return req->body; }
/**
 * @brief Perform Http Request Query.
 *
 * @param req [out] Req value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_http_request_query(SNEPPX_HttpRequest* req, const char* key) {
    if (!req || !req->query[0] || !key) return NULL;
    const char* q = req->query;
    while (*q) {
        while (*q == '&') q++;
        const char* eq = strchr(q, '=');
        const char* amp = strchr(q, '&');
        size_t klen = strlen(key);
        size_t qlen = eq ? (size_t)(eq - q) : (amp ? (size_t)(amp - q) : strlen(q));
        if (klen == qlen && memcmp(q, key, klen) == 0) {
            return eq ? eq + 1 : "";
        }
        if (amp) q = amp + 1; else break;
    }
    return NULL;
}
/**
 * @brief Perform Http Request Userdata.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_http_request_userdata(SNEPPX_HttpRequest* req) { return req ? req->userdata : NULL; }
/**
 * @brief Perform Http Request Param.
 *
 * @param req [out] Req value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_http_request_param(SNEPPX_HttpRequest* req, const char* name) {
    if (!req || !name) return NULL;
    for (int i = 0; i < req->num_params; i++) {
        if (strcmp(req->params[i][0], name) == 0)
            return req->params[i][1];
    }
    return NULL;
}

/* ---- Response setters ---- */

/**
 * @brief Perform Http Response Set Status.
 *
 * @param resp [out] Resp value.
 */
void SNEPPX_http_response_set_status(SNEPPX_HttpResponse* resp, int status_code) { if (resp && status_code >= 100 && status_code <= 599) resp->status_code = status_code; }
/**
 * @brief Perform Http Response Set Header.
 *
 * @param resp [out] Resp value.
 * @param name [in] Name value.
 */
void SNEPPX_http_response_set_header(SNEPPX_HttpResponse* resp, const char* name, const char* value) {
    add_header(resp, name, value);
}
/**
 * @brief Perform Http Response Set Body.
 *
 * @param resp [out] Resp value.
 * @param data [in] Data value.
 */
void SNEPPX_http_response_set_body(SNEPPX_HttpResponse* resp, const char* data, size_t len) {
    if (!resp || (!data && len > 0)) return;
    if (len > BUF_SIZE - 1) len = BUF_SIZE - 1;
    if (len) memcpy(resp->body, data, len);
    resp->body[len] = '\0';
    resp->body_len = len;
}
/**
 * @brief Perform Http Response Set Body Str.
 *
 * @param resp [out] Resp value.
 */
void SNEPPX_http_response_set_body_str(SNEPPX_HttpResponse* resp, const char* str) {
    if (!resp || !str) return;
    SNEPPX_http_response_set_body(resp, str, strlen(str));
}
/**
 * @brief Perform Http Response Set Json.
 *
 * @param resp [out] Resp value.
 */
void SNEPPX_http_response_set_json(SNEPPX_HttpResponse* resp, const char* json_str) {
    if (!resp || !json_str) return;
    add_header(resp, "Content-Type", "application/json");
    SNEPPX_http_response_set_body_str(resp, json_str);
}
