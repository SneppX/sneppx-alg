#include "model_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/*
 * SNEPPX - Registry
 *
 * WHAT
 *   Registry.
 *
 * CONCEPT
 *   Provides the Registry.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct ModelRegistryEntry ModelRegistryEntry;
typedef struct ModelRegistry ModelRegistry;

struct ModelRegistryEntry {
    char *name;
    char *version;
    char *architecture;
    char *description;
    char *author;
    char *license;
    char *repo_url;
    char *config_path;
    char *weights_path;
    int is_local;
    int deprecated;
    char *deprecated_reason;
};

struct ModelRegistry {
    ModelRegistryEntry **entries;
    int count;
    int capacity;
};

static ModelRegistry *g_global_registry = NULL;

static ModelRegistryEntry *entry_create(void) {
    ModelRegistryEntry *e = (ModelRegistryEntry *)calloc(1, sizeof(ModelRegistryEntry));
    return e;
}

static void entry_destroy(ModelRegistryEntry *e) {
    if (!e) return;
    free(e->name);
    free(e->version);
    free(e->architecture);
    free(e->description);
    free(e->author);
    free(e->license);
    free(e->repo_url);
    free(e->config_path);
    free(e->weights_path);
    free(e->deprecated_reason);
    free(e);
}

static char *dup_str(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *copy = (char *)malloc(len + 1);
    if (copy) memcpy(copy, s, len + 1);
    return copy;
}

ModelRegistry *model_registry_create(void) {
    ModelRegistry *r = (ModelRegistry *)calloc(1, sizeof(ModelRegistry));
    return r;
}

void model_registry_destroy(ModelRegistry *r) {
    if (!r) return;
    for (int i = 0; i < r->count; i++) {
        entry_destroy(r->entries[i]);
    }
    free(r->entries);
    free(r);
}

int model_registry_register(ModelRegistry *r,
                            const char *name,
                            const char *version,
                            const char *architecture,
                            const char *description,
                            const char *author,
                            const char *license,
                            const char *repo_url,
                            const char *config_path,
                            const char *weights_path,
                            int is_local) {
    if (!r || !name || !version) return -1;

    for (int i = 0; i < r->count; i++) {
        if (strcmp(r->entries[i]->name, name) == 0 &&
            strcmp(r->entries[i]->version, version) == 0) {
            return -1;
        }
    }

    if (r->count == r->capacity) {
        int new_cap = r->capacity ? r->capacity * 2 : 4;
        ModelRegistryEntry **new_entries = (ModelRegistryEntry **)realloc(r->entries, new_cap * sizeof(ModelRegistryEntry *));
        if (!new_entries) return -1;
        r->entries = new_entries;
        r->capacity = new_cap;
    }

    ModelRegistryEntry *e = entry_create();
    e->name = name ? strdup(name) : NULL;
    e->version = version ? strdup(version) : NULL;
    e->architecture = architecture ? strdup(architecture) : NULL;
    e->description = description ? strdup(description) : NULL;
    e->author = author ? strdup(author) : NULL;
    e->license = license ? strdup(license) : NULL;
    e->repo_url = repo_url ? strdup(repo_url) : NULL;
    e->config_path = config_path ? strdup(config_path) : NULL;
    e->weights_path = weights_path ? strdup(weights_path) : NULL;
    e->is_local = is_local;

    r->entries[r->count++] = e;
    return 0;
}

int model_registry_unregister(ModelRegistry *r, const char *name, const char *version) {
    if (!r || !name || !version) return -1;

    for (int i = 0; i < r->count; i++) {
        if (strcmp(r->entries[i]->name, name) == 0 &&
            strcmp(r->entries[i]->version, version) == 0) {
            entry_destroy(r->entries[i]);
            for (int j = i; j < r->count - 1; j++) {
                r->entries[j] = r->entries[j + 1];
            }
            r->count--;
            return 0;
        }
    }
    return -1;
}

ModelRegistryEntry *model_registry_get(ModelRegistry *r, const char *name, const char *version) {
    if (!r || !name || !version) return NULL;
    for (int i = 0; i < r->count; i++) {
        if (strcmp(r->entries[i]->name, name) == 0 &&
            strcmp(r->entries[i]->version, version) == 0) {
            return r->entries[i];
        }
    }
    return NULL;
}

ModelRegistryEntry *model_registry_get_latest(ModelRegistry *r, const char *name) {
    if (!r || !name) return NULL;
    ModelRegistryEntry *latest = NULL;
    for (int i = 0; i < r->count; i++) {
        if (strcmp(r->entries[i]->name, name) == 0) {
            if (!latest || strcmp(r->entries[i]->version, latest->version) > 0) {
                latest = r->entries[i];
            }
        }
    }
    return latest;
}

ModelRegistryEntry **model_registry_list(ModelRegistry *r, const char *architecture, int *count_out) {
    if (!r || !count_out) return NULL;

    int matches = 0;
    for (int i = 0; i < r->count; i++) {
        if (!architecture || strcmp(r->entries[i]->architecture, architecture) == 0) {
            matches++;
        }
    }

    if (matches == 0) {
        *count_out = 0;
        return NULL;
    }

    ModelRegistryEntry **result = (ModelRegistryEntry **)malloc(matches * sizeof(ModelRegistryEntry *));
    int idx = 0;
    for (int i = 0; i < r->count; i++) {
        if (!architecture || strcmp(r->entries[i]->architecture, architecture) == 0) {
            result[idx++] = r->entries[i];
        }
    }
    *count_out = matches;
    return result;
}

ModelRegistryEntry **model_registry_search(ModelRegistry *r, const char *keyword, int *count_out) {
    if (!r || !keyword || !count_out) return NULL;

    int matches = 0;
    for (int i = 0; i < r->count; i++) {
        ModelRegistryEntry *e = r->entries[i];
        if (strstr(e->name, keyword) ||
            (e->description && strstr(e->description, keyword)) ||
            (e->architecture && strstr(e->architecture, keyword)) ||
            (e->author && strstr(e->author, keyword))) {
            matches++;
        }
    }

    if (matches == 0) {
        *count_out = 0;
        return NULL;
    }

    ModelRegistryEntry **result = (ModelRegistryEntry **)malloc(matches * sizeof(ModelRegistryEntry *));
    int idx = 0;
    for (int i = 0; i < r->count; i++) {
        ModelRegistryEntry *e = r->entries[i];
        if (strstr(e->name, keyword) ||
            (e->description && strstr(e->description, keyword)) ||
            (e->architecture && strstr(e->architecture, keyword)) ||
            (e->author && strstr(e->author, keyword))) {
            result[idx++] = e;
        }
    }
    *count_out = matches;
    return result;
}

int model_registry_save(ModelRegistry *r, const char *path) {
    if (!r || !path) return -1;
    FILE *f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "%d\n", r->count);
    for (int i = 0; i < r->count; i++) {
        ModelRegistryEntry *e = r->entries[i];
        fprintf(f, "%s|%s|%s|%s|%s|%s|%s|%s|%s|%d|%d|%s\n",
                e->name ? e->name : "",
                e->version ? e->version : "",
                e->architecture ? e->architecture : "",
                e->description ? e->description : "",
                e->author ? e->author : "",
                e->license ? e->license : "",
                e->repo_url ? e->repo_url : "",
                e->config_path ? e->config_path : "",
                e->weights_path ? e->weights_path : "",
                e->is_local,
                e->deprecated,
                e->deprecated_reason ? e->deprecated_reason : "");
    }
    fclose(f);
    return 0;
}

ModelRegistry *model_registry_load(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    ModelRegistry *r = model_registry_create();
    int count;
    char buf[32];
    if (!fgets(buf, sizeof(buf), f) || sscanf(buf, "%d", &count) != 1) {
        fclose(f);
        model_registry_destroy(r);
        return NULL;
    }

    for (int i = 0; i < count; i++) {
        char line[4096];
        if (!fgets(line, sizeof(line), f)) break;
        line[strcspn(line, "\n\r")] = '\0';

        char *parts[12];
        int p = 0;
        char *tok = strtok(line, "|");
        while (tok && p < 12) {
            parts[p++] = tok;
            tok = strtok(NULL, "|");
        }

        if (p >= 10) {
            model_registry_register(r, parts[0], parts[1], parts[2], parts[3], parts[4],
                                  parts[5], parts[6], parts[7], parts[8], atoi(parts[9]));
            if (p >= 12 && atoi(parts[10])) {
                ModelRegistryEntry *e = r->entries[r->count - 1];
                e->deprecated = 1;
                e->deprecated_reason = parts[11] ? strdup(parts[11]) : NULL;
            }
        }
    }
    fclose(f);
    return r;
}

ModelRegistry *model_registry_global(void) {
    if (!g_global_registry) {
        g_global_registry = model_registry_create();
    }
    return g_global_registry;
}

void model_registry_deprecate(ModelRegistry *r, const char *name, const char *version, const char *reason) {
    ModelRegistryEntry *e = model_registry_get(r, name, version);
    if (e) {
        e->deprecated = 1;
        free(e->deprecated_reason);
        e->deprecated_reason = reason ? strdup(reason) : NULL;
    }
}

int model_registry_is_deprecated(ModelRegistryEntry *e) {
    return e ? e->deprecated : 0;
}

ModelRegistryEntry *model_registry_entry_copy(ModelRegistryEntry *e) {
    if (!e) return NULL;
    ModelRegistryEntry *copy = entry_create();
    copy->name = dup_str(e->name);
    copy->version = dup_str(e->version);
    copy->architecture = dup_str(e->architecture);
    copy->description = dup_str(e->description);
    copy->author = dup_str(e->author);
    copy->license = dup_str(e->license);
    copy->repo_url = dup_str(e->repo_url);
    copy->config_path = dup_str(e->config_path);
    copy->weights_path = dup_str(e->weights_path);
    copy->is_local = e->is_local;
    copy->deprecated = e->deprecated;
    copy->deprecated_reason = dup_str(e->deprecated_reason);
    return copy;
}
