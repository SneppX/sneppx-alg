#ifndef NEURAL_CORE_MODEL_ZOO_REGISTRY_H
#define NEURAL_CORE_MODEL_ZOO_REGISTRY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
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


extern "C" {
#endif

// Forward declarations
typedef struct ModelRegistryEntry ModelRegistryEntry;
typedef struct ModelRegistry ModelRegistry;

// Model registry entry
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

// Model registry
struct ModelRegistry {
    ModelRegistryEntry **entries;
    int count;
    int capacity;
};

// Create/destroy
ModelRegistry *model_registry_create(void);
void model_registry_destroy(ModelRegistry *registry);

// Register a model
int model_registry_register(ModelRegistry *registry,
                            const char *name,
                            const char *version,
                            const char *architecture,
                            const char *description,
                            const char *author,
                            const char *license,
                            const char *repo_url,
                            const char *config_path,
                            const char *weights_path,
                            int is_local);

// Unregister a model
int model_registry_unregister(ModelRegistry *registry, const char *name, const char *version);

// Get entry by name and version
ModelRegistryEntry *model_registry_get(ModelRegistry *registry, const char *name, const char *version);

// Get latest version of a model
ModelRegistryEntry *model_registry_get_latest(ModelRegistry *registry, const char *name);

// List all models (optionally filtered by architecture)
ModelRegistryEntry **model_registry_list(ModelRegistry *registry, const char *architecture, int *count_out);

// Search models by keyword
ModelRegistryEntry **model_registry_search(ModelRegistry *registry, const char *keyword, int *count_out);

// Save/load registry to/from file
int model_registry_save(ModelRegistry *registry, const char *path);
ModelRegistry *model_registry_load(const char *path);

// Global registry (singleton)
ModelRegistry *model_registry_global(void);

// Mark as deprecated
void model_registry_deprecate(ModelRegistry *registry, const char *name, const char *version, const char *reason);

// Check if entry is deprecated
int model_registry_is_deprecated(ModelRegistryEntry *entry);

// Copy/free entry
ModelRegistryEntry *model_registry_entry_copy(ModelRegistryEntry *entry);
void model_registry_entry_free(ModelRegistryEntry *entry);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_CORE_MODEL_ZOO_REGISTRY_H
