#include "cognitive_memory.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/*
 * SNEPPX - Semantic
 *
 * WHAT
 *   Semantic.
 *
 * CONCEPT
 *   Provides the Semantic.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    char key[SNEPPX_MAX_CONCEPT_KEY_LEN];
    SNEPPXTensor* embedding;
    size_t ref_count;
    uint64_t last_access;
    int valid;
} SNEPPXSemanticConcept;

typedef struct {
    size_t from_idx;
    size_t to_idx;
    float weight;
    int valid;
} SNEPPXSemanticEdge;

typedef struct SNEPPXSemanticGraph SNEPPXSemanticGraph;

struct SNEPPXSemanticGraph {
    SNEPPXSemanticConcept* concepts;
    size_t num_concepts;
    size_t max_concepts;
    SNEPPXSemanticEdge* edges;
    size_t num_edges;
    size_t max_edges;
    size_t embedding_dim;
    uint64_t access_counter;
    unsigned int* seed_ptr;
};

static int find_concept_idx(SNEPPXSemanticGraph* graph, const char* key) {
    for (size_t i = 0; i < graph->num_concepts; i++) {
        if (graph->concepts[i].valid && strcmp(graph->concepts[i].key, key) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int find_or_create_concept(SNEPPXSemanticGraph* graph, const char* key) {
    int idx = find_concept_idx(graph, key);
    if (idx >= 0) return idx;
    if (graph->num_concepts >= graph->max_concepts) {
        uint64_t oldest = graph->access_counter;
        int lru_idx = -1;
        for (size_t i = 0; i < graph->num_concepts; i++) {
            if (graph->concepts[i].valid && graph->concepts[i].last_access < oldest) {
                oldest = graph->concepts[i].last_access;
                lru_idx = (int)i;
            }
        }
        if (lru_idx >= 0) {
            if (graph->concepts[lru_idx].embedding) {
                SNEPPX_tensor_destroy(graph->concepts[lru_idx].embedding);
                graph->concepts[lru_idx].embedding = NULL;
            }
            for (size_t e = 0; e < graph->num_edges; e++) {
                if (graph->edges[e].valid &&
                    (graph->edges[e].from_idx == (size_t)lru_idx ||
                     graph->edges[e].to_idx == (size_t)lru_idx)) {
                    graph->edges[e].valid = 0;
                }
            }
            memset(graph->concepts[lru_idx].key, 0, SNEPPX_MAX_CONCEPT_KEY_LEN);
            graph->concepts[lru_idx].valid = 0;
            idx = lru_idx;
        } else {
            return -1;
        }
    } else {
        idx = (int)graph->num_concepts;
        graph->num_concepts++;
    }
    return idx;
}

SNEPPXSemanticGraph* sneppx_semantic_create(const SNEPPXSemanticConfig* cfg, unsigned int* seed_ptr) {
    SNEPPXSemanticGraph* graph = SNEPPX_malloc(sizeof(SNEPPXSemanticGraph), 64);
    if (!graph) return NULL;
    memset(graph, 0, sizeof(SNEPPXSemanticGraph));
    graph->max_concepts = cfg->max_concepts > 0 ? cfg->max_concepts : 100;
    graph->max_edges = cfg->max_edges > 0 ? cfg->max_edges : 500;
    graph->embedding_dim = cfg->embedding_dim > 0 ? cfg->embedding_dim : 64;
    graph->seed_ptr = seed_ptr;
    graph->concepts = SNEPPX_malloc(graph->max_concepts * sizeof(SNEPPXSemanticConcept), 64);
    if (!graph->concepts) {
        SNEPPX_free(graph, sizeof(SNEPPXSemanticGraph));
        return NULL;
    }
    memset(graph->concepts, 0, graph->max_concepts * sizeof(SNEPPXSemanticConcept));
    graph->edges = SNEPPX_malloc(graph->max_edges * sizeof(SNEPPXSemanticEdge), 64);
    if (!graph->edges) {
        SNEPPX_free(graph->concepts, graph->max_concepts * sizeof(SNEPPXSemanticConcept));
        SNEPPX_free(graph, sizeof(SNEPPXSemanticGraph));
        return NULL;
    }
    memset(graph->edges, 0, graph->max_edges * sizeof(SNEPPXSemanticEdge));
    return graph;
}

void sneppx_semantic_destroy(SNEPPXSemanticGraph* graph) {
    if (!graph) return;
    for (size_t i = 0; i < graph->num_concepts; i++) {
        if (graph->concepts[i].embedding) SNEPPX_tensor_destroy(graph->concepts[i].embedding);
    }
    SNEPPX_free(graph->concepts, graph->max_concepts * sizeof(SNEPPXSemanticConcept));
    SNEPPX_free(graph->edges, graph->max_edges * sizeof(SNEPPXSemanticEdge));
    SNEPPX_free(graph, sizeof(SNEPPXSemanticGraph));
}

int sneppx_semantic_store(SNEPPXSemanticGraph* graph, const char* key,
                          const SNEPPXTensor* embedding) {
    if (!graph || !key || !embedding) return -1;
    int idx = find_or_create_concept(graph, key);
    if (idx < 0) return -1;
    SNEPPXSemanticConcept* conc = &graph->concepts[idx];
    strncpy(conc->key, key, SNEPPX_MAX_CONCEPT_KEY_LEN - 1);
    conc->key[SNEPPX_MAX_CONCEPT_KEY_LEN - 1] = '\0';
    if (conc->embedding) SNEPPX_tensor_destroy(conc->embedding);
    conc->embedding = SNEPPX_tensor_copy(embedding);
    if (!conc->embedding) return -1;
    conc->ref_count = 0;
    conc->last_access = ++graph->access_counter;
    conc->valid = 1;
    return 0;
}

int sneppx_semantic_retrieve(SNEPPXSemanticGraph* graph, const SNEPPXTensor* query,
                             size_t top_k, SNEPPXTensor** output, char*** out_keys) {
    if (!graph || !query || !output) return -1;
    if (graph->num_concepts == 0) { *output = NULL; return 0; }
    size_t n = graph->num_concepts;
    float* scores = SNEPPX_malloc(n * sizeof(float), 64);
    size_t* indices = SNEPPX_malloc(n * sizeof(size_t), 64);
    if (!scores || !indices) {
        if (scores) SNEPPX_free(scores, n * sizeof(float));
        if (indices) SNEPPX_free(indices, n * sizeof(size_t));
        return -1;
    }
    size_t valid_count = 0;
    float query_norm = 0.0f;
    float* qd = (float*)query->data;
    size_t qlen = query->size;
    for (size_t i = 0; i < qlen; i++) query_norm += qd[i] * qd[i];
    query_norm = sqrtf(query_norm + 1e-10f);
    for (size_t i = 0; i < graph->num_concepts; i++) {
        if (!graph->concepts[i].valid || !graph->concepts[i].embedding) continue;
        float* ed = (float*)graph->concepts[i].embedding->data;
        size_t elen = graph->concepts[i].embedding->size;
        size_t min_dim = qlen < elen ? qlen : elen;
        float dot = 0.0f, norm = 0.0f;
        for (size_t j = 0; j < min_dim; j++) dot += qd[j] * ed[j];
        for (size_t j = 0; j < elen; j++) norm += ed[j] * ed[j];
        norm = sqrtf(norm + 1e-10f);
        scores[valid_count] = dot / (query_norm * norm + 1e-10f);
        indices[valid_count] = i;
        graph->concepts[i].last_access = ++graph->access_counter;
        graph->concepts[i].ref_count++;
        valid_count++;
    }
    if (valid_count == 0) { *output = NULL; SNEPPX_free(scores, n * sizeof(float)); SNEPPX_free(indices, n * sizeof(size_t)); return 0; }
    for (size_t i = 0; i < valid_count - 1; i++) {
        for (size_t j = 0; j < valid_count - 1 - i; j++) {
            if (scores[j] < scores[j + 1]) {
                float ts = scores[j]; scores[j] = scores[j + 1]; scores[j + 1] = ts;
                size_t ti = indices[j]; indices[j] = indices[j + 1]; indices[j + 1] = ti;
            }
        }
    }
    size_t actual_k = top_k < valid_count ? top_k : valid_count;
    size_t shape[2] = {actual_k, graph->embedding_dim};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    if (!result) { SNEPPX_free(scores, n * sizeof(float)); SNEPPX_free(indices, n * sizeof(size_t)); return -1; }
    float* rd = (float*)result->data;
    if (out_keys) {
        *out_keys = SNEPPX_malloc(actual_k * sizeof(char*), 64);
        if (!*out_keys) {
            SNEPPX_tensor_destroy(result);
            SNEPPX_free(scores, n * sizeof(float));
            SNEPPX_free(indices, n * sizeof(size_t));
            return -1;
        }
    }
    for (size_t i = 0; i < actual_k; i++) {
        SNEPPXSemanticConcept* c = &graph->concepts[indices[i]];
        if (c->embedding) {
            memcpy(rd + i * graph->embedding_dim, c->embedding->data,
                   (graph->embedding_dim < c->embedding->size ? graph->embedding_dim : c->embedding->size) * sizeof(float));
        }
        if (out_keys) {
            (*out_keys)[i] = SNEPPX_malloc(SNEPPX_MAX_CONCEPT_KEY_LEN, 64);
            if ((*out_keys)[i]) {
                strncpy((*out_keys)[i], c->key, SNEPPX_MAX_CONCEPT_KEY_LEN - 1);
                (*out_keys)[i][SNEPPX_MAX_CONCEPT_KEY_LEN - 1] = '\0';
            }
        }
    }
    *output = result;
    SNEPPX_free(scores, n * sizeof(float));
    SNEPPX_free(indices, n * sizeof(size_t));
    return (int)actual_k;
}

int sneppx_semantic_relate(SNEPPXSemanticGraph* graph, const char* from_key,
                           const char* to_key, float weight) {
    if (!graph || !from_key || !to_key) return -1;
    int fi = find_concept_idx(graph, from_key);
    int ti = find_concept_idx(graph, to_key);
    if (fi < 0 || ti < 0) return -1;
    if (graph->num_edges >= graph->max_edges) return -1;
    SNEPPXSemanticEdge* edge = &graph->edges[graph->num_edges++];
    edge->from_idx = (size_t)fi;
    edge->to_idx = (size_t)ti;
    edge->weight = weight;
    edge->valid = 1;
    return 0;
}

int sneppx_semantic_forget(SNEPPXSemanticGraph* graph, const char* key) {
    if (!graph || !key) return -1;
    int idx = find_concept_idx(graph, key);
    if (idx < 0) return -1;
    SNEPPXSemanticConcept* conc = &graph->concepts[idx];
    if (conc->embedding) SNEPPX_tensor_destroy(conc->embedding);
    conc->embedding = NULL;
    conc->valid = 0;
    memset(conc->key, 0, SNEPPX_MAX_CONCEPT_KEY_LEN);
    for (size_t e = 0; e < graph->num_edges; e++) {
        if (graph->edges[e].valid &&
            (graph->edges[e].from_idx == (size_t)idx ||
             graph->edges[e].to_idx == (size_t)idx)) {
            graph->edges[e].valid = 0;
        }
    }
    return 0;
}
