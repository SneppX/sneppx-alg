#ifndef SNEPPX_ATTACK_TREE_H
#define SNEPPX_ATTACK_TREE_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_AT_MAX_CHILDREN 32
#define SNEPPX_AT_MAX_TREE_DEPTH 16

/*
 * SNEPPX - Attack Tree
 *
 * WHAT
 *   Attack Tree.
 *
 * CONCEPT
 *   Provides the Attack Tree.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_AT_GOAL,
    SNEPPX_AT_OR_NODE,
    SNEPPX_AT_AND_NODE,
    SNEPPX_AT_LEAF
} SNEPPXAtNodeType;

typedef struct {
    uint64_t node_id;
    SNEPPXAtNodeType node_type;
    uint8_t* description;
    size_t desc_len;
    uint64_t parent_id;
    uint32_t child_indices[SNEPPX_AT_MAX_CHILDREN];
    uint32_t num_children;
    float likelihood;
    float impact;
    float effort;
    float skill_level;
    float detection_probability;
    float total_risk;
    uint8_t mitigated : 1;
    uint8_t* mitigation;
    size_t mitigation_len;
} SNEPPXAtNode;

typedef struct {
    uint8_t* tree_name;
    size_t name_len;
    SNEPPXAtNode* nodes;
    uint32_t num_nodes;
    uint32_t max_nodes;
    uint64_t root_node_id;
    float total_attack_risk;
} SNEPPXAttackTree;

int snepx_at_tree_create(SNEPPXAttackTree* tree, const uint8_t* name, size_t name_len);
uint64_t snepx_at_tree_add_node(SNEPPXAttackTree* tree, SNEPPXAtNodeType type, const uint8_t* desc, size_t desc_len, uint64_t parent_id);
int snepx_at_tree_compute_risk(SNEPPXAttackTree* tree);
int snepx_at_tree_find_shortest_path(const SNEPPXAttackTree* tree, uint64_t** path, uint32_t* path_len);
int snepx_at_tree_find_most_likely_path(const SNEPPXAttackTree* tree, uint64_t** path, uint32_t* path_len);
int snepx_at_tree_export_dot(const SNEPPXAttackTree* tree, uint8_t* out, size_t* out_len);
int snepx_at_tree_destroy(SNEPPXAttackTree* tree);

#endif
