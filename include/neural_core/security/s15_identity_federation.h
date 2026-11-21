#ifndef SNEPPX_S15_IDENTITY_FEDERATION_H
#define SNEPPX_S15_IDENTITY_FEDERATION_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_DID_MAX_DOCUMENTS 65536
#define SNEPPX_VC_MAX_CLAIMS 1024
#define SNEPPX_VERIFIABLE_PRESENTATION_MAX 256
#define SNEPPX_FIDO2_MAX_CREDENTIALS 128

/*
 * SNEPPX - S15 Identity Federation
 *
 * WHAT
 *   S15 Identity Federation.
 *
 * CONCEPT
 *   Provides the S15 Identity Federation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_IDENTITY_ANONYMOUS,
    SNEPPX_IDENTITY_PSEUDONYMOUS,
    SNEPPX_IDENTITY_VERIFIED,
    SNEPPX_IDENTITY_GOVERNMENT,
    SNEPPX_IDENTITY_ENTERPRISE
} SNEPPXIdentityLevel;

typedef struct {
    char did[128];
    char controller[128];
    uint8_t* verification_method;
    size_t verification_method_len;
    uint8_t* authentication_key;
    size_t authentication_key_len;
    uint8_t* assertion_key;
    size_t assertion_key_len;
    uint8_t* key_agreement_key;
    size_t key_agreement_key_len;
    uint8_t* capability_invocation_key;
    size_t capability_invocation_key_len;
    uint8_t* capability_delegation_key;
    size_t capability_delegation_key_len;
    uint64_t created_at;
    uint64_t updated_at;
    uint32_t version;
    uint8_t deactivated : 1;
    uint8_t suspended : 1;
} SNEPPXDIDDocument;

typedef struct {
    char credential_id[128];
    char issuer_did[128];
    char subject_did[128];
    uint8_t* credential_schema;
    size_t schema_len;
    uint8_t* claims;
    size_t claims_len;
    uint8_t* proof;
    size_t proof_len;
    uint64_t issued_at;
    uint64_t expires_at;
    uint8_t revoked : 1;
    uint32_t credential_type;
    char context_url[256];
} SNEPPXVerifiableCredential;

typedef struct {
    uint8_t* verifiable_presentation;
    size_t presentation_len;
    uint32_t num_credentials;
    SNEPPXVerifiableCredential* credentials;
    uint8_t* holder_proof;
    size_t holder_proof_len;
    uint64_t presentation_timestamp;
    uint8_t domain[128];
    uint8_t challenge[64];
} SNEPPXVerifiablePresentation;

typedef struct {
    uint8_t credential_id[128];
    uint8_t public_key_cose[128];
    uint8_t private_key_cose[128];
    uint64_t signature_count;
    uint8_t rp_id[256];
    uint8_t user_handle[64];
    uint32_t algorithm;
    uint8_t aa_guid[16];
    uint32_t credential_type;
} SNEPPXFido2Credential;

typedef struct {
    uint32_t num_credentials;
    SNEPPXFido2Credential credentials[SNEPPX_FIDO2_MAX_CREDENTIALS];
    uint64_t last_used;
    uint64_t created_at;
    uint8_t* relying_party_list;
    size_t rp_list_len;
} SNEPPXFido2Authenticator;

typedef struct {
    uint8_t* session_token;
    size_t session_token_len;
    uint64_t session_created;
    uint64_t session_expires;
    uint32_t session_timeout_seconds;
    uint8_t* refresh_token;
    size_t refresh_token_len;
    uint32_t max_sessions;
    uint32_t current_sessions;
    uint8_t* mfa_session;
    size_t mfa_session_len;
} SNEPPXOAuth2State;

typedef struct {
    char issuer[256];
    char authorization_endpoint[256];
    char token_endpoint[256];
    char jwks_uri[256];
    uint8_t* public_keys;
    size_t public_keys_len;
    uint32_t token_expiry_seconds;
    uint32_t refresh_expiry_seconds;
    uint8_t* supported_scopes;
    size_t scopes_len;
} SNEPPXOidcProvider;

int snepx_did_create(SNEPPXDIDDocument* doc, SNEPPXIdentityLevel level);
int snepx_did_resolve(const char* did, SNEPPXDIDDocument* doc);
int snepx_did_update(SNEPPXDIDDocument* doc, const uint8_t* new_key, size_t new_key_len);
int snepx_did_deactivate(SNEPPXDIDDocument* doc);
int snepx_did_verify(const SNEPPXDIDDocument* doc, const uint8_t* message, size_t message_len, const uint8_t* signature, size_t signature_len);

int snepx_vc_issue(SNEPPXDIDDocument* issuer, const char* subject_did, const uint8_t* claims, size_t claims_len, SNEPPXVerifiableCredential* vc);
int snepx_vc_verify(const SNEPPXDIDDocument* issuer, const SNEPPXVerifiableCredential* vc);
int snepx_vc_revoke(SNEPPXVerifiableCredential* vc);
int snepx_vc_create_presentation(const SNEPPXVerifiableCredential* vcs, uint32_t num_vcs, const SNEPPXDIDDocument* holder, SNEPPXVerifiablePresentation* vp);
int snepx_vc_verify_presentation(SNEPPXVerifiablePresentation* vp);

int snepx_fido2_register(SNEPPXFido2Authenticator* auth, const char* rp_id, const uint8_t* user_id, size_t user_id_len);
int snepx_fido2_authenticate(SNEPPXFido2Authenticator* auth, const char* rp_id, const uint8_t* challenge, size_t challenge_len, uint8_t* assertion, size_t* assertion_len);
int snepx_fido2_verify_assertion(const SNEPPXFido2Authenticator* auth, const uint8_t* assertion, size_t assertion_len, const uint8_t* challenge, size_t challenge_len);

int snepx_oauth2_create_session(SNEPPXOAuth2State* oauth, const uint8_t* user_id, size_t user_id_len);
int snepx_oauth2_validate_token(SNEPPXOAuth2State* oauth, const uint8_t* token, size_t token_len);
int snepx_oauth2_refresh_token(SNEPPXOAuth2State* oauth, const uint8_t* refresh_token, size_t refresh_token_len);
int snepx_oauth2_revoke(SNEPPXOAuth2State* oauth);
int snepx_oauth2_mfa_required(SNEPPXOAuth2State* oauth, uint8_t* mfa_challenge, size_t* challenge_len);
int snepx_oauth2_mfa_verify(SNEPPXOAuth2State* oauth, const uint8_t* mfa_response, size_t response_len);

int snepx_oidc_discover(SNEPPXOidcProvider* provider, const char* issuer_url);
int snepx_oidc_verify_token(const SNEPPXOidcProvider* provider, const uint8_t* token, size_t token_len);
int snepx_oidc_refresh(const SNEPPXOidcProvider* provider, const uint8_t* refresh_token, size_t refresh_token_len);

// Zero-knowledge proofs
typedef struct {
    uint8_t* proof;
    size_t proof_len;
    uint64_t security_parameter;
    uint8_t* public_inputs;
    size_t public_inputs_len;
    uint8_t* private_inputs;
    size_t private_inputs_len;
} SNEPPXZeroKnowledgeProof;

int snepx_zkp_generate(SNEPPXZeroKnowledgeProof* zkp, const uint8_t* public_inputs, size_t public_len, const uint8_t* private_inputs, size_t private_len);
int snepx_zkp_verify(const SNEPPXZeroKnowledgeProof* zkp, const uint8_t* public_inputs, size_t public_len);

#endif
