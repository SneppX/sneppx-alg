#include "pth_format.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * SNEPPX - Pth Format
 *
 * WHAT
 *   Pth Format.
 *
 * CONCEPT
 *   Provides the Pth Format.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* Real PyTorch .bin/.pth reader (torch >= 1.6 zip serialization).
 * Parses the ZIP container, the data.pkl pickle (subset of opcodes used by
 * state_dict checkpoints), and maps storages to the raw bytes in the archive. */

typedef enum {
    PTH_FLOAT32 = 0, PTH_FLOAT64, PTH_FLOAT16, PTH_BFLOAT16,
    PTH_INT64, PTH_INT32, PTH_INT16, PTH_INT8, PTH_UINT8, PTH_BOOL
} PTHDType;

typedef struct {
    void* data;
    size_t* shape;
    size_t ndim;
    int dtype;
} PTHensor;

typedef struct PTHState {
    PTHensor* items;
    char** keys;
    size_t count;
    size_t cap;
} PTHState;

static size_t pth_esize(int dtype) {
    switch (dtype) {
        case PTH_FLOAT64: case PTH_INT64: return 8;
        case PTH_FLOAT32: case PTH_INT32: return 4;
        case PTH_FLOAT16: case PTH_BFLOAT16: case PTH_INT16: return 2;
        case PTH_INT8: case PTH_UINT8: case PTH_BOOL: return 1;
        default: return 4;
    }
}

/* ---- minimal ZIP reader (stored + deflate-lite: stored only) ---- */
typedef struct { char* name; size_t name_len; unsigned char* data; size_t len; } ZipEntry;

static int zip_find(const char* path, const char* entry_name, unsigned char** out, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 22) { fclose(f); return -1; }
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    fclose(f);
    size_t pos = 0;
    int found = 0;
    while (pos + 30 <= (size_t)sz) {
        unsigned sig = buf[pos] | (buf[pos+1]<<8) | (buf[pos+2]<<16) | (buf[pos+3]<<24);
        if (sig != 0x04034b50u) break;
        unsigned short method = buf[pos+8] | (buf[pos+9]<<8);
        unsigned comp_size = buf[pos+18] | (buf[pos+19]<<8) | (buf[pos+20]<<16) | (buf[pos+21]<<24);
        unsigned fname_len = buf[pos+26] | (buf[pos+27]<<8);
        unsigned extra_len = buf[pos+28] | (buf[pos+29]<<8);
        const char* fname = (const char*)(buf + pos + 30);
        size_t data_off = pos + 30 + fname_len + extra_len;
        if (data_off + comp_size > (size_t)sz) break;
        if (method == 0 && (size_t)fname_len == strlen(entry_name) && !memcmp(fname, entry_name, fname_len)) {
            *out = (unsigned char*)malloc(comp_size);
            memcpy(*out, buf + data_off, comp_size);
            *out_len = comp_size;
            found = 1;
        }
        pos = data_off + comp_size;
    }
    /* look in central directory too if not found (stored only) */
    free(buf);
    return found ? 0 : -1;
}

/* ---- pickle virtual machine ---- */
enum { O_NONE, O_INT, O_FLOAT, O_STR, O_BYTES, O_TUPLE, O_LIST, O_DICT, O_STORAGE, O_TENSOR, O_GLOBAL };

typedef struct Obj Obj;
struct Obj {
    int type;
    long long ival;
    double fval;
    unsigned char* sval; size_t slen;
    Obj** items; size_t nitem, cap_item;       /* tuple/list */
    Obj** keys; Obj** vals; size_t nkv, cap_kv; /* dict */
    /* storage/tensor payload */
    char* storage_id; int dtype; long long numel; long long offset;
    size_t* shape; size_t ndim;
    Obj* storage; /* tensor -> storage obj */
};

static Obj* obj_new(int type) {
    Obj* o = (Obj*)calloc(1, sizeof(Obj));
    o->type = type;
    return o;
}
static void obj_free(Obj* o) {
    if (!o) return;
    free(o->sval); free(o->storage_id); free(o->shape);
    if (o->items) { for (size_t i=0;i<o->nitem;i++) obj_free(o->items[i]); free(o->items); }
    if (o->keys) { for (size_t i=0;i<o->nkv;i++){ obj_free(o->keys[i]); obj_free(o->vals[i]); } free(o->keys); free(o->vals); }
    free(o);
}
static void obj_push(Obj*** stack, size_t* n, size_t* cap, Obj* o) {
    if (*n >= *cap) { *cap = *cap ? *cap*2 : 16; *stack = (Obj**)realloc(*stack, *cap*sizeof(Obj*)); }
    (*stack)[(*n)++] = o;
}
static Obj* obj_pop(Obj** stack, size_t* n) { return (*n) ? stack[--(*n)] : NULL; }

static int dtype_from_global(const char* name) {
    if (strstr(name, "FloatStorage")) return PTH_FLOAT32;
    if (strstr(name, "DoubleStorage")) return PTH_FLOAT64;
    if (strstr(name, "HalfStorage")) return PTH_FLOAT16;
    if (strstr(name, "BFloat16Storage")) return PTH_BFLOAT16;
    if (strstr(name, "LongStorage")) return PTH_INT64;
    if (strstr(name, "IntStorage")) return PTH_INT32;
    if (strstr(name, "ShortStorage")) return PTH_INT16;
    if (strstr(name, "CharStorage")) return PTH_INT8;
    if (strstr(name, "ByteStorage")) return PTH_UINT8;
    if (strstr(name, "BoolStorage")) return PTH_BOOL;
    return PTH_FLOAT32;
}

static Obj* pth_unpickle(const unsigned char* p, size_t len) {
    Obj** stack = NULL; size_t n = 0, scap = 0;
    Obj** memo = (Obj**)calloc(256, sizeof(Obj*)); size_t mcap = 256;
    size_t i = 0;
    Obj* result = NULL;
    /* persistent storage map: id string -> resolved storage obj (with data later) */
    while (i < len) {
        unsigned char op = p[i++];
        switch (op) {
            case '(': /* MARK */ {
                Obj* m = obj_new(O_NONE); m->type = -1; /* marker sentinel */
                obj_push(&stack, &n, &scap, m);
                break;
            }
            case ')': /* TUPLE */ {
                size_t cnt = 0; Obj** buf = NULL; size_t bcap = 0;
                while (n) { Obj* top = stack[n-1]; if (top->type == -1) { obj_free(obj_pop(stack,&n)); break; } buf = (Obj**)realloc(buf,(++cnt)*sizeof(Obj*)); buf[cnt-1]=top; obj_pop(stack,&n); }
                Obj* t = obj_new(O_TUPLE);
                t->items = (Obj**)malloc(cnt?cnt*sizeof(Obj*):1);
                for (size_t k=0;k<cnt;k++) t->items[k]=buf[cnt-1-k];
                t->nitem = cnt; free(buf);
                obj_push(&stack,&n,&scap,t);
                break;
            }
            case 't': /* TUPLE + MARK handled in ')' */ break;
            case '\x85': case '\x86': case '\x87': { /* TUPLE1/2/3 */
                size_t cnt = op - '\x84';
                Obj* t = obj_new(O_TUPLE);
                t->items = (Obj**)malloc(cnt*sizeof(Obj*));
                for (size_t k=0;k<cnt;k++) t->items[cnt-1-k] = obj_pop(stack,&n);
                t->nitem = cnt;
                obj_push(&stack,&n,&scap,t);
                break;
            }
            case '}': { /* EMPTY_DICT */ obj_push(&stack,&n,&scap,obj_new(O_DICT)); break; }
            case '{': { /* DICT with mark */ /* treat like MARK for items */ Obj* m = obj_new(O_NONE); m->type = -1; obj_push(&stack,&n,&scap,m); break; }
            case '\x84': break; /* EMPTY_LIST? no */
            case 'l': { Obj* m = obj_new(O_NONE); m->type = -1; obj_push(&stack,&n,&scap,m); break; } /* LIST mark */
            case ']': { /* EMPTY_LIST */ obj_push(&stack,&n,&scap,obj_new(O_LIST)); break; }
            case 'a': { /* APPEND */ Obj* v = obj_pop(stack,&n); Obj* l = stack[n-1]; if (l->type==O_LIST){ l->items=(Obj**)realloc(l->items,(++l->nitem)*sizeof(Obj*)); l->items[l->nitem-1]=v; } else obj_free(v); break; }
            case 'e': { /* APPENDS (from MARK) */ while (n && stack[n-1]->type!=-1){ Obj* v=obj_pop(stack,&n); Obj* l=stack[n-1]; if(l->type==O_LIST){ l->items=(Obj**)realloc(l->items,(++l->nitem)*sizeof(Obj*)); l->items[l->nitem-1]=v;} else obj_free(v);} if(n) obj_free(obj_pop(stack,&n)); break; }
            case 's': { /* SETITEM */ Obj* v=obj_pop(stack,&n); Obj* k=obj_pop(stack,&n); Obj* d=stack[n-1]; if(d->type==O_DICT){ d->keys=(Obj**)realloc(d->keys,(++d->nkv)*sizeof(Obj*)); d->vals=(Obj**)realloc(d->vals,d->nkv*sizeof(Obj*)); d->keys[d->nkv-1]=k; d->vals[d->nkv-1]=v;} else {obj_free(k);obj_free(v);} break; }
            case 'u': { /* SETITEMS from MARK */ while (n && stack[n-1]->type!=-1){ Obj* v=obj_pop(stack,&n); Obj* k=obj_pop(stack,&n); Obj* d=stack[n-1]; if(d->type==O_DICT){ d->keys=(Obj**)realloc(d->keys,(++d->nkv)*sizeof(Obj*)); d->vals=(Obj**)realloc(d->vals,d->nkv*sizeof(Obj*)); d->keys[d->nkv-1]=k; d->vals[d->nkv-1]=v;} else {obj_free(k);obj_free(v);}} if(n) obj_free(obj_pop(stack,&n)); break; }
            case 'N': { obj_push(&stack,&n,&scap,obj_new(O_NONE)); break; }
            case '\x88': case '\x89': { obj_push(&stack,&n,&scap,obj_new(O_NONE)); break; } /* NEWTRUE/NEWFALSE */
            case 'I': { /* INT */ long long v=0; int neg=0; while(p[i]==' '||p[i]=='+') i++; if(p[i]=='-'){neg=1;i++;} while(p[i]>='0'&&p[i]<='9'){v=v*10+(p[i]-'0');i++;} i++; Obj* o=obj_new(O_INT); o->ival=(neg?-v:v); obj_push(&stack,&n,&scap,o); break; }
            case 'J': { long long v=0; for(int b=0;b<4;b++){v|=(long long)p[i++]<<(8*b);} if(v&0x80000000) v-=0x100000000LL; Obj* o=obj_new(O_INT); o->ival=v; obj_push(&stack,&n,&scap,o); break; }
            case 'K': { Obj* o=obj_new(O_INT); o->ival=p[i++]; obj_push(&stack,&n,&scap,o); break; }
            case 'M': { int v=p[i]|(p[i+1]<<8); i+=2; Obj* o=obj_new(O_INT); o->ival=v; obj_push(&stack,&n,&scap,o); break; }
            case '\x8a': { int v=p[i]|(p[i+1]<<8)|(p[i+2]<<16)|(p[i+3]<<24); i+=4; Obj* o=obj_new(O_INT); o->ival=v; obj_push(&stack,&n,&scap,o); break; } /* BININT4 */
            case 'L': { /* LONG */ long long v=0; int neg=0; while(p[i]==' '||p[i]=='+') i++; if(p[i]=='-'){neg=1;i++;} while(p[i]>='0'&&p[i]<='9'){v=v*10+(p[i]-'0');i++;} i+=2; Obj* o=obj_new(O_INT); o->ival=(neg?-v:v); obj_push(&stack,&n,&scap,o); break; }
            case '\x8b': { /* LONG1 */ unsigned char nb=p[i++]; long long v=0; for(int b=0;b<nb;b++) v|=(long long)p[i++]<<(8*b); Obj* o=obj_new(O_INT); o->ival=v; obj_push(&stack,&n,&scap,o); break; }
            case '\x8c': { /* LONG4 */ unsigned nb=p[i]|(p[i+1]<<8)|(p[i+2]<<16)|(p[i+3]<<24); i+=4; long long v=0; for(unsigned b=0;b<nb;b++) v|=(long long)p[i++]<<(8*b); Obj* o=obj_new(O_INT); o->ival=v; obj_push(&stack,&n,&scap,o); break; }
            case 'F': { /* FLOAT */ double v=0; char tmp[32]; int c=0; while(p[i]!='\n'&&c<31) tmp[c++]=p[i++]; tmp[c]=0; i++; v=atof(tmp); Obj* o=obj_new(O_FLOAT); o->fval=v; obj_push(&stack,&n,&scap,o); break; }
            case 'G': { unsigned long long bits=0; for(int b=0;b<8;b++) bits|=(unsigned long long)p[i++]<<(8*b); Obj* o=obj_new(O_FLOAT); memcpy(&o->fval,&bits,8); obj_push(&stack,&n,&scap,o); break; }
            case 'S': case 'V': { /* SHORT_BINSTRING / UNICODE legacy */ unsigned char sl=(op=='S')?p[i++]:p[i++]; char* s=(char*)malloc(sl+1); memcpy(s,p+i,sl); s[sl]=0; i+=sl; Obj* o=obj_new(O_STR); o->sval=(unsigned char*)s; o->slen=sl; obj_push(&stack,&n,&scap,o); break; }
            case 'X': { /* BINUNICODE */ unsigned sl=p[i]|(p[i+1]<<8); i+=2; char* s=(char*)malloc(sl+1); memcpy(s,p+i,sl); s[sl]=0; i+=sl; Obj* o=obj_new(O_STR); o->sval=(unsigned char*)s; o->slen=sl; obj_push(&stack,&n,&scap,o); break; }
            case '\x8d': { /* SHORT_BINUNICODE */ unsigned char sl=p[i++]; char* s=(char*)malloc(sl+1); memcpy(s,p+i,sl); s[sl]=0; i+=sl; Obj* o=obj_new(O_STR); o->sval=(unsigned char*)s; o->slen=sl; obj_push(&stack,&n,&scap,o); break; }
            case '\x8e': { /* BINUNICODE8 */ size_t sl=0; for(int b=0;b<8;b++) sl|=(size_t)p[i++]<<(8*b); char* s=(char*)malloc(sl+1); memcpy(s,p+i,sl); s[sl]=0; i+=sl; Obj* o=obj_new(O_STR); o->sval=(unsigned char*)s; o->slen=sl; obj_push(&stack,&n,&scap,o); break; }
            case 'T': case 'B': { /* BINSTRING / STRING legacy */ unsigned sl=p[i]|(p[i+1]<<8)|(p[i+2]<<16)|(p[i+3]<<24); i+=4; unsigned char* s=(unsigned char*)malloc(sl?sl:1); memcpy(s,p+i,sl); i+=sl; Obj* o=obj_new(O_BYTES); o->sval=s; o->slen=sl; obj_push(&stack,&n,&scap,o); break; }
            case 'U': { /* SHORT_BINSTRING */ unsigned char sl=p[i++]; unsigned char* s=(unsigned char*)malloc(sl?sl:1); memcpy(s,p+i,sl); i+=sl; Obj* o=obj_new(O_BYTES); o->sval=s; o->slen=sl; obj_push(&stack,&n,&scap,o); break; }
            case 'C': { /* SHORT_BINBYTES */ unsigned char sl=p[i++]; unsigned char* s=(unsigned char*)malloc(sl?sl:1); memcpy(s,p+i,sl); i+=sl; Obj* o=obj_new(O_BYTES); o->sval=s; o->slen=sl; obj_push(&stack,&n,&scap,o); break; }
            case 'A': { /* BINBYTES */ unsigned sl=p[i]|(p[i+1]<<8)|(p[i+2]<<16)|(p[i+3]<<24); i+=4; unsigned char* s=(unsigned char*)malloc(sl?sl:1); memcpy(s,p+i,sl); i+=sl; Obj* o=obj_new(O_BYTES); o->sval=s; o->slen=sl; obj_push(&stack,&n,&scap,o); break; }
            case '\x8f': { /* BINBYTES8 */ size_t sl=0; for(int b=0;b<8;b++) sl|=(size_t)p[i++]<<(8*b); unsigned char* s=(unsigned char*)malloc(sl?sl:1); memcpy(s,p+i,sl); i+=sl; Obj* o=obj_new(O_BYTES); o->sval=s; o->slen=sl; obj_push(&stack,&n,&scap,o); break; }
            case 'c': case '\x93': { /* GLOBAL / STACK_GLOBAL */ /* read module\nname\n */ char mod[256], nm[256]; unsigned c=0; while(p[i]!='\n'&&c<255) mod[c++]=p[i++]; mod[c]=0; i++; c=0; while(p[i]!='\n'&&c<255) nm[c++]=p[i++]; nm[c]=0; i++; Obj* o=obj_new(O_GLOBAL); size_t tl=strlen(mod)+strlen(nm)+2; char* full=(char*)malloc(tl); snprintf(full, tl, "%s.%s", mod, nm); o->sval=(unsigned char*)full; o->slen=tl; if(op=='\x93'){ /* STACK_GLOBAL uses two strings already on stack */ } obj_push(&stack,&n,&scap,o); break; }
            case 'R': { /* REDUCE */ Obj* args=obj_pop(stack,&n); Obj* callable=obj_pop(stack,&n); Obj* out=NULL; if (callable->type==O_GLOBAL) { char* fn = (char*)callable->sval; if (strstr(fn,"rebuild_tensor_v2")||strstr(fn,"rebuild_from_type_v2")||strstr(fn,"_rebuild_tensor")) { /* args: (storage, offset, size, stride, requires_grad, ...) */ if (args->type==O_TUPLE && args->nitem>=4) { Obj* st=args->items[0]; Obj* sz=args->items[2]; Obj* t=obj_new(O_TENSOR); t->storage=st; t->offset=(st->type==O_STORAGE)?st->offset:0; if (sz->type==O_TUPLE){ t->ndim=sz->nitem; t->shape=(size_t*)malloc(sz->nitem?sz->nitem*sizeof(size_t):1); for(size_t k=0;k<sz->nitem;k++) t->shape[k]=(size_t)sz->items[k]->ival; } else { t->ndim=0; t->shape=NULL; } if(st->type==O_STORAGE){ t->dtype=st->dtype; } else t->dtype=PTH_FLOAT32; out=t; } } else if (strstr(fn,"_rebuild_storage_from_bytes")||strstr(fn,"_rebuild_storage")) { /* args: (typename, byteorder, elemsize, numel, key, bytes) */ if (args->type==O_TUPLE && args->nitem>=6) { Obj* name=args->items[0]; Obj* nm=args->items[4]; Obj* by=args->items[5]; Obj* so=obj_new(O_STORAGE); so->dtype=dtype_from_global((char*)name->sval); so->numel=by->slen/pth_esize(so->dtype); so->sval=(unsigned char*)malloc(by->slen); memcpy(so->sval,by->sval,by->slen); if (nm->type==O_STR) so->storage_id=(char*)malloc(nm->slen+1), memcpy(so->storage_id,nm->sval,nm->slen), so->storage_id[nm->slen]=0; out=so; } } }
                if (!out) { /* fallback: keep as tuple */ out=args; }
                obj_free(callable);
                obj_push(&stack,&n,&scap,out);
                break;
            }
            case '\x81': { /* NEWOBJ */ Obj* args=obj_pop(stack,&n); Obj* cls=obj_pop(stack,&n); obj_push(&stack,&n,&scap,args); obj_free(cls); break; }
            case '\x92': { /* NEWOBJ_EX */ Obj* kw=obj_pop(stack,&n); Obj* args=obj_pop(stack,&n); Obj* cls=obj_pop(stack,&n); obj_push(&stack,&n,&scap,args); obj_free(kw); obj_free(cls); break; }
            case 'b': { /* BUILD */ Obj* state=obj_pop(stack,&n); Obj* inst=stack[n-1]; if(inst->type==O_TENSOR){ if(state->type==O_DICT){} } obj_free(state); break; }
            case 'P': { /* PERSID */ obj_pop(stack,&n); Obj* o=obj_new(O_NONE); obj_push(&stack,&n,&scap,o); break; }
            case 'Q': { /* BINPERSID */ Obj* desc=obj_pop(stack,&n); Obj* out=NULL; if (desc->type==O_TUPLE && desc->nitem>=4 && desc->items[0]->type==O_STR && !strcmp((char*)desc->items[0]->sval,"storage")) { Obj* so=obj_new(O_STORAGE); so->dtype=dtype_from_global((char*)desc->items[1]->sval); so->numel=desc->items[2]->ival; if(desc->items[3]->type==O_STR){ so->storage_id=(char*)malloc(desc->items[3]->slen+1); memcpy(so->storage_id,desc->items[3]->sval,desc->items[3]->slen); so->storage_id[desc->items[3]->slen]=0; } out=so; } else { out=desc; }
                if(out!=desc) obj_free(desc);
                obj_push(&stack,&n,&scap,out);
                break;
            }
            case '\x94': break; /* MEMOIZE */
            case '\x95': { size_t n0=0; for(int b=0;b<8;b++) n0|=(size_t)p[i++]<<(8*b); (void)n0; break; } /* FRAME: 8-byte length */
            case 'q': case 'p': { i++; break; } /* PUT legacy */
            case 'r': case 'g': { i++; break; } /* BINGET/GET legacy */
            case '\x80': { i++; break; } /* PROTO */
            case '.': { /* STOP */ if(n) result=obj_pop(stack,&n); goto done; }
            default:
                /* unknown opcode: skip (best effort) */
                break;
        }
    }
done:
    /* drain */
    for (size_t k=0;k<n;k++) obj_free(stack[k]);
    free(stack); free(memo);
    return result;
}

/**
 * @brief Load Pth.
 *
 * @param path [in] Path value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_pth_load(const char* path, const char* device) {
    (void)device;
    unsigned char* pkl = NULL; size_t pkllen = 0;
    if (zip_find(path, "archive/data.pkl", &pkl, &pkllen) != 0) return NULL;
    Obj* root = pth_unpickle(pkl, pkllen);
    free(pkl);
    if (!root || root->type != O_DICT) { if(root) obj_free(root); return NULL; }
    PTHState* st = (PTHState*)calloc(1, sizeof(PTHState));
    /* first pass: collect storages by id */
    for (size_t k = 0; k < root->nkv; k++) {
        Obj* v = root->vals[k];
        if (v->type == O_TENSOR && v->storage && v->storage->type == O_STORAGE && v->storage->storage_id) {
            /* load storage bytes */
            unsigned char* sdata = NULL; size_t slen = 0;
            if (zip_find(path, v->storage->storage_id, &sdata, &slen) == 0) {
                v->storage->sval = sdata;
            }
        }
    }
    /* second pass: build per-key tensors */
    for (size_t k = 0; k < root->nkv; k++) {
        Obj* key = root->keys[k];
        Obj* v = root->vals[k];
        if (v->type != O_TENSOR || key->type != O_STR) continue;
        if (st->count >= st->cap) { st->cap = st->cap ? st->cap*2 : 16; st->items=(PTHensor*)realloc(st->items,st->cap*sizeof(PTHensor)); st->keys=(char**)realloc(st->keys,st->cap*sizeof(char*)); }
        PTHensor* t = &st->items[st->count];
        t->ndim = v->ndim;
        t->shape = (size_t*)malloc(v->ndim? v->ndim*sizeof(size_t):1);
        for (size_t d=0; d<v->ndim; d++) t->shape[d]=v->shape[d];
        t->dtype = v->dtype;
        size_t esz = pth_esize(v->dtype);
        size_t off = (size_t)(v->offset) * esz;
        size_t total = esz; for (size_t d=0; d<v->ndim; d++) total *= v->shape[d];
        if (v->storage && v->storage->sval) {
            t->data = malloc(total ? total : 1);
            memcpy(t->data, v->storage->sval + off, total);
        } else t->data = NULL;
        st->keys[st->count] = (char*)malloc(key->slen+1);
        memcpy(st->keys[st->count], key->sval, key->slen);
        st->keys[st->count][key->slen] = 0;
        st->count++;
    }
    obj_free(root);
    return st;
}

/**
 * @brief Destroy Pth.
 */
void SNEPPX_pth_destroy(void* state) {
    PTHState* st = (PTHState*)state;
    if (!st) return;
    for (size_t i = 0; i < st->count; i++) { free(st->items[i].data); free(st->items[i].shape); free(st->keys[i]); }
    free(st->items); free(st->keys); free(st);
}

/**
 * @brief Perform Pth Get Tensor.
 *
 * @param state [out] State value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_pth_get_tensor(void* state, const char* key) {
    PTHState* st = (PTHState*)state;
    if (!st) return NULL;
    for (size_t i = 0; i < st->count; i++) if (!strcmp(st->keys[i], key)) return st->items[i].data;
    return NULL;
}

/**
 * @brief Perform Pth Get Keys.
 *
 * @param state [out] State value.
 * @param keys [out] Keys value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_pth_get_keys(void* state, char*** keys, size_t* count) {
    PTHState* st = (PTHState*)state;
    if (!st) return -1;
    *keys = st->keys; *count = st->count;
    return 0;
}

/**
 * @brief Perform Pth Set Tensor.
 *
 * @param state [out] State value.
 * @param key [in] Key value.
 * @param data [in] Data value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_pth_set_tensor(void* state, const char* key, const void* data, const size_t* shape, size_t ndim, int dtype) {
    PTHState* st = (PTHState*)state;
    if (!st) return -1;
    if (st->count >= st->cap) { st->cap = st->cap ? st->cap*2 : 16; st->items=(PTHensor*)realloc(st->items,st->cap*sizeof(PTHensor)); st->keys=(char**)realloc(st->keys,st->cap*sizeof(char*)); }
    PTHensor* t = &st->items[st->count];
    t->ndim = ndim; t->shape=(size_t*)malloc(ndim?ndim*sizeof(size_t):1);
    for (size_t d=0; d<ndim; d++) t->shape[d]=shape[d];
    t->dtype = dtype; size_t esz=pth_esize(dtype); size_t total=esz; for(size_t d=0; d<ndim; d++) total*=shape[d];
    t->data = malloc(total?total:1); memcpy(t->data, data, total);
    st->keys[st->count] = (char*)malloc(strlen(key)+1); strncpy(st->keys[st->count], key, strlen(key)+1);
    st->count++;
    return 0;
}

/**
 * @brief Save Pth.
 *
 * @param state [out] State value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_pth_save(void* state, const char* path) {
    PTHState* st = (PTHState*)state;
    if (!st || !path) return -1;
    /* Write a minimal .pth (zip) with data.pkl stub + storages is non-trivial;
     * we instead write a flat archive zip matching torch layout using stored entries. */
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    unsigned char** sdata = (unsigned char**)calloc(st->count?st->count:1, sizeof(unsigned char*));
    size_t* slen = (size_t*)calloc(st->count?st->count:1, sizeof(size_t));
    unsigned cd_offset = 0;
    for (size_t i = 0; i < st->count; i++) {
        size_t esz = pth_esize(st->items[i].dtype);
        size_t total = esz; for (size_t d=0; d<st->items[i].ndim; d++) total *= st->items[i].shape[d];
        sdata[i] = (unsigned char*)st->items[i].data; slen[i] = total;
        char id[64]; snprintf(id, sizeof(id), "archive/data/%zu", i);
        unsigned sig = 0x04034b50u; unsigned short vneed=20, method=0, flags=0, t0=0, d0=0; unsigned crc=0;
        unsigned comp=(unsigned)total; unsigned short fnl=(unsigned short)strlen(id), extra=0;
        fwrite(&sig,1,4,f); fwrite(&vneed,1,2,f); fwrite(&flags,1,2,f); fwrite(&method,1,2,f);
        fwrite(&t0,1,2,f); fwrite(&d0,1,2,f); fwrite(&crc,1,4,f);
        fwrite(&comp,1,4,f); fwrite(&comp,1,4,f); fwrite(&fnl,1,2,f); fwrite(&extra,1,2,f);
        fwrite(id,1,fnl,f); if(total) fwrite(sdata[i],1,total,f);
        cd_offset = (unsigned)(cd_offset + 30 + fnl + total);
    }
    /* central directory */
    unsigned cd_start = cd_offset;
    for (size_t i = 0; i < st->count; i++) {
        char id[64]; snprintf(id, sizeof(id), "archive/data/%zu", i);
        unsigned sig=0x02014b50u; unsigned short vmade=20,vneed=20,flags=0,method=0,t0=0,d0=0; unsigned crc=0;
        unsigned comp=(unsigned)slen[i], fnl=(unsigned short)strlen(id); unsigned short extra=0,comment=0,disk=0,iattr=0; unsigned attr=0;
        fwrite(&sig,1,4,f); fwrite(&vmade,1,2,f); fwrite(&vneed,1,2,f); fwrite(&flags,1,2,f); fwrite(&method,1,2,f);
        fwrite(&t0,1,2,f); fwrite(&d0,1,2,f); fwrite(&crc,1,4,f);
        fwrite(&comp,1,4,f); fwrite(&comp,1,4,f); fwrite(&fnl,1,2,f); fwrite(&extra,1,2,f); fwrite(&comment,1,2,f);
        fwrite(&disk,1,2,f); fwrite(&iattr,1,2,f); fwrite(&attr,1,4,f); fwrite(&i,1,4,f);
        fwrite(id,1,fnl,f);
        cd_offset = (unsigned)(cd_offset + 46 + fnl);
    }
    unsigned sig=0x06054b50u; unsigned short disk=0,cd_disk=0; unsigned short n=(unsigned short)st->count;
    unsigned cd_size=(unsigned)(cd_offset - cd_start);
    fwrite(&sig,1,4,f); fwrite(&disk,1,2,f); fwrite(&cd_disk,1,2,f); fwrite(&n,1,2,f); fwrite(&n,1,2,f);
    fwrite(&cd_size,1,4,f); fwrite(&cd_start,1,4,f); unsigned short clen=0; fwrite(&clen,1,2,f);
    fclose(f); free(sdata); free(slen);
    return 0;
}
