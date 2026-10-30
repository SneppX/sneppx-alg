/*
 * CTM Tool: Count LOC by file
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

int is_ignored_dir(const char *name) {
    if (strcmp(name, ".git")==0||strcmp(name, ".svn")==0||strcmp(name, ".hg")==0) return 1;
    if (strcmp(name, "__pycache__")==0||strcmp(name,"node_modules")==0) return 1;
    if (strcmp(name, ".venv")==0||strcmp(name,"venv")==0) return 1;
    if (strcmp(name,"build")==0||strcmp(name,"dist")==0) return 1;
    if (strcmp(name,".next")==0||strcmp(name,".cache")==0) return 1;
    if (strcmp(name,".pytest_cache")==0||strcmp(name,".ruff_cache")==0) return 1;
    return 0;
}
int is_ignored_ext(const char *name) {
    const char *ext = strrchr(name,'.');
    if(!ext) return 0;
    if(strcmp(ext,".pyc")==0||strcmp(ext,".pyo")==0||strcmp(ext,".o")==0||strcmp(ext,".obj")==0) return 1;
    if(strcmp(ext,".exe")==0||strcmp(ext,".dll")==0||strcmp(ext,".so")==0||strcmp(ext,".dylib")==0) return 1;
    if(strcmp(ext,".class")==0||strcmp(ext,".jar")==0) return 1;
    return 0;
}
#ifdef _WIN32
long count_loc_file(const char *fpath) {
    FILE *f = fopen(fpath,"rb");
    if(!f) return 0;
    fseek(f,0,SEEK_END);
    long sz = ftell(f);
    fseek(f,0,SEEK_SET);
    if(sz==0){fclose(f);return 0;}
    char *buf = (char*)malloc(sz);
    if(!buf){fclose(f);return 0;}
    fread(buf,1,sz,f); fclose(f);
    long lines=0;
    for(long i=0;i<sz;i++) if(buf[i]=='\n') lines++;
    if(sz>0 && buf[sz-1]!='\n') lines++;
    free(buf);
    return lines;
}
void walk_by_file(const char *base, const char *root) {
    char path[MAX_PATH];
    snprintf(path,MAX_PATH,"%s\\*",base);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(path,&fd);
    if(h==INVALID_HANDLE_VALUE) return;
    do {
        if(strcmp(fd.cFileName,".")==0||strcmp(fd.cFileName,"..")==0) continue;
        if(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            if(is_ignored_dir(fd.cFileName)) continue;
            char sub[MAX_PATH];
            snprintf(sub,MAX_PATH,"%s\\%s",base,fd.cFileName);
            walk_by_file(sub, root);
        } else {
            if(is_ignored_ext(fd.cFileName)) continue;
            char fpath[MAX_PATH];
            snprintf(fpath,MAX_PATH,"%s\\%s",base,fd.cFileName);
            long l = count_loc_file(fpath);
            if(l==0) continue;
            // print relative
            const char *rel = fpath;
            if(strncmp(rel, root, strlen(root))==0) {
                rel += strlen(root);
                if(rel[0]=='\\'||rel[0]=='/') rel++;
            }
            printf("%6ld %s\n", l, rel);
        }
    } while(FindNextFileA(h,&fd));
    FindClose(h);
}
#endif
int main(int argc,char **argv){
    const char *root = argc>1?argv[1]:".";
#ifdef _WIN32
    walk_by_file(root, root);
#else
#endif
    return 0;
}
