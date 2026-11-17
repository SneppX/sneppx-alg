(() => {
"use strict";
const root=document.querySelector("[data-git-browser]"); if(!root)return;
const BASE="/sneppx-alg/", DATA=BASE+"git-browser-data/", REPO="https://github.com/SneppX/sneppx-alg";
const state={manifest:null,branch:null,tree:null};
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
const link=(href,text)=>{const n=el("a","",text);n.href=href;return n;};
const url=(kind,branch,path)=>BASE+kind+"/"+branch+"/"+(path?path.split("/").map(encodeURIComponent).join("/")+"/":"");
const raw=(sha,path)=>"https://raw.githubusercontent.com/SneppX/sneppx-alg/"+sha+"/"+path.split("/").map(encodeURIComponent).join("/");
const size=n=>n==null?"—":n<1024?n+" B":n<1048576?(n/1024).toFixed(1)+" KB":(n/1048576).toFixed(1)+" MB";
const date=s=>{const d=new Date(s);return isNaN(d)?s:d.toLocaleString();};
const show=(...nodes)=>{root.replaceChildren(...nodes);root.classList.add("git-browser");};
const header=(title,detail)=>{const h=el("div","git-browser-header"),l=el("div");l.append(el("h1","git-browser-title",title));if(detail)l.append(el("div","git-browser-muted",detail));h.append(l,link(REPO,"View on GitHub ↗"));return h;};
const toolbar=()=>{const b=el("div","git-browser-toolbar"),s=el("select");s.setAttribute("aria-label","Select branch");state.manifest.branches.forEach(x=>{const o=el("option","",x.name);o.value=x.name;o.selected=x.name===state.branch;s.append(o);});s.addEventListener("change",()=>location.href=url("tree",s.value,""));b.append(s,link(BASE+"branches/","Branches"),link(BASE+"tags/","Tags"),link(BASE+"commits/"+(state.branch||"main")+"/","Commits"),link(BASE+"search/","Search"));return b;};
const crumbs=(branch,path)=>{const b=el("div","git-browser-crumbs");b.append(link(url("tree",branch,""),branch));let current="";(path?path.split("/"):[]).forEach(part=>{b.append(el("span","","/"));current=current?current+"/"+part:part;b.append(link(url("tree",branch,current),part));});return b;};
const requestCache=new Map(),REQUEST_TIMEOUT_MS=3000;
async function requestText(target){
if(requestCache.has(target))return requestCache.get(target);
const pending=(async()=>{const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),REQUEST_TIMEOUT_MS);try{const response=await fetch(target,{cache:"force-cache",signal:controller.signal});if(!response.ok)throw Error("Request failed ("+response.status+").");return await response.text();}catch(error){if(error.name==="AbortError")throw Error("Request timed out after "+(REQUEST_TIMEOUT_MS/1000)+" seconds. Check the connection and retry.");throw error;}finally{clearTimeout(timer);}})();
requestCache.set(target,pending);
try{return await pending;}catch(error){requestCache.delete(target);throw error;}
}
async function json(path){try{return JSON.parse(await requestText(path));}catch(error){throw Error("Repository index unavailable: "+error.message+" The documentation may still be deploying.");}}
async function manifest(){if(!state.manifest)state.manifest=await json(DATA+"index.json");return state.manifest;}
async function loadBranch(name){const b=state.manifest.branches.find(x=>x.name===name);if(!b)throw Error("Branch not found: "+name);state.branch=name;if(!state.tree||state.tree.branch!==name)state.tree=await json(DATA+"trees/"+b.id+".json");return b;}
const tagUrl=(kind,tag,path)=>BASE+"tag/"+tag+"/"+kind+"/"+(path?path.split("/").map(encodeURIComponent).join("/")+"/":"");
async function loadTag(name){const tag=state.manifest.tags.find(x=>x.name===name);if(!tag)throw Error("Tag not found: "+name);state.branch=name;state.tree=await json(DATA+"trees/"+tag.id+".json");return tag;}
function resolveTag(rest){for(const tag of state.manifest.tags.slice().sort((a,b)=>b.name.length-a.name.length)){if(rest.startsWith(tag.name+"/"))return{tag:tag.name,tail:rest.slice(tag.name.length+1)};}throw Error("The requested tag does not exist.");}
const tagCrumbs=(tag,path)=>{const b=el("div","git-browser-crumbs");b.append(link(tagUrl("tree",tag,""),tag));let current="";(path?path.split("/"):[]).forEach(part=>{b.append(el("span","","/"));current=current?current+"/"+part:part;b.append(link(tagUrl("tree",tag,current),part));});return b;}
function resolveBranch(rest){for(const b of state.manifest.branches.slice().sort((a,b)=>b.name.length-a.name.length)){if(rest===b.name||rest.startsWith(b.name+"/"))return{branch:b.name,path:rest===b.name?"":rest.slice(b.name.length+1).replace(/\/$/,"")};}throw Error("The requested branch does not exist.");}
function rows(path){const prefix=path?path+"/":"",map=new Map();state.tree.entries.forEach(x=>{if(!x.path.startsWith(prefix))return;const tail=x.path.slice(prefix.length);if(!tail)return;const i=tail.indexOf("/");if(i>=0){const n=tail.slice(0,i);if(!map.has(n))map.set(n,{name:n,dir:true});}else map.set(tail,{name:tail,path:x.path,sha:x.sha,size:x.size,mode:x.mode});});return [...map.values()].sort((a,b)=>Number(b.dir)-Number(a.dir)||a.name.localeCompare(b.name,undefined,{numeric:true,sensitivity:"base"}));}
async function tree(branch,path){const meta=await loadBranch(branch),items=rows(path),table=el("table","git-browser-table");table.innerHTML="<thead><tr><th>Name</th><th>Type</th><th>Size</th></tr></thead>";const body=el("tbody");if(path){const parent=path.split("/").slice(0,-1).join("/"),tr=el("tr"),td=el("td");td.append(link(url("tree",branch,parent),"↰ .."));tr.append(td,el("td","","Directory"),el("td","","—"));body.append(tr);}items.forEach(x=>{const tr=el("tr"),td=el("td","git-browser-name"),target=x.dir?url("tree",branch,path?(path+"/") + x.name:x.name):url("blob",branch,x.path);td.append(link(target,(x.dir?"📁 ":"▤ ")+x.name));tr.append(td,el("td","",x.dir?"Directory":x.mode==="120000"?"Symlink":"File"),el("td","",size(x.size)));body.append(tr);});table.append(body);const summary=el("div","git-browser-summary");summary.append(el("span","",items.filter(x=>x.dir).length+" directories"),el("span","",items.filter(x=>!x.dir).length+" files"),link(REPO+"/commit/"+meta.commit,"Indexed commit "+meta.commit.slice(0,12)));const actions=el("div","git-browser-actions");actions.append(link(REPO+"/archive/refs/heads/"+encodeURIComponent(branch)+".zip","Download ZIP"));const compare=el("button","git-browser-action-button","Compare with main");compare.type="button";compare.addEventListener("click",()=>compareBranch(branch).catch(error=>show(header("Comparison unavailable"),el("div","git-browser-error",error.message))));actions.append(compare);show(header("SneppX-ALG / "+(path?path.split("/").pop():branch),"Read-only source browser · branch "+branch),toolbar(),actions,crumbs(branch,path),summary,table);}

function language(path){const ext=(path.split(".").pop()||"").toLowerCase();return ({py:"python",pyi:"python",c:"c",h:"c",cc:"cpp",cpp:"cpp",hpp:"cpp",js:"javascript",jsx:"javascript",ts:"typescript",tsx:"typescript",json:"json",yml:"yaml",yaml:"yaml",toml:"toml",sh:"bash",ps1:"powershell",html:"html",css:"css",sql:"sql",rs:"rust",go:"go",java:"java"})[ext]||"text";}
const LANGUAGE_KEYWORDS={
python:"False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return self try while with yield",
c:"_Alignas _Alignof _Atomic _Bool _Complex _Generic _Imaginary _Noreturn _Static_assert _Thread_local auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while",
cpp:"alignas alignof and asm auto bool break case catch char class const constexpr const_cast continue decltype default delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable namespace new noexcept not nullptr operator or private protected public register reinterpret_cast return short signed sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while",
javascript:"async await break case catch class const continue debugger default delete do else export extends false finally for from function get if import in instanceof let new null of return set static super switch this throw true try typeof undefined var void while with yield",
typescript:"abstract any as asserts async await boolean break case catch class const constructor continue debugger declare default delete do else enum export extends false finally for from function get if implements import in infer instanceof interface is keyof let module namespace never new null number object of package private protected public readonly require global return set static string super switch symbol this throw true try type typeof undefined unique unknown var void while with yield",
rust:"as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while",
go:"break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var",
java:"abstract assert boolean break byte case catch char class const continue default do double else enum extends final finally float for if implements import instanceof int interface long native new package private protected public return short static strictfp super switch synchronized this throw throws transient try void volatile while",
json:"false null true",yaml:"false null true yes no on off",toml:"false true",
bash:"case coproc do done elif else esac fi for function if in select then time until while",
powershell:"begin break catch class continue data define do dynamicparam else elseif end enum exit filter finally for foreach from function if in param process return switch throw trap try until using var while"
};
const LANG_LINE_COMMENTS=new Set(["python","yaml","toml","bash","powershell"]);
const LANG_SLASH_COMMENTS=new Set(["c","cpp","javascript","typescript","rust","go","java"]);
function highlightedLine(line,lang){
const fragment=document.createDocumentFragment(),keywords=new Set((LANGUAGE_KEYWORDS[lang]||"").split(/\s+/));
let cursor=0;
while(cursor<line.length){
const rest=line.slice(cursor);
if((LANG_LINE_COMMENTS.has(lang)&&rest[0]==="#")||(LANG_SLASH_COMMENTS.has(lang)&&rest.startsWith("//"))||(rest.startsWith("/*")&&["c","cpp","javascript","typescript","rust","go","java","css","sql"].includes(lang))){
fragment.append(el("span","git-token-comment",rest));break;
}
const stringMatch=/^(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/.exec(rest);
if(stringMatch){fragment.append(el("span","git-token-string",stringMatch[0]));cursor+=stringMatch[0].length;continue;}
if(lang==="css"){
const colorMatch=/^(?:#[0-9a-f]{3,8}\b|(?:rgba?|hsla?)\([^)]*\)|[a-zA-Z]{3,20}\b)/i.exec(rest);
if(colorMatch){
const value=colorMatch[0],hex=value[0]==="#",fn=/^(?:rgba?|hsla?)\(/i.test(value);
const named=!hex&&!fn&&typeof CSS!=="undefined"&&CSS.supports("color",value)&&!["transparent","currentcolor"].includes(value.toLowerCase());
if(hex||fn||named){
const swatch=el("span","git-color-swatch");swatch.style.backgroundColor=value;swatch.setAttribute("aria-label","Color preview: "+value);swatch.title=value;
const token=el("span","git-token-color");token.append(swatch,document.createTextNode(value));fragment.append(token);cursor+=value.length;continue;
}
}
}
const numberMatch=/^(?:0x[\da-f]+|\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:%|px|rem|em|vh|vw)?/i.exec(rest);
if(numberMatch){fragment.append(el("span","git-token-number",numberMatch[0]));cursor+=numberMatch[0].length;continue;}
const wordMatch=/^[A-Za-z_$][\w$]*/.exec(rest);
if(wordMatch){
const word=wordMatch[0];
if(keywords.has(word))fragment.append(el("span","git-token-keyword",word));
else if(rest.slice(word.length).startsWith("("))fragment.append(el("span","git-token-function",word));
else fragment.append(document.createTextNode(word));
cursor+=word.length;continue;
}
fragment.append(document.createTextNode(line[cursor]));cursor++;
}
return fragment;
}

async function blob(branch,path){const meta=await loadBranch(branch),item=state.tree.entries.find(x=>x.path===path);if(!item)throw Error("File not found: "+path);const actions=el("div","git-browser-actions");actions.append(link(REPO+"/blob/"+meta.commit+"/"+path,"GitHub permalink"),link(raw(meta.commit,path),"Raw file"),link(REPO+"/commits/"+meta.commit+"/"+path,"File history"),link(REPO+"/blame/"+meta.commit+"/"+path,"Blame"));const code=el("div","git-browser-code","Loading source…");show(header(path,"Branch "+branch),toolbar(),crumbs(branch,path),actions,el("div","git-browser-muted","Blob "+item.sha+" · "+size(item.size)+" · commit "+meta.commit.slice(0,12)),code);if(item.size>1048576){code.textContent="This file exceeds 1 MB. Use the raw-file link to view or download it.";return;}const source=await requestText(raw(meta.commit,path));const lines=source.replace(/\\r\\n/g,"\\n").split("\\n"),table=el("table"),body=el("tbody"),lang=language(path);lines.forEach((line,i)=>{const tr=el("tr"),n=el("td","",String(i+1)),c=el("td"),tokenLine=el("code");tokenLine.append(highlightedLine(line||" ",lang));c.append(tokenLine);tr.append(n,c);body.append(tr);});table.append(body);code.replaceChildren(table);}
async function tagTree(tag,path){
const meta=await loadTag(tag),items=rows(path),table=el("table","git-browser-table");
table.innerHTML="<thead><tr><th>Name</th><th>Type</th><th>Size</th></tr></thead>";
const body=el("tbody");
if(path){const parent=path.split("/").slice(0,-1).join("/"),tr=el("tr"),td=el("td");td.append(link(tagUrl("tree",tag,parent),"↰ .."));tr.append(td,el("td","","Directory"),el("td","","—"));body.append(tr);}
items.forEach(x=>{const tr=el("tr"),td=el("td","git-browser-name"),target=x.dir?tagUrl("tree",tag,path?(path+"/")+x.name:x.name):tagUrl("blob",tag,x.path);td.append(link(target,(x.dir?"📁 ":"▤ ")+x.name));tr.append(td,el("td","",x.dir?"Directory":x.mode==="120000"?"Symlink":"File"),el("td","",size(x.size)));body.append(tr);});
table.append(body);
const actions=el("div","git-browser-actions");
actions.append(link(REPO+"/archive/refs/tags/"+encodeURIComponent(tag)+".zip","Download ZIP"),link(REPO+"/commit/"+meta.commit,"Tagged commit on GitHub"));
const summary=el("div","git-browser-summary");summary.append(el("span","",items.filter(x=>x.dir).length+" directories"),el("span","",items.filter(x=>!x.dir).length+" files"));
show(header("SneppX-ALG / tag / "+tag,"Read-only source browser · immutable tag"),actions,tagCrumbs(tag,path),summary,table);
}
async function tagBlob(tag,path){
const meta=await loadTag(tag),item=state.tree.entries.find(x=>x.path===path);if(!item)throw Error("File not found: "+path);
const actions=el("div","git-browser-actions");
actions.append(link(REPO+"/blob/"+meta.commit+"/"+path,"GitHub permalink"),link(raw(meta.commit,path),"Raw file"),link(REPO+"/commits/"+meta.commit+"/"+path,"File history"),link(REPO+"/blame/"+meta.commit+"/"+path,"Blame"));
const code=el("div","git-browser-code","Loading source…");
show(header(path,"Tag "+tag),actions,tagCrumbs(tag,path),el("div","git-browser-muted","Blob "+item.sha+" · "+size(item.size)+" · commit "+meta.commit.slice(0,12)),code);
if(item.size>1048576){code.textContent="This file exceeds 1 MB. Use the raw-file link to view or download it.";return;}
const source=await requestText(raw(meta.commit,path));
const lines=source.replace(/\r\n/g,"\n").split("\n"),table=el("table"),body=el("tbody"),lang=language(path);
lines.forEach((line,i)=>{const tr=el("tr"),number=el("td","",String(i+1)),cell=el("td"),code=el("code");code.append(highlightedLine(line||" ",lang));cell.append(code);tr.append(number,cell);body.append(tr);});
table.append(body);code.replaceChildren(table);
}
async function commitDetails(sha){
const data=await json("https://api.github.com/repos/SneppX/sneppx-alg/commits/"+sha),meta=data.commit||{},author=meta.author||{},files=data.files||[];
const summary=el("div","git-browser-summary");summary.append(el("span","","Commit "+sha.slice(0,12)),el("span","",files.length+" changed files"),el("span","",date(author.date||"")));
const message=el("pre","git-browser-code",meta.message||"");
const parents=el("div","git-browser-actions");(data.parents||[]).forEach(parent=>parents.append(link(BASE+"commit/"+parent.sha+"/","Parent "+parent.sha.slice(0,12))));
const table=el("table","git-browser-table");table.innerHTML="<thead><tr><th>File</th><th>Status</th><th>Changes</th></tr></thead>";const body=el("tbody");
files.forEach(file=>{const tr=el("tr"),name=el("td","git-browser-name");name.append(link(REPO+"/blob/"+sha+"/"+file.filename,file.filename));tr.append(name,el("td","",file.status||"modified"),el("td","",(file.additions||0)+" additions / "+(file.deletions||0)+" deletions"));body.append(tr);if(file.patch){const patchRow=el("tr","git-browser-patch-row"),cell=el("td");cell.colSpan=3;const details=el("details");details.append(el("summary","","View diff"),el("pre","git-browser-diff",file.patch));cell.append(details);patchRow.append(cell);body.append(patchRow);}});
table.append(body);
show(header("Commit "+sha.slice(0,12),author.name?author.name+" · "+(author.email||""):sha),summary,message,parents,table,link(REPO+"/commit/"+sha,"Open full commit on GitHub ↗"));
}
async function compareBranch(branch){
const comparison=await json("https://api.github.com/repos/SneppX/sneppx-alg/compare/main..."+encodeURIComponent(branch)),commits=comparison.commits||[],files=comparison.files||[];
const summary=el("div","git-browser-summary");summary.append(el("span","",commits.length+" commits ahead of main"),el("span","",files.length+" changed files"),el("span","",comparison.status||"comparison"));
const table=el("table","git-browser-table");table.innerHTML="<thead><tr><th>File</th><th>Status</th><th>Changes</th></tr></thead>";const body=el("tbody");
files.forEach(file=>{const tr=el("tr"),name=el("td","git-browser-name");name.append(link(REPO+"/blob/"+comparison.merge_base.sha+"/"+file.filename,file.filename));tr.append(name,el("td","",file.status||"modified"),el("td","",(file.additions||0)+" additions / "+(file.deletions||0)+" deletions"));body.append(tr);if(file.patch){const patchRow=el("tr","git-browser-patch-row"),cell=el("td");cell.colSpan=3;const details=el("details");details.append(el("summary","","View diff"),el("pre","git-browser-diff",file.patch));cell.append(details);patchRow.append(cell);body.append(patchRow);}});
table.append(body);const history=el("ol","git-browser-compare-commits");commits.forEach(commit=>{const item=el("li");item.append(link(BASE+"commit/"+commit.sha+"/",commit.sha.slice(0,12)+" · "+(commit.commit.message||"").split("\n")[0]));history.append(item);});
show(header("Compare main…"+branch,"Branch comparison · "+comparison.base_commit.sha.slice(0,12)+" → "+comparison.head_commit.sha.slice(0,12)),summary,el("h2","","Changed files"),table,el("h2","","Commits"),history,link(REPO+"/compare/main..."+encodeURIComponent(branch),"Open full comparison on GitHub ↗"));
}
async function branches(){const t=el("table","git-browser-table");t.innerHTML="<thead><tr><th>Branch</th><th>Latest commit</th><th>Files</th><th>Commits</th></tr></thead>";const b=el("tbody");state.manifest.branches.forEach(x=>{const tr=el("tr"),n=el("td","git-browser-name"),c=el("td");n.append(link(url("tree",x.name,""),x.name));c.append(link(BASE+"commit/"+x.commit+"/",x.commit.slice(0,12)));tr.append(n,c,el("td","",String(x.file_count)),el("td","",String(x.commit_count)));b.append(tr);});t.append(b);show(header("Branches",state.manifest.branches.length+" branches indexed from GitHub"),t);}
async function tags(){const t=el("table","git-browser-table");t.innerHTML="<thead><tr><th>Tag</th><th>Commit</th><th>Date</th><th>Files</th></tr></thead>";const b=el("tbody");state.manifest.tags.forEach(x=>{const tr=el("tr"),n=el("td","git-browser-name"),c=el("td");n.append(link(tagUrl("tree",x.name,""),x.name));c.append(link(BASE+"commit/"+x.commit+"/",x.commit.slice(0,12)));tr.append(n,c,el("td","",date(x.date)),el("td","",String(x.file_count)));b.append(tr);});t.append(b);show(header("Tags",state.manifest.tags.length+" tags indexed"),t);}
async function commits(branch){const meta=await loadBranch(branch),entry=state.manifest.branches.find(x=>x.name===branch),data=await json(DATA+"commits/"+entry.id+".json"),t=el("table","git-browser-table");t.innerHTML="<thead><tr><th>Commit</th><th>Message</th><th>Author</th><th>Date</th></tr></thead>";const b=el("tbody");data.commits.forEach(x=>{const tr=el("tr"),s=el("td");s.append(link(BASE+"commit/"+x.sha+"/",x.sha.slice(0,12)));tr.append(s,el("td","",x.subject),el("td","",x.author),el("td","",date(x.date)));b.append(tr);});t.append(b);show(header("Commit history","Branch "+branch+" · "+data.commits.length+" commits indexed"),toolbar(),t);}
async function search(){
const selected=new URLSearchParams(location.search).get("branch")||"main",branch=await loadBranch(selected);
const input=el("input","git-browser-search"),results=el("div","git-browser-search-results");
input.placeholder="Search source code, symbols, comments and paths…";input.setAttribute("aria-label","Search source code");
input.value=new URLSearchParams(location.search).get("q")||"";
const index=await json(DATA+"search/"+branch.id+".json"),info=el("p","git-browser-muted",index.lines.length.toLocaleString()+" indexed source lines"+(index.truncated?" · index capped":"")+" · branch "+selected);
let timer=null;
const run=()=>{const terms=input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);results.replaceChildren();if(!terms.length)return;
const matches=index.lines.filter(item=>{const value=(item.path+" "+item.text).toLowerCase();return terms.every(term=>value.includes(term));}).slice(0,200);
const list=el("div","git-browser-search-results-list");
matches.forEach(item=>{const row=el("article","git-browser-search-result"),heading=el("div","git-browser-search-path");heading.append(link(url("blob",selected,item.path)+"#L"+item.line,item.path),el("span","git-browser-muted"," : "+item.line));row.append(heading,el("pre","git-browser-search-excerpt",item.text.trim()));list.append(row);});
results.append(el("p","git-browser-muted",matches.length+" matches shown (maximum 200)"),list);};
input.addEventListener("input",()=>{clearTimeout(timer);timer=setTimeout(run,100);});
const controls=el("div","git-browser-toolbar"),select=el("select");select.setAttribute("aria-label","Search branch");
state.manifest.branches.forEach(item=>{const option=el("option","",item.name);option.value=item.name;option.selected=item.name===selected;select.append(option);});
select.addEventListener("change",()=>location.href=BASE+"search/?branch="+encodeURIComponent(select.value)+"&q="+encodeURIComponent(input.value));
controls.append(select,input);show(header("Full-text repository search","Search indexed source lines for symbols, code, comments and paths"),controls,info,results);run();
}
async function route(){const p=decodeURI(location.pathname.replace(BASE,""));if(p==="tree/"||p==="tree"||p==="code-browser/"||p.endsWith("/code-browser/"))return tree("main","");if(p==="branches/"||p==="branches")return branches();if(p==="tags/"||p==="tags")return tags();if(p.startsWith("commit/"))return commitDetails(p.slice(7).replace(/\/$/,""));if(p.startsWith("tag/")){const resolved=resolveTag(p.slice(4));const parts=resolved.tail.split("/");const kind=parts.shift();const path=parts.join("/");if(kind==="tree")return tagTree(resolved.tag,path);if(kind==="blob")return tagBlob(resolved.tag,path);throw Error("Unknown tag route.");}for(const kind of ["tree","blob","commits"]){if(p.startsWith(kind+"/")){const r=resolveBranch(p.slice(kind.length+1));return kind==="tree"?tree(r.branch,r.path):kind==="blob"?blob(r.branch,r.path):commits(r.branch);}}if(p.startsWith("search"))return search();return false;}
manifest().then(route).catch(error=>show(header("Repository browser unavailable"),el("div","git-browser-error",error.message)));
})();