(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.SneppXGitHighlighter = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";
  const extensions = {
    py: "python",
    pyi: "python",
    c: "c",
    h: "c",
    cc: "cpp",
    cpp: "cpp",
    hpp: "cpp",
    js: "javascript",
    jsx: "javascript",
    ts: "typescript",
    tsx: "typescript",
    json: "json",
    yml: "yaml",
    yaml: "yaml",
    toml: "toml",
    sh: "bash",
    ps1: "powershell",
    html: "html",
    css: "css",
    sql: "sql",
    rs: "rust",
    go: "go",
    java: "java",
  };
  const keywords = {
    python:
      "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return self try while with yield",
    c: "_Alignas _Alignof _Atomic _Bool _Complex _Generic _Imaginary _Noreturn _Static_assert _Thread_local auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while",
    cpp: "alignas alignof and asm auto bool break case catch char class const constexpr const_cast continue decltype default delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable namespace new noexcept not nullptr operator or private protected public register reinterpret_cast return short signed sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while",
    javascript:
      "async await break case catch class const continue debugger default delete do else export extends false finally for from function get if import in instanceof let new null of return set static super switch this throw true try typeof undefined var void while with yield",
    typescript:
      "abstract any as asserts async await boolean break case catch class const constructor continue debugger declare default delete do else enum export extends false finally for from function get if implements import in infer instanceof interface is keyof let module namespace never new null number object of package private protected public readonly require global return set static string super switch symbol this throw true try type typeof undefined unique unknown var void while with yield",
    rust: "as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while",
    go: "break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var",
    java: "abstract assert boolean break byte case catch char class const continue default do double else enum extends final finally float for if implements import instanceof int interface long native new package private protected public return short static strictfp super switch synchronized this throw throws transient try void volatile while",
    json: "false null true",
    yaml: "false null true yes no on off",
    toml: "false true",
    bash: "case coproc do done elif else esac fi for function if in select then time until while",
    powershell:
      "begin break catch class continue data define do dynamicparam else elseif end enum exit filter finally for foreach from function if in param process return switch throw trap try until using var while",
  };
  const lineComments = new Set(["python", "yaml", "toml", "bash", "powershell"]),
    slashComments = new Set(["c", "cpp", "javascript", "typescript", "rust", "go", "java"]);
  const cssNames = new Set(
    "aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey darkturquoise darkviolet deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory khaki lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen linen magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin navajowhite navy oldlace olive olivedrab orange orangered orchid palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru pink plum powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue slateblue slategray slategrey snow springgreen steelblue tan teal thistle tomato turquoise violet wheat white whitesmoke yellow yellowgreen".split(
      /\s+/,
    ),
  );
  const stringPattern = /^(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/,
    numberPattern = /^(?:0x[\da-f]+|\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:%|px|rem|em|vh|vw)?/i,
    wordPattern = /^[A-Za-z_$][\w$]*/;
  function language(path) {
    return extensions[(String(path).split(".").pop() || "").toLowerCase()] || "text";
  }
  function tokenizeLine(line, lang) {
    line = String(line);
    const tokens = [],
      set = new Set((keywords[lang] || "").split(/\s+/));
    let i = 0;
    function push(type, value) {
      const last = tokens[tokens.length - 1];
      if (last && last.type === type) last.value += value;
      else tokens.push({ type, value });
    }
    while (i < line.length) {
      const rest = line.slice(i);
      if (
        (lineComments.has(lang) && rest[0] === "#") ||
        (slashComments.has(lang) && rest.startsWith("//")) ||
        (rest.startsWith("/*") &&
          ["c", "cpp", "javascript", "typescript", "rust", "go", "java", "css", "sql"].includes(lang))
      ) {
        push("comment", rest);
        break;
      }
      const str = stringPattern.exec(rest);
      if (str) {
        push("string", str[0]);
        i += str[0].length;
        continue;
      }
      if (lang === "css") {
        const color =
          /^(?:#(?:[\da-f]{3}|[\da-f]{4}|[\da-f]{6}|[\da-f]{8})\b|(?:rgba?|hsla?)\([^)]*\)|[A-Za-z_$][\w$]*)/i.exec(
            rest,
          );
        if (color) {
          const value = color[0],
            isName = cssNames.has(value.toLowerCase());
          if (value[0] === "#" || /^(?:rgba?|hsla?)\(/i.test(value) || isName) {
            push("color", value);
            i += value.length;
            continue;
          }
        }
      }
      const num = numberPattern.exec(rest);
      if (num) {
        push("number", num[0]);
        i += num[0].length;
        continue;
      }
      const word = wordPattern.exec(rest);
      if (word) {
        const value = word[0];
        push(set.has(value) ? "keyword" : rest.slice(value.length).startsWith("(") ? "function" : "plain", value);
        i += value.length;
        continue;
      }
      push("plain", line[i]);
      i++;
    }
    return tokens;
  }
  return Object.freeze({ language, tokenizeLine });
});
