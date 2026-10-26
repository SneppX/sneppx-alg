/* SNEPPX-Alg — tiny scalar reverse-mode autodiff core.
 * Pure functions, no DOM. Exposed as globalThis.SxAgvCore for the
 * autograd visualizer and for unit tests (node).
 *
 * Expression grammar:
 *   expr    := term (('+'|'-') term)*
 *   term    := unary (('*'|'/') unary)*
 *   unary   := '-' unary | power
 *   power   := primary ('**' unary)?
 *   primary := number | ident | ident '(' expr ')' | '(' expr ')'
 * Functions: relu, exp, log, sqrt, square, tanh, abs
 */
(function (root) {
    "use strict";

    function tokenize(src) {
        var toks = [], i = 0;
        while (i < src.length) {
            var ch = src[i];
            if (/\s/.test(ch)) { i++; continue; }
            if (/[0-9]/.test(ch) || (ch === "." && /[0-9]/.test(src[i + 1] || ""))) {
                var m = /^[0-9]*\.?[0-9]+/.exec(src.slice(i));
                toks.push({ t: "num", v: parseFloat(m[0]) });
                i += m[0].length;
                continue;
            }
            if (/[A-Za-z_]/.test(ch)) {
                var id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i))[0];
                toks.push({ t: "id", v: id });
                i += id.length;
                continue;
            }
            if (src.substr(i, 2) === "**") { toks.push({ t: "op", v: "**" }); i += 2; continue; }
            if ("+-*/(),".indexOf(ch) >= 0) { toks.push({ t: "op", v: ch }); i++; continue; }
            throw new Error("Unexpected character '" + ch + "'");
        }
        return toks;
    }

    var FNS = ["relu", "exp", "log", "sqrt", "square", "tanh", "abs"];

    function parse(src, env) {
        var toks = tokenize(src);
        var pos = 0, nodes = [];

        function mk(op, value, inputs, label) {
            var n = { id: nodes.length, op: op, value: value, inputs: inputs || [], label: label || op, grad: 0 };
            nodes.push(n);
            return n;
        }
        function peek() { return toks[pos]; }
        function next() { return toks[pos++]; }
        function expect(v) {
            var t = next();
            if (!t || t.v !== v) throw new Error("Expected '" + v + "'");
        }
        function applyFn(name, a) {
            var v;
            switch (name) {
                case "relu": v = Math.max(0, a.value); break;
                case "exp": v = Math.exp(a.value); break;
                case "log": v = Math.log(a.value); break;
                case "sqrt": v = Math.sqrt(a.value); break;
                case "square": v = a.value * a.value; break;
                case "tanh": v = Math.tanh(a.value); break;
                case "abs": v = Math.abs(a.value); break;
            }
            return mk(name, v, [a], name);
        }
        function primary() {
            var t = peek();
            if (!t) throw new Error("Unexpected end of expression");
            if (t.t === "num") { next(); return mk("const", t.v, [], String(t.v)); }
            if (t.t === "id") {
                next();
                if (FNS.indexOf(t.v) >= 0 && peek() && peek().v === "(") {
                    next();
                    var arg = expr(0);
                    expect(")");
                    return applyFn(t.v, arg);
                }
                if (!(t.v in env)) throw new Error("Unknown variable '" + t.v + "'");
                return mk("var", env[t.v], [], t.v);
            }
            if (t.v === "(") { next(); var e = expr(0); expect(")"); return e; }
            if (t.v === "-") { next(); var u = primary(); return mk("neg", -u.value, [u], "neg"); }
            if (t.v === "+") { next(); return primary(); }
            throw new Error("Unexpected token '" + t.v + "'");
        }
        function prec(op) {
            if (op === "+" || op === "-") return 10;
            if (op === "*" || op === "/") return 20;
            if (op === "**") return 30;
            return 0;
        }
        function applyBinary(op, a, b) {
            var v;
            if (op === "+") v = a.value + b.value;
            else if (op === "-") v = a.value - b.value;
            else if (op === "*") v = a.value * b.value;
            else if (op === "/") v = a.value / b.value;
            else if (op === "**") v = Math.pow(a.value, b.value);
            return mk(op, v, [a, b], op);
        }
        function expr(minPrec) {
            var left = primary();
            for (;;) {
                var t = peek();
                if (!t || t.t !== "op" || prec(t.v) < minPrec || prec(t.v) === 0) break;
                var op = t.v;
                next();
                var right = expr(op === "**" ? prec(op) : prec(op) + 1);
                left = applyBinary(op, left, right);
            }
            return left;
        }

        var out = expr(0);
        if (pos < toks.length) throw new Error("Unexpected token '" + toks[pos].v + "'");
        return { out: out, nodes: nodes };
    }

    function localGrads(n) {
        var i0 = n.inputs[0], i1 = n.inputs[1];
        switch (n.op) {
            case "+": return [1, 1];
            case "-": return [1, -1];
            case "*": return [i1.value, i0.value];
            case "/": return [1 / i1.value, -i0.value / (i1.value * i1.value)];
            case "**": return [i1.value * Math.pow(i0.value, i1.value - 1)];
            case "neg": return [-1];
            case "relu": return [i0.value > 0 ? 1 : 0];
            case "exp": return [Math.exp(i0.value)];
            case "log": return [1 / i0.value];
            case "sqrt": return [1 / (2 * Math.sqrt(i0.value))];
            case "square": return [2 * i0.value];
            case "tanh": return [1 - Math.tanh(i0.value) * Math.tanh(i0.value)];
            case "abs": return [i0.value >= 0 ? 1 : -1];
            default: return [];
        }
    }

    function topo(out) {
        var order = [], seen = {};
        (function visit(n) {
            if (seen[n.id]) return;
            seen[n.id] = 1;
            n.inputs.forEach(visit);
            order.push(n);
        })(out);
        return order;
    }

    function backward(out) {
        var order = topo(out);
        order.forEach(function (n) { n.grad = 0; });
        out.grad = 1;
        for (var i = order.length - 1; i >= 0; i--) {
            var n = order[i];
            if (!n.inputs.length || !n.grad) continue;
            var gs = localGrads(n);
            n.inputs.forEach(function (p, k) { p.grad += n.grad * gs[k]; });
        }
        return order;
    }

    /* Gradient of the output w.r.t. each named variable. */
    function varGrads(ast) {
        var order = backward(ast.out);
        var out = {};
        order.forEach(function (n) {
            if (n.op === "var") out[n.label] = (out[n.label] || 0) + n.grad;
        });
        return out;
    }

    var api = { tokenize: tokenize, parse: parse, topo: topo, backward: backward, varGrads: varGrads, localGrads: localGrads };
    if (typeof module !== "undefined" && module.exports) module.exports = api;
    root.SxAgvCore = api;
})(typeof globalThis !== "undefined" ? globalThis : this);