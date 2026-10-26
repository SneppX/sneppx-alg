/* Node test for docs/javascripts/autograd-core.js — finite-difference check. */
"use strict";
const core = require("../../docs/javascripts/autograd-core.js");

const cases = [
    "a*b",
    "(a*b+c)*relu(a)",
    "square(a)+exp(b)",
    "tanh(a*b)/(1+square(c))",
    "sqrt(abs(w))*log(exp(x)+1)",
    "a**2 + b**3",
    "a - b/c",
    "relu(a)*relu(b) + c",
    "abs(a)*b - c/2",
];

let pass = 0, fail = 0;
const BASE = { a: 2, b: 3, c: 1, x: 1.5, w: -0.5 };

function valueOf(expr, env) {
    return core.parse(expr, env).out.value;
}

for (const expr of cases) {
    const ast = core.parse(expr, BASE);
    const grads = core.varGrads(ast);
    for (const v of Object.keys(BASE)) {
        // skip vars not present
        if (!(v in grads)) continue;
        const h = 1e-6;
        const e1 = Object.assign({}, BASE); e1[v] += h;
        const e2 = Object.assign({}, BASE); e2[v] -= h;
        const fd = (valueOf(expr, e1) - valueOf(expr, e2)) / (2 * h);
        const an = grads[v];
        const ok = Math.abs(fd - an) < 1e-4;
        if (ok) pass++;
        else { fail++; console.error(`FAIL ${expr} d/d${v}: analytic=${an} fd=${fd}`); }
    }
}

// parse error handling
try {
    core.parse("a +", BASE);
    fail++; console.error("FAIL: expected parse error for 'a +'");
} catch (e) { pass++; }

try {
    core.parse("zzz * 2", BASE);
    fail++; console.error("FAIL: expected error for unknown var");
} catch (e) { pass++; }

console.log(`autograd-core: ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);