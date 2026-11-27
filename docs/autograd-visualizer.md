---
description: "Interactive SneppX-ALG autograd visualizer showing forward values and reverse-mode gradient propagation for scalar expressions."
---

# Autograd visualizer

Type a scalar expression, then watch the **forward values** appear and the
**backward pass** propagate gradients through the graph. No installation —
this runs entirely in your browser.

<div id="sx-agv" class="sx-agv">
  <div class="sx-agv-controls">
    <input class="sx-agv-expr" type="text" value="(a * b + c) * relu(a)" aria-label="Expression" />
    <button type="button" class="sx-agv-run">▶ Run</button>
    <button type="button" class="sx-agv-step">Step backward</button>
    <button type="button" class="sx-agv-reset">Reset</button>
  </div>
  <div class="sx-agv-presets">
    <span>Presets:</span>
    <button type="button" class="sx-agv-preset" data-expr="(a * b + c) * relu(a)">(a*b+c)*relu(a)</button>
    <button type="button" class="sx-agv-preset" data-expr="square(a) + exp(b)">square(a)+exp(b)</button>
    <button type="button" class="sx-agv-preset" data-expr="tanh(a * b) / (1 + square(c))">tanh(ab)/(1+c²)</button>
    <button type="button" class="sx-agv-preset" data-expr="sqrt(abs(w)) * log(exp(x) + 1)">sqrt|w|·log(eˣ+1)</button>
  </div>
  <div class="sx-agv-vars"></div>
  <div class="sx-agv-error" role="alert"></div>
  <svg class="sx-agv-svg" role="img" aria-label="Computation graph"></svg>
  <div class="sx-agv-out"></div>
</div>

## How to read it

- Each **box** is a node: the operator on top, its forward value in the middle,
  and the gradient `∂f/∂node` above it once the backward pass reaches it.
- **Edges** point from inputs to their consumer, so the backward pass runs
  right-to-left.
- `relu` has a subgradient of 0 for negative inputs — that's why a gradient
  can vanish at a node.

## How reverse-mode autodiff works

Automatic differentiation applies the chain rule to the graph above. The
forward pass computes and stores each node's value. The backward pass starts at
the output with gradient `1` and walks the graph in reverse. At each node it
multiplies the incoming gradient by the node's local derivative and adds the
result to each input's gradient.

Two details explain most of what you see in the visualizer:

- **Gradients add up.** When a variable feeds several nodes, its gradient is
  the sum of the contributions from every path.
- **One backward pass gives every gradient.** Reverse mode computes `∂f/∂x`
  for all inputs at once, which is why it is the standard choice for training
  networks with millions of parameters and a single scalar loss.

SneppX-ALG's tensor engine uses the same tape-based approach and also supports
higher-order gradients; see [Higher-order AD](higher-order-ad.md).

## Supported syntax

| Element | Examples |
|---------|----------|
| Variables | `a`, `b`, `c`, `x`, `w` |
| Arithmetic | `+ - * / **` and unary `-` |
| Functions | `relu`, `exp`, `log`, `sqrt`, `square`, `tanh`, `abs` |
| Grouping | `( ... )` |

Everything is scalar-valued, so the graph stays small and readable. For the
real tensor engine, see the [Autograd guide](autograd-guide.md) and the
[Cookbook](cookbook.md).
