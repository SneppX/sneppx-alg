"""Explicit opt-in operations routed through the native C tensor engine.

This mixin intentionally does not alter Python operators or autograd behavior.
All methods return detached public Tensor instances through the existing bridge.
"""


class NativeTensorOps:
    """Native CPU operations available explicitly on the public Tensor API."""

    def native_sub(self, other):
        """Subtract tensors with the native C engine (explicit opt-in)."""
        return self._native_binary(other, "sub")

    def native_mul(self, other):
        """Multiply tensors element-wise with the native C engine."""
        return self._native_binary(other, "mul")

    def native_div(self, other):
        """Divide tensors element-wise with the native C engine."""
        return self._native_binary(other, "div")

    def native_pow(self, other):
        """Raise tensor elements to tensor powers with the native C engine."""
        return self._native_binary(other, "pow_")

    def _native_unary(self, operation, *args):
        native = self.to_native()
        result = getattr(native, operation)(*args)
        return type(self).from_native(result)

    def native_relu(self):
        """Apply the native C ReLU operation."""
        return self._native_unary("relu")

    def native_sigmoid(self):
        """Apply the native C sigmoid operation."""
        return self._native_unary("sigmoid")

    def native_tanh(self):
        """Apply the native C tanh operation."""
        return self._native_unary("tanh_act")

    def native_softmax(self, dim=0):
        """Apply native C softmax along an explicit dimension."""
        return self._native_unary("softmax", dim)

    def native_sum(self, dim=0):
        """Reduce a tensor with the native C sum operation."""
        return self._native_unary("sum", dim)

    def native_mean(self, dim=0):
        """Reduce a tensor with the native C mean operation."""
        return self._native_unary("mean", dim)
