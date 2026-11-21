"""
SneppX Edge Model Converter.
Serializes a compiled graph to the lightweight edge binary format.
"""
import struct
import numpy as np
from SneppX_ALG.interface_bindings.graph_compiler import CompiledGraph

def sneppx_to_edge(compiled: CompiledGraph, output_file: str):
    with open(output_file, 'wb') as f:
        # Minimal binary header
        f.write(struct.pack('4sI', b'EDGE', 1))
        
        # Serialize nodes (simplified)
        for node in compiled.nodes:
            op_code = {'input': 1, 'const': 2, 'matmul': 3, 'fused': 4}.get(node.op, 0)
            f.write(struct.pack('I', op_code))
            # ... add more serialization logic for params/shape
    print(f"Model exported to {output_file}")
