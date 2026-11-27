import sys
from SneppX_ALG.interface_bindings import graph_optimizer as go
print('trace ->', repr(go.trace(lambda a,b:a*b,1,2)), flush=True)
from SneppX_ALG.interface_bindings.s4_network import TransportSecurity
ts=TransportSecurity()
print('verify bogus ->', ts.verify_certificate('', 'wrong.example'), flush=True)
from SneppX_ALG.interface_bindings.runtime_monitor import AdvancedMonitor
print('anomaly {} ->', AdvancedMonitor.detect_anomaly({}), flush=True)
print('anomaly ext ->', AdvancedMonitor.detect_anomaly({'cpu':999.0}), flush=True)
