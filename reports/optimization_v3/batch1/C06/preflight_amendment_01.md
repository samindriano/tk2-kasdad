# C06 preflight amendment 01

No candidate fit was started under the initial `freeze.json`. The first frozen-pilot invocation failed closed before training because the in-memory recipe represented evaluation segments as tuples while the persisted JSON represented them as lists.

The recipe builder now emits JSON-native lists, and a round-trip regression test covers the representation. The original freeze remains preserved for provenance. `freeze_amendment_01.json` is the active source-bound freeze for the unchanged C06 recipe; the model, features, control route, thresholds, temporal views, and support gate are unchanged.
