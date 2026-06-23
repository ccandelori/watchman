## Residual Review Findings

Source review run: CIFT runtime hardening LFG review on branch `codex/trace-collection-harness`.

- P2 `introspection/src/aegis_introspection/trace_record_adapter.py:15` - Selected-choice span extraction is coupled to prompt prose. The trace adapter still locates selected-choice geometry by searching for a semantic-indirection sentence and then parsing the next semicolon-delimited clause. Fix by having trace generation emit structured selected-choice span metadata, or a profile-owned structured marker, so the adapter copies and validates metadata instead of reparsing rendered prompt prose.
