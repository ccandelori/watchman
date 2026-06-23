## Summary

Describe the change and the runtime contract it touches.

## Work Lane

Select the primary lane and describe any intentional boundary crossing.

- [ ] Runtime spine / contracts / orchestrator
- [ ] DP-HONEY injection, registry, or canary detection
- [ ] CIFT extraction, artifact promotion, or runtime scoring
- [ ] NIMBUS critic, session state, or leakage scoring
- [ ] Proxy / SDK / audit / dashboard / eval
- [ ] Documentation / governance only

## Quality Gates

- [ ] `make quality` passes locally.
- [ ] New behavior has tests.
- [ ] Detector changes emit `DetectorResult`, not `PolicyDecision`.
- [ ] Detector changes cover active, degraded, and unavailable capability states where applicable.
- [ ] Policy changes are isolated to policy modules.
- [ ] Research or introspection code crosses into runtime only through an adapter.
- [ ] Raw `.pt`, `.pkl`, generated trace JSONL, and local corpora are not staged.
- [ ] No raw production secrets cross runtime seams.

## Notes

Call out any deferred work, unsupported capability mode, or intentional contract change.
