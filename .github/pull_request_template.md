## Summary

Describe the change and the runtime contract it touches.

## Quality Gates

- [ ] `make quality` passes locally.
- [ ] New behavior has tests.
- [ ] Detector changes emit `DetectorResult`, not `PolicyDecision`.
- [ ] Policy changes are isolated to policy modules.
- [ ] Research or introspection code crosses into runtime only through an adapter.
- [ ] No raw production secrets cross runtime seams.

## Notes

Call out any deferred work, unsupported capability mode, or intentional contract change.
