# Release checklist

- [ ] Version bumped according to SemVer
- [ ] Changelog updated from template
- [ ] Unit/integration tests green (`pytest -q`)
- [ ] Perf smoke green (`python tools/perf_smoke.py`)
- [ ] Security/model-loader checks reviewed (`SECURITY_MODEL_LOADING_CHECKLIST.md`)
- [ ] Bundle compatibility fixtures/tests updated
- [ ] Migration notes prepared for contract changes
