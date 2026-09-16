# Decisions

Record research and architecture decisions with their context, rationale, and consequences.

## 2026-09-16 — Keep large and generated ML artifacts outside Git

- **Status:** Accepted
- **Context:** The working tree contains raw and processed datasets, generated image crops, model weights, checkpoints, experiment outputs, local environments, and files above GitHub's normal file-size limits.
- **Decision:** Track reproducible source, configuration, documentation, notebooks, lockfiles, and small textual result summaries. Keep datasets, generated media, model artifacts, environments, runtime credentials, and bulky experiment outputs in external storage.
- **Reason:** This reduces repository size and prevents accidental publication of sensitive or non-reproducible local artifacts.
- **Alternatives considered:** Commit all local artifacts directly to Git; configure Git LFS for large binaries; or keep large/generated artifacts outside Git. Direct Git storage was rejected because of size and sensitivity, and Git LFS was deferred by explicit project decision.
- **Consequences:** External artifact locations and retrieval instructions should be documented before collaborators need to reproduce an experiment. Git LFS is not enabled unless separately approved.

## Decision Template

### YYYY-MM-DD — Decision title

- **Status:** Proposed | Accepted | Superseded
- **Context:** Needs review
- **Decision:** Needs review
- **Reason:** Needs review
- **Alternatives considered:** Needs review
- **Consequences:** Needs review
