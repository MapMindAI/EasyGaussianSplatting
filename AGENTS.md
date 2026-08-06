Mandatory rules for AI coding agents contributing to this repo. Direct user instructions win — flag the conflict when they do.

## Repository layout

* `README.md` — user-facing entry point: what the pipeline does, Docker image contents, usage.
* `artifacts/docker/` — `dev.dockerfile` (COLMAP + Insta360 Media SDK + ExifTool + conda env) and its `installers/` scripts. CI (`.github/workflows/docker.yml`) builds and publishes this to `ghcr.io/mapmindai/gaussiansplatting`.
* `arguments/` — vendored parameter-parsing classes (`ModelParams`, `PipelineParams`, `OptimizationParams`) from upstream `graphdeco-inria/gaussian-splatting`, for the training step.
* `data/` — local dev datasets/captures (see §5); not repo-managed.

## 1. Read the docs before starting a task

## 2. Update docs after changing code

## 3. Run the simplify skill before opening a PR

Once the diff is functionally complete, run /simplify. Apply the legitimate findings; note false positives in the PR summary.

## 4. Keep prose and comments concise

* Docs: tight prose. One sentence beats two. Cut hedges, cut narration, cut sentences that restate a heading.
* Code comments: add one only when a reader can't derive the why from the identifiers and structure. Skip comments that restate what the code does, narrate the task, or reference the PR / caller. One short line, not a docstring paragraph.

## 5. Destructive-action discipline

Follow the harness's default git-safety protocol (no force-push, git reset --hard, branch deletion, or --no-verify without explicit user confirmation this session). Repo-specific additions:

* Treat LFS-tracked files (`*.onnx`) as append-only unless the user asks for a rewrite — they're expensive to re-upload and easy to corrupt with `git add` on a host without LFS configured.
* Never commit or overwrite anything under `data/` unless the user explicitly asks — it holds local datasets/captures used for dev testing, not repo-managed content.

## 6. Scope discipline

* Broad reshuffles ship separately. A drive-by within the files you're already touching is fine; a sweep across unrelated packages is its own PR.
* No backwards-compatibility shims or feature flags for hypothetical callers. Change every call site in the same PR.
* Keep the codebase clean as you go. When you spot duplicated logic — same conversion, same math helper, same boilerplate block — hoist it into a shared helper in the same PR if the lift is small (a header, a single function). Refactor when it makes the diff smaller and cleaner; don't refactor for its own sake. The boundary: if the cleanup touches only the files you're already changing plus a new helper file, it's in scope; if it would ripple across unrelated packages, ship separately.

## 7. Prefer the simplest design that solves the concrete problem

Pick the design that solves only the requirement on the table. Don't add abstraction layers, configurable knobs, extra primitives, or extension points "in case we need them later." If a future requirement materializes, the design can grow then — when its actual constraints are known, not guessed.

When proposing or reviewing a design, drop the bullet that starts with "this also lets us…" or "leaves room for…". Two-tier mechanisms, pluggable backends, and speculative interfaces are debt: they widen the API surface, multiply test cases, and lock in assumptions that may turn out wrong. Pairs with §7 (no scope creep within a PR) and §5 (no narrative comments about hypothetical callers).

## 8. Name things fully; avoid abbreviations

Prefer the unabbreviated English word for packages, directories, modules, types, and class / struct / field names. Keep the short form only when it's already standard (src, docs, id, unit suffixes like `_ns` / `_ms`, well-known initialisms like `xml` / `json` / `url`, and domain terms already standard in SLAM literature like `imu`, `ikfom`, `esdf`, `jps`). When in doubt, spell it out — a name that makes sense on first read beats one that saves three characters.
