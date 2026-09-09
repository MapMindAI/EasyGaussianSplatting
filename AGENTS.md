Mandatory rules for AI coding agents contributing to this repo. Direct user instructions win — flag the conflict when they do.

## Repository layout

```text
.
├── .github/workflows/
│   ├── docker.yml                # Builds and publishes the Docker image.
│   └── tests.yml                 # Runs the tests on a slim Python image.
├── artifacts/
│   ├── requirements-test.txt     # Test-only dependencies.
│   └── docker/
│       ├── dev.dockerfile        # COLMAP, Insta360 SDK, ExifTool, and gsplat.
│       ├── jetson.dockerfile     # arm64 image: JetPack 6 CUDA and gsplat.
│       └── installers/           # Docker image installation scripts.
├── data/                         # Local datasets and captures; not repo-managed (see §5).
├── doc/                          # Pipeline, tooling, and server guides.
├── gsplat_server/                # gRPC training server (see doc/gsplat_server.md).
│   ├── server.py                 # gRPC service and job queue.
│   ├── server_test.py            # Job store, uploads, and service preconditions.
│   ├── serve.sh                  # Container: start the service.
│   ├── run_server.sh             # Host: start the service via Docker.
│   ├── client.py                 # Command-line client.
│   ├── parameters.py             # JobParameters text-protobuf helpers.
│   ├── parameters_test.py        # Defaults layering and round trips.
│   ├── proto/                    # gsplat.proto and its generated bindings.
│   └── config/                   # Shipped JobParameters defaults.
├── mapping/
│   ├── benchmark/                # Parameter sweeps and their Markdown report.
│   │   ├── summarize_gsplat_sweep_test.py # Report sections and PLY geometry.
│   │   └── configurations/       # One JobParameters file per swept configuration.
│   ├── features/                 # Learned features (doc/panorama_mapping.md).
│   │   ├── triton_models.py      # SuperPoint/LightGlue/SALAD clients.
│   │   ├── extraction.py         # Features into the database and its sidecar.
│   │   ├── matching.py           # Pair selection and LightGlue matching.
│   │   └── progress.py           # Thread pool with progress logging.
│   ├── mapping_pipeline.py       # The four mapping stages end to end.
│   ├── gps.py                    # Capture GPS as UTM priors and map alignment.
│   ├── gps_test.py               # Track parsing, the fit, and the alignment.
│   ├── mapping_pipeline_test.py  # Gravity levelling of the solved map.
│   ├── panorama_database.py      # Panorama video to a cube-map rig database.
│   ├── segment_people.py         # Person masks for the training images.
│   └── train_gsplat_with_masks.py # simple_trainer entrypoint; reads JobParameters.
├── scripts/
│   ├── run_pipeline.sh           # Host: the whole pipeline in one `docker run`.
│   ├── run_stitch.sh             # Host: stitching on its own.
│   ├── stitch_video.sh           # Container: Insta360 capture to panorama video.
│   ├── docker_common.sh          # Shared helpers for the host-side scripts.
│   └── gsplat_env.sh             # Container: activate the gsplat conda env.
├── third_party/
│   ├── gsplat/                   # gsplat submodule the training stage uses.
│   ├── colmap/                   # COLMAP source the Jetson image builds from.
│   └── EasyTensorRT/             # Triton server and clients for the features.
├── conftest.py                   # Puts the repo root on sys.path for the tests.
├── pytest.ini                    # Points pytest at the tested packages.
└── README.md                     # Pipeline, Docker image, and usage overview.
```

Tests sit beside the module they cover, named `<module>_test.py`, and run in
CI on a slim Python image (see README).

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
