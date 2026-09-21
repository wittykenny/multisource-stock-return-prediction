# Research repository release report

## 1. Suggested repository name

`multisource-stock-return-prediction`, initially **PRIVATE**. The GitHub name's
availability has not been checked because GitHub CLI is unavailable locally.

## 2. Release tree

This is the Git-selected release, not the much larger unchanged local workspace.

```text
.
├── .gitattributes
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── config/
│   ├── default.yaml
│   ├── direction_accuracy_check_20260518.yaml
│   ├── direction_accuracy_quick_check_20260518.yaml
│   ├── optimized_55.yaml
│   ├── red_comment_revision.yaml
│   ├── reproduction.yaml
│   └── run_optimized_experiment_20260507.yaml
├── stock_predictor/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── pipeline.py
│   ├── text_processing.py
│   └── utils.py
├── scripts/audit_release.py
├── tests/test_strategy_significance_selection.py
├── data/README.md
├── results/
│   ├── README.md
│   ├── main_results.csv
│   ├── reported_metrics.json
│   ├── ablation.csv
│   ├── ablation_significance.csv
│   ├── experiment_config.json
│   ├── data_summary.json
│   ├── leakage_audit.json
│   └── provenance.json
└── docs/
    ├── REPRODUCIBILITY.md
    └── RELEASE_REPORT.md
```

## 3. Added files

`.gitattributes`, `.gitignore`, `config/reproduction.yaml`,
`scripts/audit_release.py`, `data/README.md`, all files under `results/`,
and the two top-level release documents in `docs/` are new. Local audit backups
and the execution plan under `docs/superpowers/plans/` are ignored.

## 4. Modified files

Only the existing `README.md` and `requirements.txt` were rewritten. Their
original bytes are backed up in the ignored `_release_audit/` directory. All 16
original core Python, configuration and test files were hash-checked unchanged.
No original file, experiment output, model or data file was deleted or moved.
New reproduction paths prevent new runs from overwriting saved experiment output.

## 5. Excluded local material

The root `.gitignore` is an explicit allowlist with nested safety exclusions.
Excluded material includes environments, bytecode, IDE/agent settings, raw data,
text extraction caches, full prediction/feature files, logs, pretrained weights,
external dictionary, every original `data/output/` experiment directory, thesis
documents, proposal, manuscript extracts, renders, figures of unverified run
provenance, temporary scripts and audit backups. No figure was repackaged because
its selected-experiment provenance was not established; numerical tables provide
the evidence instead. Future intentional additions require allowlist review.

`.gitattributes` preserves result CSV/JSON bytes so the published evidence hashes
remain valid even with automatic line-ending conversion enabled for Git.

## 6. Sensitive-information review

- Candidate release files are scanned for credential formats/assignments,
  private keys, bearer/database credentials, personal paths, email addresses,
  possible phone numbers, unexpected payload types and files over 10 MiB.
- Requested keywords were reviewed in core source. Matches concerned text tokens,
  tokenizer APIs and leakage-check identifiers, not API credentials.
- A broader workspace text scan found personal paths and email/phone-like strings
  in excluded local material. No key-format or literal-credential-assignment match
  was found in that scan. Matches are recorded locally without copying their
  values into release documentation.
- That broader scan excludes virtual environments, Git internals, bytecode,
  audit backups, binaries and text files over 25 MiB; two temporary LibreOffice
  directories were unreadable. It is not a claim that the entire workspace is
  free of private information. All such directories are excluded from release.
- There are 47 local files larger than 10 MiB outside the environment, including
  two pretrained weight files of 437,961,700 and 409,103,316 bytes. None are selected.
- Git was initially configured with a non-no-reply address. The user supplied a
  GitHub settings screenshot; the repository-local author identity now uses the
  shown GitHub no-reply address and the supplied author name. The screenshot and
  private address are not repository files. Global Git identity was not changed.
- No `.env.example` was introduced because this pipeline has no API-key setup to
  configure and no embedded credential required replacement.

Heuristic scans supplement manual review; they are not a comprehensive security
or privacy certification.

## 7. Data and licensing

Raw market data, news, announcements, policy text, sentiment corpora and the local
dictionary are withheld because redistribution rights were not established.
Model download instructions link to the upstream distribution without bundling
weights. No license is assigned to this project pending ownership/permission
confirmation. These decisions apply even to a private upload.

## 8. Dependencies and environment

Direct imports drive `requirements.txt`; no environment-wide freeze was used.
Observed direct dependencies are pinned, while four optional backend/PDF imports
remain unpinned because their versions could not be established. Python 3.13.3,
PyTorch 2.11.0+cu128 and Transformers 5.5.4 are observed in the existing project
environment. CLI help and two existing tests passed there. Fresh installation,
full acquisition and retraining have not been validated. See the separate
reproducibility record for exact boundaries.

## 9. Shortest supported experiment recipe

After installation and external-input preparation in `data/README.md`:

```bash
python main.py --config config/reproduction.yaml fetch
python main.py --config config/reproduction.yaml prepare
python main.py --config config/reproduction.yaml ablation
python main.py --config config/reproduction.yaml ablation-significance
```

This is a workflow recipe, not a guarantee of identical historical numbers.

## 10. Git state

The starting directory had no Git repository; there was no existing history or
remote to preserve. A local `main` branch was initialized for this release.
The requested initial commit message is `Initial public release`; it describes
the release contents and does not change the intended PRIVATE remote visibility.
The staged file list and diff are reviewed and the complete index is audited;
the release contains 34 files, approximately 0.57 MB, with no blocking scan flags.
No force push, remote replacement or history deletion is part of the workflow.

## 11. GitHub upload state

Not uploaded. `gh` is not installed or available on PATH, and `gh auth status`
could not execute. No remote repository was created and no push was attempted.
The name and complete upload scope have been communicated. Exact CLI installation,
login, collision check, private creation and push commands are provided in
`docs/REPRODUCIBILITY.md`. The user can complete these after installing the CLI.

## 12. Remaining author decisions

- Confirm code ownership and choose a license only after confirming reuse rights.
- Confirm that the selected saved experiment is the intended headline experiment;
  this release identifies it explicitly instead of merging results from different
  strategy/configuration directories.
- Supply an exact historical corpus/model revision/environment record if exact
  numerical replication is required; do not imply these already exist.
- Complete GitHub CLI installation/login and verify repository-name availability.
- Decide on public visibility only after reviewing the private repository.

## Research Postgraduate Application Review

A 60-second review now exposes the question, CATL sample and time span, implemented
method, measured result and its main caveat in the first screen. The project is
presented as an undergraduate integration/evaluation effort, not a published
method or a new language model. The first screen links directly to reproduction
and limitations. The method/data sections expose the experiment design; all
headline numbers have checked-in numerical evidence.

The most material research limitations are retained: all-non-up direction
predictions, negative R-squared/IC, a composite optimized trading strategy,
asymmetric strategy settings in ablations, incomplete historical text coverage,
and no proven untouched holdout. No improvement, publication, end-to-end FinBERT
training or complete clean-environment reproduction is invented. These are
substantive research boundaries that documentation alone cannot fix.
