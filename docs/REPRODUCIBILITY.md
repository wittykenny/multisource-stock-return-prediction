# Reproduction record

## What was validated

Release packaging uses the existing Windows project environment, not a newly
created environment. CLI help and the two existing unittest strategy-selection
tests run with Python 3.13.3. Syntax, configuration parsing, original-code hashes
and release evidence are checked separately. No fetch, prepare, backtest or
training command was executed during packaging because these write experimental
outputs, access third-party services and can be expensive.

These smoke checks establish import/CLI/test viability only. They do not establish
clean-install compatibility, numerical replication or predictive performance.

## Dependency record

| Dependency | Version observed in project environment |
|---|---|
| Python | 3.13.3 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.5.4 |
| NumPy / pandas | 2.4.4 / 3.0.2 |
| scikit-learn | 1.8.0 |
| AKShare | 1.18.55 |
| PyWavelets / vmdpy | 1.9.0 / 0.2 |
| PyYAML | 6.0.3 |
| requests / urllib3 | 2.33.1 / 2.6.3 |
| Beautiful Soup / lxml | 4.14.3 / 6.0.4 |

These versions were read from installed package metadata; they are not proof of
the environment used to produce historical results. `requirements.txt` records
direct runtime imports rather than an unrelated environment freeze. The portable
PyTorch version pin does not force the observed CUDA build. A compatible driver
and device are needed for GPU execution; CPU fallback is implemented. CUDA
availability is not inferred from a package version. The release smoke check
separately returned `torch.cuda.is_available() == True` and CUDA build 12.8;
no GPU training or numerical reproducibility test was run.

Optional imports for XGBoost, statsmodels, arch and PDF extraction (`pypdf`) exist
in the code but their distributions were absent from the project environment's
metadata. They remain unpinned, explicitly unresolved requirements. Their absence
can change baseline fallback behavior or text extraction. SciPy is a transitive
dependency rather than a direct import. No direct networkx or torch-geometric
dependency was found. Unused direct `tqdm` was removed; direct `urllib3` was added.

## Configurations and execution

The configuration loader merges the chosen YAML into dataclass defaults in
`stock_predictor/config.py`; it does **not** inherit `config/default.yaml`.
Always run from the repository root because data/model/config paths are relative
to the working directory. The preserved core Python/YAML files contain no
machine-specific absolute filesystem paths requiring a runtime change.

`config/reproduction.yaml` copies `config/red_comment_revision.yaml`, changing
only cache/output roots. Original experiments remain intact. The full-model
report in the release was produced under the `ablation` workflow, which disables
external baselines. Run:

```bash
python main.py --config config/reproduction.yaml fetch
python main.py --config config/reproduction.yaml prepare
python main.py --config config/reproduction.yaml ablation
python main.py --config config/reproduction.yaml ablation-significance
```

First follow `data/README.md` to install the local FinBERT model and external
dictionary. Fresh data acquisition requires network access to the implemented
providers. New ablation summaries are written under
`data/output/reproduction/300750/`; full-run details are under
`data/output/reproduction/ablations/full/300750/`.

Random seed: 2026 in the selected experiment. Python, NumPy and PyTorch are seeded;
full deterministic CUDA execution is not guaranteed. Existing configs also
contain multi-seed schedules, but a configured schedule alone is not evidence that
the selected final experiment was evaluated across those seeds.

## Exact-replication gaps

- No distributable frozen corpus, immutable provider snapshot or model revision.
- No complete verified historical dependency lock or hardware record.
- No forecasting checkpoints included.
- Current source files were not tied to a historical commit when the experiment
  ran. The first release commit establishes provenance going forward, not backward.
- Multiple strategy explorations and no established untouched final holdout.

## Release checks

Before staging and again before uploading:

```bash
python scripts/audit_release.py
git diff --check
git add .
python scripts/audit_release.py --staged
git diff --cached --check
git diff --cached --stat
git status --short
```

Inspect flagged files locally; the scanner prints locations/categories, never
matched secret values. Its allowlist and 10 MiB limit complement manual review.
Keyword hits such as tokenizer names or audit pattern definitions are expected
and need contextual review. This is a heuristic scan, not a security guarantee.

## Private GitHub upload

Suggested name: `multisource-stock-return-prediction`. Upload only the reviewed
tracked code, configs, test, audit script, documentation and aggregate results.
On Windows, install GitHub CLI if absent with
`winget install --id GitHub.cli --exact`, reopen the terminal, then run from this
repository:

```powershell
gh auth login
gh auth status
python scripts/audit_release.py
python scripts/audit_release.py --staged
git status
git log --oneline -5
$owner = gh api user --jq .login
gh repo list $owner --limit 1000 --json name --jq '.[].name'
```

If the exact name already exists, stop and choose a different name. If listing
fails, resolve authentication/network access before continuing. Do not reuse or
overwrite an existing repository. Check `git remote -v` is still empty; otherwise
inspect the existing remote rather than replacing it. After these checks:

```powershell
gh repo create "$owner/multisource-stock-return-prediction" --private --source . --remote origin
git push -u origin HEAD
```

`gh repo create` fails on a name collision rather than overwriting it. If a local
initial commit could not be created because Git identity is unset, first set a
user-approved author identity (prefer the account's GitHub-provided no-reply
address), rerun staged checks, then commit with `git commit -m "Initial public
release"`. Do not invent an email address. Never force push. Private visibility
does not make secrets or unlicensed raw data acceptable to upload.
