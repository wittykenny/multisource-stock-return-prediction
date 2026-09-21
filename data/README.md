# Data and pretrained-model preparation

No raw data, text corpus, lexicon, pretrained weights or prepared feature cache
is distributed. Public access to an endpoint does not establish permission to
redistribute its contents. Review provider terms before acquiring and using data.

## Inputs and local layout

| Input | Acquisition implemented in `stock_predictor/data.py` | Local artifacts |
|---|---|---|
| CATL price history | AKShare with source fallbacks including Tencent | `stock.csv` |
| Financial statements | AKShare financial interfaces; disclosure-date/lag alignment | `financial.csv` |
| Industry series | Industry index/valuation interfaces and proxy construction | `industry.csv` |
| Macroeconomic releases | AKShare interfaces and release-date cleaning | `macro.csv`, `macro_cleaned.csv` |
| Company/industry news | Eastmoney, Sina and Google News RSS searches; GDELT is configurable | `news.csv` |
| Announcements | CNInfo full-text search and extraction | `notices.csv`, `notices_cleaned.csv` |
| Policy | Official government/ministry pages and full-text extraction | `policy.csv` |
| Sentiment | Investor-source retrieval and historical text proxies | `sentiment.csv`, `text_sentiment_events.csv`, `text_sentiment_daily.csv` |

`fetch` builds these caches under the configured cache root and symbol directory.
The reproduction configuration uses `data/cache_reproduction/300750/` and
`data/output/reproduction/`; the older cache directories remain local. Acquisition
can also create text-extraction caches and call the text sentiment builder, so
prepare the language model before running `fetch`, not only before training.

The historical saved run reports 1,454 prepared rows, 1,999 text sentiment events
and 1,029 sentiment daily rows. These are different units, not one sample count.
See `results/data_summary.json` for recorded coverage limitations. Fresh retrieval
can differ in row counts and contents and may fail if a provider changes an API.

## Chinese FinBERT

The configured model is `yiyanghkust/finbert-tone-chinese`, loaded only from
`data/models/yiyanghkust_finbert-tone-chinese/`. The earlier README's reference to
English ProsusAI FinBERT did not match the current experiment configuration.
The upstream [model card](https://huggingface.co/yiyanghkust/finbert-tone-chinese)
identifies a Chinese financial sentiment model and currently labels its license
Apache-2.0. Model weights are still excluded from this repository.

Download the model files through its upstream distribution. The following uses
`huggingface_hub`, a dependency of Transformers, after installing requirements:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='yiyanghkust/finbert-tone-chinese', local_dir='data/models/yiyanghkust_finbert-tone-chinese')"
```

This fetches the current upstream revision. The revision of the historical local
weights was not recorded, so this is not an exact reconstruction. For a new study,
record the downloaded commit and file hashes and pass an explicit `revision` to
`snapshot_download`. The existing local model directory contains `config.json`,
`model.safetensors`, `tokenizer_config.json`, `special_tokens_map.json` and
`vocab.txt`. The saved model uses 768-dimensional encodings and neutral/positive/
negative labels. Do not substitute another language model without treating the
result as a new experiment.

## Lexicon

The configured external resource is
`data/resources/loughran_mcdonald_master_dictionary.csv`. Obtain it from the
Loughran–McDonald dictionary authors under their applicable terms; redistribution
permission and the exact local edition were not established during this release.
The loader expects `Word`, `Positive` and `Negative` columns. The code has built-in
fallback word lists if the dictionary is unavailable; a run using that fallback
must be recorded as a changed input condition, not claimed as an exact replica.

## Run acquisition and preparation

```bash
python main.py --config config/reproduction.yaml fetch
python main.py --config config/reproduction.yaml prepare
```

Examine `fetch_summary.json`, `prepare_summary.json` and provider warnings before
running experiments. A successfully completed request does not guarantee complete
historical news coverage. No confidential API credential is required by the
current configured pipeline. Do not commit credentials if adding a new provider.
