# CrisisLens Architecture Document

## 1. System Overview

**CrisisLens** is an end-to-end decision-support platform designed to ingest high-velocity disaster tweet streams, filter ambient noise, extract structured emergency intelligence, and present real-time triage queues for humanitarian operators.

```
+-----------------------------------------------------------------------------------+
|                                  INCOMING DATA                                    |
|                      HumAID Benchmark (13 Events, ~43.4k Tweets)                   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                               ETL & PREPROCESSING                                 |
|  - HTML unescaping & normalization    - Tokenization & cleaning                   |
|  - Hashtag / mention splitting        - Snowflake timestamp recovery              |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         INTELLIGENCE EXTRACTION ENGINES                           |
|  +--------------------+   +-----------------------+   +------------------------+  |
|  |     Classifier     |   |    NER & Geocoding    |   |     Need Taxonomy      |  |
|  | TF-IDF + Calibrated|   |  spaCy + GeoNamesCache|   | 10 Multi-label Need    |  |
|  | Linear Support Vec |   |   Prior Disambiguation|   | Lexicons (Regex Match) |  |
|  +--------------------+   +-----------------------+   +------------------------+  |
|                                         |                                         |
|                                         v                                         |
|  +-----------------------------------------------------------------------------+  |
|  |                   PRIORITY TRIAGE & URGENCY ENGINE                          |  |
|  |  - Urgency Score (0 - 10)          - Category Base Weight                   |  |
|  |  - Critical/High Keyword Boost     - Punctuation & Retweet Attenuation      |  |
|  |  - Donation/Sympathy Damping       - Additive Waterfall Signal Breakdown    |  |
|  +-----------------------------------------------------------------------------+  |
|                                         |                                         |
|                                         v                                         |
|  +--------------------+   +-----------------------+   +------------------------+  |
|  |  Surge Detection   |   |  Incident Clustering  |   | 2D Semantic Projection |  |
|  | Robust Median/MAD  |   |   Cosine Sim (TF-IDF) |   | TruncatedSVD +         |  |
|  | Volume & Crit Spik |   |   Ranked Incidents    |   | MiniBatchKMeans        |  |
|  +--------------------+   +-----------------------+   +------------------------+  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                             STORAGE & CACHING LAYER                               |
|  - processed_tweets.parquet (fast columnar read, sub-second Streamlit reload)     |
|  - disaster_tweets.db (SQLite: processed_tweets + operator feedback audit table)  |
|  - reports/ (ROC/PR curves, ECE calibration JSON, per_event_metrics.csv)           |
+-----------------------------------------------------------------------------------+
                                         |
                   +---------------------+---------------------+
                   v                                           v
+------------------------------------+   +------------------------------------------+
|       FASTAPI REST SERVICE         |   |          STREAMLIT COMMAND CENTER        |
|  - GET  /health                    |   |  1. Situation Room (KPIs & Surges)       |
|  - GET  /version                   |   |  2. Replay Mode (Time Machine Playback)  |
|  - POST /analyze                   |   |  3. Priority Triage Queue (Waterfall)    |
|  - POST /analyze/batch             |   |  4. Geospatial Intel (Offline Clusters)  |
|                                    |   |  5. Model Observatory (ROC, PR, ECE)     |
|                                    |   |  6. Human Review Queue (Active Feedback) |
|                                    |   |  7. About & Data (Dataset Provenance)    |
+------------------------------------+   +------------------------------------------+
```

## 2. Component Directory Structure

- `src/config.py`: Central configuration, paths, event geo priors, NER stopwords, urgency weights, need taxonomy.
- `src/data_loader.py`: HumAID dataset discovery, snowflake timestamp derivation, schema validation.
- `src/preprocessing.py`: Text cleaning, entity preservation, feature extraction (caps, exclamations, length).
- `src/features.py`: TF-IDF feature extraction pipeline.
- `src/classifier.py`: Model candidate training (NB, LR, LinearSVM, RF), probability calibration, cross-validation.
- `src/urgency.py`: Urgency scoring, severity categorization, and additive waterfall breakdown.
- `src/ner.py`: spaCy entity recognition for geopolitical locations.
- `src/geocoding.py`: Offline GeoNamesCache resolution, event prior disambiguation, NER junk entity cleaning.
- `src/needs.py`: Multi-label humanitarian need taxonomy extraction and raw label evaluation.
- `src/anomaly.py`: Statistical surge detection using rolling median and Median Absolute Deviation (MAD).
- `src/clustering.py`: TF-IDF cosine similarity clustering into ranked incidents and 2D TruncatedSVD semantic projections.
- `src/explainability.py`: Linear feature token attribution, confidence margin calculation, safe HTML highlighting.
- `src/nlquery.py`: Deterministic natural language query parser for filter synthesis.
- `src/briefing.py`: Executive Markdown and self-contained HTML situation report generator.
- `src/database.py`: SQLite schema management, batch insertion, parameterized human feedback storage.
- `src/pipeline.py`: End-to-end processing pipeline orchestrator; stamps every row with the model version that scored it.
- `src/model_registry.py`: Model content fingerprints, metadata sidecar (`best_classifier.meta.json`), and the health check that flags a missing, implausibly small, or out-of-sync model.
- `src/entities.py`: Extraction of reported casualty / displacement figures ("death toll 413", "5 people trapped").
- `src/utils.py`: `as_list` / `normalize_list_columns`, which make list columns behave the same whether loaded from Parquet (numpy arrays), SQLite or CSV (JSON text).
- `retrain_with_feedback.py`: Applies operator corrections by split and retrains behind a regression gate.
- `api/main.py`: Production FastAPI REST microservice (analysis, feedback, surges, incidents, health/readiness).
- `app/`: Multi-page command-center Streamlit user interface. `app/ui/theme.py` is the only place colours are defined; `app/ui/cache.py` caches the expensive analytics.

## 3. Key Design Decisions

1. **Parquet-First Caching**:
   Reading processed tweets from columnar Apache Parquet is ~10-20x faster than SQLite/CSV, keeping Streamlit filter re-computations snappy under interactive load.

2. **Zero External API / Air-Gapped Offline Geocoding**:
   Geocoding uses `geonamescache` without external API calls or internet dependencies. Ambiguous names (e.g. "Athens" or "Hamilton") are disambiguated using the disaster event's country/region priors.

3. **Additive Waterfall Scoring**:
   Urgency scores are not opaque black boxes: each score is decomposed into explicit positive and negative contributions (category base, critical keywords, shout ratio, donation damping) that sum exactly to the final score.

4. **Honest Confidences & ECE**:
   Confidence is the model's predicted probability, and the Model Observatory reports its expected calibration error. Class-balanced training deliberately trades some calibration for recall on rare classes, so the dashboard shows the measured ECE rather than claiming calibration.

5. **Model Integrity**:
   A model is identified by a content fingerprint. The processed dataset is stamped with the fingerprint of the model that produced it, and the dashboard and API compare it with the model currently loaded. A model with an implausibly small vocabulary (the signature of a test fixture overwriting production) is reported as `degraded`, and the API answers `503` rather than guessing when no model is loaded. Model selection prefers the simpler model whenever candidates are within 0.005 dev macro-F1, because differences that small are noise on the dev split.

6. **Test Isolation**:
   `tests/conftest.py` redirects the data, models and reports directories to a temp location (`CRISISLENS_*_DIR`) before any project module is imported, so the suite can never overwrite production artifacts. `tests/test_isolation.py` fails if that regresses.

7. **Feedback Loop Hygiene**:
   Operator corrections are applied by split: train-split corrections relabel training data, live (API) corrections add new rows, and dev/test corrections are never trained on so the benchmark stays clean. Tweets already reviewed are hidden from the review queue, and the latest decision per tweet wins.

8. **Geocoding Precedence**:
   Countries and US states are resolved before cities, curated places fill gaps in the GeoNames cache, macro-regions and storm names are never plotted, and a city that only matches outside the event's expected countries keeps its name but is not drawn on the map.
