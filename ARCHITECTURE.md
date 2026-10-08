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
- `src/pipeline.py`: End-to-end processing pipeline orchestrator.
- `api/main.py`: Production FastAPI REST microservice.
- `app/`: Multi-page command-center Streamlit user interface.

## 3. Key Design Decisions

1. **Parquet-First Caching**:
   Reading processed tweets from columnar Apache Parquet is ~10-20x faster than SQLite/CSV, keeping Streamlit filter re-computations snappy under interactive load.

2. **Zero External API / Air-Gapped Offline Geocoding**:
   Geocoding uses `geonamescache` without external API calls or internet dependencies. Ambiguous names (e.g. "Athens" or "Hamilton") are disambiguated using the disaster event's country/region priors.

3. **Additive Waterfall Scoring**:
   Urgency scores are not opaque black boxes: each score is decomposed into explicit positive and negative contributions (category base, critical keywords, shout ratio, donation damping) that sum exactly to the final score.

4. **Calibrated Confidences & ECE**:
   Decision-support tools must communicate calibrated probabilities. Linear models use Platt scaling or Sigmoid calibration (`CalibratedClassifierCV`) to output accurate posterior probabilities.
