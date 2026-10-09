# CrisisLens 🚨

**Crisis intelligence from social signals**

CrisisLens is a decision-support platform designed to transform noisy, unstructured social media signals into structured, prioritized emergency dispatches during humanitarian disasters.

> **Operational Decision-Support Notice**:  
> CrisisLens is a decision-support and research system. It is not an operational emergency dispatch tool; outputs must be reviewed by a human responder.

---

## 🏛️ System Architecture

```
                               ┌────────────────────────────────┐
                               │   HumAID Social Signals Stream │
                               │     (13 Historical Disasters)  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │      Data Ingestion & Snowflake │
                               │      UTC Timestamp Derivation   │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │     Preprocessing & Signals    │
                               │   (Text Normalization/Regex)   │
                               └───────────────┬────────────────┘
                                               │
                     ┌─────────────────────────┼─────────────────────────┐
                     │                         │                         │
                     ▼                         ▼                         ▼
           ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
           │ Category Classifier│    │  NER & Geocoding  │     │ Explainable Rules │
           │   (Calibrated LR) │     │ (spaCy + GeoNames)│     │  (0-10 Urgency)   │
           └─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘
                     │                         │                         │
                     └─────────────────────────┼─────────────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │  Surge Anomaly & Clustering    │
                               │  (Robust Z-Scores, Incidents)  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │ Storage, Analytics & Fast Parquet│
                               │  (SQLite DB + PyArrow Parquet) │
                               └───────────────┬────────────────┘
                                               │
                     ┌─────────────────────────┴─────────────────────────┐
                     │                                                   │
                     ▼                                                   ▼
       ┌───────────────────────────┐                       ┌───────────────────────────┐
       │   FastAPI REST Service    │                       │  Streamlit Command Center │
       │    (POST /analyze, batch) │                       │  (Situation Room, Replay) │
       └───────────────────────────┘                       └───────────────────────────┘
```

---

## 🚀 Key Capabilities

1. **Category Triage**: Classifies unstructured social dispatches into 5 operational categories (`Emergency`, `Help Request`, `Damage Report`, `Information/Update`, `General`).
2. **Offline Geospatial Resolution**: Resolves extracted entity mentions (spaCy `en_core_web_sm`) to latitude/longitude coordinates using `geonamescache`, a small curated list of towns and islands, and disaster-specific geographic priors. Countries and US states take precedence over same-named towns (so "Mexico" is the country, not a town in the Philippines), storm names and macro-regions are never plotted, and low-confidence matches keep their name but are not drawn on the map.
3. **Multi-Label Humanitarian Needs**: Identifies critical needs (`Medical`, `Water`, `Food`, `Shelter`, `Rescue/Evacuation`, `Power/Utilities`, `Transport/Roads`, `Communication`, `Missing persons`, `Donations/Volunteering`) with phrase lexicons, evaluated against HumAID labels as proxies in `reports/needs_evaluation.json`.
4. **Explainable Priority Urgency**: Rule-based scoring engine producing bounded priority scores (0 to 10), severity levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), and waterfall attributions. Features damping rules for donation campaigns, and a casualty-report signal that lifts death-toll news to MEDIUM without double-counting or making it CRITICAL.
5. **Reported Figures**: Extracts claimed counts such as "death toll 413" or "5 people trapped" as structured data. These are claims made in tweets, not verified counts.
6. **Statistical Surge Detection**: Detects hourly volume spikes and critical ratio anomalies using robust median and Median Absolute Deviation (MAD) z-scores; each alert carries the top terms and most-mentioned place of its window.
7. **Incident Aggregation**: Clusters near-duplicate tweets into disaster incidents using chunked TF-IDF cosine similarity, seeding from the most urgent tweets.
8. **Interactive Command Center**: Streamlit light-theme command center featuring a Situation Room with activity timeline and surge markers, Chronological Replay with play/pause, Triage Queue, Geospatial Map with hotspot drill-down, Model Observatory, and a Human Review Queue.
9. **Model Integrity Checks**: Every model has a content fingerprint and metadata sidecar. The dataset is stamped with the model that scored it, and the dashboard and API warn when the loaded model is missing, implausibly small, or out of sync with the data.
10. **Production REST API**: FastAPI microservice (`/health`, `/ready`, `/version`, `/analyze`, `/analyze/batch`, `/feedback`, `/surges`, `/incidents`) with Pydantic v2 schemas, optional API-key auth, and request IDs.

---

## ⚙️ Quick Start & Installation

### 1. Prerequisites
Python 3.10+ on Windows, Linux, or macOS.

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ritusri0308/Disaster-Tweet-Intelligence.git
cd DisasterTweetIntelligence

# Create and activate virtual environment
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English NLP model
python -m spacy download en_core_web_sm
```

---

## 🛠️ CLI Operations

| Command | Description |
| :--- | :--- |
| `python run_pipeline.py` | Runs end-to-end data ingestion, classification, NER, geocoding, needs, and urgency pipeline; generates Parquet, CSV, and SQLite, stamped with the model version. Add `--demo` to use synthetic sample data when the HumAID files are absent. |
| `python train.py` | Trains classifier suite (Naive Bayes, Logistic Regression, Linear SVM, Random Forest) and saves the best model by dev Macro-F1 (models within 0.005 are tied and the simpler one wins). Also writes `models/best_classifier.meta.json`. |
| `python train.py --experiments` | Executes ablation and Leave-One-Event-Out experiments, including word vs word+character n-grams. |
| `python evaluate.py` | Generates ROC/PR curves, reliability calibration data, and per-event test metrics into `reports/`. |
| `python retrain_with_feedback.py` | Applies operator corrections (train-split relabels and live API feedback) and retrains with a regression safeguard. Dev/test corrections are never trained on. |
| `streamlit run app.py` | Launches the Streamlit Command Center GUI at `http://localhost:8501`. |
| `uvicorn api.main:app --port 8000` | Launches the FastAPI REST service at `http://localhost:8000`. |
| `pytest --cov=src --cov=api` | Executes the offline test suite. Tests run against throwaway temp directories and never touch `models/` or `reports/`. |

After retraining a model, re-run `python run_pipeline.py` and `python evaluate.py` so the dashboard data and evaluation reports describe the model that is loaded.

---

## 🔧 Configuration

All settings are optional environment variables (see `.env.example`).

| Variable | Purpose |
| :--- | :--- |
| `CRISISLENS_DATA_DIR`, `CRISISLENS_MODELS_DIR`, `CRISISLENS_REPORTS_DIR` | Redirect where data, models and reports live (defaults: `data/`, `models/`, `reports/`). |
| `CRISISLENS_API_KEY` | When set, `/analyze`, `/analyze/batch`, `/feedback`, `/surges` and `/incidents` require an `X-API-Key` header. |
| `CRISISLENS_CORS_ORIGINS` | Comma-separated browser origins allowed to call the API (CORS is off when unset). |
| `CRISISLENS_ALLOW_SYNTHETIC` | Same as `--demo`: permit synthetic data when no HumAID TSVs exist. Without it the pipeline stops with a clear error instead of silently training on fake data. |
| `CRISISLENS_MIN_VOCAB` | Models with fewer vocabulary terms are reported as `degraded` (default 1000). |

---

## 📡 REST API Reference

### Health & Metadata
- **`GET /health`**: Returns `healthy`, `degraded` (model implausibly small) or `unavailable` (no model), plus the model version and vocabulary size.
- **`GET /ready`**: Readiness probe; `200` only when a healthy model is loaded, otherwise `503`.
- **`GET /version`**: Returns service version, dataset provenance, model version, and disclaimer.

### Operations
- **`POST /analyze`** and **`POST /analyze/batch`** (max 500): classify, geocode, extract needs and figures, and score urgency. Batches run in one classifier pass. Both return `503` when no model is loaded rather than guessing a category.
- **`POST /feedback`**: record an operator correction (category and urgency) with the tweet text so it can be used by `retrain_with_feedback.py`.
- **`GET /surges?event=&limit=`** and **`GET /incidents?event=&limit=`**: surge alerts and ranked incident clusters from the processed dataset.
- Every response carries `X-Request-ID` and `X-Process-Time-ms` headers.

### Analyze Single Dispatch
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "SOS! 5 people trapped inside collapsed building near Houston Central. Need medical help immediately!",
       "event": "hurricane_harvey_2017"
     }'
```

**Response**:
```json
{
  "clean_text": "sos 5 people trapped inside collapsed building near houston central need medical help immediately",
  "category": "Emergency",
  "confidence": 0.952,
  "urgency_score": 9.5,
  "urgency_level": "CRITICAL",
  "locations": ["Houston Central"],
  "geocoded_locations": [
    {"name": "Houston", "lat": 29.7633, "lon": -95.3633, "country_code": "US", "population": 2314157,
     "confidence": 0.95, "is_ambiguous": true, "entity_type": "city"}
  ],
  "needs": ["Medical", "Rescue/Evacuation"],
  "quantities": [{"kind": "trapped", "count": 5}],
  "model_version": "706a502aa89f"
}
```

---

## 📊 Dataset Provenance & Humanitarian License

### HumAID Benchmark
- **Scale**: 13 historical natural disaster events (~43,400 tweets, 2016–2019).
- **Mode**: **HISTORICAL REPLAY**. All records are historical benchmark events.
- **Licensing**: Licensed exclusively for non-commercial humanitarian computing research. Raw tweet data and internal database files are strictly git-ignored and not redistributed.
- **Official Citation**:
> Firoj Alam, Hassan Sajjad, Muhammad Imran, and Ferda Ofli.  
> *HumAID: Human-Annotated Disaster Incident Data from Twitter with Multimodal Features*.  
> Proceedings of the International AAAI Conference on Web and Social Media (ICWSM), 2021.

### Category Taxonomy Mapping
The 10 raw HumAID classes are normalized into 5 actionable emergency categories:

| Raw HumAID Label | Project Category | Definition |
| :--- | :--- | :--- |
| `injured_or_dead_people` | **Emergency** | Casualty reports (injured or deceased individuals) |
| `requests_or_urgent_needs` | **Help Request** | Urgent calls for resources, food, water, or rescue |
| `infrastructure_and_utility_damage` | **Damage Report** | Damaged roads, collapsed bridges, utility grid failures |
| `caution_and_advice` | **Information/Update** | Official advisories, evacuation instructions |
| `displaced_people_and_evacuations` | **Information/Update** | Shelter status, temporary camps |
| `missing_or_found_people` | **Information/Update** | Reports of missing individuals |
| `other_relevant_information` | **Information/Update** | Disaster news and meteorology reports |
| `rescue_volunteering_or_donation_effort` | **Information/Update** | Volunteer mobilization and donation efforts |
| `not_humanitarian` | **General** | Irrelevant or non-crisis content |
| `sympathy_and_support` | **General** | Emotional solidarity, condolences |
| `dont_know_cant_judge` | **General** | Ambiguous or unverifiable content |

---

## ⚠️ System Limitations & Future Roadmap

### Known Limitations
1. **Decision-Support Scope**: Outputs must be verified by trained human coordinators before dispatching emergency teams.
2. **Geocoding Precision**: Locations represent city and state centroids identified in text mentions, not real-time GPS telemetry.
3. **Keyword Heuristics**: Urgency scores are derived through deterministic rules and keyword attribution.

### Roadmap
- Integration of multilingual transformer models (e.g. XLM-RoBERTa) for non-English crisis events.
- Real-time streaming ingestion connectors for official emergency feeds.
- Live geocoding verification using satellite imagery and GIS layers.
