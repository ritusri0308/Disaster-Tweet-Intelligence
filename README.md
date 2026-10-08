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
2. **Offline Geospatial Resolution**: Resolves extracted entity mentions (spaCy `en_core_web_sm`) to latitude/longitude coordinates using `geonamescache` and disaster-specific geographic priors. Filters false-positive noise words (e.g. "Truly").
3. **Multi-Label Humanitarian Needs**: Identifies critical needs (`Medical`, `Water`, `Food`, `Shelter`, `Rescue/Evacuation`, `Power/Utilities`, `Transport/Roads`, `Communication`, `Missing persons`, `Donations/Volunteering`).
4. **Explainable Priority Urgency**: Rule-based scoring engine producing bounded priority scores (0 to 10), severity levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), and waterfall attributions. Features damping rules for donation campaigns.
5. **Statistical Surge Detection**: Detects hourly volume spikes and critical ratio anomalies using robust median and Median Absolute Deviation (MAD) z-scores.
6. **Incident Aggregation**: Clusters near-duplicate tweets into disaster incidents using chunked TF-IDF cosine similarity.
7. **Interactive Command Center**: Streamlit dark-mode command center featuring Situation Room, Chronological Replay, Triage Queue, Geospatial Map, and Model Diagnostics.
8. **Production REST API**: FastAPI microservice (`/health`, `/version`, `/analyze`, `/analyze/batch`) with Pydantic v2 schemas.

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
| `python run_pipeline.py` | Runs end-to-end data ingestion, classification, NER, geocoding, needs, and urgency pipeline; generates Parquet, CSV, and SQLite. |
| `python train.py` | Trains classifier suite (Naive Bayes, Logistic Regression, Linear SVM, Random Forest) and saves best model by Macro-F1. |
| `python train.py --experiments` | Executes ablation and Leave-One-Event-Out cross-validation experiments. |
| `python evaluate.py` | Generates ROC/PR curves, reliability calibration data, and per-event test metrics into `reports/`. |
| `python retrain_with_feedback.py` | Ingests operator corrections from SQLite and retrains production model with regression safeguards. |
| `streamlit run app.py` | Launches the Streamlit Command Center GUI at `http://localhost:8501`. |
| `uvicorn api.main:app --port 8000` | Launches the FastAPI REST service at `http://localhost:8000`. |
| `pytest --cov=src` | Executes the complete offline unit test suite. |

---

## 📡 REST API Reference

### Health & Metadata
- **`GET /health`**: Returns service status and model loading state.
- **`GET /version`**: Returns service version, dataset provenance, and disclaimer.

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
    {"name": "Houston", "lat": 29.7633, "lon": -95.3633, "country_code": "US", "confidence": 0.95}
  ],
  "needs": ["Medical", "Rescue/Evacuation"]
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
