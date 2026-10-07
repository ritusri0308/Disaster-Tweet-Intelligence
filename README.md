# Disaster Tweet Intelligence & Emergency Prioritization System 🚨

An end-to-end, modular Python NLP, Machine Learning, and Streamlit analytics application developed for the university course **Advanced Python Programming (CSI3007)**.

The system ingests unstructured social media disaster tweets (HumAID dataset) and converts them into structured emergency intelligence:
- **Category Classification**: What is happening (`Emergency`, `Help Request`, `Damage Report`, `Information/Update`, `General`).
- **Location Extraction (NER)**: Where it is occurring (`GPE`, `LOC`, `FAC` entities via spaCy + gazetteer fallback).
- **Urgency Prioritization**: How urgent it is (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL` via an explainable score engine).
- **Aggregated Analytics**: Dashboard with KPIs, situation summaries, volume over time, and filtered reports.

> **Disclaimer**: This is an academic prototype developed for educational and humanitarian computing research purposes. It is **not** an operational emergency tool or dispatch system.

---

## 🏛️ System Architecture

```
                               ┌─────────────────────────────┐
                               │  HumAID Raw TSV Dataset     │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │   data_loader & Snowflake   │
                               │     Timestamp Derivation    │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │        preprocessing        │
                               │   (Cleaning & Features)     │
                               └──────────────┬──────────────┘
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        │                     │                     │
                        ▼                     ▼                     ▼
               ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
               │   classifier    │   │      ner        │   │     urgency     │
               │ (TF-IDF + ML)   │   │ (spaCy + Gaz)   │   │  (Rule Engine)  │
               └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
                        │                     │                     │
                        └─────────────────────┼─────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │     database & Analytics    │
                               │   (SQLite & Aggregations)   │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │    Streamlit Web App GUI    │
                               └─────────────────────────────┘
```

---

## 📂 Project Layout

```
DisasterTweetIntelligence/
├── app.py                      # Streamlit dashboard GUI entry point
├── train.py                    # Model training & experiment CLI
├── run_pipeline.py             # End-to-end processing pipeline CLI
├── requirements.txt            # Pinned Python package dependencies
├── pytest.ini                  # Pytest configuration
├── .gitignore                  # Git ignore rules (raw/processed data & DB excluded)
├── README.md                   # Comprehensive project documentation
├── data/
│   ├── raw/events_set1/        # Raw HumAID dataset (git-ignored)
│   └── processed/              # Processed CSV & SQLite DB (git-ignored)
├── models/
│   └── best_classifier.joblib  # Saved best machine learning model
├── reports/                    # Metrics, confusion matrices, and experiment tables
├── scripts/
│   ├── make_sample_data.py     # Synthetic data generator fallback
│   ├── export_urgency_sheet.py # Urgency hand-annotation exporter & evaluator
│   └── export_ner_sheet.py     # NER check sheet exporter & evaluator
├── src/
│   ├── __init__.py
│   ├── config.py               # Configurable mapping, thresholds, paths
│   ├── data_loader.py          # TSV loading, timestamp derivation, validation
│   ├── preprocessing.py       # Cleaning, regex, text feature extraction
│   ├── features.py            # TF-IDF vectorizer configuration
│   ├── classifier.py         # Sklearn classifier pipelines & metrics
│   ├── ner.py                 # spaCy NER & gazetteer fallback location extraction
│   ├── urgency.py             # Explainable urgency scoring engine
│   ├── analytics.py           # Crosstabs, location explosion, situation summary
│   ├── database.py            # Parameterized SQLite storage & helper queries
│   └── pipeline.py            # Complete end-to-end pipeline execution
└── tests/                      # Fast offline pytest suite
    ├── conftest.py             # Shared fixtures
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_classifier.py
    ├── test_ner.py
    ├── test_urgency.py
    ├── test_analytics.py
    ├── test_database.py
    ├── test_pipeline.py
    └── test_app.py
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Python 3.10+ installed on Windows, macOS, or Linux.

### 2. Environment Setup & Dependencies
```bash
# Clone the repository and navigate to root directory
cd DisasterTweetIntelligence

# Create virtual environment (optional but recommended)
python -m venv venv
# Activate on Windows: venv\Scripts\activate
# Activate on Linux/macOS: source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Download spaCy English NLP model
python -m spacy download en_core_web_sm
```

---

## 📊 Dataset & License

### HumAID Dataset (events_set1)
- **Scale**: 13 historical disaster events (~43,400 tweets, 2016–2019).
- **Structure**: Discovered recursively via `pathlib.Path.rglob("*.tsv")`.
- **License**: Licensed strictly for non-commercial humanitarian computing research. Raw data (`data/raw/`), processed data (`data/processed/`), and SQLite databases are strictly git-ignored.
- **Citation**:
  > Firoj Alam, Hassan Sajjad, Muhammad Imran, and Ferda Ofli. *HumAID: Human-Annotated Disaster Incident Data from Twitter with Multimodal Features*. ICWSM 2021.

### 🏷️ Category Mapping Table
Raw HumAID class labels are mapped into 5 project categories in `src/config.py`:

| Raw HumAID Label | Project 5 Category | Description |
| :--- | :--- | :--- |
| `injured_or_dead_people` | **Emergency** | Casualty reports (injured or deceased people) |
| `requests_or_urgent_needs` | **Help Request** | Urgent requests for rescue, food, water, medical assistance |
| `infrastructure_and_utility_damage` | **Damage Report** | Road, bridge, power grid, or building damage |
| `caution_and_advice` | **Information/Update** | Advisories, warnings, safety tips |
| `displaced_people_and_evacuations` | **Information/Update** | Shelters, evacuations, displaced populations |
| `missing_or_found_people` | **Information/Update** | Reports of missing persons or found individuals |
| `other_relevant_information` | **Information/Update** | General disaster news updates |
| `rescue_volunteering_or_donation_effort` | **Information/Update** | Volunteer drives & donation campaigns |
| `not_humanitarian` | **General** | Irrelevant or non-humanitarian tweets |
| `sympathy_and_support` | **General** | Condolences, prayers, emotional support |
| `dont_know_cant_judge` | **General** | Ambiguous content |

### ⏰ Snowflake ID Timestamp Derivation
Twitter IDs are Snowflake IDs encoding timestamp information:
$$\text{timestamp\_ms} = (\text{tweet\_id} \gg 22) + 1288834974657$$
Converted to UTC datetime in `data_loader.py` via `derive_timestamp_from_id()`. Non-numeric IDs gracefully set to `NaT`.

---

## 🚀 Execution Instructions

### 1. Run Unit & Smoke Tests
```bash
pytest
```
Or run with coverage reporting:
```bash
pytest --cov=src --cov-report=term-missing
```

### 2. Train Classifiers & Run Experiments
Train Naive Bayes, Logistic Regression, Linear SVM, and Random Forest models, evaluate macro-F1 on dev set, and save the best model to `models/best_classifier.joblib`:
```bash
python train.py
```
To run full ablation and Leave-One-Event-Out cross-validation experiments:
```bash
python train.py --experiments
```

### 3. Run End-to-End Pipeline
Execute data loading, preprocessing, classification, location extraction (NER), urgency scoring, and save results to CSV & SQLite DB:
```bash
python run_pipeline.py
```

### 4. Launch Streamlit Dashboard App
```bash
streamlit run app.py
```

---

## 🧪 Experiments & Evaluation

The `train.py --experiments` command executes 5 quantitative experiments and saves results to `reports/`:
1. **Exp (a)**: Unigrams `(1,1)` vs Unigrams + Bigrams `(1,2)` feature comparison.
2. **Exp (b)**: Raw vs Cleaned text training comparison.
3. **Exp (c)**: With vs Without Class Balancing (`class_weight='balanced'`).
4. **Exp (d)**: **Leave-One-Event-Out Cross-Validation** held out on `srilanka_floods_2017`, `cyclone_idai_2019`, `hurricane_harvey_2017`, and `kaikoura_earthquake_2016`, reporting mean ± std across held-out events.
5. **Exp (e)**: `project_5` (5 classes) vs `raw_10` (10 classes) classification target comparison.

### Hand Annotation & Evaluation Sheets
- **Urgency Evaluation**: Export sheet via `python scripts/export_urgency_sheet.py`. Hand-annotate the `human_urgency` column in `reports/urgency_annotation_sheet.csv`, then call `evaluate_urgency("reports/urgency_annotation_sheet.csv")`.
- **NER Evaluation**: Export sheet via `python scripts/export_ner_sheet.py`. Fill `correct_locations` in `reports/ner_check_sheet.csv`, then call `evaluate_ner("reports/ner_check_sheet.csv")`.

---

## ⚠️ System Limitations & Future Work

### Limitations
1. **Academic Prototype**: Designed for academic evaluation (CSI3007), not for real-life emergency response dispatch.
2. **Category Definitions**: "Emergency" strictly represents casualty reports (`injured_or_dead_people`).
3. **No Real Coordinates**: HumAID dataset does not contain GPS latitude/longitude columns; map visualizers are omitted per guidelines.
4. **Rule Engine Urgency**: Urgency prioritization relies on keyword heuristics, text feature boosts, and retweet/donation damping rules.

### Future Work
- Integration of Transformer models (RoBERTa / DeBERTa) for fine-grained semantic classification.
- Multilingual NLP support for non-English disaster tweets.
- Real-time Twitter / X API streaming ingestion pipeline.
- Geocoding API integration (e.g., Nominatim / Google Maps API) to convert extracted location names into GPS coordinates for map plotting.
