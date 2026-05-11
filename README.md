<div align="center">

<h1>Hybrid Movie Recommendation System</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--surprise-4B8BBE?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Dataset-MovieLens%20100K-blueviolet?style=flat-square"/>
  <img src="https://img.shields.io/badge/Model-SVD%20%2B%20TF--IDF-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Approach-Weighted%20Hybrid-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/CF%20RMSE-0.8807-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Precision%4010-55.4%25-green?style=flat-square"/>
</p>

<p><strong>A production-grade hybrid recommendation engine combining Collaborative Filtering (SVD matrix factorization) and Content-Based Filtering (TF-IDF cosine similarity) on the MovieLens 100K dataset — deployed as an interactive Streamlit application.</strong></p>

<a href="https://hybrid-movie-recommender-euiu52ebvg9u5qkpkrmoy6.streamlit.app">
  <img src="https://img.shields.io/badge/Live%20Demo-Open%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</a>

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Modules](#modules)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Team](#team)

---

## Overview

This project implements a **Hybrid Movie Recommendation System** that intelligently combines two complementary filtering strategies:

| Strategy | Method | Signal Used |
|---|---|---|
| Collaborative Filtering | SVD Matrix Factorization (Surprise) | User-item rating interactions |
| Content-Based Filtering | TF-IDF + Cosine Similarity | Movie genre metadata |
| Hybrid Engine | Weighted Average (alpha blending) | Both signals combined |

The hybrid approach overcomes the individual limitations of each method:
- Collaborative filtering alone suffers from the **cold-start problem**
- Content-based filtering alone leads to **over-specialization**
- The hybrid balances both to deliver **broader, more accurate recommendations**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                           │
│              MovieLens 100K  (movies.csv + ratings.csv)         │
│         9,724 movies   ·   100,836 ratings   ·   610 users      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
              ▼                            ▼
┌─────────────────────────┐  ┌─────────────────────────────────┐
│   CONTENT-BASED MODULE  │  │    COLLABORATIVE FILTERING      │
│                         │  │           MODULE                │
│  TF-IDF Vectorization   │  │                                 │
│  on movie genres        │  │  Surprise SVD                   │
│         ↓               │  │  n_factors=100 · n_epochs=20    │
│  Cosine Similarity      │  │         ↓                       │
│  Matrix (9724 × 9724)   │  │  Predicted Rating Matrix        │
│         ↓               │  │  (user × item)                  │
│  Item Similarity Scores │  │         ↓                       │
│  normalized → [0, 1]    │  │  CF Scores normalized → [0, 1]  │
└───────────┬─────────────┘  └──────────────┬──────────────────┘
            │                               │
            └──────────────┬────────────────┘
                           ▼
            ┌──────────────────────────────┐
            │       HYBRID ENGINE          │
            │                              │
            │  score = α · CF + (1-α) · CB │
            │  default α = 0.6             │
            │  α adjustable via UI slider  │
            └──────────────┬───────────────┘
                           ▼
            ┌──────────────────────────────┐
            │      STREAMLIT UI            │
            │                              │
            │  · Personalized recs by user │
            │  · Similar movies by title   │
            │  · Evaluation metrics tab    │
            └──────────────────────────────┘
```

---

## Modules

### 1. Data Ingestion and Preprocessing
- Loads `movies.csv` and `ratings.csv`
- Drops nulls and deduplicates
- Filters movies to only those present in ratings
- Builds internal movie index mapping for fast lookups

### 2. Content-Based Filtering
- Applies **TF-IDF vectorization** on pipe-separated genre strings
- Computes a full **9724 × 9724 cosine similarity matrix**
- Averages similarity across a user's top-5 rated movies to produce a personalized CB score vector

### 3. Collaborative Filtering
- Uses **Surprise library's SVD** algorithm (matrix factorization)
- 80/20 train-test split with random seed 42
- Batch prediction via internal factor matrices (`bu`, `bi`, `pu`, `qi`) for speed
- Handles unknown users via global mean fallback

### 4. Hybrid Recommendation Engine
- Normalizes CF scores (rating scale 0.5–5.0 → 0–1)
- Normalizes CB scores (already cosine similarity 0–1, then min-max)
- Combines via: `hybrid = α × CF_norm + (1 − α) × CB_norm`
- Alpha is user-controllable in the UI (default 0.6)

### 5. Evaluation
- **80/20 hold-out split** — all metrics on unseen test data
- RMSE and MAE for both CF and Hybrid
- Precision@10, Recall@10, F1@10 at relevance threshold 3.5 stars

---

## Dataset

**MovieLens Latest Small (100K)**

| Property | Value |
|---|---|
| Total ratings | 100,836 |
| Unique users | 610 |
| Unique movies | 9,742 |
| Rating scale | 0.5 to 5.0 (half-star increments) |
| Rating sparsity | ~98.3% |
| Features used | userId, movieId, rating, title, genres |

Genre representation uses **TF-IDF** over the pipe-separated genre strings (e.g., `Action|Adventure|Sci-Fi`), treating each pipe-separated genre as a term.

---

## Evaluation Metrics

All metrics are computed on the held-out **20% test set** (random seed 42).

### Rating Prediction Accuracy

| Model | RMSE | MAE |
|---|---|---|
| Collaborative Filtering (SVD) | **0.8807** | **0.6766** |
| Hybrid (α = 0.5) | 1.3418 | 1.1614 |

> The hybrid model shows higher RMSE/MAE than pure SVD by design. The content-based component is not trained to minimize rating error — it contributes genre-based similarity. The hybrid trades raw rating accuracy for improved recommendation coverage and serendipity.

### Ranking Quality (CF Component, threshold = 3.5 stars)

| Metric | Score |
|---|---|
| Precision@10 | **0.5541** |
| Recall@10 | **0.5086** |
| F1@10 | **0.5304** |

> A Precision@10 of **55.4%** means more than half of every 10 recommendations are movies the user would genuinely rate highly — strong performance for a recommendation system on this dataset.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.12 | Core runtime |
| UI Framework | Streamlit | Interactive web application |
| Collaborative Filtering | scikit-surprise (SVD) | Matrix factorization |
| Content-Based Filtering | scikit-learn (TF-IDF, cosine similarity) | Genre vectorization |
| Data Processing | pandas, NumPy | Data manipulation |
| Deployment | Streamlit Community Cloud | Public cloud hosting |
| Version Control | Git + GitHub | Source control |

---

## Project Structure

```
hybrid-movie-recommender/
│
├── app.py                  # Streamlit application (UI — 3 tabs)
├── recommender.py          # Core engine (preprocessing, models, evaluation)
├── movies.csv              # MovieLens movie metadata
├── ratings.csv             # MovieLens user ratings
├── requirements.txt        # Python dependencies
├── .python-version         # Python 3.12 specification for Streamlit Cloud
└── .gitignore
```

---

## Installation

**Requirements:** Python 3.12, pip

```bash
# Clone the repository
git clone https://github.com/nabil234g/hybrid-movie-recommender.git
cd hybrid-movie-recommender

# Create and activate virtual environment
py -3.12 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Run the Streamlit app
venv\Scripts\streamlit.exe run app.py      # Windows
# streamlit run app.py                     # macOS / Linux
```

The app opens at `http://localhost:8501` with three tabs:

| Tab | Description |
|---|---|
| Personalized Recommendations | Select a user ID, adjust the CF weight slider, get top-N hybrid recommendations |
| Similar Movies | Search any movie title, get content-based similar movies ranked by cosine similarity |
| Model Evaluation | View all computed metrics — RMSE, MAE, Precision@10, Recall@10, F1@10 |

Or use the **live deployment** directly — no installation required:
**[hybrid-movie-recommender-euiu52ebvg9u5qkpkrmoy6.streamlit.app](https://hybrid-movie-recommender-euiu52ebvg9u5qkpkrmoy6.streamlit.app)**

---

## Team

<div align="center">

| Name | Role |
|---|---|
| **Nabil Adel** | Collaborative Filtering · SVD Implementation · Hybrid Engine |
| **Mariam Hany** | Content-Based Filtering · TF-IDF · Cosine Similarity |
| **Manar Ali Mohammed** | Data Preprocessing · Evaluation Metrics |
| **Tasneem Yosry** | Streamlit UI · Deployment |
| **ANAS Mohamed** | Model Integration · Testing · Documentation |

</div>

---

<div align="center">

**Final Project — Hybrid Recommendation Systems**

<sub>Built with scikit-surprise SVD · TF-IDF Cosine Similarity · Streamlit · MovieLens 100K</sub>

</div>
