<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:8B5CF6,25:6366F1,50:3B82F6,75:0EA5E9,100:06B6D4&height=300&section=header&text=Hybrid%20Movie%20Recommendation%20System&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=40&desc=SVD%20Matrix%20Factorization%20%2B%20TF-IDF%20Cosine%20Similarity%20%2B%20Weighted%20Hybrid%20Engine&descSize=16&descAlignY=58&descFontColor=ddd6fe"/>

<div align="center">

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=22&duration=3000&pause=800&color=8B5CF6&center=true&vCenter=true&width=750&lines=Collaborative+Filtering+%2B+Content-Based+Filtering;SVD+Matrix+Factorization+via+Surprise+Library;TF-IDF+Vectorization+%2B+Cosine+Similarity;Deployed+on+Streamlit+Community+Cloud)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![scikit--surprise](https://img.shields.io/badge/scikit--surprise-8B5CF6?style=for-the-badge&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

<br/>

![Dataset](https://img.shields.io/badge/Dataset-MovieLens%20100K-7C3AED?style=flat-square&logoColor=white)
![Ratings](https://img.shields.io/badge/Ratings-100%2C836-4F46E5?style=flat-square)
![Movies](https://img.shields.io/badge/Movies-9%2C724-2563EB?style=flat-square)
![Users](https://img.shields.io/badge/Users-610-0891B2?style=flat-square)
![CF RMSE](https://img.shields.io/badge/CF%20RMSE-0.8807-059669?style=flat-square)
![Precision@10](https://img.shields.io/badge/Precision%4010-55.4%25-D97706?style=flat-square)
![F1@10](https://img.shields.io/badge/F1%4010-53.0%25-DC2626?style=flat-square)

<br/>

[![Live Demo](https://img.shields.io/badge/LIVE%20DEMO-Open%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://hybrid-movie-recommender-euiu52ebvg9u5qkpkrmoy6.streamlit.app)
&nbsp;
[![GitHub](https://img.shields.io/badge/SOURCE-View%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nabil234g/hybrid-movie-recommender)

<br/>

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

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

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

## Overview

This project implements a **Hybrid Movie Recommendation System** that intelligently combines two complementary filtering strategies:

| Strategy | Method | Signal Used |
|:---|:---|:---|
| Collaborative Filtering | SVD Matrix Factorization (Surprise) | User-item rating interactions |
| Content-Based Filtering | TF-IDF + Cosine Similarity | Movie genre metadata |
| Hybrid Engine | Weighted Average (alpha blending) | Both signals combined |

The hybrid approach overcomes the individual limitations of each method:

- Collaborative filtering alone suffers from the **cold-start problem**
- Content-based filtering alone leads to **over-specialization**
- The hybrid balances both to deliver **broader, more accurate recommendations**

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION                             │
│               MovieLens 100K  (movies.csv + ratings.csv)            │
│          9,724 movies   ·   100,836 ratings   ·   610 users         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
               ┌─────────────┴──────────────┐
               │                            │
               ▼                            ▼
┌──────────────────────────┐  ┌─────────────────────────────────────┐
│   CONTENT-BASED MODULE   │  │       COLLABORATIVE FILTERING       │
│                          │  │               MODULE                │
│  TF-IDF Vectorization    │  │                                     │
│  on movie genres         │  │  Surprise SVD                       │
│          ↓               │  │  n_factors=100  ·  n_epochs=20      │
│  Cosine Similarity       │  │          ↓                          │
│  Matrix (9724 × 9724)    │  │  Predicted Rating Matrix            │
│          ↓               │  │  (user × item)                      │
│  Item Similarity Scores  │  │          ↓                          │
│  normalized → [0, 1]     │  │  CF Scores  normalized → [0, 1]     │
└────────────┬─────────────┘  └───────────────┬─────────────────────┘
             │                                │
             └─────────────┬──────────────────┘
                           ▼
            ┌──────────────────────────────────┐
            │          HYBRID ENGINE           │
            │                                  │
            │   score = α · CF + (1−α) · CB   │
            │   default α = 0.6                │
            │   α fully adjustable via slider  │
            └──────────────┬───────────────────┘
                           ▼
            ┌──────────────────────────────────┐
            │         STREAMLIT UI             │
            │                                  │
            │  · Personalized recs by user     │
            │  · Similar movies by title       │
            │  · Evaluation metrics tab        │
            └──────────────────────────────────┘
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

## Modules

### 1 — Data Ingestion and Preprocessing
- Loads `movies.csv` and `ratings.csv`
- Drops nulls and deduplicates on `movieId` and `(userId, movieId)`
- Filters movies to only those present in the ratings data
- Builds an internal dictionary index for O(1) movie lookups

### 2 — Content-Based Filtering
- Applies **TF-IDF vectorization** on pipe-separated genre strings
- Computes a full **9,724 × 9,724 cosine similarity matrix**
- Averages similarity across a user's top-5 rated movies to form a personalized CB score vector

### 3 — Collaborative Filtering
- Uses **Surprise library's SVD** algorithm with `n_factors=100`, `n_epochs=20`
- 80/20 train-test split with fixed random seed for reproducibility
- Batch prediction via internal factor matrices (`bu`, `bi`, `pu`, `qi`) for fast inference
- Graceful fallback to global mean for unknown users

### 4 — Hybrid Recommendation Engine
- Normalizes CF scores from rating scale `[0.5, 5.0]` → `[0, 1]`
- Normalizes CB scores via min-max scaling → `[0, 1]`
- Combines: `hybrid = α × CF_norm + (1 − α) × CB_norm`
- Alpha is user-controllable via the UI slider (default **0.6**)

### 5 — Evaluation
- All metrics on **held-out 20% test split**
- RMSE and MAE for both CF and Hybrid models
- Precision@10, Recall@10, F1@10 at relevance threshold of **3.5 stars**

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

## Dataset

**MovieLens Latest Small (100K)**

| Property | Value |
|:---|:---|
| Total ratings | 100,836 |
| Unique users | 610 |
| Unique movies | 9,724 |
| Rating scale | 0.5 to 5.0 (half-star increments) |
| Matrix sparsity | ~98.3% |
| Features used | `userId`, `movieId`, `rating`, `title`, `genres` |

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

## Evaluation Metrics

All metrics computed on the **20% hold-out test set** (random seed 42).

### Rating Prediction Accuracy

<div align="center">

![CF RMSE](https://img.shields.io/badge/CF%20RMSE-0.8807-2563EB?style=for-the-badge)
![CF MAE](https://img.shields.io/badge/CF%20MAE-0.6766-2563EB?style=for-the-badge)
![Hybrid RMSE](https://img.shields.io/badge/Hybrid%20RMSE-1.3418-7C3AED?style=for-the-badge)
![Hybrid MAE](https://img.shields.io/badge/Hybrid%20MAE-1.1614-7C3AED?style=for-the-badge)

</div>

| Model | RMSE | MAE |
|:---|:---:|:---:|
| Collaborative Filtering (SVD) | **0.8807** | **0.6766** |
| Hybrid (α = 0.5) | 1.3418 | 1.1614 |

> The hybrid shows higher RMSE than pure SVD by design. The content-based component (genre similarity) is not optimized for rating prediction — it improves recommendation coverage and diversity, not rating accuracy.

### Ranking Quality

<div align="center">

![Precision@10](https://img.shields.io/badge/Precision%4010-55.41%25-059669?style=for-the-badge)
![Recall@10](https://img.shields.io/badge/Recall%4010-50.86%25-059669?style=for-the-badge)
![F1@10](https://img.shields.io/badge/F1%4010-53.04%25-D97706?style=for-the-badge)

</div>

| Metric | Score | Meaning |
|:---|:---:|:---|
| Precision@10 | **0.5541** | 55% of top-10 recommendations are genuinely relevant |
| Recall@10 | **0.5086** | 51% of all relevant items are captured in top-10 |
| F1@10 | **0.5304** | Balanced harmonic mean of Precision and Recall |

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

## Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| Language | Python 3.12 | Core runtime |
| UI Framework | Streamlit | Interactive web application |
| Collaborative Filtering | scikit-surprise (SVD) | Matrix factorization |
| Content-Based Filtering | scikit-learn (TF-IDF, cosine similarity) | Genre vectorization |
| Data Manipulation | pandas | DataFrame operations |
| Numerical Computing | NumPy | Vector and matrix operations |
| Deployment | Streamlit Community Cloud | Public cloud hosting |
| Version Control | Git + GitHub | Source control |

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

## Project Structure

```
hybrid-movie-recommender/
│
├── app.py                  Streamlit application — 3 tabs (Recommendations, Similar, Metrics)
├── recommender.py          Core engine — preprocessing, TF-IDF, SVD, hybrid, evaluation
├── movies.csv              MovieLens movie metadata (movieId, title, genres)
├── ratings.csv             MovieLens user ratings (userId, movieId, rating, timestamp)
├── requirements.txt        Python package dependencies
├── .python-version         Python 3.12 pin for Streamlit Community Cloud
└── .gitignore              Excludes venv and cache files
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

## Installation

**Requirements:** Python 3.12 · pip

```bash
git clone https://github.com/nabil234g/hybrid-movie-recommender.git
cd hybrid-movie-recommender

py -3.12 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

## Usage

```bash
venv\Scripts\streamlit.exe run app.py
```

Opens at `http://localhost:8501`

| Tab | What it does |
|:---|:---|
| Personalized Recommendations | Select a user ID, adjust the CF weight (alpha), choose N — get a hybrid ranked list |
| Similar Movies | Search any movie title — get content-based similar movies by cosine similarity |
| Model Evaluation | View all evaluation metrics computed on the 20% hold-out test set |

**Or skip installation entirely and use the live app:**

[![Open App](https://img.shields.io/badge/Open%20Live%20App-hybrid--movie--recommender-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://hybrid-movie-recommender-euiu52ebvg9u5qkpkrmoy6.streamlit.app)

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:8B5CF6,100:06B6D4&height=4" width="100%"/>

<br/>

## Team

<div align="center">

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&duration=2000&pause=500&color=06B6D4&center=true&vCenter=true&width=500&lines=Nabil+Adel;Mariam+Hany;Manar+Ali+Mohammed;Tasneem+Yosry;ANAS+Mohamed)](https://git.io/typing-svg)

<br/>

| Name |
|:---:|
| **Nabil Adel** |
| **Mariam Hany** |
| **Manar Ali Mohammed** |
| **Tasneem Yosry** |
| **ANAS Mohamed** |

<br/>

</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:06B6D4,25:0EA5E9,50:3B82F6,75:6366F1,100:8B5CF6&height=200&section=footer"/>
