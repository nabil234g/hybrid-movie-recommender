import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommender import HybridRecommender

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_PATH = os.path.join(BASE_DIR, "movies.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "ratings.csv")

st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; }
        [data-testid="stMetricLabel"] { font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading data and training models. This may take a minute...")
def initialize():
    rec = HybridRecommender(MOVIES_PATH, RATINGS_PATH)
    rec.load_and_preprocess()
    rec.build_content_model()
    rec.build_collaborative_model()
    return rec


rec = initialize()

st.title("Hybrid Movie Recommendation System")
st.caption(
    "Combines collaborative filtering (SVD matrix factorization) and content-based filtering "
    "(TF-IDF cosine similarity) via weighted averaging."
)

tab1, tab2, tab3 = st.tabs(
    ["Personalized Recommendations", "Similar Movies", "Model Evaluation"]
)

with tab1:
    left, right = st.columns([1, 3], gap="large")

    with left:
        st.subheader("Parameters")
        user_id = st.selectbox("User ID", rec.get_all_users())
        n_recs = st.slider("Number of results", min_value=5, max_value=20, value=10)
        alpha = st.slider(
            "Collaborative filtering weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Controls the blend between CF (SVD) and content-based scores. "
                 "0.0 = content only, 1.0 = collaborative only.",
        )
        run_btn = st.button("Generate Recommendations", type="primary", use_container_width=True)

    with right:
        if run_btn:
            with st.spinner("Computing recommendations..."):
                recs_df = rec.recommend(user_id, n=n_recs, alpha=alpha)

            st.subheader(f"Top {n_recs} recommendations for User {user_id}")
            st.dataframe(
                recs_df,
                column_config={
                    "title": st.column_config.TextColumn("Title", width="large"),
                    "genres": st.column_config.TextColumn("Genres", width="medium"),
                    "predicted_rating": st.column_config.NumberColumn(
                        "Predicted Rating", format="%.2f", width="small"
                    ),
                    "content_similarity": st.column_config.NumberColumn(
                        "Content Similarity", format="%.4f", width="small"
                    ),
                    "hybrid_score": st.column_config.NumberColumn(
                        "Hybrid Score", format="%.4f", width="small"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Rating History")
            history_df = rec.get_user_history(user_id)
            st.dataframe(
                history_df,
                column_config={
                    "title": st.column_config.TextColumn("Title", width="large"),
                    "genres": st.column_config.TextColumn("Genres", width="medium"),
                    "rating": st.column_config.NumberColumn("Rating", format="%.1f", width="small"),
                },
                use_container_width=True,
                hide_index=True,
                height=300,
            )
        else:
            st.info("Select a user ID and click Generate Recommendations.")

with tab2:
    left, right = st.columns([1, 3], gap="large")

    with left:
        st.subheader("Parameters")
        query = st.text_input("Movie title", placeholder="e.g. Toy Story")
        n_similar = st.slider("Number of results", min_value=5, max_value=20, value=10, key="n_sim")
        search_btn = st.button("Find Similar Movies", type="primary", use_container_width=True)

    with right:
        if search_btn:
            if not query.strip():
                st.warning("Enter a movie title to search.")
            else:
                with st.spinner("Searching..."):
                    sim_df, ref_title = rec.get_similar_movies(query.strip(), n=n_similar)
                if sim_df.empty:
                    st.error(f'No movies found matching "{query}".')
                else:
                    st.subheader(f"Movies similar to: {ref_title}")
                    st.dataframe(
                        sim_df,
                        column_config={
                            "title": st.column_config.TextColumn("Title", width="large"),
                            "genres": st.column_config.TextColumn("Genres", width="medium"),
                            "similarity": st.column_config.NumberColumn(
                                "Similarity Score", format="%.4f", width="small"
                            ),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.info("Enter a movie title and click Find Similar Movies.")

with tab3:
    st.subheader("Evaluation Metrics")
    st.caption(
        "All metrics are computed on a held-out 20% test split (random seed 42). "
        "Hybrid metrics use alpha = 0.5."
    )

    metrics = rec.evaluation_results

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CF RMSE", metrics.get("CF_RMSE", "N/A"))
    c2.metric("CF MAE", metrics.get("CF_MAE", "N/A"))
    c3.metric("Hybrid RMSE", metrics.get("Hybrid_RMSE", "N/A"))
    c4.metric("Hybrid MAE", metrics.get("Hybrid_MAE", "N/A"))

    st.divider()

    c5, c6, c7, _ = st.columns(4)
    c5.metric("Precision@10", metrics.get("Precision@10", "N/A"))
    c6.metric("Recall@10", metrics.get("Recall@10", "N/A"))
    c7.metric("F1@10", metrics.get("F1@10", "N/A"))

    st.divider()

    st.subheader("Metric Definitions")
    ref_df = pd.DataFrame(
        {
            "Metric": [
                "CF RMSE",
                "CF MAE",
                "Hybrid RMSE",
                "Hybrid MAE",
                "Precision@10",
                "Recall@10",
                "F1@10",
            ],
            "Description": [
                "Root Mean Square Error of SVD-predicted ratings vs actual ratings on test set",
                "Mean Absolute Error of SVD-predicted ratings vs actual ratings on test set",
                "Root Mean Square Error of hybrid-predicted ratings vs actual ratings on test set",
                "Mean Absolute Error of hybrid-predicted ratings vs actual ratings on test set",
                "Average fraction of top-10 recommendations rated >= 3.5 that the user also rated >= 3.5",
                "Average fraction of all relevant items (rating >= 3.5) that appear in top-10 recommendations",
                "Harmonic mean of Precision@10 and Recall@10",
            ],
        }
    )
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
