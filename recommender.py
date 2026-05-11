import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split as surprise_split
import warnings

warnings.filterwarnings("ignore")


class HybridRecommender:
    def __init__(self, movies_path, ratings_path):
        self.movies_path = movies_path
        self.ratings_path = ratings_path
        self.movies = None
        self.ratings = None
        self.cosine_sim = None
        self.movie_idx = None
        self.svd_model = None
        self.global_mean = None
        self.evaluation_results = {}
        self._train_user_items = None

    def load_and_preprocess(self):
        self.movies = pd.read_csv(self.movies_path)
        self.ratings = pd.read_csv(self.ratings_path)

        self.movies.dropna(inplace=True)
        self.ratings.dropna(inplace=True)
        self.movies.drop_duplicates(subset="movieId", inplace=True)
        self.ratings.drop_duplicates(subset=["userId", "movieId"], inplace=True)

        rated_ids = set(self.ratings["movieId"].unique())
        self.movies = self.movies[self.movies["movieId"].isin(rated_ids)].reset_index(drop=True)
        self.movie_idx = pd.Series(self.movies.index, index=self.movies["movieId"]).to_dict()
        self.global_mean = float(self.ratings["rating"].mean())

    def build_content_model(self):
        genres_text = (
            self.movies["genres"]
            .str.replace("|", " ", regex=False)
            .str.replace("(no genres listed)", "", regex=False)
        )
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(genres_text)
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def build_collaborative_model(self):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings[["userId", "movieId", "rating"]], reader)
        trainset, testset = surprise_split(data, test_size=0.2, random_state=42)

        self.svd_model = SVD(n_factors=100, n_epochs=20, random_state=42)
        self.svd_model.fit(trainset)

        cf_predictions = self.svd_model.test(testset)
        self.evaluation_results["CF_RMSE"] = round(
            float(accuracy.rmse(cf_predictions, verbose=False)), 4
        )
        self.evaluation_results["CF_MAE"] = round(
            float(accuracy.mae(cf_predictions, verbose=False)), 4
        )

        self._build_train_user_items(trainset)
        self._evaluate_hybrid(cf_predictions)
        self._evaluate_precision_recall(cf_predictions)

    def _build_train_user_items(self, trainset):
        self._train_user_items = {}
        for uid_inner, iid_inner, _ in trainset.all_ratings():
            raw_uid = trainset.to_raw_uid(uid_inner)
            raw_iid = int(trainset.to_raw_iid(iid_inner))
            self._train_user_items.setdefault(raw_uid, []).append(raw_iid)

    def _build_user_cb_vectors(self, user_items_dict):
        n_movies = len(self.movies)
        user_cb_vecs = {}
        for uid, items in user_items_dict.items():
            valid_idx = [self.movie_idx[mid] for mid in items if mid in self.movie_idx]
            if valid_idx:
                user_cb_vecs[uid] = np.mean(self.cosine_sim[valid_idx], axis=0)
            else:
                user_cb_vecs[uid] = np.zeros(n_movies)
        return user_cb_vecs

    def _evaluate_hybrid(self, cf_predictions, alpha=0.5):
        user_cb_vecs = self._build_user_cb_vectors(self._train_user_items)
        zero_vec = np.zeros(len(self.movies))
        true_ratings, hybrid_ratings = [], []

        for pred in cf_predictions:
            uid = pred.uid
            mid = int(pred.iid)
            true_r = pred.r_ui
            cf_est = pred.est

            cb_vec = user_cb_vecs.get(uid, zero_vec)
            target_idx = self.movie_idx.get(mid, -1)
            cb_score = float(cb_vec[target_idx]) if target_idx >= 0 else 0.0

            cf_norm = (cf_est - 0.5) / 4.5
            hybrid_norm = alpha * cf_norm + (1.0 - alpha) * cb_score
            hybrid_rating = float(np.clip(hybrid_norm * 4.5 + 0.5, 0.5, 5.0))

            true_ratings.append(true_r)
            hybrid_ratings.append(hybrid_rating)

        self.evaluation_results["Hybrid_RMSE"] = round(
            float(np.sqrt(mean_squared_error(true_ratings, hybrid_ratings))), 4
        )
        self.evaluation_results["Hybrid_MAE"] = round(
            float(mean_absolute_error(true_ratings, hybrid_ratings)), 4
        )

    def _evaluate_precision_recall(self, predictions, threshold=3.5, k=10):
        user_preds = {}
        for pred in predictions:
            user_preds.setdefault(pred.uid, []).append((pred.est, pred.r_ui))

        precisions, recalls = [], []
        for preds in user_preds.values():
            preds.sort(key=lambda x: x[0], reverse=True)
            top_k = preds[:k]
            n_relevant = sum(1 for _, r in preds if r >= threshold)
            n_hit = sum(1 for est, r in top_k if est >= threshold and r >= threshold)
            precisions.append(n_hit / k)
            recalls.append(n_hit / n_relevant if n_relevant > 0 else 0.0)

        p = float(np.mean(precisions))
        r = float(np.mean(recalls))
        f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
        self.evaluation_results["Precision@10"] = round(p, 4)
        self.evaluation_results["Recall@10"] = round(r, 4)
        self.evaluation_results["F1@10"] = round(f1, 4)

    def _batch_cf_predict(self, user_id, movie_ids):
        trainset = self.svd_model.trainset
        global_mean = trainset.global_mean

        try:
            inner_uid = trainset.to_inner_uid(user_id)
            bu = float(self.svd_model.bu[inner_uid])
            pu = self.svd_model.pu[inner_uid]
        except ValueError:
            bu = 0.0
            pu = np.zeros(self.svd_model.pu.shape[1])

        known_mids, known_inner_iids, unknown_mids = [], [], []
        for mid in movie_ids:
            try:
                known_inner_iids.append(trainset.to_inner_iid(mid))
                known_mids.append(mid)
            except ValueError:
                unknown_mids.append(mid)

        predictions = {}
        if known_inner_iids:
            bi_vec = self.svd_model.bi[known_inner_iids]
            qi_mat = self.svd_model.qi[known_inner_iids]
            raw_preds = np.clip(global_mean + bu + bi_vec + qi_mat.dot(pu), 0.5, 5.0)
            predictions.update(zip(known_mids, raw_preds.tolist()))

        for mid in unknown_mids:
            predictions[mid] = global_mean

        return predictions

    def recommend(self, user_id, n=10, alpha=0.5):
        rated = set(self.ratings[self.ratings["userId"] == user_id]["movieId"].tolist())

        top_rated_ids = (
            self.ratings[self.ratings["userId"] == user_id]
            .sort_values("rating", ascending=False)
            .head(5)["movieId"]
            .tolist()
        )

        candidate_df = self.movies[~self.movies["movieId"].isin(rated)].copy()
        candidate_ids = candidate_df["movieId"].tolist()
        candidate_positions = candidate_df.index.tolist()

        valid_ref_idx = [self.movie_idx[mid] for mid in top_rated_ids if mid in self.movie_idx]
        cb_vec = (
            np.mean(self.cosine_sim[valid_ref_idx], axis=0)
            if valid_ref_idx
            else np.zeros(len(self.movies))
        )
        cb_candidates = cb_vec[candidate_positions]

        cf_preds = self._batch_cf_predict(user_id, candidate_ids)
        cf_candidates = np.array([cf_preds[mid] for mid in candidate_ids])

        cb_min, cb_max = cb_candidates.min(), cb_candidates.max()
        cb_norm = (
            (cb_candidates - cb_min) / (cb_max - cb_min)
            if cb_max > cb_min
            else np.full(len(cb_candidates), 0.5)
        )
        cf_norm = (cf_candidates - 0.5) / 4.5
        hybrid = alpha * cf_norm + (1.0 - alpha) * cb_norm

        top_n_arg = np.argsort(hybrid)[::-1][:n]
        result_df = candidate_df.iloc[top_n_arg].copy()
        result_df["hybrid_score"] = np.round(hybrid[top_n_arg], 4)
        result_df["predicted_rating"] = np.round(cf_candidates[top_n_arg], 2)
        result_df["content_similarity"] = np.round(cb_candidates[top_n_arg], 4)

        return result_df[
            ["title", "genres", "predicted_rating", "content_similarity", "hybrid_score"]
        ].reset_index(drop=True)

    def get_similar_movies(self, movie_title, n=10):
        mask = self.movies["title"].str.contains(movie_title, case=False, na=False, regex=False)
        matched = self.movies[mask]
        if matched.empty:
            return pd.DataFrame(), None

        ref_movie = matched.iloc[0]
        ref_idx = self.movie_idx[ref_movie["movieId"]]
        sim_scores = self.cosine_sim[ref_idx].copy()
        sim_scores[ref_idx] = -1.0

        top_n_idx = np.argsort(sim_scores)[::-1][:n]
        result_df = self.movies.iloc[top_n_idx].copy()
        result_df["similarity"] = np.round(sim_scores[top_n_idx], 4)
        return result_df[["title", "genres", "similarity"]].reset_index(drop=True), ref_movie["title"]

    def get_user_history(self, user_id):
        user_df = self.ratings[self.ratings["userId"] == user_id].copy()
        user_df = user_df.merge(
            self.movies[["movieId", "title", "genres"]], on="movieId", how="left"
        )
        return (
            user_df[["title", "genres", "rating"]]
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )

    def get_all_users(self):
        return sorted(self.ratings["userId"].unique().tolist())
