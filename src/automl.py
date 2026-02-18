from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
from src.models import get_models
from src.preprocessing import build_preprocessor


def detect_problem_type(y):

    if y.dtype == "object" or y.nunique() <= 20:
        return "classification"
    else:
        return "regression"


def run_automl(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    problem_type = detect_problem_type(y)

    preprocessor = build_preprocessor(X)

    models = get_models(problem_type)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    leaderboard = []

    best_model = None
    best_score = float("-inf")

    for name, (model, params) in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        grid = RandomizedSearchCV(
            pipeline,
            params,
            cv=2,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

        if problem_type == "classification":
            score = accuracy_score(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)

        leaderboard.append((name, score))

        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_

    leaderboard_df = pd.DataFrame(
        leaderboard,
        columns=["Model", "Score"]
    ).sort_values(by="Score", ascending=False)

    return best_model, leaderboard_df, problem_type
