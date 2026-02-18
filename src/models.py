from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

def get_models(problem_type):

    if problem_type == "classification":

        models = {

            "Logistic Regression": (
                LogisticRegression(max_iter=1000),
                {"model__C": [0.1, 1, 10]}
            ),

            "Random Forest": (
                RandomForestClassifier(),
                {"model__n_estimators": [100, 200]}
            ),

            "Gradient Boosting": (
                GradientBoostingClassifier(),
                {"model__n_estimators": [100, 200]}
            ),

            "SVM": (
                SVC(),
                {"model__C": [0.1, 1, 10]}
            ),

            "KNN": (
                KNeighborsClassifier(),
                {"model__n_neighbors": [3, 5, 7]}
            ),

            "Decision Tree": (
                DecisionTreeClassifier(),
                {"model__max_depth": [None, 10, 20]}
            ),

            "Naive Bayes": (
                GaussianNB(),
                {}
            ),

            
        }

    else:  # regression

        models = {

            "Linear Regression": (
                LinearRegression(),
                {}
            ),

            "Ridge": (
                Ridge(),
                {"model__alpha": [0.1, 1, 10]}
            ),

            "Lasso": (
                Lasso(),
                {"model__alpha": [0.1, 1, 10]}
            ),

            "Random Forest": (
                RandomForestRegressor(),
                {"model__n_estimators": [100, 200]}
            ),

            "Gradient Boosting": (
                GradientBoostingRegressor(),
                {"model__n_estimators": [100, 200]}
            ),

            "SVR": (
                SVR(),
                {"model__C": [0.1, 1, 10]}
            ),

            "KNN": (
                KNeighborsRegressor(),
                {"model__n_neighbors": [3, 5, 7]}
            ),

            "Decision Tree": (
                DecisionTreeRegressor(),
                {"model__max_depth": [None, 10, 20]}
            ),

            
        }

    return models
