import pandas as pd
from src.automl import run_automl
from src.serializer import save_model

df = pd.read_csv("data.csv")

target_column = "target"  # change this

best_model, leaderboard, problem_type = run_automl(df, target_column)

print("Problem Type:", problem_type)
print("\nLeaderboard:")
print(leaderboard)

save_model(best_model)
