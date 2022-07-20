import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from joblib import Parallel, delayed
import joblib

# Get Directory of Training Dataset
#dirData = input("Enter directory to training data: ")
os.chdir("D:/GitHub/Fantasy_Sports_Optimizers/SportsOutcomePredictor/MLB/training_data")

# Read Data File
games = pd.read_csv("mlb_games_final.csv", index_col=0)

# Import the Model (This case its Random Forest)
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1) #experiment to find optimal parameters

# Parse Data
train = games[games["Season"] < 2020]
test = games[games["Season"] >= 2020]

# Determine Predictors
predictors = ["venue_code", "opp_code", "TOD","Rank"]

# Train Model
rf.fit(train[predictors], train["bResult"])
preds = rf.predict(test[predictors])

# Error Score
error = accuracy_score(test["bResult"], preds)
print("Error Score: " + str(error))

# Precision Score
combined = pd.DataFrame(dict(actual=test["bResult"], predicted=preds))
pd.crosstab(index=combined["actual"], columns=combined["predicted"])
precision_score = precision_score(test["bResult"], preds)
print("Precision Score: " + str(precision_score))

# Save Model
joblib.dump(rf, 'D:/GitHub/Fantasy_Sports_Optimizers/SportsOutcomePredictor/MLB/model/mlb_prediction_model.pkl')