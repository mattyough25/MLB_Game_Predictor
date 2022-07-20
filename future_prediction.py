import os
import requests
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup
from datetime import datetime
from joblib import Parallel, delayed
import joblib

# Define Years to Scrape 
year = 2022
all_matches = []

# Base URL
standings_url = "https://www.baseball-reference.com/leagues/majors/2022-standard-batting.shtml"

# Import Previously Scraped Data
scraped_games = pd.read_csv("D:/GitHub/Fantasy_Sports_Optimizers/SportsOutcomePredictor/MLB/training_data/mlb_games_final.csv", index_col=0)

# Scrape Data
data = requests.get(standings_url)
soup = BeautifulSoup(data.text)
print('Scraping ' + str(year))
standings_table = soup.select('table.stats_table')[0]
links = [l.get("href") for l in standings_table.find_all('a')]
links = [l for l in links if '/teams/' in l]
links = {x.replace('.shtml', '') for x in links}
team_urls = [f"https://www.baseball-reference.com{l}-schedule-scores.shtml" for l in links]
previous_season = soup.select("a.prev")[0].get("href")

# Organize Data in a Dataframe
for team_url in team_urls:
    team_name = team_url.split("/")[4]
    data = requests.get(team_url)
    team_data = pd.read_html(data.text)[0]
    team_data["Season"] = year

    dummyPrev     = team_data.loc[team_data['Unnamed: 2'] == 'boxscore']
    dummyUpcoming = team_data.loc[team_data['Unnamed: 2'] == 'preview']

    nDataPrev = dummyPrev.tail(1)
    nDataUpcoming = dummyUpcoming.head(1)

    nDataUpcoming["Rank"] = int(nDataPrev["Rank"])
    sGameTime = str(nDataUpcoming["W/L"])
    tGameTime = int(sGameTime.split(":")[0].split(" ")[4])

    if tGameTime >= 5:
        nDataUpcoming["TOD"] = 1
    else:
        nDataUpcoming["TOD"] = 0
    
    #print(dummyData.head(1))
    all_matches.append(nDataUpcoming)
    time.sleep(1)

# Define DataFrame
match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]

# Rename columns and Set venue code
match_df.rename(columns = {'gm#':'Game #', 'date':'Date', 'tm':'Team', 'unnamed: 4':'H/A', 'opp':'Opp', 'w/l':'W/L', 'r':'RS', 'ra':'RA', 'w-l':'W-L', 'rank':'Rank', 'gb':'GB', 'win':'Win', 'loss':'Loss', 'save':'Save', 'time':'Time', 'd/n':'D/N', 'attendance':'Attendance', 'streak':'Streak', 'season':'Season','tod':'TOD'}, inplace = True)
match_df["venue_code"] = match_df["H/A"].astype("category").cat.codes
match_df["venue_code"] = match_df["venue_code"].astype("category").cat.codes

# Set opponent and opponent code data
nDataOpponent = scraped_games[1:len(scraped_games.index)][['Opp', 'opp_code']]

# Match opponents with their codes
lines = len(scraped_games.index)
d_opp = {'Opp': [], 'opp_code': []}
df_opp = pd.DataFrame(data=d_opp)
for line in range(lines - 2):
    sTeam = nDataOpponent.iloc[line,0]
    nTeam = nDataOpponent.iloc[line,1]

    bPresent = sTeam in df_opp['Opp'].unique()
    if bPresent is False:
        df2 = {'Opp': sTeam, 'opp_code': nTeam}
        df_opp = df_opp.append(df2, ignore_index = True)



# Create opp_code column and reset index
match_df["opp_code"] = np.nan
match_df = match_df.reset_index()

# Fill opp_code column
lines_opp = len(df_opp.index)
lines_predict = len(match_df.index)
for line_opp in range(lines_opp):
    sTeam2 = df_opp.iloc[line_opp,0]
    nTeam2  = df_opp.iloc[line_opp,1]
    for line_predict in range(lines_predict):
        if sTeam2 == match_df.iloc[line_predict,6]:
            match_df.at[line_predict,"opp_code"] = nTeam2
        else:
            continue

# Determine Predictors
predictors = ["venue_code", "opp_code", "TOD","Rank"]

# Load Trained Model
rf = joblib.load("D:/GitHub/Fantasy_Sports_Optimizers/SportsOutcomePredictor/MLB/model/mlb_prediction_model.pkl")

# Run Model
preds = rf.predict(match_df[predictors])

# Create Predictions DataFrame
predictions = {'Team' : match_df['Team'],'bResult': preds}
predictions_df = pd.DataFrame(data=predictions)

# Generate DateTime for File Naming
now = datetime.now()
date_time = now.strftime("%m_%d_%Y")

# Save Predictions as csv
predictions_df.to_csv("D:/GitHub/Fantasy_Sports_Optimizers/SportsOutcomePredictor/MLB/predictions/" + "mlb_game_predictions_" + date_time + ".csv",index=False)


