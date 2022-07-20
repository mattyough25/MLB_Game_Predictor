# Import Libraries
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

# Define Years to Scrape 
years = list(range(2022, 2014, -1))
all_matches = []

# Base URL
standings_url = "https://www.baseball-reference.com/leagues/majors/2021-standard-batting.shtml"

# Scrape Data
for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    print('Scraping ' + str(year))
    standings_table = soup.select('table.stats_table')[0]
    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/teams/' in l]
    links = {x.replace('.shtml', '') for x in links}
    team_urls = [f"https://www.baseball-reference.com{l}-schedule-scores.shtml" for l in links]
    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://www.baseball-reference.com{previous_season}"
    
    for team_url in team_urls:
        team_name = team_url.split("/")[4]
        data = requests.get(team_url)
        team_data = pd.read_html(data.text)[0]
        team_data["Season"] = year
        all_matches.append(team_data)
        time.sleep(1)
        
# Define DataFraem
match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]

# Delete unwanted columns
match_df.drop(match_df.columns[2], axis=1, inplace=True)
match_df.drop(match_df.columns[8], axis=1, inplace=True)
match_df.drop(match_df.columns[17], axis=1, inplace=True)
match_df.drop(match_df.columns[18], axis=1, inplace=True)

# Rename columns
match_df.rename(columns = {'gm#':'Game #', 'date':'Date', 'tm':'Team', 'unnamed: 4':'H/A', 'opp':'Opp', 'w/l':'W/L', 'r':'RS', 'ra':'RA', 'w-l':'W-L', 'rank':'Rank', 'gb':'GB', 'win':'Win', 'loss':'Loss', 'save':'Save', 'time':'Time', 'd/n':'D/N', 'attendance':'Attendance', 'streak':'Streak', 'season':'Season'}, inplace = True)

# Output Initial Data
match_df.to_csv("D:/GitHub/Fantasy_Sports_Optimizers/SportsOutcomePredictor/MLB/training_data/raw_mlb_games.csv",index=False)

# Read Initial Data
dfGames = pd.read_csv("D:/GitHub/Fantasy_Sports_Optimizers/SportsOutcomePredictor/MLB/training_data/raw_mlb_games.csv")

#Delete unwanted rows
rows = len(dfGames.index)

for row in range(rows):
    try:
        if dfGames.iat[row,0] == 'Gm#':
            dfGames.drop(dfGames.index[row], axis=0, inplace=True)            
    except:
        continue

#Change W/L column to exlude walk off wins
dfGames.replace(to_replace ="W-wo",value ="W",inplace=True)
dfGames.replace(to_replace ="L-wo",value ="L",inplace=True)

# Further Organize the Data
dfGames["Team"].value_counts()
dfGames["bResult"] = (dfGames["W/L"] == "W").astype("int")
dfGames["venue_code"] = dfGames["H/A"].astype("category").cat.codes
dfGames["venue_code"] = dfGames["venue_code"].astype("category").cat.codes
dfGames["opp_code"] = dfGames["Opp"].astype("category").cat.codes
dfGames["team_code"] = dfGames["Team"].astype("category").cat.codes
dfGames["TOD"] = dfGames["D/N"].astype("category").cat.codes

dfGames.to_csv("D:/GitHub/Fantasy_Sports_Optimizers/SportsOutcomePredictor/MLB/training_data/mlb_games_final.csv",index=False)
print("******Finished Scraping******")
