import requests
import pandas as pd
from bs4 import BeautifulSoup

def fetch_latest_data():
    # Example using mock API (you can replace with Sportsdata.io, The Odds API, ESPN scraping, etc.)
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    response = requests.get(url).json()

    games = []
    for event in response["events"]:
        home = event["competitions"][0]["competitors"][0]["team"]["displayName"]
        away = event["competitions"][0]["competitors"][1]["team"]["displayName"]
        spread = -3.5  # placeholder (should fetch from odds API)

        games.append({"game": f"{away} vs {home}", "home": home, "away": away, "spread_favorite": spread})

    return pd.DataFrame(games)
