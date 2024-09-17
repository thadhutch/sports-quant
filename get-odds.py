from sbrscrape import Scoreboard
import datetime

def get_nfl_odds(start_year: int):
    # Set the sport to NFL
    sport = "NFL"
    
    # Define the start and end dates
    start_date = datetime.datetime(start_year, 1, 1)
    end_date = datetime.datetime.today()
    
    # Initialize an empty list to store the odds data
    all_odds = []

    # Loop through each day from start_date to end_date
    current_date = start_date
    while current_date <= end_date:
        try:
            # Get the scoreboard for the given day
            scoreboard = Scoreboard(sport=sport, date=current_date.strftime("%Y-%m-%d"))
            games = scoreboard.games
            
            # Append the games' odds data to the list
            for game in games:
                all_odds.append({
                    'date': game['date'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'home_spread': game.get('home_spread', {}),
                    'away_spread': game.get('away_spread', {}),
                    'home_moneyline': game.get('home_ml', {}),
                    'away_moneyline': game.get('away_ml', {}),
                    'total': game.get('total', {})
                })

            print(all_odds)
        except Exception as e:
            print(f"Error fetching data for {current_date.strftime('%Y-%m-%d')}: {e}")
        
        # Move to the next day
        current_date += datetime.timedelta(days=1)
    
    return all_odds

if __name__ == "__main__":
    # Get NFL odds data from 2015 to today
    nfl_odds = get_nfl_odds(2015)
    
    # Print or save the data to a file
    for odds in nfl_odds:
        print(odds)
