import requests
from bs4 import BeautifulSoup
import random
import time
import csv

output_path = "2024_boxscore_urls.txt"

# Base URL structure
base_url = "https://www.pro-football-reference.com/years/{}/week_{}.htm"

# Years and weeks to loop through
start_year = 2024
end_year = 2024
max_week = 6

# Function to read proxies from a CSV file
def load_proxies_from_csv(file_path):
    proxies = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Each row is a single string like 'proxy_address:port:user:password'
            proxy_parts = row[0].split(':')
            if len(proxy_parts) == 4:
                address, port, user, password = proxy_parts
                proxy_url = f"http://{user}:{password}@{address}:{port}"
                proxies.append({
                    'http': proxy_url,
                    'https': proxy_url
                })
    return proxies

# Load proxies from the CSV file
proxies_list = load_proxies_from_csv('proxies/proxies.csv')

# Function to get a random proxy from the list
def get_random_proxy():
    return random.choice(proxies_list)

# Function to extract boxscores from a specific URL using a proxy
def get_boxscores(year, week):
    url = base_url.format(year, week)
    proxy = get_random_proxy()  # Get a random proxy

    try:
        # Send request using a proxy
        response = requests.get(url, proxies=proxy, timeout=5)
        
        # Check if the page was successfully fetched
        if response.status_code != 200:
            print(f"Failed to fetch {url} with proxy {proxy['http']}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        boxscores = []
        
        # Find all td elements with class 'right gamelink'
        for gamelink in soup.find_all('td', class_='right gamelink'):
            a_tag = gamelink.find('a')
            if a_tag and 'href' in a_tag.attrs:
                boxscore_url = "https://www.pro-football-reference.com" + a_tag['href']
                boxscores.append(boxscore_url)

        return boxscores

    except Exception as e:
        print(f"Error fetching {url} with proxy {proxy['http']}: {e}")
        return []

# Main loop to gather boxscores
all_boxscores = []

for year in range(start_year, end_year + 1):
    # Determine the range of weeks to scrape based on the year
    max_weeks = max_week if year == end_year else 17  # Typically 17 weeks per year
    for week in range(1, max_weeks + 1):
        print(f"Scraping year {year}, week {week}...")
        print(f"Total boxscores scraped so far: {len(all_boxscores)}")
        boxscores = get_boxscores(year, week)
        print(boxscores)
        all_boxscores.extend(boxscores)
        time.sleep(1)  # Add a small delay to avoid overwhelming the server

# Output the gathered boxscores
for boxscore in all_boxscores:
    print(boxscore)

# Optionally save the results to a file
with open(output_path, 'w') as f:
    for boxscore in all_boxscores:
        f.write(boxscore + '\n')

print(f"Scraped {len(all_boxscores)} boxscores.")
