from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
import time
import csv

# Path to the boxscores URLs file
boxscores_file_path = "data/pfr/boxscores_urls.txt"

# Path to your WebDriver (update this if necessary)
webdriver_path = "/path/to/chromedriver"


# Function to read proxies from a CSV file
def load_proxies_from_csv(file_path):
    proxies = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            # Each row is a single string like 'proxy_address:port:user:password'
            proxy_parts = row[0].split(":")
            if len(proxy_parts) == 4:
                address, port, user, password = proxy_parts
                proxy_url = f"{address}:{port}"
                proxy_auth = f"{user}:{password}"
                proxies.append((proxy_url, proxy_auth))
    return proxies


# Load proxies from the CSV file
proxies_list = load_proxies_from_csv("proxies/proxies.csv")


# Function to get a random proxy from the list
def get_random_proxy():
    return random.choice(proxies_list)


# Function to configure WebDriver with proxy
def configure_driver_with_proxy(proxy_url, proxy_auth):
    proxy_user, proxy_pass = proxy_auth.split(":")

    # Define the proxy options with credentials
    proxy_options = {
        "proxy": {
            "http": f"http://{proxy_user}:{proxy_pass}@{proxy_url}",
            "https": f"https://{proxy_user}:{proxy_pass}@{proxy_url}",
            "no_proxy": "localhost,127.0.0.1",  # Exclude local addresses
        },
        "verify_ssl": False,  # Add this option to disable SSL verification
    }

    # Set Chrome options
    options = Options()

    # List of user-agents for random selection (desktop and mobile)
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A372 Safari/604.1",
    ]
    user_agent = random.choice(user_agents)

    # Set random user-agent
    options.add_argument(f"user-agent={user_agent}")

    # Disable automation flags and spoof properties
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("window-size=1280x800")
    options.add_argument("--disable-webgl")  # Disable WebGL (optional, for evasion)
    options.add_argument("--disable-gpu")  # Disable GPU rendering (optional)

    # Set up Chrome driver with proxy and user-agent
    driver = webdriver.Chrome(seleniumwire_options=proxy_options, options=options)

    # Remove navigator.webdriver property to avoid detection
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    return driver


# Function to auto-scroll the page
def auto_scroll(driver):
    scroll_pause_time = 1  # Time to wait between scrolls in seconds

    # Get the initial scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to the bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load the page
        time.sleep(scroll_pause_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break  # If the scroll height is the same, we have reached the bottom
        last_height = new_height


# Function to extract game info and header from a single URL using Selenium and a proxy
def scrape_game_info(url):
    full_url = f"{url}"  # Use the URL as is without a base URL
    proxy_url, proxy_auth = get_random_proxy()  # Get a random proxy

    driver = None
    try:
        # Start WebDriver with proxy
        driver = configure_driver_with_proxy(proxy_url, proxy_auth)
        driver.get(full_url)

        # Auto-scroll the page to ensure all content is loaded
        auto_scroll(driver)

        # Wait for the #content div to be visible
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "content"))
        )
        content_div = driver.find_element(By.ID, "content")

        # Wait for the first h1 within #content div to be visible
        h1_tag = WebDriverWait(content_div, 10).until(
            EC.visibility_of_element_located((By.TAG_NAME, "h1"))
        )
        game_title = h1_tag.text.strip() if h1_tag else "N/A"

        # Wait for the game info table to be visible
        game_info_table = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "game_info"))
        )

        # Extract the rows within the game_info_table
        rows = game_info_table.find_elements(By.TAG_NAME, "tr")

        game_info = {}

        # We are interested in these specific rows: Roof, Surface, Vegas Line, and Over/Under
        required_rows = ["Roof", "Surface", "Vegas Line", "Over/Under"]

        for row in rows:
            try:
                # Extract the <th> and <td> elements from each row
                info_key = row.find_element(By.TAG_NAME, "th").text.strip()  # Get the key
                info_value = row.find_element(By.TAG_NAME, "td").text.strip()  # Get the value

                # Print the key-value pair for debugging
                print(f"Key: {info_key}, Value: {info_value}")

                # Only store the data if it's one of the required rows
                if info_key in required_rows:
                    game_info[info_key] = info_value

            except Exception as e:
                print(f"Error extracting data from row: {e}")  # Continue on error

        # Return the extracted game info
        return {
            "title": game_title,
            "roof": game_info.get("Roof", "N/A"),
            "surface": game_info.get("Surface", "N/A"),
            "vegas_line": game_info.get("Vegas Line", "N/A"),
            "over_under": game_info.get("Over/Under", "N/A"),
        }

    except Exception as e:
        print(f"Error fetching {full_url} with proxy {proxy_url}: {e}")
        return None

    finally:
        if driver:
            driver.quit()  # Make sure to close the browser session


# New global variable to store failed URLs
failed_urls = []


# Modified scrape_all_game_info function with retry logic
def scrape_all_game_info():
    # Load all boxscores URLs
    with open(boxscores_file_path, "r") as f:
        boxscores_urls = [line.strip() for line in f.readlines()]

    # List to store all game data
    all_game_data = []
    max_retries = 5

    for url in boxscores_urls:
        print(f"Scraping URL: {url}")
        success = False
        retry_count = 0

        while not success and retry_count < max_retries:
            try:
                game_data = scrape_game_info(url)
                if game_data:
                    all_game_data.append(game_data)
                    success = True
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                retry_count += 1

            if not success:
                failed_urls.append(url)
            time.sleep(1)  # Add a small delay to avoid overwhelming the server
            
        # Save the results to a CSV file
        csv_file_path = "data/pfr/regular_game_data.csv"
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["Title", "Roof", "Surface", "Vegas Line", "Over/Under"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()  # Write the header row
            for game in all_game_data:
                writer.writerow(
                    {
                        "Title": game["title"],
                        "Roof": game["roof"],
                        "Surface": game["surface"],
                        "Vegas Line": game["vegas_line"],
                        "Over/Under": game["over_under"],
                    }
                )

    # Save failed URLs to a file for later retrying
    if failed_urls:
        with open("failed_urls.txt", "w") as f:
            f.write("\n".join(failed_urls))

    print(f"Scraped data saved")
    if failed_urls:
        print(f"Failed URLs saved to failed_urls.txt")


# Run the main function
if __name__ == "__main__":
    scrape_all_game_info()
