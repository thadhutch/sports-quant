"""Scrape PFF team grades from premium.pff.com."""

import logging
import time

import pandas as pd
from sports_quant.scrapers.auth import login_to_pff, navigate_and_sign_in
from sports_quant.teams import url_teams, encoded_teams, url_decoded_teams
from sports_quant import _config as config

logger = logging.getLogger(__name__)

"""

This section of code created unique game id's (dictionary key) for each game by home/away teams, game date, and season.
The value for each key is another dictionary of each key-value pair for team grade by category (key) and the numeric value.

NAME DISCREPENCIES
# Oakland Raiders moved Las Vegas in 2020
# Washington Redskins became Washington Football Team in 2020, and Commanders in 2022
# St. Louis Rams moved in 2016 to Los Angeles
# San Diego Chargers moved in 2017 to Los Angeles
"""


def scrape_pff_data() -> None:
    """Scrape PFF team schedule data for all teams across configured seasons."""
    driver = login_to_pff()

    time.sleep(5)

    games_dict = {}
    total_teams = len(url_teams)

    for szn_idx, szn in enumerate(config.SEASONS, 1):
        logger.info("=== Season %s (%d/%d) ===", szn, szn_idx, len(config.SEASONS))
        for team_idx, url_team in enumerate(url_teams, 1):

            if url_team == "washington-commanders":
                if szn in ["2020", "2021"]:
                    url_team = "washington-football-team"
                elif szn < "2020":
                    url_team = "washington-redskins"

            elif url_team == "las-vegas-raiders" and szn < "2020":
                url_team = "oakland-raiders"

            elif url_team == "los-angeles-rams" and szn < "2016":
                url_team = "st-louis-rams"

            elif url_team == "los-angeles-chargers" and szn < "2017":
                url_team = "san-diego-chargers"

            team = url_decoded_teams[url_team]
            logger.info("Scraping %s [%d/%d] (season %s)", team, team_idx, total_teams, szn)

            url = f"https://premium.pff.com/nfl/teams/{szn}/REGPO/{url_team}/schedule"
            navigate_and_sign_in(driver, url)
            time.sleep(5)

            home_stats = [
                "",
                "",
                "",
                "",
                "",
                "home-score",
                "",
                "",
                "home-off",
                "home-pass",
                "home-pblk",
                "home-recv",
                "home-run",
                "home-rblk",
                "home-def",
                "home-rdef",
                "home-tack",
                "home-prsh",
                "home-cov",
            ]
            away_stats = [
                "",
                "",
                "",
                "",
                "",
                "away-score",
                "",
                "",
                "away-off",
                "away-pass",
                "away-pblk",
                "away-recv",
                "away-run",
                "away-rblk",
                "away-def",
                "away-rdef",
                "away-tack",
                "away-prsh",
                "away-cov",
            ]

            rows = driver.find_elements("class name", "kyber-table-body__row")
            row_nums = range(
                1, 22
            )  # should be 1,22 for all possible games including playoffs
            stat_nums = range(1, 19)  # should be 1,20 to include special teams

            for row in row_nums:
                try:
                    # if away = '@', the current team is away. if empty, the current team is home
                    away = driver.find_element(
                        "xpath",
                        f'//*[@id="react-root"]/div/div[2]/div/div/div[3]/div/div/div[2]/div/div[1]/div/div[1]/div/div[{row}]/div[2]',
                    ).text
                except Exception as e:
                    logger.debug("Row %d not found, skipping: %s", row, e)
                    continue

                is_empty = driver.find_element(
                    "xpath",
                    f'//*[@id="react-root"]/div/div[2]/div/div/div[3]/div/div/div[2]/div/div[1]/div/div[2]/div/div[{row}]/div[4]',
                ).text
                if is_empty == "-" or is_empty == "":
                    break

                for stat in stat_nums:
                    try:
                        if stat == 1:
                            opp_team = driver.find_element(
                                "xpath",
                                f'//*[@id="react-root"]/div/div[2]/div/div/div[3]/div/div/div[2]/div/div[1]/div/div[2]/div/div[{row}]/div[{stat}]',
                            ).text
                            if away == "@":
                                game_id = (
                                    encoded_teams[team] + "-" + encoded_teams[opp_team]
                                )
                            else:
                                game_id = (
                                    encoded_teams[opp_team] + "-" + encoded_teams[team]
                                )
                        elif stat == 2:
                            game_date = driver.find_element(
                                "xpath",
                                f'//*[@id="react-root"]/div/div[2]/div/div/div[3]/div/div/div[2]/div/div[1]/div/div[2]/div/div[{row}]/div[{stat}]',
                            ).text
                            if game_date == "":
                                break

                            game_id = game_id + "-" + game_date + "/" + szn
                        elif stat in (3, 4, 6, 7):
                            continue
                        else:

                            if game_id not in games_dict:
                                games_dict[game_id] = {
                                    "home-score": "",
                                    "away-score": "",
                                    "home-off": "",
                                    "home-pass": "",
                                    "home-pblk": "",
                                    "home-recv": "",
                                    "home-run": "",
                                    "home-rblk": "",
                                    "away-off": "",
                                    "away-pass": "",
                                    "away-pblk": "",
                                    "away-recv": "",
                                    "away-run": "",
                                    "away-rblk": "",
                                    "home-def": "",
                                    "home-rdef": "",
                                    "home-tack": "",
                                    "home-prsh": "",
                                    "home-cov": "",
                                    "away-def": "",
                                    "away-rdef": "",
                                    "away-tack": "",
                                    "away-prsh": "",
                                    "away-cov": "",
                                }

                            current_cell = driver.find_element(
                                "xpath",
                                f'//*[@id="react-root"]/div/div[2]/div/div/div[3]/div/div/div[2]/div/div[1]/div/div[2]/div/div[{row}]/div[{stat}]',
                            ).text

                            if away == "@":
                                games_dict[game_id][away_stats[stat]] = current_cell
                            else:
                                games_dict[game_id][home_stats[stat]] = current_cell

                    except Exception as e:
                        logger.debug("Error scraping stat %d in row %d: %s", stat, row, e)
                        continue

    logger.info("Scraping complete. %d games collected.", len(games_dict))
    df = pd.DataFrame(games_dict)
    df = df.T
    df.to_csv(config.PFF_RAW_FILE)
    logger.info("PFF data saved to %s", config.PFF_RAW_FILE)
