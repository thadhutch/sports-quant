"""NFL team logo downloading and caching via ESPN CDN."""

import logging
import urllib.request
from pathlib import Path

from matplotlib.image import imread
from matplotlib.offsetbox import OffsetImage
from PIL import Image

from sports_quant import _config as config

logger = logging.getLogger(__name__)

ESPN_LOGO_ABBRS: dict[str, str] = {
    # Current 32 teams
    "Arizona Cardinals": "ari",
    "Atlanta Falcons": "atl",
    "Baltimore Ravens": "bal",
    "Buffalo Bills": "buf",
    "Carolina Panthers": "car",
    "Chicago Bears": "chi",
    "Cincinnati Bengals": "cin",
    "Cleveland Browns": "cle",
    "Dallas Cowboys": "dal",
    "Denver Broncos": "den",
    "Detroit Lions": "det",
    "Green Bay Packers": "gb",
    "Houston Texans": "hou",
    "Indianapolis Colts": "ind",
    "Jacksonville Jaguars": "jax",
    "Kansas City Chiefs": "kc",
    "Las Vegas Raiders": "lv",
    "Los Angeles Chargers": "lac",
    "Los Angeles Rams": "lar",
    "Miami Dolphins": "mia",
    "Minnesota Vikings": "min",
    "New England Patriots": "ne",
    "New Orleans Saints": "no",
    "New York Giants": "nyg",
    "New York Jets": "nyj",
    "Philadelphia Eagles": "phi",
    "Pittsburgh Steelers": "pit",
    "San Francisco 49ers": "sf",
    "Seattle Seahawks": "sea",
    "Tampa Bay Buccaneers": "tb",
    "Tennessee Titans": "ten",
    "Washington Commanders": "wsh",
    # Historical names → current franchise abbreviation
    "Oakland Raiders": "lv",
    "San Diego Chargers": "lac",
    "St. Louis Rams": "lar",
    "Washington Football Team": "wsh",
    "Washington Redskins": "wsh",
}

_ESPN_URL = "https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png"


def get_logo_path(team_name: str) -> Path:
    """Return the local cached path for a team logo, downloading if missing."""
    abbr = ESPN_LOGO_ABBRS[team_name]
    path = config.LOGOS_DIR / f"{abbr}.png"
    if not path.exists():
        config.LOGOS_DIR.mkdir(parents=True, exist_ok=True)
        url = _ESPN_URL.format(abbr=abbr)
        logger.info("Downloading logo for %s from %s", team_name, url)
        urllib.request.urlretrieve(url, path)
    return path


_LOGO_TARGET_PX = 30  # target height in pixels for chart logos


def get_logo_image(team_name: str) -> OffsetImage:
    """Return a matplotlib OffsetImage of the team logo, normalized to a fixed height."""
    path = get_logo_path(team_name)
    img = Image.open(path)
    # Resize so height == _LOGO_TARGET_PX, preserving aspect ratio
    w, h = img.size
    new_h = _LOGO_TARGET_PX
    new_w = int(w * new_h / h)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return OffsetImage(img, zoom=1.0)
