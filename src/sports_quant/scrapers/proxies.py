"""Shared proxy loading utilities."""

import csv


def load_proxies_from_csv(file_path: str, format: str = "requests") -> list:
    """Read proxy configurations from a CSV file.

    Args:
        file_path: Path to the CSV file containing proxy configurations.
            Each row should be a single string like 'proxy_address:port:user:password'.
        format: Output format. "requests" returns dicts suitable for the requests
            library. "selenium" returns (url, auth) tuples for Selenium Wire.

    Returns:
        List of proxy configurations in the requested format.
    """
    proxies = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            proxy_parts = row[0].split(":")
            if len(proxy_parts) == 4:
                address, port, user, password = proxy_parts
                if format == "selenium":
                    proxy_url = f"{address}:{port}"
                    proxy_auth = f"{user}:{password}"
                    proxies.append((proxy_url, proxy_auth))
                else:
                    proxy_url = f"http://{user}:{password}@{address}:{port}"
                    proxies.append({"http": proxy_url, "https": proxy_url})
    return proxies
