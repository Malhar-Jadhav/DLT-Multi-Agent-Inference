"""Download the SRDD requirements table from the ChatDev repository (chatdev1.0 branch).

The file has 1200 rows with Name, Description and Category columns and is saved as
data/raw/chatdev_srdd/srdd.csv.

Usage:
    python -m data.chatdev_srdd.download
"""

import urllib.request
from pathlib import Path

from data.dataset_config import raw_dir

SRDD_URL = "https://raw.githubusercontent.com/OpenBMB/ChatDev/chatdev1.0/SRDD/data/data_attribute_format.csv"


def download(out_dir: Path) -> None:
    """Fetch the SRDD CSV over HTTPS and write it to out_dir/srdd.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "srdd.csv"
    with urllib.request.urlopen(SRDD_URL, timeout=60) as response:
        target.write_bytes(response.read())
    print(f"SRDD downloaded to {target}")


if __name__ == "__main__":
    download(raw_dir("chatdev_srdd"))
