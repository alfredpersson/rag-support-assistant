import json
import os
from datetime import date

RATE_LIMIT_FILE = os.path.join(os.path.dirname(__file__), "..", "rate_limit.json")
DAILY_LIMIT = 100


class RateLimitExceeded(Exception):
    pass


def _load() -> dict:
    if not os.path.exists(RATE_LIMIT_FILE):
        return {}
    with open(RATE_LIMIT_FILE) as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(RATE_LIMIT_FILE, "w") as f:
        json.dump(data, f)


# NOTE: read-check-write is not atomic — concurrent requests can both pass
# the check before either writes. Fine for a single-process demo; production
# would use Redis INCR + EXPIRE (see ARCHITECTURE.md §15).
def check_and_increment() -> None:
    if os.environ.get("SKIP_RATE_LIMIT") == "true":
        return
    today = date.today().isoformat()
    data = _load()
    count = data.get(today, 0)
    if count >= DAILY_LIMIT:
        raise RateLimitExceeded(f"Daily limit of {DAILY_LIMIT} requests reached.")
    data[today] = count + 1
    _save(data)
