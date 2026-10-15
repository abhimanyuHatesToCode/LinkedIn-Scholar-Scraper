import json

# ── Stroing JSON data ───────────────────────────────────────────────────────────

def save_data(data):
    with open('scraped_data.json', 'w', encoding='utf-8') as f:
        json.dump(json.loads(data), f, indent=2)

# made this earlier in main.py but Classic circular import trap got me ;)