import json
import time
from huggingface_hub import HfApi
from datetime import datetime
import os

STATE_FILE = os.path.join(os.path.expanduser("~"), ".hf_monitor_state.json")


def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def get_user_assets(username):
    api = HfApi()
    models = api.list_models(author=username, full=True)
    datasets = api.list_datasets(author=username, full=True)
    spaces = api.list_spaces(author=username, full=True)

    def to_dict(items):
        return {
            item.id: item.lastModified.isoformat() if item.lastModified else None
            for item in items
        }

    return {
        "models": to_dict(models),
        "datasets": to_dict(datasets),
        "spaces": to_dict(spaces),
    }

def check_changes(username, old, new):
    categories = ["models", "datasets", "spaces"]
    changes = []

    for cat in categories:
        old_items = old.get(cat, {})
        new_items = new.get(cat, {})

        # Detect new uploads
        for item_id in new_items:
            if item_id not in old_items:
                changes.append(f"🆕 NEW {cat[:-1].upper()}: {item_id}")

        # Detect updates
        for item_id, new_time in new_items.items():
            old_time = old_items.get(item_id)
            if old_time and new_time != old_time:
                changes.append(f"✏️ UPDATED {cat[:-1].upper()}: {item_id}")

    return changes

def main():
    USERS = ["grose99111", "speechmaster"]  # ← Add users here
    CHECK_EVERY = 300  # seconds (1 hour)

    print("🔍 Hugging Face Monitor Started")
    print(f"Tracking: {', '.join(USERS)}")
    print(f"Checking every {CHECK_EVERY} sec")
    print("-" * 40)

    state = load_state()

    while True:
        for user in USERS:
            print(f"\n👤 Checking user: {user}")
            previous = state.get(user, {})
            current = get_user_assets(user)

            # Compare
            changes = check_changes(user, previous, current)

            # Print changes
            if changes:
                print("\n".join(changes))
            else:
                print("No changes found.")

            # Update state
            state[user] = current
            save_state(state)

        time.sleep(CHECK_EVERY)

if __name__ == "__main__":
    main()
