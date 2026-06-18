import requests
import os
import time
from dotenv import load_dotenv
import pandas as pd

# Load full subscriber list and filter
# Compute CTOR
# Filter to subscribers who has received emails before
# Remove NAN values in CTOR column
df = pd.read_csv('active_subscriber_raw.csv')
df['ctor'] = (df['clicks_count'] / df['opens_count']) * 100
filtered_df = df[df['sent'] > 0].copy()
filtered_df = filtered_df[filtered_df['ctor'].notna()]

# ---------------------------------------------------------------
# Loading .env file
load_dotenv()
API_KEY = os.getenv("MAILERLITE_API_KEY")

# Set up headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# Main loop
# Take only the subscriber IDs list
all_ids = filtered_df['id'].tolist()
print(f"Total subscriber to fetch: {len(all_ids)}")

activity_log = []
counter = 0

for id in all_ids:

    page = 1
    while True:
        try:
            response = requests.get(
                url=f"https://connect.mailerlite.com/api/subscribers/{id}/activity-log",
                headers=headers,
                params={"limit":100, "page":page},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.ConnectionError:
            print(f"Connection Failed - Check your internet or the URL. Retryin in 30s...")
            time.sleep(30)
            continue

        except requests.exceptions.Timeout:
            print(f"Request timed out - subscriber {id}, page {page}. Retrying in 30s...")
            time.sleep(30)
            continue

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 400:
                print(f"Bad request. Check request again or retrying in 30s...")
                time.sleep(30)
                continue

            if status_code == 404:
                print(f"Subscriber {id} not found. Skipping.")
                break # move to the next subscriber
            
            if status_code == 408:
                print(f"Server timed out on subscriber {id}. Retrying in 30s...")
                time.sleep(30)
                continue

            elif status_code == 429:
                retry_after = int(e.response.headers.get('Retry-After'))
                print(f"Rate limited. Retrying in {retry_after}s...")
                time.sleep(retry_after)
                continue

            elif status_code >= 500:
                print(f"Server error {status_code} - subscriber {id}. Retrying in 30s...")
                time.sleep(30)
                continue

            else:
                print(f"HTTP {status_code} - subscriber{id}. Skipping.")
                break

        except requests.exceptions.RequestException as e:
            print("Error {e}. Unexpected error on subscriber {id}. Skipping")
            break
    
        # Attatch each activity log with subscriber ID before adding to list
        for entry in data['data']:
            entry['subscriber_id'] = id
        activity_log.extend(data['data'])

        # Pagination
        if len(data['data']) < 100:
            break
        page += 1

    time.sleep(0.5) # keep rate limit under 120 requests/min
    counter += 1

    # Progress update
    if counter % 100 == 0:
        print(f"Subscriber activity log fetched: {counter}/{len(all_ids)}")


all_activity_df = pd.DataFrame(activity_log)
all_activity_df.to_csv('all_activity_log.csv', index=False)
print(f"Done. {len(all_activity_df)} rows total.")