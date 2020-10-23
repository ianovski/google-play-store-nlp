""" Extract reviews from their source - the Google Play Store"""

from google_play_scraper import Sort, reviews_all, reviews
import pandas as pd
import json
import sys

# result = reviews_all(
#     sys.argv[1],
#     sleep_milliseconds=10, # defaults to 0
#     lang='en', # defaults to 'en'
#     country='us', # defaults to 'us'
#     sort=Sort.NEWEST # defaults to Sort.MOST_RELEVANT
# )

result, continuation_token = reviews(
    sys.argv[1],
    lang='en', # defaults to 'en'
    country='us', # defaults to 'us'
    sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT
    count=3, # defaults to 100
    filter_score_with=5 # defaults to None(means all score)
)

data = pd.DataFrame.from_dict(result)

data.to
