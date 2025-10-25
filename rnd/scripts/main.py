import pandas as pd
from nixtla import NixtlaClient


API_KEY = "nixak-BPWuiu0QLaDocnyGH7oFOutH821mnpHI5jFwgujKGyPGiLCAqNkQGUQ0vp11ZSOXX9msKcsCZgVM8cRu"
nixtla_client = NixtlaClient(
    api_key=API_KEY
)
isValid = nixtla_client.validate_api_key()

if isValid:
    print("API key is valid")
else:
    print("API key is invalid")


df = pd.read_csv(
    'https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv')
df.head()
