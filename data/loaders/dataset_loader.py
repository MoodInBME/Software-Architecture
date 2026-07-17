import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Token Kaggle
os.environ["KAGGLE_API_TOKEN"] = "***REMOVED***"

api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    "malekdinarito/ubfc-rppg-dataset",
    path="dataset",
    unzip=True
)

print("Dataset berhasil diunduh.")
