import requests
import zipfile
import io
import os

def download_and_unzip_data():
    """Downloads and unzips the bike sharing dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    raw_data_path = "data/raw"

    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
        print(f"Created directory: {raw_data_path}")

    print("Downloading data...")
    response = requests.get(url)
    if response.status_code == 200:
        print("Download successful. Unzipping files...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(raw_data_path)
        print(f"Data unzipped to {raw_data_path}")
    else:
        print(f"Failed to download data. Status code: {response.status_code}")

if __name__ == "__main__":
    download_and_unzip_data()