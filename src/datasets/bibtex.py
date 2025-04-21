


'''
Load youtube deepwalk multilabel classification dataset.
'''

import bz2
import io
import logging
import os

import requests
from sklearn.datasets import load_svmlight_file
from sklearn.utils import Bunch
from tqdm import tqdm

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BIBTEX_DATASET_URL = (
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/bibtex.bz2"
)

def load_youtube_dataset() -> Bunch:
    """
    Loads youtube deepwalk multilabel classification dataset.

    Returns:
        Bunch: Sklearn Bunch with xs as data and ys as labels.
    """
    extracted_path = "bibtex.svm"

    if not os.path.exists(extracted_path):
        logger.info("Youtube dataset not found, downloading and decompressing in memory...")

        response = requests.get(BIBTEX_DATASET_URL, stream=True, verify=False)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        compressed_data = io.BytesIO()

        with tqdm(
            desc="Downloading dataset",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                compressed_data.write(chunk)
                bar.update(len(chunk))

        compressed_data.seek(0)
        decompressed_data = bz2.decompress(compressed_data.read())

        with open(extracted_path, "wb") as f_out:
            f_out.write(decompressed_data)
        logger.info("Decompression complete.")
    else:
        logger.info("Found decompressed dataset.")

    logger.info("Loading dataset...")
    x, y = load_svmlight_file(extracted_path, multilabel=True)

    return Bunch(data=x, target=y)


if __name__ == "__main__":
    dataset = load_youtube_dataset()
    logger.info(f"Dataset loaded with shape: {dataset.data.shape}, {len(dataset.target)} samples.")

    for i, (x, y) in enumerate(zip(dataset.data, dataset.target)):
        logger.debug(f"Sample {i}:\nData: {x}\nLabels: {y}")
        if i > 0:
            break
