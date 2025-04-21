"""
Load multilabel classification datasets.
"""

import bz2
import io
import logging
import os

import requests
from sklearn.datasets import load_svmlight_file
from sklearn.utils import Bunch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

YOUTUBE_DATASET_URL = (
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/"
    "datasets/multilabel/youtube_deepwalk.svm.bz2"
)
BIBTEX_DATASET_URL = (
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/bibtex.bz2"
)


def load_dataset(url: str) -> Bunch:
    """
    Load a given multilabel dataset by fetching it from a given url.

    Args:
        url (str): URL to download the dataset from.

    Returns:
        Bunch: Sklearn Bunch containing data and labels.
    """
    extracted_path = url.split("/")[-1].split(".")[0] + ".svm"

    if not os.path.exists(extracted_path):
        logger.info("Dataset not found locally, downloading...")

        response = requests.get(url, stream=True, verify=False, timeout=10)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        compressed_data = io.BytesIO()

        with tqdm(
            desc="Downloading dataset",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                compressed_data.write(chunk)
                progress_bar.update(len(chunk))

        compressed_data.seek(0)
        decompressed_data = bz2.decompress(compressed_data.read())

        with open(extracted_path, "wb") as f_out:
            f_out.write(decompressed_data)

        logger.info("Decompression complete.")
    else:
        logger.info("Found decompressed dataset.")

    logger.info("Loading dataset...")

    # pylint: disable=unbalanced-tuple-unpacking
    x, y = load_svmlight_file(extracted_path, multilabel=True)

    return Bunch(data=x, target=y)


def load_youtube_dataset() -> Bunch:
    """
    Load youtube deepwalk multilabel dataset.

    Returns:
        Bunch: Sklearn Bunch containing dataset data and labels.
    """
    return load_dataset(YOUTUBE_DATASET_URL)


def load_bibtex_dataset() -> Bunch:
    """
    Load Bibtex multilabel dataset.

    Returns:
        Bunch: Sklearn Bunch containing dataset data and labels.
    """
    return load_dataset(BIBTEX_DATASET_URL)
