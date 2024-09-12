"""The builtin_data module houses the functions that are necessary for tutorials.

The data provided in this module are needed for specific tutorials, and
are a good place to start when learning the modules.
"""

import os
from typing import Literal

import requests
from tqdm import tqdm

# Static data paths, update when necessary
DATA_PATHS = {
    "Vu": [
        "https://drive.google.com/uc?export=download&id=1quCP3403hOPG5Q8cy1KWZui_a0mJrQ5J",
        "vu_2022_ay_wh.h5ad",
    ],
    "Ji": [
        "https://drive.google.com/uc?export=download&id=1QVjRyZmMArI0ex0NvpJcAvTxvt8JgbwA",
        "ji_2020_tumor_ct.h5ad",
    ],
    "Mascharak": [
        "https://drive.google.com/uc?export=download&id=1EEUefuOWAXo5pgSEgMTsfIGT2ik64pJN",
        "mascharak_2022_tn_wh.h5ad",
    ],
}

# Static parameter paths, update when necessary
PARAM_PATHS = {
    "SCVI": (
        "https://drive.google.com/uc?export=download&id=1p85P0o0aJ47tWJc7SoWycjWKBdqFOy9i",
        "https://drive.google.com/uc?export=download&id=1RRJA_UdRogJptyama3_-ViCWb1TwBzLI",
    ),
    "SCANVI": (
        "https://drive.google.com/uc?export=download&id=1YUZUtGqwQrreG-qUvgb3WGTj3ZB_WrAN",
        "https://drive.google.com/uc?export=download&id=1d2m1bvN8hzAP4tL-GxecX8-bd1OBOWJq",
    ),
    "Patches": (
        "https://drive.google.com/uc?export=download&id=1Si2HyvArOKNh0efn2DLg2F7d6RjcY-4S",
        "https://drive.google.com/uc?export=download&id=1zX1K7oyBKF9ArQ7vUpqk1ehrO81a6rHO",
    ),
}


def _download_data(
    response: requests.Response,
    save_path: str,
    smoke_test: bool,
):
    ## Good
    if response.status_code == 200 and not smoke_test:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Write with progress bar
        with tqdm(
            total=int(response.headers.get("content-length", 0)),
            unit="B",
            unit_scale=True,
        ) as progress_bar:
            with open(save_path, "wb") as f:
                for data in response.iter_content(1024):
                    progress_bar.update(len(data))
                    f.write(data)

        print(f"Object saved at {save_path}")

    ## Catch test
    elif smoke_test:
        pass

    ## Unexpected
    else:
        raise Exception("Object not found at URL")


def get_data(
    dataset: Literal["Vu", "Ji", "Mascharak"],
    save_path: str = "./data/",
    smoke_test: bool = False,
):
    """Used to download data for tutorials.

    Parameters
    ----------
    dataset : {"Vu", "Ji", "Mascharak"}
        Specifies which dataset is to be downloaded.

    save_path : str, default: "./data/"
        Specifies the directory in which the dataset will be saved. Defaults to `./data/`.

    smoke_test : bool, default: False
        Used when testing to pass through without actually unpacking the response from server.
    """
    assert dataset in DATA_PATHS.keys(), f"No link found for {dataset}"

    # Reorganize param paths
    save_path = save_path + DATA_PATHS[dataset][1]

    # Send download request
    response = requests.get(DATA_PATHS[dataset][0], allow_redirects=True)

    # Get and process response
    _download_data(response, save_path, smoke_test)
