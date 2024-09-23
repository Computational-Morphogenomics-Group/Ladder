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
        print(response.__dict__)
        raise Exception("Object not found at URL")


def get_data(
    dataset: Literal["Vu", "Ji", "Mascharak"],
    save_path: str = "./data/",
    smoke_test: bool = False,
) -> None:
    """Used to download data for tutorials.

    Parameters
    ----------
    dataset : :class:`Literal["Vu", "Ji", "Mascharak"]`
        Specifies which dataset is to be downloaded.

    save_path : :class:`str`, default: "./data/"
        Specifies the directory in which the dataset will be saved. Defaults to `./data/`.

    smoke_test : :class:`bool`, default: False
        Used when testing to pass through without actually unpacking the response from server.

    Returns
    -------
    None
    """
    assert dataset in DATA_PATHS.keys(), f"No link found for {dataset}"

    # Reorganize param paths
    save_path = save_path + DATA_PATHS[dataset][1]

    # Send download request
    response = requests.get(DATA_PATHS[dataset][0], allow_redirects=True)

    # Get and process response
    _download_data(response, save_path, smoke_test)
