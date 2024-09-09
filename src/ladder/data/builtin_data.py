#######################################
### Functions to download test data ###
#######################################

import os, requests
from typing import Literal
from tqdm import tqdm

# Static data paths, update when necessary
DATA_PATHS = {
    "Vu": " https://drive.google.com/uc?export=download&id=1quCP3403hOPG5Q8cy1KWZui_a0mJrQ5J",
    "Ji": " https://drive.google.com/uc?export=download&id=1QVjRyZmMArI0ex0NvpJcAvTxvt8JgbwA",
}


def get_data(
    dataset: Literal["Vu", "Ji"],
    save_path: str = "./data/vu_2022_ay_wh.h5ad",
    smoke_test: bool = False,
):
    """
    Used to fetch Vu (2022, 10.1016/j.celrep.2022.111155) data for tutorials.
    """

    assert dataset in DATA_PATHS.keys(), f"No link found for {dataset}"

    # Send download request
    response = requests.get(DATA_PATHS[dataset], allow_redirects=True)

    # Get response

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

        print(f"Dataset saved at {save_path}")

    ## Unexpected, means download link broke
    else:
        raise Exception(f"Dataset not found at URL for {dataset}")
