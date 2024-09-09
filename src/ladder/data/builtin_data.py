#######################################
### Functions to download test data ###
#######################################

import os, requests
from typing import Literal
from tqdm import tqdm

# Static data paths, update when necessary
DATA_PATHS = {
    "Vu": "https://drive.google.com/uc?export=download&id=1quCP3403hOPG5Q8cy1KWZui_a0mJrQ5J",
    "Ji": "https://drive.google.com/uc?export=download&id=1QVjRyZmMArI0ex0NvpJcAvTxvt8JgbwA",
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

    ## Unexpected, means download link broke
    else:
        raise Exception(f"Object not found at URL for {dataset}")


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

    # Get and process response
    _download_data(response, save_path, smoke_test)


def get_params(
    params: Literal["SCVI", "SCANVI", "Patches"],
    save_path: str = "./params/",
    smoke_test: bool = False,
):
    """
    Used to fetch the parameters for reproducibility in case retraining is not desired.
    """

    assert params in PARAM_PATHS.keys(), f"No link found for {params}"

    # Reorganize param paths
    save_path_torch, save_path_pyro = (
        save_path + f"{params}_reprod_torch.pth",
        save_path + f"{params}_reprod_pyro.pth",
    )

    # Get requests
    response_torch, response_pyro = requests.get(
        PARAM_PATHS[params][0], allow_redirects=True
    ), requests.get(PARAM_PATHS[params][1], allow_redirects=True)

    # Download the data
    _download_data(response_torch, save_path_torch, smoke_test)
    _download_data(response_pyro, save_path_pyro, smoke_test)
