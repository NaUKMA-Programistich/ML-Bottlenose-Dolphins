import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dataset-loc", action="store", default=None, help="Path to dataset file"
    )


@pytest.fixture(scope="module")
def fins_dataset_path(request):
    return request.config.getoption("--dataset-loc")
