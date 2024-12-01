import os

import pytest
from ultralytics import YOLO


def pytest_addoption(parser):
    """
    Add a custom command-line option for specifying the YOLO model location.

    Args:
        parser (pytest.Parser): The pytest parser object.
    """
    parser.addoption(
        "--yolo-loc",
        action="store",
        default=None,
        help="Path to YOLO directory"
    )


@pytest.fixture(scope="module")
def loaded_yolo_model(request):
    """
    Fixture to load the YOLO model from the specified directory.

    Args:
        request (pytest.FixtureRequest): The pytest fixture request object.

    Returns:
        YOLO: An instance of the YOLO model loaded from the specified path.
    """
    yolo_location = request.config.getoption("--yolo-loc")
    yolo_model_loc = os.path.join(yolo_location, "best.pt")
    return YOLO(yolo_model_loc)
