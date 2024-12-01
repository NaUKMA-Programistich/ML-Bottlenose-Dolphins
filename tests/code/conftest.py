import os
import pytest
from ultralytics import YOLO


def pytest_addoption(parser):
    parser.addoption(
        "--yolo-loc", action="store", default=None, help="Path to yolo directory"
    )


@pytest.fixture(scope="module")
def loaded_yolo_model(request):
    yolo_location = request.config.getoption("--yolo-loc")
    yolo_model_loc = os.path.join(yolo_location, "best.pt")
    return YOLO(yolo_model_loc)
