import pytest


def pytest_addoption(parser):
    """Add custom command-line option for pytest."""
    parser.addoption(
        "--dataset-loc",
        action="store",
        default=None,
        help="Path to dataset file"
    )


@pytest.fixture(scope="module")
def fins_dataset_path(request):
    """Fixture to retrieve the dataset location from pytest options."""
    return request.config.getoption("--dataset-loc")
