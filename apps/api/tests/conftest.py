import pytest
from fastapi.testclient import TestClient

from llm_lab.config import Settings
from llm_lab.main import create_app
from llm_lab.models.manager import ModelService


def pytest_addoption(parser):
    parser.addoption(
        "--real-model", action="store_true", help="Run actual pinned Qwen integration tests"
    )


@pytest.fixture
def unloaded_service():
    service = ModelService(Settings(auto_load=False))
    yield service
    service.close()


@pytest.fixture
def client(unloaded_service):
    with TestClient(create_app(Settings(auto_load=False), unloaded_service)) as client:
        yield client


@pytest.fixture(scope="session")
def real_service(request):
    if not request.config.getoption("--real-model"):
        pytest.skip("Actual Qwen test requires --real-model; no numerical mock replaces it")
    service = ModelService(Settings(auto_load=False))
    service.start_loading()
    service.wait_until_ready()
    yield service
    service.close()


@pytest.fixture(scope="session")
def real_result(real_service):
    return real_service.infer_for_validation("The capital of Germany is", 10).result(timeout=180)
