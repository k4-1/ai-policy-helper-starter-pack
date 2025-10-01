import pytest
import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure the app module can be imported from the container environment
current_dir = Path(__file__).parent
app_dir = current_dir.parent.parent
sys.path.insert(0, str(app_dir))

from app.main import app

@pytest.fixture(scope="session")
def client():
    return TestClient(app)
