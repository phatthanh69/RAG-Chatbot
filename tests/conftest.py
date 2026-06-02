import os

import pytest

# Force the in-repo Testing config and avoid real external calls at import time.
os.environ.setdefault("FLASK_ENV", "testing")
os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-used")

from tests._entry import create_app  # noqa: E402


@pytest.fixture(scope="session")
def app():
    application = create_app()
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()
