import pytest

from my_harness.ai import FauxProvider, register_provider


@pytest.fixture
def faux():
    provider = FauxProvider()
    register_provider("faux", provider)
    return provider
