"""Transport-security checks for the Model Hub API client."""

import pytest

from model_hub.api.client import HubAPIClient


@pytest.mark.parametrize("url", ["http://example.com", "http://127.0.0.1.evil"])
def test_api_key_is_rejected_for_remote_plain_http(url):
    with pytest.raises(ValueError, match="HTTPS"):
        HubAPIClient(base_url=url, api_key="test-api-key")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://[::1]:8000",
        "https://models.example.com",
    ],
)
def test_api_key_is_allowed_for_tls_or_loopback(url):
    client = HubAPIClient(base_url=url, api_key="test-api-key")
    assert client._api_key == "test-api-key"


def test_remote_plain_http_without_credentials_remains_available():
    client = HubAPIClient(base_url="http://models.example.com")
    assert client._base == "http://models.example.com"


@pytest.mark.parametrize("url", ["ftp://models.example.com", "not-a-url"])
def test_non_http_url_schemes_are_rejected(url):
    with pytest.raises(ValueError, match="HTTP"):
        HubAPIClient(base_url=url)
