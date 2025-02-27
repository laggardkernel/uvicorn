import contextlib
import logging

import httpx
import pytest

from tests.utils import run_server
from uvicorn import Config


@contextlib.contextmanager
def caplog_for_logger(caplog, logger_name):
    logger = logging.getLogger(logger_name)
    if logger.propagate:
        logger.propagate = False
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


async def app(scope, receive, send):
    assert scope["type"] == "http"
    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b"", "more_body": False})


@pytest.mark.asyncio
async def test_trace_logging(caplog):
    config = Config(app=app, log_level="trace")
    with caplog_for_logger(caplog, "uvicorn.asgi"):
        async with run_server(config):
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:8000")
        assert response.status_code == 204
        messages = [
            record.message for record in caplog.records if record.name == "uvicorn.asgi"
        ]
        assert "ASGI [1] Started scope=" in messages.pop(0)
        assert "ASGI [1] Raised exception" in messages.pop(0)
        assert "ASGI [2] Started scope=" in messages.pop(0)
        assert "ASGI [2] Send " in messages.pop(0)
        assert "ASGI [2] Send " in messages.pop(0)
        assert "ASGI [2] Completed" in messages.pop(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_colors", [(True), (False), (None)])
async def test_access_logging(use_colors, caplog):
    config = Config(app=app, use_colors=use_colors)
    with caplog_for_logger(caplog, "uvicorn.access"):
        async with run_server(config):
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:8000")

        assert response.status_code == 204
        messages = [
            record.message
            for record in caplog.records
            if record.name == "uvicorn.access"
        ]
        assert '"GET / HTTP/1.1" 204' in messages.pop()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_colors", [(True), (False)])
async def test_default_logging(use_colors, caplog):
    config = Config(app=app, use_colors=use_colors)
    with caplog_for_logger(caplog, "uvicorn.access"):
        async with run_server(config):
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:8000")
        assert response.status_code == 204
        messages = [
            record.message for record in caplog.records if "uvicorn" in record.name
        ]
        assert "Started server process" in messages.pop(0)
        assert "Waiting for application startup" in messages.pop(0)
        assert "ASGI 'lifespan' protocol appears unsupported" in messages.pop(0)
        assert "Application startup complete" in messages.pop(0)
        assert "Uvicorn running on http://127.0.0.1:8000" in messages.pop(0)
        assert '"GET / HTTP/1.1" 204' in messages.pop(0)
        assert "Shutting down" in messages.pop(0)


@pytest.mark.asyncio
async def test_unknown_status_code(caplog):
    async def app(scope, receive, send):
        assert scope["type"] == "http"
        await send({"type": "http.response.start", "status": 599, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    config = Config(app=app)
    with caplog_for_logger(caplog, "uvicorn.access"):
        async with run_server(config):
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:8000")

        assert response.status_code == 599
        messages = [
            record.message
            for record in caplog.records
            if record.name == "uvicorn.access"
        ]
        assert '"GET / HTTP/1.1" 599' in messages.pop()
