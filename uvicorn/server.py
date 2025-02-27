import asyncio
import logging
import os
import platform
import signal
import socket
import sys
import threading
import time
from email.utils import formatdate
from typing import List

import click

from ._handlers.http import handle_http

HANDLED_SIGNALS = (
    signal.SIGINT,  # Unix signal 2. Sent by Ctrl+C.
    signal.SIGTERM,  # Unix signal 15. Sent by `kill <pid>`.
)

logger = logging.getLogger("uvicorn.error")


class ServerState:
    """
    Shared servers state that is available between all protocol instances.
    """

    def __init__(self):
        self.total_requests = 0
        self.connections = set()
        self.tasks = set()
        self.default_headers = []


class Server:
    def __init__(self, config):
        self.config = config
        self.server_state = ServerState()

        self.started = False
        self.should_exit = False
        self.force_exit = False
        self.last_notified = 0

    def run(self, sockets=None):
        self.config.setup_event_loop()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.serve(sockets=sockets))

    async def serve(self, sockets=None):
        process_id = os.getpid()

        config = self.config
        if not config.loaded:
            config.load()

        self.lifespan = config.lifespan_class(config)

        self.install_signal_handlers()

        message = "Started server process [%d]"
        color_message = "Started server process [" + click.style("%d", fg="cyan") + "]"
        logger.info(message, process_id, extra={"color_message": color_message})

        await self.startup(sockets=sockets)
        if self.should_exit:
            return
        await self.main_loop()
        await self.shutdown(sockets=sockets)

        message = "Finished server process [%d]"
        color_message = "Finished server process [" + click.style("%d", fg="cyan") + "]"
        logger.info(
            "Finished server process [%d]",
            process_id,
            extra={"color_message": color_message},
        )

    async def startup(self, sockets: list = None) -> None:
        await self.lifespan.startup()
        if self.lifespan.should_exit:
            self.should_exit = True
            return

        config = self.config

        async def handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await handle_http(
                reader, writer, server_state=self.server_state, config=config
            )

        if sockets is not None:
            # Explicitly passed a list of open sockets.
            # We use this when the server is run from a Gunicorn worker.

            def _share_socket(sock: socket.SocketType) -> socket.SocketType:
                # Windows requires the socket be explicitly shared across
                # multiple workers (processes).
                from socket import fromshare  # type: ignore

                sock_data = sock.share(os.getpid())  # type: ignore
                return fromshare(sock_data)

            self.servers = []
            for sock in sockets:
                if config.workers > 1 and platform.system() == "Windows":
                    sock = _share_socket(sock)
                server = await asyncio.start_server(
                    handler, sock=sock, ssl=config.ssl, backlog=config.backlog
                )
                self.servers.append(server)
            listeners = sockets

        elif config.fd is not None:
            # Use an existing socket, from a file descriptor.
            sock = socket.fromfd(config.fd, socket.AF_UNIX, socket.SOCK_STREAM)
            server = await asyncio.start_server(
                handler, sock=sock, ssl=config.ssl, backlog=config.backlog
            )
            assert server.sockets is not None  # mypy
            listeners = server.sockets
            self.servers = [server]

        elif config.uds is not None:
            # Create a socket using UNIX domain socket.
            uds_perms = 0o666
            if os.path.exists(config.uds):
                uds_perms = os.stat(config.uds).st_mode
            server = await asyncio.start_unix_server(
                handler, path=config.uds, ssl=config.ssl, backlog=config.backlog
            )
            os.chmod(config.uds, uds_perms)
            assert server.sockets is not None  # mypy
            listeners = server.sockets
            self.servers = [server]

        else:
            # Standard case. Create a socket from a host/port pair.
            try:
                server = await asyncio.start_server(
                    handler,
                    host=config.host,
                    port=config.port,
                    ssl=config.ssl,
                    backlog=config.backlog,
                )
            except OSError as exc:
                logger.error(exc)
                await self.lifespan.shutdown()
                sys.exit(1)

            assert server.sockets is not None
            listeners = server.sockets
            self.servers = [server]

        if sockets is None:
            self._log_started_message(listeners)
        else:
            # We're most likely running multiple workers, so a message has already been
            # logged by `config.bind_socket()`.
            pass

        self.started = True

    def _log_started_message(self, listeners: List[socket.SocketType]) -> None:
        config = self.config

        if config.fd is not None:
            sock = listeners[0]
            logger.info(
                "Uvicorn running on socket %s (Press CTRL+C to quit)",
                sock.getsockname(),
            )

        elif config.uds is not None:
            logger.info(
                "Uvicorn running on unix socket %s (Press CTRL+C to quit)", config.uds
            )

        else:
            addr_format = "%s://%s:%d"
            host = "0.0.0.0" if config.host is None else config.host
            if ":" in host:
                # It's an IPv6 address.
                addr_format = "%s://[%s]:%d"

            port = config.port
            if port == 0:
                port = listeners[0].getsockname()[1]

            protocol_name = "https" if config.ssl else "http"
            message = f"Uvicorn running on {addr_format} (Press CTRL+C to quit)"
            color_message = (
                "Uvicorn running on "
                + click.style(addr_format, bold=True)
                + " (Press CTRL+C to quit)"
            )
            logger.info(
                message,
                protocol_name,
                host,
                port,
                extra={"color_message": color_message},
            )

    async def main_loop(self):
        counter = 0
        should_exit = await self.on_tick(counter)
        while not should_exit:
            counter += 1
            counter = counter % 864000
            await asyncio.sleep(0.1)
            should_exit = await self.on_tick(counter)

    async def on_tick(self, counter) -> bool:
        # Update the default headers, once per second.
        if counter % 10 == 0:
            current_time = time.time()
            current_date = formatdate(current_time, usegmt=True).encode()

            if self.config.date_header:
                date_header = [(b"date", current_date)]
            else:
                date_header = []

            self.server_state.default_headers = (
                date_header + self.config.encoded_headers
            )

            # Callback to `callback_notify` once every `timeout_notify` seconds.
            if self.config.callback_notify is not None:
                if current_time - self.last_notified > self.config.timeout_notify:
                    self.last_notified = current_time
                    await self.config.callback_notify()

        # Determine if we should exit.
        if self.should_exit:
            return True
        if self.config.limit_max_requests is not None:
            return self.server_state.total_requests >= self.config.limit_max_requests
        return False

    async def shutdown(self, sockets=None):
        logger.info("Shutting down")

        # Stop accepting new connections.
        for server in self.servers:
            server.close()
        for sock in sockets or []:
            sock.close()
        for server in self.servers:
            await server.wait_closed()

        # Request shutdown on all existing connections.
        for connection in list(self.server_state.connections):
            connection.shutdown()
        await asyncio.sleep(0.1)

        # Wait for existing connections to finish sending responses.
        if self.server_state.connections and not self.force_exit:
            msg = "Waiting for connections to close. (CTRL+C to force quit)"
            logger.info(msg)
            while self.server_state.connections and not self.force_exit:
                await asyncio.sleep(0.1)

        # Wait for existing tasks to complete.
        if self.server_state.tasks and not self.force_exit:
            msg = "Waiting for background tasks to complete. (CTRL+C to force quit)"
            logger.info(msg)
            while self.server_state.tasks and not self.force_exit:
                await asyncio.sleep(0.1)

        # Send the lifespan shutdown event, and wait for application shutdown.
        if not self.force_exit:
            await self.lifespan.shutdown()

    def install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            # Signals can only be listened to from the main thread.
            return

        loop = asyncio.get_event_loop()

        try:
            for sig in HANDLED_SIGNALS:
                loop.add_signal_handler(sig, self.handle_exit, sig, None)
        except NotImplementedError:  # pragma: no cover
            # Windows
            for sig in HANDLED_SIGNALS:
                signal.signal(sig, self.handle_exit)

    def handle_exit(self, sig, frame):
        if self.should_exit:
            self.force_exit = True
        else:
            self.should_exit = True
