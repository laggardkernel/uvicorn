import asyncio
import http
import logging
import re
import urllib
from typing import Callable

import httptools

from uvicorn.protocols.http.flow_control import (
    CLOSE_HEADER,
    HIGH_WATER_LIMIT,
    TRACE_LOG_LEVEL,
    FlowControl,
    service_unavailable,
)
from uvicorn.protocols.utils import (
    get_client_addr,
    get_local_addr,
    get_path_with_query_string,
    get_remote_addr,
    is_ssl,
)

# TODO(lk): Invalid matching?
HEADER_RE = re.compile(b'[\x00-\x1F\x7F()<>@,;:[]={} \t\\"]')
HEADER_VALUE_RE = re.compile(b"[\x00-\x1F\x7F]")


def _get_status_line(status_code):
    try:
        phrase = http.HTTPStatus(status_code).phrase.encode()
    except ValueError:
        phrase = b""
    return b"".join([b"HTTP/1.1 ", str(status_code).encode(), b" ", phrase, b"\r\n"])


STATUS_LINE = {
    status_code: _get_status_line(status_code) for status_code in range(100, 600)
}


class HttpToolsProtocol(asyncio.Protocol):
    def __init__(
        self, config, server_state, on_connection_lost: Callable = None, _loop=None
    ):
        if not config.loaded:
            config.load()

        self.config = config
        self.app = config.loaded_app
        self.on_connection_lost = on_connection_lost
        self.loop = _loop or asyncio.get_event_loop()
        self.logger = logging.getLogger("uvicorn.error")
        self.access_logger = logging.getLogger("uvicorn.access")
        self.access_log = self.access_logger.hasHandlers()
        # CO(lk): parse and bind onto current protocol instance
        self.parser = httptools.HttpRequestParser(self)
        self.ws_protocol_class = config.ws_protocol_class
        self.root_path = config.root_path
        self.limit_concurrency = config.limit_concurrency

        # Timeouts
        self.timeout_keep_alive_task = None
        # CO(lk): after a while call .timeout_keep_alive_handler(), to close transp
        self.timeout_keep_alive = config.timeout_keep_alive

        # Global state
        self.server_state = server_state
        # CO(lk): share conn, task with ServerState.
        #  cycle.run_asgi(), a req handling
        self.connections = server_state.connections
        self.tasks = server_state.tasks
        self.default_headers = server_state.default_headers

        # Per-connection state
        self.transport = None
        self.flow = None
        self.server = None
        self.client = None
        self.scheme = None
        # CO(lk): req waiting to be handled in one conn
        self.pipeline = []

        # Per-request state
        self.url = None
        self.scope = None
        self.headers = None
        self.expect_100_continue = False
        self.cycle = None

    # Protocol interface
    def connection_made(self, transport):
        # CO(lk): share conn to ServeState
        self.connections.add(self)

        self.transport = transport
        self.flow = FlowControl(transport)
        self.server = get_local_addr(transport)
        self.client = get_remote_addr(transport)
        self.scheme = "https" if is_ssl(transport) else "http"

        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = "%s:%d - " % tuple(self.client) if self.client else ""
            self.logger.log(TRACE_LOG_LEVEL, "%sConnection made", prefix)

    def connection_lost(self, exc):
        self.connections.discard(self)

        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = "%s:%d - " % tuple(self.client) if self.client else ""
            self.logger.log(TRACE_LOG_LEVEL, "%sConnection lost", prefix)

        if self.cycle and not self.cycle.response_complete:
            # NOTE(lk): notify inner cycle stop handling req
            self.cycle.disconnected = True
        if self.cycle is not None:
            # NOTE(lk): notify inner cycle stop waiting data in receive()
            self.cycle.message_event.set()
        if self.flow is not None:
            # NOTE(lk): resume trans writing, cause inner cycle may wait in send()
            self.flow.resume_writing()
        if exc is None:
            self.transport.close()

        if self.on_connection_lost is not None:
            # CO(lk): set future passed from http.handle_http()
            self.on_connection_lost()

    def eof_received(self):
        pass

    def _unset_keepalive_if_required(self):
        if self.timeout_keep_alive_task is not None:
            self.timeout_keep_alive_task.cancel()
            self.timeout_keep_alive_task = None

    def data_received(self, data):
        self._unset_keepalive_if_required()

        try:
            # TODO(lk): parsed into .scope and call callbacks on_*() methods
            self.parser.feed_data(data)
        except httptools.HttpParserError as exc:
            msg = "Invalid HTTP request received."
            self.logger.warning(msg, exc_info=exc)
            self.transport.close()
        except httptools.HttpParserUpgrade:
            self.handle_upgrade()

    def handle_upgrade(self):
        upgrade_value = None
        for name, value in self.headers:
            if name == b"upgrade":
                upgrade_value = value.lower()

        if upgrade_value != b"websocket" or self.ws_protocol_class is None:
            msg = "Unsupported upgrade request."
            self.logger.warning(msg)

            from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol

            if AutoWebSocketsProtocol is None:
                msg = "No supported WebSocket library detected. Please use 'pip install uvicorn[standard]', or install 'websockets' or 'wsproto' manually."  # noqa: E501
                self.logger.warning(msg)

            content = [STATUS_LINE[400]]
            for name, value in self.default_headers:
                content.extend([name, b": ", value, b"\r\n"])
            content.extend(
                [
                    b"content-type: text/plain; charset=utf-8\r\n",
                    b"content-length: " + str(len(msg)).encode("ascii") + b"\r\n",
                    b"connection: close\r\n",
                    b"\r\n",
                    msg.encode("ascii"),
                ]
            )
            self.transport.write(b"".join(content))
            self.transport.close()
            return

        self.connections.discard(self)
        method = self.scope["method"].encode()
        output = [method, b" ", self.url, b" HTTP/1.1\r\n"]
        for name, value in self.scope["headers"]:
            output += [name, b": ", value, b"\r\n"]
        output.append(b"\r\n")
        protocol = self.ws_protocol_class(
            config=self.config,
            server_state=self.server_state,
            on_connection_lost=self.on_connection_lost,
        )
        # CO(lk): replace transp.protocol and start the new protocol
        protocol.connection_made(self.transport)
        protocol.data_received(b"".join(output))
        self.transport.set_protocol(protocol)

    # Parser callbacks
    def on_url(self, url):
        method = self.parser.get_method()
        parsed_url = httptools.parse_url(url)
        raw_path = parsed_url.path
        path = raw_path.decode("ascii")
        if "%" in path:
            path = urllib.parse.unquote(path)
        self.url = url
        self.expect_100_continue = False
        self.headers = []
        self.scope = {
            "type": "http",
            "asgi": {"version": self.config.asgi_version, "spec_version": "2.1"},
            "http_version": "1.1",
            "server": self.server,
            "client": self.client,
            "scheme": self.scheme,
            "method": method.decode("ascii"),
            "root_path": self.root_path,
            "path": path,
            "raw_path": raw_path,
            "query_string": parsed_url.query if parsed_url.query else b"",
            "headers": self.headers,
        }

    def on_header(self, name: bytes, value: bytes):
        name = name.lower()
        if name == b"expect" and value.lower() == b"100-continue":
            self.expect_100_continue = True
        self.headers.append((name, value))

    def on_headers_complete(self):
        http_version = self.parser.get_http_version()
        if http_version != "1.1":
            self.scope["http_version"] = http_version
        if self.parser.should_upgrade():
            return

        # Handle 503 responses when 'limit_concurrency' is exceeded.
        if self.limit_concurrency is not None and (
            len(self.connections) >= self.limit_concurrency
            or len(self.tasks) >= self.limit_concurrency
        ):
            app = service_unavailable
            message = "Exceeded concurrency limit."
            self.logger.warning(message)
        else:
            app = self.app

        existing_cycle = self.cycle
        self.cycle = RequestResponseCycle(
            scope=self.scope,
            transport=self.transport,
            flow=self.flow,
            logger=self.logger,
            access_logger=self.access_logger,
            access_log=self.access_log,
            default_headers=self.default_headers,
            message_event=asyncio.Event(),
            expect_100_continue=self.expect_100_continue,
            keep_alive=http_version != "1.0",
            on_response=self.on_response_complete,
        )
        if existing_cycle is None or existing_cycle.response_complete:
            # Standard case - start processing the request.
            task = self.loop.create_task(self.cycle.run_asgi(app))
            task.add_done_callback(self.tasks.discard)
            self.tasks.add(task)
        else:
            # CO(lk): use backpressure to stop accepting more requests
            # Pipelined HTTP requests need to be queued up.
            self.flow.pause_reading()
            self.pipeline.insert(0, (self.cycle, app))

    def on_body(self, body: bytes):
        if self.parser.should_upgrade() or self.cycle.response_complete:
            return
        self.cycle.body += body
        if len(self.cycle.body) > HIGH_WATER_LIMIT:
            self.flow.pause_reading()
        # CO(lk): .message_event: sync data received event between proto and req cycle
        self.cycle.message_event.set()

    def on_message_complete(self):
        if self.parser.should_upgrade() or self.cycle.response_complete:
            return
        self.cycle.more_body = False
        self.cycle.message_event.set()

    def on_response_complete(self):
        # Callback for pipelined HTTP requests to be started.
        self.server_state.total_requests += 1

        if self.transport.is_closing():
            return

        # Set a short Keep-Alive timeout.
        self._unset_keepalive_if_required()

        # CO(lk): close trasnp only after a while, which implements keepalive
        self.timeout_keep_alive_task = self.loop.call_later(
            self.timeout_keep_alive, self.timeout_keep_alive_handler
        )

        # Unpause data reads if needed.
        self.flow.resume_reading()

        # Unblock any pipelined events.
        if self.pipeline:
            cycle, app = self.pipeline.pop()
            task = self.loop.create_task(cycle.run_asgi(app))
            task.add_done_callback(self.tasks.discard)
            self.tasks.add(task)

    def shutdown(self):
        """
        Called by the server to commence a graceful shutdown.
        """
        if self.cycle is None or self.cycle.response_complete:
            self.transport.close()
        else:
            self.cycle.keep_alive = False

    def pause_writing(self):
        """
        Called by the transport when the write buffer exceeds the high water mark.
        """
        self.flow.pause_writing()

    def resume_writing(self):
        """
        Called by the transport when the write buffer drops below the low water mark.
        """
        self.flow.resume_writing()

    def timeout_keep_alive_handler(self):
        """
        Called on a keep-alive connection if no new data is received after a short
        delay.
        """
        if not self.transport.is_closing():
            self.transport.close()


class RequestResponseCycle:
    def __init__(
        self,
        scope,
        transport,
        flow,
        logger,
        access_logger,
        access_log,
        default_headers,
        message_event,
        expect_100_continue,
        keep_alive,
        on_response,
    ):
        self.scope = scope
        self.transport = transport
        self.flow = flow
        self.logger = logger
        self.access_logger = access_logger
        self.access_log = access_log
        self.default_headers = default_headers
        # CO(lk): .message_event: sync data received event between proto and req cycle
        #         on response complete func
        self.message_event = message_event
        self.on_response = on_response

        # Connection state
        # CO(lk): disconnected flag set by proto.CL()
        self.disconnected = False
        self.keep_alive = keep_alive
        self.waiting_for_100_continue = expect_100_continue

        # Request state
        self.body = b""
        self.more_body = True

        # Response state
        # CO(lk): if resp sent, completed?
        self.response_started = False
        self.response_complete = False
        self.chunked_encoding = None
        self.expected_content_length = 0

    # ASGI exception wrapper
    async def run_asgi(self, app):
        try:
            result = await app(self.scope, self.receive, self.send)
        except BaseException as exc:
            msg = "Exception in ASGI application\n"
            self.logger.error(msg, exc_info=exc)
            if not self.response_started:
                await self.send_500_response()
            else:
                self.transport.close()
        else:
            if result is not None:
                msg = "ASGI callable should return None, but returned '%s'."
                self.logger.error(msg, result)
                self.transport.close()
            elif not self.response_started and not self.disconnected:
                msg = "ASGI callable returned without starting response."
                self.logger.error(msg)
                await self.send_500_response()
            elif not self.response_complete and not self.disconnected:
                msg = "ASGI callable returned without completing response."
                self.logger.error(msg)
                self.transport.close()
        finally:
            # CO(lk): on_response() from proto: called in .send() and cleared here
            self.on_response = None

    async def send_500_response(self):
        await self.send(
            {
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"connection", b"close"),
                ],
            }
        )
        await self.send(
            {"type": "http.response.body", "body": b"Internal Server Error"}
        )

    # ASGI interface
    async def send(self, message):
        message_type = message["type"]

        if self.flow.write_paused and not self.disconnected:
            # CO(lk): wait for writing resuming
            await self.flow.drain()

        if self.disconnected:
            return

        if not self.response_started:
            # Sending response status line and headers
            if message_type != "http.response.start":
                msg = "Expected ASGI message 'http.response.start', but got '%s'."
                raise RuntimeError(msg % message_type)

            self.response_started = True
            # TODO(lk): "100 Continue" must be sent as 1st resp
            self.waiting_for_100_continue = False

            status_code = message["status"]
            headers = self.default_headers + list(message.get("headers", []))

            if CLOSE_HEADER in self.scope["headers"] and CLOSE_HEADER not in headers:
                headers = headers + [CLOSE_HEADER]

            if self.access_log:
                self.access_logger.info(
                    '%s - "%s %s HTTP/%s" %d',
                    get_client_addr(self.scope),
                    self.scope["method"],
                    get_path_with_query_string(self.scope),
                    self.scope["http_version"],
                    status_code,
                )

            # Write response status line and headers
            content = [STATUS_LINE[status_code]]

            for name, value in headers:
                if HEADER_RE.search(name):
                    raise RuntimeError("Invalid HTTP header name.")
                if HEADER_VALUE_RE.search(value):
                    raise RuntimeError("Invalid HTTP header value.")

                name = name.lower()
                if name == b"content-length" and self.chunked_encoding is None:
                    self.expected_content_length = int(value.decode())
                    self.chunked_encoding = False
                elif name == b"transfer-encoding" and value.lower() == b"chunked":
                    self.expected_content_length = 0
                    # CO(lk): chunk encoding will be used in later send()
                    self.chunked_encoding = True
                elif name == b"connection" and value.lower() == b"close":
                    self.keep_alive = False
                content.extend([name, b": ", value, b"\r\n"])

            if (
                self.chunked_encoding is None
                and self.scope["method"] != "HEAD"
                # CO(lk): if not no content, not modified
                and status_code not in (204, 304)
            ):
                # Neither content-length nor transfer-encoding specified
                self.chunked_encoding = True
                content.append(b"transfer-encoding: chunked\r\n")

            content.append(b"\r\n")
            self.transport.write(b"".join(content))

        elif not self.response_complete:
            # Sending response body
            if message_type != "http.response.body":
                msg = "Expected ASGI message 'http.response.body', but got '%s'."
                raise RuntimeError(msg % message_type)

            body = message.get("body", b"")
            more_body = message.get("more_body", False)

            # Write response body
            if self.scope["method"] == "HEAD":
                self.expected_content_length = 0
            elif self.chunked_encoding:
                if body:
                    content = [b"%x\r\n" % len(body), body, b"\r\n"]
                else:
                    content = []
                if not more_body:
                    content.append(b"0\r\n\r\n")
                self.transport.write(b"".join(content))
            else:
                num_bytes = len(body)
                if num_bytes > self.expected_content_length:
                    raise RuntimeError("Response content longer than Content-Length")
                else:
                    self.expected_content_length -= num_bytes
                self.transport.write(body)

            # Handle response completion
            if not more_body:
                if self.expected_content_length != 0:
                    raise RuntimeError("Response content shorter than Content-Length")
                self.response_complete = True
                # CO(lk): wakeup receive()
                self.message_event.set()
                if not self.keep_alive:
                    self.transport.close()
                self.on_response()

        else:
            # Response already sent
            msg = "Unexpected ASGI message '%s' sent, after response already completed."
            raise RuntimeError(msg % message_type)

    async def receive(self):
        if self.waiting_for_100_continue and not self.transport.is_closing():
            self.transport.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            self.waiting_for_100_continue = False

        if not self.disconnected and not self.response_complete:
            self.flow.resume_reading()
            # CO(lk): waiting data from proto/trasnp and passed into asgi app
            await self.message_event.wait()
            self.message_event.clear()

        if self.disconnected or self.response_complete:
            message = {"type": "http.disconnect"}
        else:
            message = {
                "type": "http.request",
                "body": self.body,
                "more_body": self.more_body,
            }
            self.body = b""

        return message
