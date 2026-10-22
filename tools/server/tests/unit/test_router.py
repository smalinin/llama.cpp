import threading
import signal
import socket
import pytest
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


def test_router_props():
    global server
    server.models_max = 2
    server.no_models_autoload = True
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["role"] == "router"
    assert res.body["max_instances"] == 2
    assert res.body["models_autoload"] is False
    assert res.body["build_info"].startswith("b")


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals and process groups")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("state", ["idle", "sse", "loading", "loaded", "loaded_group", "busy", "queued", "detached", "group", "double", "expired"])
def test_router_shutdown(tmp_path, signum, state):
    model = os.environ.get("LLAMA_ROUTER_TEST_MODEL")
    if state not in ("idle", "sse", "expired") and not model:
        pytest.skip("set LLAMA_ROUTER_TEST_MODEL to a small local GGUF")
    preset = tmp_path / "models.ini"
    preset.write_text("" if not model else "\n".join(
        f"[{name}]\nmodel = {model}\nn-gpu-layers = 0\nfit = off\n"
        "ctx-size = 128\nparallel = 1\nthreads = 2\nno-warmup = true\nstop-timeout = 1\n"
        for name in ("first", "second")
    ))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    binary = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    env.pop("LLAMA_SERVER_DEBUG_FAKE_TIMING", None)
    requests_in_flight = []
    children = []
    sse = None
    with open(tmp_path / "server.log", "w") as log:
        proc = subprocess.Popen([
            binary, "--models-preset", str(preset), "--models-max", "1",
            "--host", "127.0.0.1", "--port", str(port), "--no-ui",
        ], stdout=log, stderr=log, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + 20
            while True:
                assert proc.poll() is None, "router exited before becoming ready"
                try:
                    if requests.get(url + "/health", timeout=0.2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, "router did not become ready"
                time.sleep(0.02)

            if state == "sse":
                sse = requests.get(url + "/models/sse", stream=True, timeout=5)
                assert sse.status_code == 200
            elif state not in ("idle", "expired"):
                res = requests.post(url + "/models/load", json={"model": "first"}, timeout=20)
                assert res.status_code == 200
                deadline = time.monotonic() + 20
                while state != "loading":
                    models = requests.get(url + "/models", timeout=2).json()["data"]
                    first = next(m for m in models if m["id"] == "first")
                    if first["status"]["value"] == "loaded":
                        break
                    assert time.monotonic() < deadline, "child did not load"
                    time.sleep(0.02)
                # Freeze the child so an active proxy cannot complete on its own.
                if not os.path.isdir(f"/proc/{proc.pid}/task/{proc.pid}"):
                    pytest.skip("child PID lookup needs Linux procfs")
                child_pids = set()
                for tid in os.listdir(f"/proc/{proc.pid}/task"):
                    try:
                        with open(f"/proc/{proc.pid}/task/{tid}/children") as f:
                            child_pids.update(int(pid) for pid in f.read().split())
                    except FileNotFoundError:
                        pass
                children = sorted(child_pids)
                assert len(children) == 1, children
                os.kill(children[0], signal.SIGSTOP)
                if state == "loading":
                    models = requests.get(url + "/models", timeout=2).json()["data"]
                    first = next(m for m in models if m["id"] == "first")
                    assert first["status"]["value"] == "loading"

                def request(model_name):
                    try:
                        if state.startswith("loaded"):
                            requests.get(url + "/props", params={"model": model_name}, timeout=15)
                            return
                        requests.post(url + "/tokenize", json={"model": model_name, "content": "hello"},
                                      headers={"X-Conversation-Id": "shutdown-test"} if state == "detached" else {},
                                      timeout=15)
                    except requests.RequestException:
                        pass

                request_thread = threading.Thread(target=request, args=("first",), daemon=True)
                requests_in_flight.append(request_thread)
                request_thread.start()
                time.sleep(0.2)
                assert request_thread.is_alive()
                if state == "queued":
                    queued = threading.Thread(target=request, args=("second",), daemon=True)
                    requests_in_flight.append(queued)
                    queued.start()
                    time.sleep(0.2)
                    assert queued.is_alive()
                if state.startswith("loaded"):
                    os.kill(children[0], signal.SIGCONT)

            if state in ("group", "loaded_group"):
                os.killpg(proc.pid, signum)
            else:
                proc.send_signal(signum)
            if signum == signal.SIGINT:
                time.sleep(0.1)
                assert proc.poll() is None, "one Ctrl+C stopped the router"
                assert requests.get(url + "/health", timeout=2).status_code == 200
                for child in children:
                    assert os.path.exists(f"/proc/{child}"), "one Ctrl+C stopped a child"
                if state == "expired":
                    time.sleep(2.1)
                    proc.send_signal(signum)
                    time.sleep(0.1)
                    assert proc.poll() is None, "expired Ctrl+C confirmation stopped the router"
                    assert requests.get(url + "/health", timeout=2).status_code == 200
                if state in ("group", "loaded_group"):
                    os.killpg(proc.pid, signum)
                else:
                    proc.send_signal(signum)
            if state == "double":
                time.sleep(0.1)
                proc.send_signal(signum)
                assert proc.wait(timeout=1) == 1
            else:
                assert proc.wait(timeout=6) == 0
                if state.startswith("loaded"):
                    assert "instance name=first exited with status 0" in (tmp_path / "server.log").read_text()
                for child in children:
                    assert not os.path.exists(f"/proc/{child}"), f"child {child} survived router shutdown"
                for thread in requests_in_flight:
                    thread.join(timeout=2)
                    assert not thread.is_alive(), "proxy request survived shutdown"
        finally:
            if sse is not None:
                sse.close()
            # Only the process group created by this test is eligible for forced cleanup.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals and FIFO")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_interrupt_during_model_loading(tmp_path, signum):
    model = tmp_path / "blocked.gguf"
    os.mkfifo(model)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    binary = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    with open(tmp_path / "server.log", "w") as log:
        proc = subprocess.Popen([
            binary, "--model", str(model), "--fit", "off", "--n-gpu-layers", "0",
            "--host", "127.0.0.1", "--port", str(port), "--no-ui", "--no-mmproj-auto",
        ], stdout=log, stderr=log, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, start_new_session=True)
        try:
            deadline = time.monotonic() + 20
            while True:
                assert proc.poll() is None, "server exited before loading"
                try:
                    if requests.get(url + "/health", timeout=0.2).status_code == 503:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, "server did not start loading"
                time.sleep(0.02)
            time.sleep(0.1)
            proc.send_signal(signum)
            if signum == signal.SIGINT:
                time.sleep(0.1)
                assert proc.poll() is None, "one Ctrl+C interrupted model I/O"
                assert requests.get(url + "/health", timeout=2).status_code == 503
                proc.send_signal(signum)
            assert proc.wait(timeout=3) == (130 if signum == signal.SIGINT else -signal.SIGTERM)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)


@pytest.mark.parametrize(
    "model,success",
    [
        ("ggml-org/tinygemma3-GGUF:Q8_0", True),
        ("non-existent/model", False),
    ]
)
def test_router_chat_completion_stream(model: str, success: bool):
    global server
    server.start()
    content = ""
    ex: ServerError | None = None
    try:
        res = server.make_stream_request("POST", "/chat/completions", data={
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "stream": True,
        })
        for data in res:
            if data["choices"]:
                choice = data["choices"][0]
                if choice["finish_reason"] in ["stop", "length"]:
                    assert "content" not in choice["delta"]
                else:
                    assert choice["finish_reason"] is None
                    content += choice["delta"]["content"] or ''
    except ServerError as e:
        ex = e

    if success:
        assert ex is None
        assert len(content) > 0
    else:
        assert ex is not None
        assert content == ""


def _get_model_ids(is_reload: bool, headers: dict | None = None) -> set[str]:
    res = server.make_request(
        "GET", "/models" + ("?reload=1" if is_reload else ""), headers=headers
    )
    assert res.status_code == 200
    return {item["id"] for item in res.body.get("data", [])}


def _get_model_status(model_id: str, headers: dict | None = None) -> str:
    res = server.make_request("GET", "/models", headers=headers)
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id or item.get("model") == model_id:
            return item["status"]["value"]
    raise AssertionError(f"Model {model_id} not found in /models response")


def _wait_for_model_status(model_id: str, desired: set[str], timeout: int = 60, headers: dict | None = None) -> str:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        last_status = _get_model_status(model_id, headers=headers)
        if last_status in desired:
            return last_status
        time.sleep(0.01)
    raise AssertionError(
        f"Timed out waiting for {model_id} to reach {desired}, last status: {last_status}"
    )


def _load_model_and_wait(
    model_id: str, timeout: int = 60, headers: dict | None = None
) -> None:
    load_res = server.make_request(
        "POST", "/models/load", data={"model": model_id}, headers=headers
    )
    assert load_res.status_code == 200
    assert isinstance(load_res.body, dict)
    assert load_res.body.get("success") is True
    _wait_for_model_status(model_id, {"loaded"}, timeout=timeout, headers=headers)


def test_router_unload_model():
    global server
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    _load_model_and_wait(model_id)

    unload_res = server.make_request("POST", "/models/unload", data={"model": model_id})
    assert unload_res.status_code == 200
    assert unload_res.body.get("success") is True
    _wait_for_model_status(model_id, {"unloaded"})


def test_router_models_max_evicts_lru():
    global server
    server.models_max = 2
    server.start()

    candidate_models = [
        "ggml-org/tinygemma3-GGUF:Q8_0",
        "ggml-org/test-model-stories260K:F32",
        "ggml-org/test-model-stories260K-infill:F32",
    ]

    # Load only the first 2 models to fill the cache
    first, second, third = candidate_models[:3]

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)

    # Verify both models are loaded
    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"

    # Load the third model - this should trigger LRU eviction of the first model
    _load_model_and_wait(third, timeout=120)

    # Verify eviction: third is loaded, first was evicted
    assert _get_model_status(third) == "loaded"
    assert _get_model_status(first) == "unloaded"


# server_lru_sched tests (relying on LLAMA_SERVER_DEBUG_FAKE_TIMING)

MODEL_A = "ggml-org/tinygemma3-GGUF:Q8_0"
MODEL_B = "ggml-org/test-model-stories260K:F32"
MODEL_C = "ggml-org/test-model-stories260K-infill:F32"


def _tokenize(model_id: str, timeout: float | None = DEFAULT_REQUEST_TIMEOUT) -> ServerResponse:
    return server.make_request(
        "POST", "/tokenize", data={"model": model_id, "content": "hello world"}, timeout=timeout
    )


class _Bg:
    """runs one request in a thread, keeps its result, error and finish time"""

    def __init__(self, fn):
        self.result = None
        self.error: Exception | None = None
        self.done_at: float = 0.0
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)

    def _run(self, fn):
        try:
            self.result = fn()
        except Exception as e:
            self.error = e
        self.done_at = time.time()

    def start(self):
        self._thread.start()
        return self

    def join(self, timeout: int = 180):
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "background request did not finish in time"
        return self

    def assert_ok(self, what: str):
        assert self.error is None, f"{what} raised {self.error!r}"
        assert self.result is not None and self.result.status_code == 200, \
            f"{what} failed: {self.result.status_code if self.result else None} {self.result.body if self.result else None}"


def test_router_queue_does_not_evict_busy_model():
    """a request that finds no free slot waits, and the model serving a request survives it"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)  # let the request reach the child and take the only slot

    # no slot free and MODEL_A is busy, so this queues instead of evicting mid-request
    queued = _Bg(lambda: _tokenize(MODEL_B)).start()

    busy.join()
    queued.join()

    # had MODEL_A been evicted while serving, its own request would have died
    busy.assert_ok("request against the busy model")
    queued.assert_ok("queued request")

    _wait_for_model_status(MODEL_B, {"loaded"}, timeout=120)
    assert _get_model_status(MODEL_A) == "unloaded"


def test_router_queue_coalesces_requests_for_same_model():
    """many requests for one missing model share a slot, so only one model is given up"""
    global server
    server.models_max = 2
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)
    _load_model_and_wait(MODEL_B, timeout=120)

    # keep MODEL_A busy so MODEL_B is the only model that can be given up
    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)

    waiters = [_Bg(lambda: _tokenize(MODEL_C)).start() for _ in range(3)]

    busy.join()
    for w in waiters:
        w.join()

    busy.assert_ok("request against the busy model")
    for i, w in enumerate(waiters):
        w.assert_ok(f"queued request {i}")

    _wait_for_model_status(MODEL_C, {"loaded"}, timeout=120)
    # one entry for 3 requests means one eviction: MODEL_B goes, MODEL_A is left alone.
    # without coalescing the leftover entries still ask for a slot,
    # and MODEL_A is taken too as soon as it goes idle
    assert _get_model_status(MODEL_A) == "loaded"
    assert _get_model_status(MODEL_B) == "unloaded"


def test_router_queue_client_disconnect_keeps_model():
    """a client that leaves while queued must not cost a running model its slot"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)

    # queues behind MODEL_A, then gives up long before MODEL_A goes idle
    with pytest.raises(requests.exceptions.RequestException):
        _tokenize(MODEL_B, timeout=1)

    busy.join()
    busy.assert_ok("request against the busy model")

    # nobody is waiting anymore, so MODEL_A keeps its slot
    time.sleep(3)
    assert _get_model_status(MODEL_A) == "loaded"
    assert _get_model_status(MODEL_B) == "unloaded"


def test_router_queue_is_fifo():
    """the queue is served in arrival order"""
    global server
    server.models_max = 1
    server.start()

    _load_model_and_wait(MODEL_A, timeout=120)

    busy = _Bg(lambda: _tokenize(MODEL_A)).start()
    time.sleep(0.5)

    first = _Bg(lambda: _tokenize(MODEL_B)).start()
    time.sleep(1)  # keep the arrival order unambiguous
    second = _Bg(lambda: _tokenize(MODEL_C)).start()

    busy.join()
    first.join()
    second.join()

    busy.assert_ok("request against the busy model")
    first.assert_ok("first queued request")
    second.assert_ok("second queued request")

    assert first.done_at < second.done_at, "queue was not served in arrival order"


def test_router_no_models_autoload():
    global server
    server.no_models_autoload = True
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 400
    assert "error" in res.body

    _load_model_and_wait(model_id)

    success_res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert success_res.status_code == 200
    assert "error" not in success_res.body


def test_router_api_key_required():
    global server
    server.api_key = "sk-router-secret"
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"
    auth_headers = {"Authorization": f"Bearer {server.api_key}"}

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 401
    assert res.body.get("error", {}).get("type") == "authentication_error"

    _load_model_and_wait(model_id, headers=auth_headers)

    authed = server.make_request(
        "POST",
        "/v1/chat/completions",
        headers=auth_headers,
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert authed.status_code == 200
    assert "error" not in authed.body


def test_router_reload_models():
    """POST /models/reload re-reads the INI preset and updates the model list."""
    global server

    preset_path = os.path.join(TMP_DIR, "test_reload.ini")

    # Initial preset: two models
    with open(preset_path, "w") as f:
        f.write(
            "[model-reload-a]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            "\n"
            "[model-reload-b]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
        )

    server.models_preset = preset_path
    server.start()

    ids = _get_model_ids(is_reload=False)
    assert "model-reload-a" in ids
    assert "model-reload-b" in ids

    # Updated preset: remove a, keep b unchanged, add c
    with open(preset_path, "w") as f:
        f.write(
            "[model-reload-b]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
            "\n"
            "[model-reload-c]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
        )

    try:
        ids = _get_model_ids(is_reload=True)
        assert "model-reload-a" not in ids, "removed model should no longer appear"
        assert "model-reload-b" in ids, "unchanged model should still appear"
        assert "model-reload-c" in ids, "newly added model should appear"
    finally:
        os.remove(preset_path)


def test_router_dedup_cache_models():
    """dedup-cache-models hides the cache entry backing a preset from GET /models"""
    global server

    preset_path = os.path.join(TMP_DIR, "test_dedup.ini")
    cache_id = "ggml-org/test-model-stories260K:F32"

    with open(preset_path, "w") as f:
        f.write(
            "[model-dedup]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            "dedup-cache-models = 1\n"
        )

    server.models_preset = preset_path
    server.start()

    try:
        ids = _get_model_ids(is_reload=False)
        assert "model-dedup" in ids
        assert cache_id not in ids, "cache model should be hidden by dedup"
        # other cache models are unaffected
        assert "ggml-org/tinygemma3-GGUF:Q8_0" in ids

        # the hidden model is only hidden from the listing, it can still be used
        res = server.make_request("POST", "/tokenize", data={"model": cache_id, "content": "hello"})
        assert res.status_code == 200

        # disabling the flag brings the cache entry back on reload
        with open(preset_path, "w") as f:
            f.write(
                "[model-dedup]\n"
                "hf-repo = ggml-org/test-model-stories260K\n"
            )
        ids = _get_model_ids(is_reload=True)
        assert cache_id in ids

        # the flag also works from the global section
        with open(preset_path, "w") as f:
            f.write(
                "[*]\n"
                "dedup-cache-models = 1\n"
                "\n"
                "[model-dedup]\n"
                "hf-repo = ggml-org/test-model-stories260K\n"
            )
        ids = _get_model_ids(is_reload=True)
        assert "model-dedup" in ids
        assert cache_id not in ids, "cache model should be hidden by global dedup"
    finally:
        os.remove(preset_path)


def test_router_remote_preset():
    global server
    server.model_hf_repo = "ggml-org/test-preset-ci"
    server.model_hf_file = None
    server.offline = False
    server.start()

    # Should see preset models in GET /models
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    ids = {item["id"] for item in res.body.get("data", [])}
    assert "tinygemma3-preset" in ids
    assert "stories260K-test" in ids

    # Should be able to load a preset model
    model_id = "tinygemma3-preset"
    _load_model_and_wait(model_id)


MODEL_DOWNLOAD_ID = "ggml-org/test-model-router-download:F16"
MODEL_DOWNLOAD_TIMEOUT = 30


def _listen_sse(
    server: ServerProcess, collected: list, stop: threading.Event, ready: threading.Event | None = None
):
    """Collect /models/sse events into `collected` until `stop` is set.

    When `ready` is provided, it is set once the streaming response is open,
    i.e. the server has accepted the connection and registered us as a
    subscriber. Callers that trigger one-shot events (e.g. download_finished)
    must wait on `ready` before acting, otherwise the event can be broadcast
    before this client is subscribed and be lost.
    """
    url = f"http://{server.server_host}:{server.server_port}/models/sse"
    try:
        with requests.get(url, stream=True, timeout=MODEL_DOWNLOAD_TIMEOUT) as resp:
            if ready is not None:
                ready.set()
            for line_bytes in resp.iter_lines():
                if stop.is_set():
                    break
                line = line_bytes.decode("utf-8")
                if line.startswith("data: "):
                    collected.append(json.loads(line[6:]))
    except Exception:
        pass


def _wait_for_sse_event(collected: list, event_type: str, model: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(e.get("event") == event_type and e.get("model") == model for e in collected):
            return True
        time.sleep(0.01)
    return False


def test_router_download_model():
    """Case 1: download a model, verify SSE events and GET /models."""
    global server
    server.start()

    # Ensure the model is not present before we start
    server.make_request("DELETE", f"/models?model={MODEL_DOWNLOAD_ID}")

    sse_events: list = []
    stop = threading.Event()
    sse_ready = threading.Event()
    sse_thread = threading.Thread(
        target=_listen_sse, args=(server, sse_events, stop, sse_ready), daemon=True
    )
    sse_thread.start()

    # wait for the SSE client to be subscribed before triggering the download,
    # otherwise the one-shot download_finished event can be broadcast before
    # this client is registered and be lost
    assert sse_ready.wait(10), "SSE client failed to connect"

    # Trigger the download
    res = server.make_request("POST", "/models", data={"model": MODEL_DOWNLOAD_ID})
    assert res.status_code == 200
    assert res.body.get("success") is True

    # Wait for download_finished SSE event
    finished = _wait_for_sse_event(
        sse_events, "download_finished", MODEL_DOWNLOAD_ID, MODEL_DOWNLOAD_TIMEOUT
    )
    stop.set()

    assert finished, "Never received download_finished SSE event"
    assert any(
        e.get("event") == "download_progress" and e.get("model") == MODEL_DOWNLOAD_ID
        for e in sse_events
    ), "No download_progress events received"

    # Model should now appear in GET /models
    ids = _get_model_ids(is_reload=False)
    assert MODEL_DOWNLOAD_ID in ids, f"{MODEL_DOWNLOAD_ID} not found in /models after download"


def test_router_delete_model():
    """Case 2: delete the downloaded model, verify it disappears from GET /models."""
    global server
    server.start()

    # Ensure the model exists (download it if needed)
    if MODEL_DOWNLOAD_ID not in _get_model_ids(is_reload=False):
        sse_events: list = []
        stop = threading.Event()
        sse_ready = threading.Event()
        threading.Thread(
            target=_listen_sse, args=(server, sse_events, stop, sse_ready), daemon=True
        ).start()
        # subscribe before triggering the download so the one-shot
        # download_finished event is not lost (see test_router_download_model)
        assert sse_ready.wait(10), "SSE client failed to connect"
        res = server.make_request("POST", "/models", data={"model": MODEL_DOWNLOAD_ID})
        assert res.status_code == 200
        finished = _wait_for_sse_event(
            sse_events, "download_finished", MODEL_DOWNLOAD_ID, MODEL_DOWNLOAD_TIMEOUT
        )
        stop.set()
        assert finished, "Model did not finish downloading before delete test"

    # Delete the model
    del_res = server.make_request("DELETE", f"/models?model={MODEL_DOWNLOAD_ID}")
    assert del_res.status_code == 200
    assert del_res.body.get("success") is True

    # Model should no longer appear in GET /models
    ids = _get_model_ids(is_reload=False)
    assert MODEL_DOWNLOAD_ID not in ids, f"{MODEL_DOWNLOAD_ID} still present after deletion"
