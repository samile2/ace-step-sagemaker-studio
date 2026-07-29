import json
import mimetypes
import os
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response


ACESTEP_URL = os.getenv("ACESTEP_URL", "http://127.0.0.1:8001").rstrip("/")
INVOCATION_TIMEOUT = float(os.getenv("INVOCATION_TIMEOUT", "55"))
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "0.5"))
MODEL_INIT_TIMEOUT = float(os.getenv("MODEL_INIT_TIMEOUT", "1800"))
MODEL_NAME = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
LM_MODEL_NAME = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
INIT_LLM = os.getenv("ACESTEP_INIT_LLM", "true").lower() == "true"

app = FastAPI()
model_ready = threading.Event()
model_init_error: str | None = None


def backend_request(
    path: str,
    payload: dict | None = None,
    timeout: float = 5,
) -> tuple[bytes, str]:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = UrlRequest(
        f"{ACESTEP_URL}{path}",
        data=body,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get_content_type()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"ACE-Step returned HTTP {error.code}: {detail}",
        ) from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"ACE-Step is unavailable: {error}",
        ) from error


def backend_json(
    path: str,
    payload: dict | None = None,
    timeout: float = 5,
) -> dict:
    content, _ = backend_request(path, payload, timeout)
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=502,
            detail="ACE-Step returned invalid JSON",
        ) from error

    if result.get("code") != 200:
        raise HTTPException(
            status_code=502,
            detail=result.get("error") or "ACE-Step request failed",
        )
    return result


def initialize_models():
    """Attend l'API interne puis charge les modèles avant le premier ping 200."""
    global model_init_error

    deadline = time.monotonic() + MODEL_INIT_TIMEOUT
    while time.monotonic() < deadline:
        try:
            backend_json("/health", timeout=2)
            break
        except HTTPException:
            time.sleep(1)
    else:
        model_init_error = "ACE-Step API did not start before the deadline"
        return

    try:
        backend_json(
            "/v1/init",
            {
                "model": MODEL_NAME,
                "init_llm": INIT_LLM,
                "lm_model_path": LM_MODEL_NAME if INIT_LLM else None,
            },
            timeout=max(1, deadline - time.monotonic()),
        )
    except HTTPException as error:
        model_init_error = str(error.detail)
        return

    model_ready.set()


@app.on_event("startup")
def start_model_initialization():
    threading.Thread(
        target=initialize_models,
        name="acestep-model-initializer",
        daemon=True,
    ).start()


@app.get("/ping")
def ping():
    """Health check attendu par SageMaker."""
    if not model_ready.is_set():
        return Response(
            content=json.dumps({
                "status": "loading",
                "error": model_init_error,
            }),
            status_code=503,
            media_type="application/json",
        )

    try:
        result = backend_json("/health", timeout=2)
    except HTTPException:
        return Response(
            content=json.dumps({"status": "unavailable"}),
            status_code=503,
            media_type="application/json",
        )

    status = (result.get("data") or {}).get("status")
    if status != "ok":
        return Response(
            content=json.dumps({"status": "unavailable"}),
            status_code=503,
            media_type="application/json",
        )
    return {"status": "ok"}


@app.post("/invocations")
def invocations(payload: dict):
    """Attend la génération ACE-Step et renvoie le premier fichier complet."""
    if not model_ready.is_set():
        raise HTTPException(
            status_code=503,
            detail=model_init_error or "ACE-Step models are loading",
        )

    if not payload.get("prompt") and not payload.get("sample_query"):
        raise HTTPException(
            status_code=400,
            detail="'prompt' or 'sample_query' is required",
        )

    # Un appel SageMaker renvoie un seul fichier. MP3 limite aussi la taille de
    # la réponse par rapport au WAV.
    task_payload = dict(payload)
    task_payload["batch_size"] = 1
    task_payload.setdefault("audio_format", "mp3")

    release = backend_json(
        "/release_task",
        task_payload,
        timeout=min(INVOCATION_TIMEOUT, 30),
    )
    task_id = (release.get("data") or {}).get("task_id")
    if not task_id:
        raise HTTPException(
            status_code=502,
            detail="ACE-Step did not return a task_id",
        )

    deadline = time.monotonic() + INVOCATION_TIMEOUT
    task_result = None
    while time.monotonic() < deadline:
        query = backend_json(
            "/query_result",
            {"task_id_list": [task_id]},
            timeout=min(10, max(1, deadline - time.monotonic())),
        )
        tasks = query.get("data") or []
        current = tasks[0] if tasks else {}
        status = current.get("status")
        if status == 1:
            task_result = current.get("result")
            break
        if status == 2:
            raise HTTPException(
                status_code=500,
                detail=current.get("error") or "ACE-Step generation failed",
            )
        time.sleep(POLL_INTERVAL)

    if task_result is None:
        raise HTTPException(
            status_code=504,
            detail="ACE-Step generation timed out",
        )

    if isinstance(task_result, str):
        try:
            task_result = json.loads(task_result)
        except json.JSONDecodeError as error:
            raise HTTPException(
                status_code=502,
                detail="ACE-Step returned an invalid task result",
            ) from error

    generated = task_result[0] if isinstance(task_result, list) else task_result
    file_reference = (generated or {}).get("file")
    if not file_reference:
        raise HTTPException(
            status_code=502,
            detail="ACE-Step did not return an audio file",
        )

    parsed = urlsplit(file_reference)
    if parsed.scheme:
        audio_path = parsed.path
        if parsed.query:
            audio_path += f"?{parsed.query}"
    elif file_reference.startswith("/"):
        audio_path = file_reference
    else:
        audio_path = f"/v1/audio?path={quote(file_reference)}"

    remaining = max(1, deadline - time.monotonic())
    audio, media_type = backend_request(audio_path, timeout=remaining)

    extension = task_payload["audio_format"].lower()
    if not media_type.startswith("audio/"):
        media_type = mimetypes.types_map.get(f".{extension}", "audio/mpeg")
    filename = f"music.{extension}"

    return Response(
        content=audio,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(audio)),
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
