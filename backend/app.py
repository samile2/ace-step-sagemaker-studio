import html
import json
import logging
import mimetypes
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import boto3
from botocore.exceptions import ClientError, WaiterError
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("musicai")

STATE_FILE = Path(
    os.getenv("SAGEMAKER_STATE_FILE", "/config/sagemaker-state.json")
)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/output"))
HEARTBEAT_TIMEOUT_SECONDS = int(
    os.getenv("WATCHDOG_IDLE_SECONDS", "90")
)
WATCHDOG_INTERVAL_SECONDS = int(
    os.getenv("WATCHDOG_INTERVAL_SECONDS", "10")
)
ASYNC_TIMEOUT_SECONDS = int(
    os.getenv("SAGEMAKER_ASYNC_TIMEOUT_SECONDS", "3600")
)
ESTIMATED_HOURLY_PRICE_USD = float(
    os.getenv("SAGEMAKER_G5_XLARGE_USD_PER_HOUR", "1.80")
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MusicAI SageMaker client")
sessions: dict[str, float] = {}
sessions_lock = threading.Lock()
endpoint_lock = threading.Lock()
jobs_lock = threading.Lock()
active_jobs = 0
last_browser_activity = time.monotonic()


def load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Configuration SageMaker absente : {STATE_FILE}"
        ) from error

    required = {
        "region",
        "bucket",
        "endpoint_name",
        "endpoint_config_name",
    }
    missing = required.difference(state)
    if missing:
        raise RuntimeError(
            f"Configuration SageMaker incomplète : {', '.join(sorted(missing))}"
        )
    return state


def sagemaker_client(state: dict):
    return boto3.client("sagemaker", region_name=state["region"])


def runtime_client(state: dict):
    return boto3.client("sagemaker-runtime", region_name=state["region"])


def s3_client(state: dict):
    return boto3.client("s3", region_name=state["region"])


def is_not_found(error: ClientError) -> bool:
    code = error.response.get("Error", {}).get("Code")
    return code in {"ValidationException", "ResourceNotFound"}


def describe_endpoint() -> dict:
    state = load_state()
    try:
        description = sagemaker_client(state).describe_endpoint(
            EndpointName=state["endpoint_name"],
        )
    except ClientError as error:
        if is_not_found(error):
            return {
                "endpoint_name": state["endpoint_name"],
                "status": "Stopped",
                "ready": False,
            }
        raise

    result = {
        "endpoint_name": description["EndpointName"],
        "status": description["EndpointStatus"],
        "ready": description["EndpointStatus"] == "InService",
        "creation_time": description["CreationTime"].isoformat(),
        "last_modified_time": description["LastModifiedTime"].isoformat(),
    }
    if description.get("FailureReason"):
        result["failure_reason"] = description["FailureReason"]
    return result


def start_endpoint() -> dict:
    state = load_state()
    with endpoint_lock:
        current = describe_endpoint()
        if current["status"] != "Stopped":
            return current

        logger.info("Démarrage de l'endpoint %s", state["endpoint_name"])
        sagemaker_client(state).create_endpoint(
            EndpointName=state["endpoint_name"],
            EndpointConfigName=state["endpoint_config_name"],
        )
        return describe_endpoint()


def stop_endpoint(reason: str, wait: bool = False) -> dict:
    state = load_state()
    with endpoint_lock:
        current = describe_endpoint()
        if current["status"] == "Stopped":
            return {
                "status": "Stopped",
                "duration_seconds": None,
                "estimated_cost_usd": None,
                "hourly_rate_usd": ESTIMATED_HOURLY_PRICE_USD,
            }

        started_at = datetime.fromisoformat(current["creation_time"])
        client = sagemaker_client(state)

        if current["status"] != "Deleting":
            logger.info(
                "Arrêt de l'endpoint %s (%s)",
                state["endpoint_name"],
                reason,
            )
            try:
                client.delete_endpoint(
                    EndpointName=state["endpoint_name"],
                )
            except ClientError as error:
                if not is_not_found(error):
                    raise

        if not wait:
            return {
                "status": "Deleting",
                "duration_seconds": None,
                "estimated_cost_usd": None,
                "hourly_rate_usd": ESTIMATED_HOURLY_PRICE_USD,
            }

        client.get_waiter("endpoint_deleted").wait(
            EndpointName=state["endpoint_name"],
            WaiterConfig={"Delay": 5, "MaxAttempts": 120},
        )
        stopped_at = datetime.now(timezone.utc)
        duration_seconds = max(
            0,
            (stopped_at - started_at).total_seconds(),
        )
        estimated_cost = (
            duration_seconds / 3600 * ESTIMATED_HOURLY_PRICE_USD
        )
        logger.info(
            "Studio fermé après %.0f s, coût estimé %.2f USD",
            duration_seconds,
            estimated_cost,
        )
        return {
            "status": "Stopped",
            "duration_seconds": round(duration_seconds),
            "estimated_cost_usd": round(estimated_cost, 4),
            "hourly_rate_usd": ESTIMATED_HOURLY_PRICE_USD,
        }


def register_session(session_id: str):
    global last_browser_activity
    if not session_id or len(session_id) > 128:
        raise HTTPException(status_code=400, detail="Session invalide")
    with sessions_lock:
        now = time.monotonic()
        sessions[session_id] = now
        last_browser_activity = now


def unregister_session(session_id: str):
    global last_browser_activity
    with sessions_lock:
        sessions.pop(session_id, None)
        last_browser_activity = time.monotonic()


def live_session_count() -> int:
    deadline = time.monotonic() - HEARTBEAT_TIMEOUT_SECONDS
    with sessions_lock:
        expired = [
            session_id
            for session_id, last_seen in sessions.items()
            if last_seen < deadline
        ]
        for session_id in expired:
            sessions.pop(session_id, None)
        return len(sessions)


def browser_idle_seconds() -> float:
    with sessions_lock:
        return max(0, time.monotonic() - last_browser_activity)


def job_started():
    global active_jobs
    with jobs_lock:
        active_jobs += 1


def job_finished():
    global active_jobs
    with jobs_lock:
        active_jobs = max(0, active_jobs - 1)


def active_job_count() -> int:
    with jobs_lock:
        return active_jobs


def watchdog():
    while True:
        try:
            clients = live_session_count()
            jobs = active_job_count()
            idle_seconds = browser_idle_seconds()
            if (
                clients == 0
                and jobs == 0
                and idle_seconds >= HEARTBEAT_TIMEOUT_SECONDS
            ):
                stop_endpoint(
                    f"watchdog : aucun navigateur depuis "
                    f"{round(idle_seconds)} s"
                )
        except Exception:
            logger.exception("Erreur du watchdog")
        time.sleep(WATCHDOG_INTERVAL_SECONDS)


@app.on_event("startup")
def start_watchdog():
    load_state()
    threading.Thread(
        target=watchdog,
        name="sagemaker-watchdog",
        daemon=True,
    ).start()


@app.get("/api/status")
def api_status():
    try:
        result = describe_endpoint()
        result["active_sessions"] = live_session_count()
        result["active_jobs"] = active_job_count()
        return result
    except (ClientError, RuntimeError) as error:
        logger.exception("Lecture du statut impossible")
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/api/server/start")
def api_start(payload: dict):
    session_id = str(payload.get("session_id", ""))
    register_session(session_id)
    try:
        return start_endpoint()
    except (ClientError, RuntimeError) as error:
        logger.exception("Démarrage SageMaker impossible")
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/api/server/stop")
def api_stop(payload: dict):
    session_id = str(payload.get("session_id", ""))
    if active_job_count() > 0:
        raise HTTPException(
            status_code=409,
            detail="Une génération est en cours. Attendez sa fin avant de fermer.",
        )
    unregister_session(session_id)
    try:
        return stop_endpoint("fermeture manuelle du studio", wait=True)
    except (ClientError, RuntimeError, WaiterError) as error:
        logger.exception("Arrêt SageMaker impossible")
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/api/session/heartbeat")
def api_heartbeat(payload: dict):
    register_session(str(payload.get("session_id", "")))
    return {"ok": True}


@app.post("/api/session/close")
def api_close(payload: dict):
    unregister_session(str(payload.get("session_id", "")))
    return {"ok": True}


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"URI S3 invalide : {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def object_exists(client, uri: str) -> bool:
    bucket, key = split_s3_uri(uri)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def wait_for_result(
    client,
    output_location: str,
    failure_location: str,
) -> bytes:
    deadline = time.monotonic() + ASYNC_TIMEOUT_SECONDS + 300
    while time.monotonic() < deadline:
        if object_exists(client, output_location):
            bucket, key = split_s3_uri(output_location)
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()

        if failure_location and object_exists(client, failure_location):
            bucket, key = split_s3_uri(failure_location)
            failure = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            raise RuntimeError(
                failure.decode("utf-8", errors="replace")
            )
        time.sleep(5)
    raise TimeoutError("La génération SageMaker a dépassé le délai autorisé")


def generation_metadata(audio_file: Path) -> dict:
    metadata_file = audio_file.with_suffix(".json")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}

    return {
        "id": audio_file.stem,
        "filename": audio_file.name,
        "prompt": metadata.get("prompt") or "Génération précédente",
        "lyrics": metadata.get("lyrics"),
        "duration_seconds": metadata.get("duration_seconds"),
        "bpm": metadata.get("bpm"),
        "vocal_language": metadata.get("vocal_language"),
        "has_vocals": metadata.get("has_vocals"),
        "model": "acestep-v15-turbo",
        "task_type": "text2music",
        "created_at": metadata.get("created_at")
        or datetime.fromtimestamp(
            audio_file.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "size_bytes": audio_file.stat().st_size,
    }


@app.get("/generations")
def list_generations(request: Request):
    base_url = str(request.base_url).rstrip("/")
    generations = []
    for audio_file in OUTPUT_DIR.glob("*.mp3"):
        metadata = generation_metadata(audio_file)
        metadata["file"] = f"{base_url}/static/{audio_file.name}"
        metadata["player"] = f"{base_url}/player/{audio_file.name}"
        generations.append(metadata)
    generations.sort(key=lambda item: item["created_at"], reverse=True)
    return {"generations": generations}


@app.post("/generate")
def generate(
    prompt: str,
    duration_seconds: int = Query(default=30, ge=10, le=600),
    bpm: int | None = Query(default=None, ge=30, le=300),
    lyrics: str = Query(default="[Instrumental]", max_length=4096),
    vocal_language: str = Query(default="fr", min_length=2, max_length=16),
    remix_from: str | None = None,
    modification: str = "",
    cover_strength: float = Query(default=0.7, ge=0.0, le=1.0),
):
    del modification, cover_strength
    if remix_from:
        raise HTTPException(
            status_code=400,
            detail="Les variantes audio ne sont pas encore disponibles via SageMaker",
        )

    status = describe_endpoint()
    if not status["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"Le serveur GPU n'est pas prêt ({status['status']})",
        )

    state = load_state()
    inference_id = uuid.uuid4().hex
    payload = {
        "prompt": prompt.strip(),
        "lyrics": lyrics.strip(),
        "vocal_language": vocal_language.strip().lower(),
        "audio_duration": duration_seconds,
        "audio_format": "mp3",
        "inference_steps": 8,
        "thinking": True,
        "batch_size": 1,
    }
    if bpm is not None:
        payload["bpm"] = bpm

    client = s3_client(state)
    input_key = f"inputs/web-{inference_id}.json"
    client.put_object(
        Bucket=state["bucket"],
        Key=input_key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )

    job_started()
    try:
        response = runtime_client(state).invoke_endpoint_async(
            EndpointName=state["endpoint_name"],
            InputLocation=f"s3://{state['bucket']}/{input_key}",
            ContentType="application/json",
            Accept="audio/mpeg",
            InferenceId=inference_id,
            InvocationTimeoutSeconds=ASYNC_TIMEOUT_SECONDS,
            RequestTTLSeconds=21600,
        )
        audio = wait_for_result(
            client,
            response["OutputLocation"],
            response.get("FailureLocation", ""),
        )
    except (ClientError, RuntimeError, TimeoutError) as error:
        logger.exception("Génération impossible")
        raise HTTPException(status_code=502, detail=str(error)) from error
    finally:
        job_finished()

    filename = f"{inference_id}.mp3"
    output_file = OUTPUT_DIR / filename
    output_file.write_bytes(audio)
    output_file.with_suffix(".json").write_text(
        json.dumps(
            {
                "prompt": prompt.strip(),
                "lyrics": lyrics.strip(),
                "duration_seconds": duration_seconds,
                "bpm": bpm,
                "vocal_language": vocal_language.strip().lower(),
                "has_vocals": lyrics.strip().casefold() != "[instrumental]",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "s3_output": response["OutputLocation"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "file": f"/static/{filename}",
        "player": f"/player/{filename}",
        "duration_seconds": duration_seconds,
        "bpm": bpm,
        "model": "acestep-v15-turbo",
    }


@app.get("/player/{filename}", response_class=HTMLResponse)
def player(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Nom invalide")
    if not (OUTPUT_DIR / filename).is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    safe_filename = html.escape(filename, quote=True)
    return HTMLResponse(
        f'<audio controls autoplay src="/static/{safe_filename}"></audio>'
    )


@app.get("/static/{filename}")
def static_file(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Nom invalide")
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return FileResponse(
        path,
        media_type=mimetypes.guess_type(path.name)[0] or "audio/mpeg",
    )
