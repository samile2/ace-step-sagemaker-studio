import requests
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
import os
from urllib.parse import urljoin

# Configuration générale du service (doit être synchronisée avec app.py)
ACE_STEP_BASE_URL = os.getenv(
    "ACE_STEP_BASE_URL",
    "http://ace-step:8001"
).rstrip("/")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://ollama:11434"
).rstrip("/")
GENERATION_TIMEOUT_SECONDS = int(
    os.getenv("ACE_STEP_GENERATION_TIMEOUT", "900")
)

import logging
logger = logging.getLogger(__name__)


def unload_ollama_models() -> List[str]:
    """Tente de décharger les modèles Ollama actifs."""
    if not os.getenv(
        "UNLOAD_OLLAMA_BEFORE_GENERATION", "false"
).lower() in {"1", "true", "yes", "on"}:
        return []

    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/ps", timeout=5
        )
        response.raise_for_status()
        models = response.json().get("models", [])

        unloaded_models = []
        for model in models:
            model_name = model.get("name") or model.get("model")
            if not model_name:
                continue

            # Tentative de génération pour forcer le déchargement (méthode brute)
            unload_response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=30
            )
            unload_response.raise_for_status()
            unloaded_models.append(model_name)

        return unloaded_models
    except requests.RequestException as error:
        logger.warning("Échec de la déconnexion des modèles Ollama, le service peut continuer sans eux.", exc_info=True)
        return []


def ace_step_request(
    method: str,
    endpoint: str,
    **kwargs
) -> dict:
    """Wrapper pour effectuer un appel générique à ACE-Step."""
    try:
        response = requests.request(
            method,
            f"{ACE_STEP_BASE_URL}{endpoint}",
            timeout=kwargs.pop("timeout", 30),
            **kwargs
        )
        response.raise_for_status()
    except requests.RequestException as error:
        logger.error(
            f"Échec de l'appel ACE-Step vers {endpoint} : {type(error).__name__}",
            exc_info=True,
            extra={"url": f"{ACE_STEP_BASE_URL}{endpoint}"}
        )
        # Le FastAPI gérera la conversion en HTTPException plus tard
        raise error

    payload = response.json()
    if payload.get("code") != 200 or payload.get("error"):
        error_msg = payload.get("error", "Réponse ACE-Step invalide ou code non 200.")
        logger.warning(f"API ACE-Step a renvoyé une erreur: {error_msg}")
        raise Exception(error_msg) # Lever une exception générique à capturer par le layer FastAPI


def wait_for_generation(task_id: str) -> dict:
    """Attend la complétion de la tâche de génération auprès d'ACE-Step."""
    deadline = time.monotonic() + GENERATION_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        try:
            tasks = ace_step_request(
                "POST",
                "/query_result",
                timeout=120,
                json={"task_id_list": [task_id]}
            )
        except Exception as e:
            # Si l'API est injoignable pendant la boucle d'attente
            raise ConnectionError(f"Impossible de contacter ACE-Step pour vérifier le statut. Détail: {e}") from e

        if not tasks:
            time.sleep(1)
            continue

        task = tasks[0]
        status = task.get("status")

        if status == 1:
            results_data = task.get("result") or "[]"
            try:
                results = json.loads(results_data)
                if not results:
                    raise ValueError("ACE-Step n'a retourné aucun fichier dans le résultat.")
                return results[0]
            except (json.JSONDecodeError, ValueError) as e:
                 logger.error(f"Erreur lors du parsing des résultats ACE-Step pour {task_id}", exc_info=True)
                 raise Exception("Le service ACE-Step a retourné un résultat non parsable.") from e

        if status == 2:
            raise ConnectionError(
                f"La génération ACE-Step a échoué. Erreur reçue : {task.get('error')}"
            )

        time.sleep(1)

    raise TimeoutError("La génération ACE-Step a dépassé le délai autorisé.")


def get_ace_step_status() -> Dict[str, Any]:
    """Vérifie la disponibilité du service ACE-Step."""
    try:
        response = requests.get(f"{ACE_STEP_BASE_URL}/health", timeout=5)
        ace_step_ready = response.ok
    except requests.RequestException as e:
        logger.warning("Échec de la vérification du statut ACE-Step.", exc_info=True)
        ace_step_ready = False

    return {
        "status": "OK" if ace_step_ready else "ACE-Step indisponible",
        "model": os.getenv(
            "ACE_STEP_MODEL", "acestep-v15-turbo"
        ),
        "ace_step_ready": ace_step_ready
    }


def generation_metadata(audio_file: Path) -> Dict[str, Any]:
    """Récupère et structure les métadonnées d'un fichier audio."""
    metadata_file = audio_file.with_suffix(".json")
    metadata: Dict[str, Any] = {}

    if metadata_file.is_file():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(f"Impossible de lire ou décoder les métadonnées pour {audio_file.name}. Initialisation avec des valeurs par défaut.", exc_info=True)

    # Logique de fallback et construction des métadonnées
    created_at = metadata.get("created_at")
    if not created_at:
        try:
            created_at = datetime.fromtimestamp(
                audio_file.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        except Exception:
            created_at = datetime.now(timezone.utc).isoformat()

    return {
        "id": audio_file.stem,
        "filename": audio_file.name,
        "prompt": metadata.get("prompt") or "Génération précédente",
        "lyrics": metadata.get("lyrics"),
        "duration_seconds": metadata.get("duration_seconds"),
        "bpm": metadata.get("bpm"),
        "vocal_language": metadata.get("vocal_language"),
        "has_vocals": metadata.get("has_vocals"),
        "model": metadata.get("model") or os.getenv(
            "ACE_STEP_MODEL", "acestep-v15-turbo"
        ),
        "task_type": metadata.get("task_type") or "text2music",
        "remix_from": metadata.get("remix_from"),
        "modification": metadata.get("modification"),
        "cover_strength": metadata.get("cover_strength"),
        "created_at": created_at,
        "size_bytes": audio_file.stat().st_size
    }

# Fonctions de gestion de génération (à être appelées par les endpoints)

def run_generation(
    prompt: str,
    duration_seconds: int,
    bpm: int | None,
    lyrics: str,
    vocal_language: str,
    remix_from: str | None,
    modification: str,
    cover_strength: float
) -> Dict[str, Any]:
    """Exécute le workflow complet de génération musicale."""
    logger.info("Début du processus de génération musicale.")
    
    # 1. Déchargement Ollama (Méthode qui pourrait échouer sans bloquer l'API principale)
    unloaded_models = unload_ollama_models()

    request_data: Dict[str, Any] = {
        "prompt": prompt.strip(),
        "lyrics": lyrics.strip(),
        "vocal_language": vocal_language.strip().lower(),
        "audio_duration": duration_seconds,
        "audio_format": "mp3",
        "model": os.getenv(
            "ACE_STEP_MODEL", "acestep-v15-turbo"
        ),
        "thinking": True,
        "inference_steps": 8,
        "batch_size": 1
    }
    if bpm is not None:
        request_data["bpm"] = bpm

    task_type = "text2music"
    source_file = None

    if remix_from:
        # Sécurité : validation stricte du nom de fichier source
        source_path_obj = Path(remix_from)
        if source_path_obj.name != remix_from and not source_path_obj.is_relative_to(Path(".")) :
            raise ValueError("Identifiant de morceau source invalide ou non résolu.")

        source_file = Path(remix_from) # On suppose que l'ID est le chemin complet ici
        if not source_file.is_file():
            # Si ce n'est pas un fichier réel, on assume qu'il s'agit juste du nom d'un fichier dans OUTPUT_DIR
            source_file = Path(f"{Path('output')}/{remix_from}.mp3")

        if not source_file.is_file():
             raise FileNotFoundError(f"Morceau source introuvable à : {source_file.resolve()}")


        task_type = "cover"
        # Mise à jour du prompt si remixage ET modification
        new_prompt = f"{prompt}. Modification demandée : {modification}" if modification else prompt
        request_data["prompt"] = new_prompt
        request_data["task_type"] = task_type
        request_data["audio_cover_strength"] = cover_strength
        request_data["thinking"] = False

        with source_file.open("rb") as source_audio:
            task = ace_step_request(
                "POST",
                "/release_task",
                timeout=GENERATION_TIMEOUT_SECONDS,
                data=request_data,
                files={
                    "src_audio": (
                        source_file.name, # Utiliser juste le nom pour l'en-tête de fichier
                        source_audio,
                        "audio/mpeg"
                    )
                }
            )
    else:
        task = ace_step_request(
            "POST",
            "/release_task",
            timeout=GENERATION_TIMEOUT_SECONDS,
            json=request_data
        )

    task_id = task["task_id"]
    result = wait_for_generation(task_id)
    audio_path = result.get("file")

    if not audio_path:
        raise ConnectionError("ACE-Step n'a retourné aucune URL audio de résultat.")

    # Téléchargement et sauvegarde du fichier
    try:
        audio_url = urljoin(f"{ACE_STEP_BASE_URL}/", audio_path)
        logger.info(f"Téléchargement du résultat depuis : {audio_url}")
        audio_response = requests.get(audio_url, timeout=120)
        audio_response.raise_for_status()
    except requests.RequestException as e:
        logger.error("Échec du téléchargement audio final.", exc_info=True)
        raise ConnectionError(f"Impossible de télécharger le résultat audio : {e}") from e

    filename = f"{task_id}.mp3"
    output_file = Path("/output") / filename
    output_file.write_bytes(audio_response.content)
    logger.info(f"Audio enregistré avec succès dans {output_file}")


    # Création des métadonnées
    result_metas = result.get("metas") or {}
    detected_bpm = bpm if bpm is not None else result_metas.get("bpm")

    metadata_file = output_file.with_suffix(".json")
    final_metadata = {
        "prompt": prompt.strip(),
        "lyrics": lyrics.strip(),
        "duration_seconds": duration_seconds,
        "bpm": detected_bpm,
        "vocal_language": vocal_language.strip().lower(),
        "has_vocals": lyrics.strip().casefold() != "[instrumental]",
        "model": os.getenv(
            "ACE_STEP_MODEL", "acestep-v15-turbo"
        ),
        "task_type": task_type,
        "remix_from": Path(remix_from).stem if remix_from else None,
        "modification": modification.strip() or None,
        "cover_strength": cover_strength,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_metadata": result_metas # Inclure les métadonnées brutes de l'API pour la traçabilité
    }

    metadata_file.write_text(
        json.dumps(final_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info("Métadonnées enregistrées avec succès.")


    public_base_url = os.getenv("MUSICGEN_PUBLIC_BASE_URL", "").rstrip("/")
    file_url = f"{public_base_url}/static/{filename}"
    player_url = f"{public_base_url}/player/{filename}"

    return {
        "file": file_url,
        "player": player_url,
        "duration_seconds": duration_seconds,
        "bpm": detected_bpm,
        "model": os.getenv(
            "ACE_STEP_MODEL", "acestep-v15-turbo"
        ),
        "task_type": task_type,
        "remix_from": Path(remix_from).stem if remix_from else None,
        "has_vocals": final_metadata["has_vocals"],
        "vocal_language": final_metadata["vocal_language"],
        "unloaded_ollama_models": unloaded_models
    }