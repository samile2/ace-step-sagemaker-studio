import argparse
from datetime import datetime, timezone
import json
import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import boto3
from botocore.exceptions import ClientError


REGION = os.getenv("AWS_REGION", "eu-west-3")
ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "").strip()
IMAGE = os.getenv("SAGEMAKER_IMAGE") or (
    f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/ace-step:0.0.1"
    if ACCOUNT_ID else ""
)
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN") or (
    f"arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole"
    if ACCOUNT_ID else ""
)
BUCKET = os.getenv("SAGEMAKER_ASYNC_BUCKET") or (
    f"ace-step-async-{ACCOUNT_ID}-{REGION}"
    if ACCOUNT_ID else ""
)
OUTPUT_PREFIX = "outputs"
INPUT_PREFIX = "inputs"
STATE_FILE = Path(__file__).resolve().parent / ".sagemaker-state.json"
ASYNC_TIMEOUT_SECONDS = 3600
INSTANCE_TYPE = "ml.g5.xlarge"
# Estimation volontairement simple pour eu-west-3. Ce n'est pas une facture AWS.
ESTIMATED_HOURLY_PRICE_USD = float(
    os.getenv("SAGEMAKER_G5_XLARGE_USD_PER_HOUR", "1.80"),
)


def ensure_bucket(s3):
    """Crée un bucket S3 privé si celui-ci n'existe pas encore."""
    try:
        s3.head_bucket(Bucket=BUCKET)
        return
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            raise

    create_args = {"Bucket": BUCKET}
    if REGION != "us-east-1":
        create_args["CreateBucketConfiguration"] = {
            "LocationConstraint": REGION,
        }
    s3.create_bucket(**create_args)
    s3.put_public_access_block(
        Bucket=BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=BUCKET,
        ServerSideEncryptionConfiguration={
            "Rules": [{
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256",
                },
            }],
        },
    )


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI returned by SageMaker: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def object_exists(s3, uri: str) -> bool:
    bucket, key = split_s3_uri(uri)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def download_s3_object(s3, uri: str, destination: Path):
    bucket, key = split_s3_uri(uri)
    with destination.open("wb") as output:
        s3.download_fileobj(bucket, key, output)


def wait_for_async_result(
    s3,
    output_location: str,
    failure_location: str,
    timeout: int,
    destination: Path,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if object_exists(s3, output_location):
            download_s3_object(s3, output_location, destination)
            return

        if failure_location and object_exists(s3, failure_location):
            bucket, key = split_s3_uri(failure_location)
            failure = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            raise RuntimeError(
                f"SageMaker asynchronous inference failed: "
                f"{failure.decode('utf-8', errors='replace')}"
            )
        time.sleep(5)

    raise TimeoutError(
        f"No asynchronous result after {timeout} seconds. "
        f"Expected output: {output_location}"
    )


def save_state(state: dict):
    STATE_FILE.write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )


def load_state() -> dict:
    if not STATE_FILE.exists():
        raise RuntimeError(
            f"{STATE_FILE} is missing. Run "
            f"'python3 musicgen_service.py deploy' first."
        )
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def is_not_found(error: ClientError) -> bool:
    code = error.response.get("Error", {}).get("Code")
    return code in {"ValidationException", "ResourceNotFound"}


def validate_deploy_settings():
    missing = [
        name
        for name, value in (
            ("SAGEMAKER_IMAGE", IMAGE),
            ("SAGEMAKER_ROLE_ARN", ROLE_ARN),
            ("SAGEMAKER_ASYNC_BUCKET", BUCKET),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Configuration de déploiement incomplète. Définis "
            "AWS_ACCOUNT_ID, ou configure séparément : "
            + ", ".join(missing)
        )


def print_cost_estimate(started_at: datetime, stopped_at: datetime):
    duration_seconds = max(0, (stopped_at - started_at).total_seconds())
    duration_hours = duration_seconds / 3600
    estimated_cost = duration_hours * ESTIMATED_HOURLY_PRICE_USD
    minutes, seconds = divmod(round(duration_seconds), 60)
    hours, minutes = divmod(minutes, 60)

    print(
        f"Durée approximative : {hours} h {minutes:02d} min {seconds:02d} s"
    )
    print(
        f"Coût GPU estimé : ~{estimated_cost:.2f} USD "
        f"({INSTANCE_TYPE} à ~{ESTIMATED_HOURLY_PRICE_USD:.2f} USD/h)"
    )
    print(
        "Estimation indicative hors stockage S3/ECR et transfert ; "
        "la facture AWS fait foi."
    )


def deploy():
    validate_deploy_settings()
    if STATE_FILE.exists():
        state = load_state()
        raise RuntimeError(
            f"A deployment is already recorded for "
            f"{state.get('endpoint_name')}. Use 'status', 'start', 'stop', "
            f"or 'cleanup'."
        )

    sm = boto3.client("sagemaker", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    ensure_bucket(s3)

    suffix = uuid.uuid4().hex[:8]
    model_name = f"ace-step-{suffix}"
    endpoint_config_name = f"ace-step-async-config-{suffix}"
    endpoint_name = f"ace-step-async-{suffix}"
    state = {
        "region": REGION,
        "bucket": BUCKET,
        "model_name": model_name,
        "endpoint_config_name": endpoint_config_name,
        "endpoint_name": endpoint_name,
    }

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer={
            "Image": IMAGE,
            "Environment": {
                "ACESTEP_MODE": "api",
                "ACESTEP_API_HOST": "127.0.0.1",
                "ACESTEP_API_PORT": "8001",
                "ACESTEP_CONFIG_PATH": "acestep-v15-turbo",
                "ACESTEP_INIT_LLM": "true",
                "ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-0.6B",
                "ACESTEP_LM_BACKEND": "pt",
                "ACESTEP_OFFLOAD_TO_CPU": "true",
                "ACESTEP_LM_OFFLOAD_TO_CPU": "true",
                "MODEL_INIT_TIMEOUT": "3600",
                # Garde une marge avant la limite SageMaker de 3600 secondes.
                "INVOCATION_TIMEOUT": "3500",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            },
        },
    )
    save_state(state)

    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": INSTANCE_TYPE,
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 1,
            "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
        }],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": f"s3://{BUCKET}/{OUTPUT_PREFIX}/",
            },
            # ACE-Step utilise un seul GPU par génération.
            "ClientConfig": {
                "MaxConcurrentInvocationsPerInstance": 1,
            },
        },
    )
    save_state(state)

    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )
    sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
    print(f"Endpoint asynchrone prêt : {endpoint_name}")
    print(f"État sauvegardé dans : {STATE_FILE.resolve()}")
    print(
        "Lance une génération avec : "
        "python3 musicgen_service.py invoke"
    )


def start():
    state = load_state()
    sm = boto3.client("sagemaker", region_name=state["region"])
    endpoint_name = state["endpoint_name"]

    try:
        description = sm.describe_endpoint(EndpointName=endpoint_name)
        print(
            f"Endpoint déjà présent : {endpoint_name} "
            f"({description['EndpointStatus']})"
        )
        return
    except ClientError as error:
        if not is_not_found(error):
            raise

    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=state["endpoint_config_name"],
    )
    print(f"Démarrage de {endpoint_name}...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)
    print(f"Endpoint prêt : {endpoint_name}")


def invoke(args):
    state = load_state()
    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=state["region"],
    )
    s3 = boto3.client("s3", region_name=state["region"])

    payload = {
        "prompt": args.prompt,
        "audio_duration": args.duration,
        "audio_format": args.format,
        "inference_steps": args.steps,
    }
    inference_id = uuid.uuid4().hex
    input_key = f"{INPUT_PREFIX}/{inference_id}.json"
    s3.put_object(
        Bucket=state["bucket"],
        Key=input_key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )

    response = runtime.invoke_endpoint_async(
        EndpointName=state["endpoint_name"],
        InputLocation=f"s3://{state['bucket']}/{input_key}",
        ContentType="application/json",
        Accept="audio/mpeg" if args.format == "mp3" else f"audio/{args.format}",
        InferenceId=inference_id,
        InvocationTimeoutSeconds=ASYNC_TIMEOUT_SECONDS,
        RequestTTLSeconds=21600,
    )

    output_location = response["OutputLocation"]
    failure_location = response.get("FailureLocation", "")
    print(f"Requête en file : {response['InferenceId']}")
    print(f"Résultat S3 attendu : {output_location}")

    destination = Path(args.output)
    wait_for_async_result(
        s3,
        output_location,
        failure_location,
        timeout=ASYNC_TIMEOUT_SECONDS + 300,
        destination=destination,
    )
    print(f"Fichier téléchargé : {destination.resolve()}")
    print(f"Fichier conservé dans S3 : {output_location}")


def status():
    state = load_state()
    sm = boto3.client("sagemaker", region_name=state["region"])
    try:
        description = sm.describe_endpoint(
            EndpointName=state["endpoint_name"],
        )
    except ClientError as error:
        if is_not_found(error):
            print(f"Endpoint arrêté : {state['endpoint_name']}")
            return
        raise

    print(json.dumps({
        "EndpointName": description["EndpointName"],
        "EndpointStatus": description["EndpointStatus"],
        "EndpointConfigName": description["EndpointConfigName"],
        "CreationTime": description["CreationTime"].isoformat(),
        "LastModifiedTime": description["LastModifiedTime"].isoformat(),
    }, indent=2))


def logs(args):
    state = load_state()
    client = boto3.client("logs", region_name=state["region"])
    log_group = f"/aws/sagemaker/Endpoints/{state['endpoint_name']}"
    start_time = int((time.time() - args.since * 60) * 1000)
    print(f"CloudWatch : {log_group}")

    while True:
        token = None
        newest_timestamp = start_time
        while True:
            request = {
                "logGroupName": log_group,
                "startTime": start_time,
                "interleaved": True,
            }
            if token:
                request["nextToken"] = token
            try:
                response = client.filter_log_events(**request)
            except client.exceptions.ResourceNotFoundException:
                print("Le groupe de logs n'existe pas encore.")
                if not args.follow:
                    return
                time.sleep(5)
                break

            for event in response.get("events", []):
                timestamp = datetime.fromtimestamp(
                    event["timestamp"] / 1000,
                    tz=timezone.utc,
                ).astimezone()
                print(
                    f"{timestamp.isoformat(timespec='seconds')} "
                    f"{event['message'].rstrip()}"
                )
                newest_timestamp = max(
                    newest_timestamp,
                    event["timestamp"] + 1,
                )

            next_token = response.get("nextToken")
            if not next_token or next_token == token:
                break
            token = next_token

        if not args.follow:
            return
        start_time = newest_timestamp
        time.sleep(3)


def stop():
    state = load_state()
    sm = boto3.client("sagemaker", region_name=state["region"])
    endpoint_name = state["endpoint_name"]
    try:
        description = sm.describe_endpoint(EndpointName=endpoint_name)
        sm.delete_endpoint(EndpointName=endpoint_name)
    except ClientError as error:
        if not is_not_found(error):
            raise
        print(f"Endpoint déjà arrêté : {endpoint_name}")
        return

    print(f"Arrêt de {endpoint_name}...")
    sm.get_waiter("endpoint_deleted").wait(EndpointName=endpoint_name)
    stopped_at = datetime.now(timezone.utc)
    print("Endpoint supprimé : l'instance GPU n'est plus facturée.")
    print_cost_estimate(description["CreationTime"], stopped_at)
    print("Pour le relancer : python3 musicgen_service.py start")


def cleanup():
    state = load_state()
    sm = boto3.client("sagemaker", region_name=state["region"])

    try:
        sm.delete_endpoint(EndpointName=state["endpoint_name"])
        sm.get_waiter("endpoint_deleted").wait(
            EndpointName=state["endpoint_name"],
        )
    except ClientError as error:
        if not is_not_found(error):
            raise

    for operation, parameter, value in (
        (
            sm.delete_endpoint_config,
            "EndpointConfigName",
            state["endpoint_config_name"],
        ),
        (sm.delete_model, "ModelName", state["model_name"]),
    ):
        try:
            operation(**{parameter: value})
        except ClientError as error:
            if not is_not_found(error):
                raise

    STATE_FILE.unlink(missing_ok=True)
    print("Endpoint, configuration et modèle SageMaker supprimés.")
    print("L'image ECR et les fichiers S3 sont conservés.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gestion de l'endpoint SageMaker Async ACE-Step",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("deploy", help="Créer et démarrer l'endpoint")
    commands.add_parser("start", help="Redémarrer un endpoint arrêté")
    commands.add_parser("status", help="Afficher l'état de l'endpoint")
    commands.add_parser("stop", help="Supprimer l'endpoint et arrêter le GPU")
    commands.add_parser(
        "cleanup",
        help="Supprimer endpoint, configuration et modèle",
    )

    invoke_parser = commands.add_parser(
        "invoke",
        help="Générer un fichier audio",
    )
    invoke_parser.add_argument(
        "--prompt",
        default="Generate piano music",
    )
    invoke_parser.add_argument("--duration", type=float, default=120)
    invoke_parser.add_argument("--steps", type=int, default=8)
    invoke_parser.add_argument(
        "--format",
        choices=("mp3", "wav", "flac"),
        default="mp3",
    )
    invoke_parser.add_argument("--output", default="music.mp3")

    logs_parser = commands.add_parser(
        "logs",
        help="Afficher les logs CloudWatch",
    )
    logs_parser.add_argument(
        "--follow",
        action="store_true",
        help="Continuer à suivre les nouveaux logs",
    )
    logs_parser.add_argument(
        "--since",
        type=int,
        default=30,
        help="Minutes de logs à afficher",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    actions = {
        "deploy": deploy,
        "start": start,
        "invoke": lambda: invoke(args),
        "status": status,
        "logs": lambda: logs(args),
        "stop": stop,
        "cleanup": cleanup,
    }
    actions[args.command]()

if __name__ == "__main__":
    main()
