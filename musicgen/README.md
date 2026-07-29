# Conteneur ACE-Step pour SageMaker

Ce répertoire sert exclusivement à construire l’image GPU utilisée par
SageMaker Async Inference. Il est indépendant des deux conteneurs de
l’application web :

```text
musicAI/
├── frontend/   # Interface web Nginx
├── backend/    # API FastAPI, boto3, SageMaker et watchdog
├── musicgen/   # Construction de l’image GPU SageMaker
└── docker-compose.yml
```

La commande `docker compose up -d --build` construit uniquement `frontend` et
`backend`. Elle ne reconstruit donc pas cette image d’environ 16 Go.

## Contenu

- `dockerfile` : image basée sur ACE-Step 1.5 avec les checkpoints préchargés ;
- `app.py` : adaptateur SageMaker exposant `/ping` et `/invocations` ;
- `sagemaker-entrypoint.sh` : lancement d’ACE-Step et de l’adaptateur ;
- `build.sh` : construction Linux AMD64 compatible SageMaker ;
- `run.sh` : essai local avec un GPU NVIDIA.

Les modèles Hugging Face sont téléchargés pendant le build et intégrés à
l’image. Cela évite de les télécharger à chaque lancement SageMaker, mais le
chargement de l’image et des modèles prend encore plusieurs minutes.

## Construire l’image

Depuis ce répertoire :

```bash
cd /home/sami/musicAI/musicgen
./build.sh
```

Le script produit actuellement l’image locale :

```text
ace-step-sagemaker:0.0.1
```

Il utilise `linux/amd64`, désactive la provenance et le SBOM, puis charge une
image Docker classique. Ces options évitent de pousser un manifeste OCI
`application/vnd.oci.image.index.v1+json`, que SageMaker refuse.

## Se connecter à ECR

```bash
export AWS_PROFILE=musicai
export AWS_REGION=eu-west-3
export AWS_ACCOUNT_ID="$(
  aws sts get-caller-identity \
    --profile "$AWS_PROFILE" \
    --query Account \
    --output text
)"

aws ecr get-login-password \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE" |
docker login \
  --username AWS \
  --password-stdin \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
```

## Taguer et pousser

```bash
docker tag ace-step-sagemaker:0.0.1 \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ace-step:0.0.1"

docker push \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ace-step:0.0.1"
```

Vérifier ensuite le type du manifeste :

```bash
docker buildx imagetools inspect \
  "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ace-step:0.0.1"
```

Le résultat attendu est :

```text
MediaType: application/vnd.docker.distribution.manifest.v2+json
```

## Publier une nouvelle version

Pour publier par exemple `0.0.2`, modifier le tag dans `build.sh`, reconstruire,
puis remplacer `0.0.1` par `0.0.2` dans les commandes de tag et de push.

Il faut également utiliser cette nouvelle URI lors de la création du modèle
SageMaker. Un modèle SageMaker existant qui référence `0.0.1` ne bascule pas
automatiquement vers `0.0.2`.

## Essai local

Sur une machine équipée de Docker, du NVIDIA Container Toolkit et d’un GPU
compatible :

```bash
./run.sh
```

Puis vérifier le endpoint SageMaker :

```bash
curl http://127.0.0.1:8080/ping
```

Le guide du déploiement SageMaker et de l’application web se trouve dans
[le README principal](../README.md).
