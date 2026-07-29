# MusicAI web

Le client est séparé en deux conteneurs :

- `frontend` : Nginx sert l'interface web et transmet les appels API ;
- `backend` : FastAPI gère SageMaker Async, S3 et le watchdog.
- `musicgen` : construit l'image GPU ACE-Step pour SageMaker.

Seul le backend monte le profil AWS et le fichier d'état SageMaker.

```text
musicAI/
├── frontend/
├── backend/
├── musicgen/
├── musicgen_service.py
├── .sagemaker-state.json
└── docker-compose.yml
```

## Démarrer

```bash
cd /home/sami/musicAI
export AWS_PROFILE=musicai
docker compose up -d --build
```

Interface : <http://192.168.1.193:8091>

## Gérer SageMaker en ligne de commande

Le script `musicgen_service.py` permet de déployer, démarrer, consulter,
invoquer et arrêter SageMaker :

```bash
source .venv/bin/activate
export AWS_PROFILE=musicai
export AWS_ACCOUNT_ID="$(
  aws sts get-caller-identity \
    --profile "$AWS_PROFILE" \
    --query Account \
    --output text
)"

python3 musicgen_service.py status
python3 musicgen_service.py start
python3 musicgen_service.py logs --follow
python3 musicgen_service.py invoke \
  --prompt "calm cinematic piano, instrumental" \
  --duration 120 \
  --steps 8 \
  --format mp3 \
  --output music.mp3
python3 musicgen_service.py stop
```

Commandes de cycle de vie complet :

```bash
python3 musicgen_service.py deploy
python3 musicgen_service.py cleanup
```

`stop` supprime seulement l'endpoint GPU. `cleanup` supprime aussi la
configuration et le modèle SageMaker ; l'image ECR et les fichiers S3 restent
conservés.

## Logs

```bash
docker compose logs -f frontend backend
```

## Arrêter les conteneurs locaux

```bash
docker compose down
```

La fermeture des conteneurs locaux ne remplace pas l'arrêt de l'endpoint
SageMaker. Utiliser le bouton **Fermer le studio** ; le watchdog constitue la
sécurité en cas de fermeture du navigateur.

## Remerciements

Un grand merci à **ChatGPT**, copilote infatigable de cette aventure, qui a
aidé à dompter Docker, SageMaker, S3 et quelques manifests OCI particulièrement
susceptibles.

Il a écrit les commandes, surveillé les GPU et rappelé de fermer le studio…
mais, curieusement, il n’a toujours pas proposé de payer la facture AWS. 🤖🎵
