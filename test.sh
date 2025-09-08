#!/usr/bin/env bash
set -euo pipefail

RG="jemg-personal-test"
ACR="azcreu2labt01"
APP="wapceu2labt01"
IMAGE="streamlit-chat-eval"
TAG="latest"

ACR_LOGIN_SERVER="$(az acr show -g "$RG" -n "$ACR" --query loginServer -o tsv)"

# build & push to ACR
az acr build -r "$ACR" -t "${ACR_LOGIN_SERVER}/${IMAGE}:${TAG}" .

# point Web App to the new image
az webapp config container set -g "$RG" -n "$APP" \
  --container-image-name "${ACR_LOGIN_SERVER}/${IMAGE}:${TAG}" \
  --container-registry-url "https://${ACR_LOGIN_SERVER}"

# restart the app
az webapp restart -g "$RG" -n "$APP"
