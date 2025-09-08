#!/usr/bin/env bash
$RG       = "jemg-personal-test"
$ACR  = "azcreu2labt01"
$ACR_LOGIN_SERVER = az acr show -g $RG -n $ACR --query loginServer -o tsv
$TAG = "latest"
$IMAGE = "streamlit-chat-eval"
$APP  = "wapceu2labt01"
az acr build -r $ACR -t "${ACR_LOGIN_SERVER}/${IMAGE}:${TAG}" .

az webapp config container set -g $RG -n $APP `
  --container-image-name "${ACR_LOGIN_SERVER}/${IMAGE}:${TAG}" `
  --container-registry-url "https://${ACR_LOGIN_SERVER}"
az webapp restart -g $RG -n $APP