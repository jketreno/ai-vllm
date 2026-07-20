#!/bin/bash
set -e
PROJECT=ai-vllm

# If dirty, prompt to continue
git diff --quiet || {
  read -p "There are uncommitted changes. Do you want to continue? (y/n) " yn
  case $yn in
    [Yy]* ) echo "Continuing...";;
    [Nn]* ) echo "Aborting."; exit 1;;
    * ) echo "Please answer yes or no."; exit 1;;
  esac
} 

git push

ssh ai.ketrenos.com "bash -lc 'cd docker/${PROJECT} && git pull && docker compose build && docker compose up -d'"
