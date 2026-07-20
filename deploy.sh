#!/bin/bash
set -e
. .env

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

ssh -t ${DEPLOYMENT} "bash -lic 'cd ${PROJECT} && git pull && docker compose build && docker compose up -d'"
