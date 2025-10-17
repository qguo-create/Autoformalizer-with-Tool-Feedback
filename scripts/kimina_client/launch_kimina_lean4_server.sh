#!/bin/bash
set -e

export PATH="/home/hadoop-aipnlp/.elan/bin:${PATH}"

which elan
which lean
which lake
elan toolchain list

cp kimina-lean-server.env /home/kimina-lean-server/.env
cd /home/kimina-lean-server

unset http_proxy https_proxy all_proxy
nohup python3 -m server > /dev/null 2>&1 &

cd -
# python3 verify_kimina_solutions.py