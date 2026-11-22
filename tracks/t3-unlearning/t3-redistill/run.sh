#!/bin/bash
# usage: run.sh NOM args...   lance rd.py detache, journal logs/NOM.log
cd ~/eac/t3/t3-redistill; n=$1; shift
nohup /home/hermes/eac/kit/.venv/bin/python code/rd.py --name $n "$@" > logs/$n.log 2>&1 &
echo lance $n pid $!
