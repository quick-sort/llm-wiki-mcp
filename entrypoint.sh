#!/bin/sh
set -e
mkdir -p /data/wikis
chown -R agent:agent /data 2>/dev/null || true
exec gosu agent "$@"
