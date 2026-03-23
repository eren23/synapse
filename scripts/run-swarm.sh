#!/usr/bin/env bash
set -euo pipefail
attocode swarm start .attocode/swarm.hybrid.yaml "$(cat tasks/goal.md)"
