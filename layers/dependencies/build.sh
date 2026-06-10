#!/bin/bash
# Rebuild the Lambda layer dependencies.
# Run this before `cdk deploy` if layers/dependencies/python/ is missing or stale.
set -e
cd "$(dirname "$0")"
rm -rf python
pip install -r requirements.txt -t python/ --upgrade
echo "Layer rebuilt at: $(pwd)/python"
