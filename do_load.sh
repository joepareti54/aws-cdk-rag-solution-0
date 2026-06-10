#!/bin/bash
# NOTE: Change the path below to point to your local data directory.
# Expected structure: <root>/finance-YYYY/press/*.pdf

FUNCTION_NAME=$(aws lambda list-functions --query "Functions[?contains(FunctionName, 'IndexManager')].FunctionName" --output text)

for year in 2021 2022 2023 2024 2025 2026; do
  python scripts/upload_documents_fixed.py /path/to/your/finance-${year}/press --year $year
done

# Rebuild index after all uploads complete
aws lambda invoke --function-name $FUNCTION_NAME --payload '{}' response.json
