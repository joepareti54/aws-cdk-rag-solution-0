# RAG System on AWS with FAISS and Bedrock

A serverless Retrieval-Augmented Generation (RAG) system on AWS using FAISS for vector search and Amazon Bedrock for LLM inference, deployed via AWS CDK.

The system ingests a corpus of financial press releases and answers natural-language questions about them using semantic retrieval combined with a foundation model.

---

## Architecture

The pipeline has four stages:

1. **Upload** — A local script pushes PDF documents to an S3 bucket.
2. **Process** — An S3 event triggers the `document_processor` Lambda, which extracts text, splits it into chunks (1000 chars, 200 overlap), generates embeddings via Bedrock Titan, and writes chunk metadata back to S3.
3. **Index** — The `index_manager` Lambda builds a FAISS IndexFlatIP index from the embeddings and stores the index file in S3.
4. **Query** — A user POSTs a question to API Gateway. The `query_handler` Lambda loads the FAISS index, retrieves the top-k relevant chunks, builds a prompt, and calls Claude 4.5 Sonnet via Bedrock to generate the answer.

All AWS resources are provisioned with AWS CDK.

---

## Tech Stack

- **IaC**: AWS CDK (Python)
- **Compute**: AWS Lambda (Python 3.9)
- **API**: Amazon API Gateway (REST, API key auth)
- **Storage**: Amazon S3
- **Embeddings**: Amazon Bedrock - amazon.titan-embed-text-v2:0 (1024-dim)
- **LLM**: Amazon Bedrock - anthropic.claude-4.5-sonnet-20240229-v1:0
- **Vector search**: FAISS (IndexFlatIP)

---

## Project Structure

    rag-solution/
    |- app.py                          CDK app entry point
    |- cdk.json                        CDK configuration
    |- stacks/
    |   |- rag_stack.py                Main CDK stack
    |- lambdas/
    |   |- document_processor/         Extracts text, chunks, embeds
    |   |- index_manager/              Builds FAISS index
    |   |- query_handler/              Handles RAG queries
    |- layers/
    |   |- dependencies/               Lambda layer (FAISS, numpy, etc.)
    |- scripts/
    |   |- upload_documents_fixed.py   Local PDF upload helper
    |- config/
    |   |- config.py                   Centralized config
    |- do_load.sh                      Convenience loader script
    |- .env.example                    Environment template
    |- requirements.txt
    |- requirements-dev.txt
    |- README.md

---

## Prerequisites

- AWS account with Bedrock model access enabled for Titan Embeddings v2 and Claude 3 Sonnet
- AWS CLI configured (aws configure)
- Node.js (for the AWS CDK CLI)
- Python 3.9
- AWS CDK v2 (npm install -g aws-cdk)

---

## Setup

### 1. Clone and create a virtual environment

    git clone https://github.com/joepareti54/aws-cdk-rag-solution-0.git
    cd rag-solution

    python -m venv .venv
    source .venv/bin/activate

    pip install -r requirements.txt
    pip install -r requirements-dev.txt

### 2. Configure environment

    cp .env.example .env

Edit .env and set your AWS account ID and region.

### 3. Build the Lambda dependency layer

    cd layers/dependencies
    bash build.sh
    cd ../..

### 4. Bootstrap CDK (first time only)

    cdk bootstrap

### 5. Deploy

    cdk deploy

The stack outputs the API Gateway endpoint URL and the API Key ID. Retrieve the API key value with:

    aws apigateway get-api-key --api-key <ApiKeyId> --include-value --query value --output text

---

## Usage

### Upload documents

    bash do_load.sh

This script uploads the PDF corpus year by year. Each upload to S3 automatically triggers the document_processor Lambda (S3 event on the uploads/ prefix), which extracts text, chunks it, and writes embeddings back to S3. After all uploads complete, do_load.sh invokes the index_manager Lambda once to (re)build the FAISS index from all embeddings. Before running, edit do_load.sh to set the local path to your PDF corpus.

### Query the system

    curl -X POST https://<api-id>.execute-api.<region>.amazonaws.com/prod/query \
      -H "Content-Type: application/json" \
      -H "x-api-key: <your-api-key>" \
      -d '{"query": "What was the revenue trend in 2023?"}'

The response contains the generated answer and the source chunks used.

---

## Configuration

All tunable parameters live in config/config.py:

- Chunking: chunk_size=1000, chunk_overlap=200
- FAISS: IVFFlat index with nlist=100
- Lambda memory/timeout: per-function settings
- Model IDs: embedding and LLM model IDs

---

## Cleanup

To remove all AWS resources created by this stack:

    cdk destroy

S3 buckets may need to be emptied manually if they contain objects.

---

## License

MIT - see LICENSE file.

---

## Author

Built by joepareti54 as a portfolio project demonstration
Built on the foundation of aws-cdk-hello-exercise-1
