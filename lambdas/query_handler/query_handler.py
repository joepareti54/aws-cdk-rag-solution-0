import json
import os
import boto3
import numpy as np
import faiss
import tempfile

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")

BUCKET_NAME = os.environ["BUCKET_NAME"]
INDEX_PREFIX = os.environ.get("INDEX_PREFIX", "faiss-index/")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

index_cache = {}

def get_embedding(text):
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

def load_index():
    if "index" in index_cache and "metadata" in index_cache:
        return index_cache["index"], index_cache["metadata"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "index.faiss")
        metadata_path = os.path.join(tmpdir, "metadata.json")
        
        s3.download_file(BUCKET_NAME, f"{INDEX_PREFIX}index.faiss", index_path)
        s3.download_file(BUCKET_NAME, f"{INDEX_PREFIX}metadata.json", metadata_path)
        
        index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    index_cache["index"] = index
    index_cache["metadata"] = metadata
    
    return index, metadata

def search(query, k=5):
    index, metadata = load_index()
    
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result["score"] = float(distances[0][i])
            results.append(result)
    
    return results

def generate_response(query, context):
    context_text = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in context])
    
    prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say so.

Context:
{context_text}

Question: {query}

Answer:"""

    response = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }),
        contentType="application/json",
    )
    
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        query = body.get("query")
        k = body.get("k", 5)
        
        if not query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Query is required"}),
            }
        
        search_results = search(query, k)
        response_text = generate_response(query, search_results)
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "query": query,
                "answer": response_text,
                "sources": [{"source": r["source"], "score": r["score"]} for r in search_results],
            }),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
