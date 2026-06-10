import json
import os
import boto3
import numpy as np
import faiss
import tempfile

s3 = boto3.client("s3")

BUCKET_NAME = os.environ["BUCKET_NAME"]
INDEX_PREFIX = os.environ.get("INDEX_PREFIX", "faiss-index/")

def handler(event, context):
    try:
        action = event.get("action", "build")
        
        if action == "build":
            return build_index(event)
        elif action == "update":
            return update_index(event)
        else:
            return {"statusCode": 400, "body": json.dumps({"error": "Invalid action"})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

def build_index(event):
    processed_prefix = event.get("processed_prefix", "processed/")
    
    all_embeddings = []
    all_metadata = []
    
    # Paginate through all objects
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=processed_prefix)
    
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                data = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                chunks = json.loads(data["Body"].read().decode("utf-8"))
                
                for chunk in chunks:
                    all_embeddings.append(chunk["embedding"])
                    all_metadata.append({
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                    })
    
    if not all_embeddings:
        return {"statusCode": 400, "body": json.dumps({"error": "No embeddings found"})}
    
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "index.faiss")
        metadata_path = os.path.join(tmpdir, "metadata.json")
        
        faiss.write_index(index, index_path)
        with open(metadata_path, "w") as f:
            json.dump(all_metadata, f)
        
        s3.upload_file(index_path, BUCKET_NAME, f"{INDEX_PREFIX}index.faiss")
        s3.upload_file(metadata_path, BUCKET_NAME, f"{INDEX_PREFIX}metadata.json")
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Index built successfully",
            "total_vectors": len(all_embeddings),
            "dimension": dimension,
        }),
    }

def update_index(event):
    return build_index(event)
