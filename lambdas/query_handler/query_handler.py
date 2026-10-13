# new version August 2026 - with chunk-to-source mapping
# see  https://poe.com/chat/66tmdly17ajaz0hs7k

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


# ============================================================
# GROUNDED RAG SYSTEM PROMPT
# ============================================================
GROUNDED_RAG_SYSTEM_PROMPT = """You are a research assistant answering questions using ONLY the retrieved document chunks provided below.

## GROUNDING RULES (CRITICAL)

1. **EXTRACTION BEFORE DISCLAIMER**: Before stating that information is "not available" or "not in the context," you MUST:
   - Read each chunk COMPLETELY, not just headlines or first sentences
   - Look for the requested information expressed in DIFFERENT TERMINOLOGY
   - Check for relevant facts EMBEDDED in narrative paragraphs
   - Consider synonyms and related concepts (e.g., "margin" includes: collateral, haircut, margin call, margin requirement, initial margin, variation margin)

2. **PARTIAL RELEVANCE RULE**: If chunks contain SOME relevant information but not all:
   - Extract and present what IS available with specific facts and figures
   - State SPECIFICALLY what is missing (e.g., "The chunks provide the margin reduction percentage but not the exact dates")
   - NEVER use blanket disclaimers like "no information available" when partial information exists

3. **TERMINOLOGY FLEXIBILITY**: Query terms may not match document vocabulary exactly:
   - "Margins" → margin requirements, collateral, margin calls, haircuts, collateral return
   - "Leverage" → exposure, gross notional, debt-to-equity, concentration limits
   - Look for the CONCEPT, not just the exact WORD

4. **QUANTITATIVE EXTRACTION**: When documents contain specific numbers, percentages, or dollar amounts related to the query, you MUST extract and report them.

## OUTPUT FORMAT

You must return valid JSON with this exact structure:
{{
  "answer": "Your comprehensive answer here, with specific facts and figures from the chunks",
  "sources_used": [
    {{
      "chunk_id": "CHUNK_0",
      "facts_extracted": ["specific fact 1", "specific fact 2"]
    }}
  ],
  "retrieval_status": "sources_used",
  "information_gaps": ["specific item 1 not found in any chunk", "specific item 2 not found"]
}}

The retrieval_status field must be one of:
- "sources_used" — at least one chunk contributed to the answer
- "sources_retrieved_but_irrelevant" — chunks were retrieved but none relevant to query
- "no_sources_retrieved" — no chunks provided (system error)

## RETRIEVED CHUNKS

{chunks}

## USER QUERY

{query}

Return ONLY the JSON object, no other text."""


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


def format_chunks_with_ids(chunks):
    """Format chunks with explicit IDs for attribution tracking."""
    formatted = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"CHUNK_{i}"
        source_name = chunk.get('source', 'unknown')
        
        # Show FULL source path for better traceability
        formatted.append(f"""--- {chunk_id} ---
Source: {source_name}
Score: {chunk['score']:.3f}
Content:
{chunk['text']}
--- END {chunk_id} ---""")
    
    return "\n\n".join(formatted)


def generate_response(query, context_chunks):
    """Generate response with grounding enforcement and structured output."""
    
    formatted_chunks = format_chunks_with_ids(context_chunks)
    
    prompt = GROUNDED_RAG_SYSTEM_PROMPT.format(
        chunks=formatted_chunks,
        query=query
    )

    response = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }),
        contentType="application/json",
    )

    result = json.loads(response["body"].read())
    response_text = result["content"][0]["text"]
    
    # Parse the structured response
    try:
        # Handle case where model wraps JSON in markdown code blocks
        parse_text = response_text
        if "```json" in parse_text:
            json_start = parse_text.find("```json") + 7
            json_end = parse_text.find("```", json_start)
            parse_text = parse_text[json_start:json_end].strip()
        elif "```" in parse_text:
            json_start = parse_text.find("```") + 3
            json_end = parse_text.find("```", json_start)
            parse_text = parse_text[json_start:json_end].strip()
        
        structured_response = json.loads(parse_text)
        
        # Validate required fields exist
        if "answer" not in structured_response:
            structured_response["answer"] = response_text
        if "retrieval_status" not in structured_response:
            structured_response["retrieval_status"] = "sources_used"
        if "sources_used" not in structured_response:
            structured_response["sources_used"] = []
        if "information_gaps" not in structured_response:
            structured_response["information_gaps"] = []
            
    except json.JSONDecodeError:
        # Fallback: treat entire response as answer
        structured_response = {
            "answer": response_text,
            "sources_used": [],
            "retrieval_status": "sources_used",
            "information_gaps": [],
            "_parse_error": "Response was not valid JSON, using raw text"
        }
    
    return structured_response


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
        
        if not search_results:
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "query": query,
                    "answer": "No relevant documents were retrieved for this query.",
                    "sources": [],
                    "retrieval_status": "no_sources_retrieved",
                    "sources_used": [],
                    "information_gaps": [query]
                }),
            }

        structured_response = generate_response(query, search_results)

        # Build a mapping from CHUNK_X to source filename
        chunk_id_to_source = {f"CHUNK_{i}": r["source"] for i, r in enumerate(search_results)}
        
        # Enrich sources_used with actual filenames
        enriched_sources_used = []
        for src in structured_response.get("sources_used", []):
            chunk_id = src.get("chunk_id", "")
            enriched_sources_used.append({
                "chunk_id": chunk_id,
                "source_file": chunk_id_to_source.get(chunk_id, "unknown"),
                "facts_extracted": src.get("facts_extracted", [])
            })

        response_body = {
            "query": query,
            "answer": structured_response.get("answer", ""),
            "sources": [
                {
                    "index": i,
                    "chunk_id": f"CHUNK_{i}",
                    "source": r["source"], 
                    "score": r["score"]
#                    "text": r["text"]
                } 
                for i, r in enumerate(search_results)
            ],
            "retrieval_status": structured_response.get("retrieval_status", "sources_used"),
            "sources_used": enriched_sources_used,
            "information_gaps": structured_response.get("information_gaps", []),
        }
        
        if "_parse_error" in structured_response:
            response_body["_parse_error"] = structured_response["_parse_error"]

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response_body),
        }

    except Exception as e:
        import traceback
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "trace": traceback.format_exc()
            }),
        }
