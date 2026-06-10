import json
import os
import boto3
import re
import io
from urllib.parse import unquote_plus

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")

BUCKET_NAME = os.environ["BUCKET_NAME"]
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text = re.sub(r'\s+', ' ', text).strip()
    
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            break_point = max(last_period, last_newline)
            if break_point > start:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    
    return chunks

def get_embedding(text):
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

def handler(event, context):
    print(f"Received event: {json.dumps(event)}")
    
    try:
        # Handle S3 event notification format
        if "Records" in event:
            record = event["Records"][0]
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"])
        else:
            # Direct invocation format
            bucket = event.get("bucket", BUCKET_NAME)
            key = event["key"]
        
        print(f"Processing file: s3://{bucket}/{key}")
        
        # Skip non-PDF files
        if not key.lower().endswith('.pdf'):
            print(f"Skipping non-PDF file: {key}")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "Skipped non-PDF file", "key": key}),
            }
        
        # Download PDF
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read()
        print(f"Downloaded {len(content)} bytes")
        
        # Extract text from PDF using pypdf
        from pypdf import PdfReader
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        print(f"Extracted {len(text)} characters from PDF")
        
        if not text.strip():
            print(f"No text extracted from PDF: {key}")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No text extracted from PDF", "key": key}),
            }
        
        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks")
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            processed_chunks.append({
                "chunk_id": f"{key}_{i}",
                "text": chunk,
                "embedding": embedding,
                "source": key,
                "chunk_index": i,
            })
        
        output_key = f"processed/{key.replace('/', '_')}.json"
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=json.dumps(processed_chunks),
            ContentType="application/json",
        )
        print(f"Saved processed chunks to {output_key}")
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Document processed successfully",
                "chunks_created": len(processed_chunks),
                "output_key": output_key,
            }),
        }
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
