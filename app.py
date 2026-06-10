#!/usr/bin/env python3
import os
import aws_cdk as cdk
from stacks.rag_stack import RAGStack

app = cdk.App()
RAGStack(
    app,
    "RAGStack",
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION", "us-east-1")
    ),
    description="RAG Solution with FAISS vector store and Bedrock"
)
app.synth()
