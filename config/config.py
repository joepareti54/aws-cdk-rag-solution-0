from dataclasses import dataclass

@dataclass
class LambdaConfig:
    document_processor_memory: int = 1024
    document_processor_timeout: int = 300
    index_manager_memory: int = 3008
    index_manager_timeout: int = 600
    query_handler_memory: int = 3008
    query_handler_timeout: int = 30

@dataclass
class BedrockConfig:
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    llm_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    embedding_dimension: int = 1024

@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class FAISSConfig:
    index_file: str = "faiss_index.bin"
    metadata_file: str = "metadata.json"
    index_type: str = "IVFFlatIP"
#    nlist: int = 100

@dataclass
class RAGConfig:
    lambda_config: LambdaConfig
    bedrock: BedrockConfig
    chunking: ChunkingConfig
    faiss: FAISSConfig

    @classmethod
    def default(cls) -> "RAGConfig":
        return cls(
            lambda_config=LambdaConfig(),
            bedrock=BedrockConfig(),
            chunking=ChunkingConfig(),
            faiss=FAISSConfig()
        )

def get_config() -> RAGConfig:
    return RAGConfig.default()
