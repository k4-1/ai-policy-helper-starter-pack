from pydantic import BaseModel
from pydantic import field_validator
import os
import re

class Settings(BaseModel):
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "local-384")
    llm_provider: str = os.getenv("LLM_PROVIDER", "stub")  # stub | openai | ollama
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    vector_store: str = os.getenv("VECTOR_STORE", "qdrant")  # qdrant | memory
    collection_name: str = os.getenv("COLLECTION_NAME", "policy_helper")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    data_dir: str = os.getenv("DATA_DIR", "/app/data")
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.5"))
    mmr_candidates: int = int(os.getenv("MMR_CANDIDATES", "12"))
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
    retrieval_cache_size: int = int(os.getenv("RETRIEVAL_CACHE_SIZE", "128"))
    generation_cache_size: int = int(os.getenv("GENERATION_CACHE_SIZE", "64"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "120"))
    streaming_enabled: bool = os.getenv("STREAMING_ENABLED", "true").lower() in ("1", "true", "yes")
    pdpa_redaction_enabled: bool = os.getenv("PDPA_REDACTION_ENABLED", "true").lower() in ("1", "true", "yes")
    feedback_db_path: str = os.getenv("FEEDBACK_DB_PATH", "/app/app/feedback.db")

    @field_validator('openai_api_key')
    def validate_openai_key(cls, v, info):
        llm_provider = info.data.get('llm_provider', 'stub')
        
        if llm_provider == 'openai':
            if not v:
                raise ValueError("OpenAI API key is required when using OpenAI as LLM provider")
            
            # Basic format validation for OpenAI API keys
            if not v.startswith('sk-'):
                raise ValueError("OpenAI API key must start with 'sk-'")
            
            # Check minimum length (OpenAI keys are typically 51+ characters)
            if len(v) < 20:
                raise ValueError("OpenAI API key appears to be too short")
                
        return v

    @field_validator('relevance_threshold')
    def validate_threshold(cls, v):
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    @field_validator('mmr_lambda')
    def validate_mmr_lambda(cls, v):
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v
    
    def validate_configuration(self):
        """Validate the complete configuration and return user-friendly error messages"""
        errors = []
        
        if self.llm_provider == 'openai':
            if not self.openai_api_key:
                errors.append({
                    "type": "authentication_error",
                    "message": "OpenAI API key is missing",
                    "user_message": "The AI service needs to be configured with a valid OpenAI API key.",
                    "suggestions": [
                        "Add your OpenAI API key to the environment variables",
                        "Get an API key from https://platform.openai.com/api-keys",
                        "Contact your administrator for configuration help"
                    ]
                })
            elif not self.openai_api_key.startswith('sk-'):
                errors.append({
                    "type": "authentication_error", 
                    "message": "Invalid OpenAI API key format",
                    "user_message": "The configured OpenAI API key appears to be invalid.",
                    "suggestions": [
                        "Check that the API key starts with 'sk-'",
                        "Verify the key was copied correctly from OpenAI",
                        "Generate a new API key if needed"
                    ]
                })
        
        return errors

settings = Settings()
