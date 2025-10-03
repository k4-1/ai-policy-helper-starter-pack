import time, os, math, json, hashlib, uuid, logging, re
from typing import List, Dict, Tuple, Optional
import numpy as np
from .settings import settings
from .ingest import chunk_text, doc_hash
from qdrant_client import QdrantClient, models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Semantic embedder using sentence-transformers ----
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

class SemanticEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic embedder with sentence-transformers model.
        all-MiniLM-L6-v2: 384 dimensions, fast, good quality
        """
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        # Generate semantic embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype("float32")

class LocalEmbedder:
    """Fallback hash-based embedder for testing/offline scenarios"""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        # Hash-based repeatable pseudo-embedding
        h = hashlib.sha1(text.encode("utf-8")).digest()
        rng_seed = int.from_bytes(h[:8], "big") % (2**32-1)
        rng = np.random.default_rng(rng_seed)
        v = rng.standard_normal(self.dim).astype("float32")
        # L2 normalize
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

# ---- Vector store abstraction ----
class InMemoryStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict] = []
        self._hashes = set()

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(v.astype("float32"))
            self.meta.append(m)
            if h:
                self._hashes.add(h)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.vecs:
            return []
        A = np.vstack(self.vecs)  # [N, d]
        q = query.reshape(1, -1)  # [1, d]
        # cosine similarity
        sims = (A @ q.T).ravel() / (np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

class QdrantStore:
    def __init__(self, collection: str, dim: int = 384):
        self.collection = collection
        self.dim = dim
        self.client = None
        self._connection_healthy = False
        self._fallback_store = None
        
        # Initialize with robust error handling
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Qdrant client with comprehensive error handling and fallback."""
        try:
            self.client = QdrantClient(url="http://qdrant:6333", timeout=10.0)
            # Test connection
            self.client.get_collections()
            self._connection_healthy = True
            self._ensure_collection()
            logger.info(f"Successfully connected to Qdrant for collection '{self.collection}'")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self._connection_healthy = False
            self._setup_fallback()

    def _setup_fallback(self):
        """Setup in-memory fallback store when Qdrant is unavailable."""
        logger.warning("Setting up in-memory fallback store due to Qdrant connection failure")
        self._fallback_store = InMemoryStore(dim=self.dim)

    def _ensure_collection(self):
        """Ensure collection exists with proper error handling."""
        if not self._connection_healthy or not self.client:
            return
            
        try:
            self.client.get_collection(self.collection)
            logger.info(f"Collection '{self.collection}' already exists")
        except UnexpectedResponse as e:
            if e.status_code == 404:
                try:
                    self.client.recreate_collection(
                        collection_name=self.collection,
                        vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
                    )
                    logger.info(f"Created new collection '{self.collection}'")
                except Exception as create_error:
                    logger.error(f"Failed to create collection '{self.collection}': {create_error}")
                    self._connection_healthy = False
                    self._setup_fallback()
            else:
                logger.error(f"Unexpected error checking collection: {e}")
                self._connection_healthy = False
                self._setup_fallback()
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            self._connection_healthy = False
            self._setup_fallback()

    def _health_check(self) -> bool:
        """Perform health check and attempt reconnection if needed."""
        if self._connection_healthy and self.client:
            try:
                self.client.get_collections()
                return True
            except Exception as e:
                logger.warning(f"Qdrant health check failed: {e}")
                self._connection_healthy = False
                
        # Attempt reconnection
        if not self._connection_healthy:
            logger.info("Attempting to reconnect to Qdrant...")
            self._initialize_client()
            
        return self._connection_healthy

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        """Upsert vectors with fallback handling."""
        # Health check and potential reconnection
        if not self._health_check():
            if self._fallback_store:
                logger.warning("Using fallback store for upsert operation")
                self._fallback_store.upsert(vectors, metadatas)
                return
            else:
                raise RuntimeError("Qdrant unavailable and no fallback store configured")

        try:
            points = []
            for i, (v, m) in enumerate(zip(vectors, metadatas)):
                # Use a valid UUID string as point ID; if a stable hex hash exists, map it to a UUID
                h = m.get("hash")
                if h:
                    # Convert hex digest to UUID format deterministically
                    try:
                        # Use the first 32 hex chars to form a UUID
                        hex32 = (h.replace("-", "") + ("0" * 32))[:32]
                        point_id = str(uuid.UUID(hex=hex32))
                    except Exception:
                        point_id = str(uuid.uuid4())
                else:
                    point_id = str(uuid.uuid4())
                points.append(qm.PointStruct(id=point_id, vector=v.tolist(), payload=m))
            
            self.client.upsert(collection_name=self.collection, points=points)
            logger.debug(f"Successfully upserted {len(points)} points to Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to upsert to Qdrant: {e}")
            self._connection_healthy = False
            
            # Fallback to in-memory store
            if self._fallback_store:
                logger.warning("Falling back to in-memory store for upsert")
                self._fallback_store.upsert(vectors, metadatas)
            else:
                self._setup_fallback()
                self._fallback_store.upsert(vectors, metadatas)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        """Search vectors with fallback handling."""
        # Health check and potential reconnection
        if not self._health_check():
            if self._fallback_store:
                logger.warning("Using fallback store for search operation")
                return self._fallback_store.search(query, k)
            else:
                logger.error("Qdrant unavailable and no fallback store configured")
                return []

        try:
            res = self.client.search(
                collection_name=self.collection,
                query_vector=query.tolist(),
                limit=k,
                with_payload=True
            )
            
            out = []
            for r in res:
                out.append((float(r.score), dict(r.payload)))
            
            logger.debug(f"Successfully retrieved {len(out)} results from Qdrant")
            return out
            
        except Exception as e:
            logger.error(f"Failed to search in Qdrant: {e}")
            self._connection_healthy = False
            
            # Fallback to in-memory store
            if self._fallback_store:
                logger.warning("Falling back to in-memory store for search")
                return self._fallback_store.search(query, k)
            else:
                logger.error("No fallback available for search operation")
                return []

    def get_status(self) -> Dict:
        """Get detailed status information about the vector store."""
        status = {
            "type": "qdrant",
            "healthy": self._connection_healthy,
            "collection": self.collection,
            "dimension": self.dim,
            "fallback_active": self._fallback_store is not None
        }
        
        if self._connection_healthy and self.client:
            try:
                collection_info = self.client.get_collection(self.collection)
                status["points_count"] = collection_info.points_count
                status["vectors_count"] = collection_info.vectors_count
            except Exception as e:
                logger.warning(f"Could not retrieve collection stats: {e}")
                status["error"] = str(e)
        
        return status

# ---- LLM provider ----
class StubLLM:
    def generate(self, query: str, contexts: List[Dict]) -> str:
        lines = [f"Answer (stub): Based on the following sources:"]
        for c in contexts:
            sec = c.get("section") or "Section"
            lines.append(f"- {c.get('title')} — {sec}")
        lines.append("Summary:")
        # naive summary of top contexts
        joined = " ".join([c.get("text", "") for c in contexts])
        lines.append(joined[:600] + ("..." if len(joined) > 600 else ""))
        return "\n".join(lines)

class OpenAILLM:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, query: str, contexts: List[Dict]) -> str:
        prompt = f"You are a helpful company policy assistant. Cite sources by title and section when relevant.\nQuestion: {query}\nSources:\n"
        for c in contexts:
            prompt += f"- {c.get('title')} | {c.get('section')}\n{c.get('text')[:600]}\n---\n"
        prompt += "Write a concise, accurate answer grounded in the sources. If unsure, say so."
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.1
        )
        return resp.choices[0].message.content

# ---- RAG Orchestrator & Metrics ----
class Metrics:
    def __init__(self):
        self.t_retrieval = []
        self.t_generation = []

    def add_retrieval(self, ms: float):
        self.t_retrieval.append(ms)

    def add_generation(self, ms: float):
        self.t_generation.append(ms)

    def summary(self) -> Dict:
        avg_r = sum(self.t_retrieval)/len(self.t_retrieval) if self.t_retrieval else 0.0
        avg_g = sum(self.t_generation)/len(self.t_generation) if self.t_generation else 0.0
        return {
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
        }

class RAGEngine:
    def __init__(self):
        # Embedder selection with graceful fallback
        if settings.embedding_model == "semantic-384" and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")
                self.embedding_name = "semantic-384"
                logger.info("Successfully initialized SemanticEmbedder")
            except Exception as e:
                logger.warning(f"Failed to initialize SemanticEmbedder: {e}. Falling back to LocalEmbedder.")
                self.embedder = LocalEmbedder(dim=384)
                self.embedding_name = "local-384"
        else:
            # Use local embedder if semantic not available or not requested
            self.embedder = LocalEmbedder(dim=384)
            self.embedding_name = "local-384"
            if settings.embedding_model == "semantic-384" and not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("Semantic embeddings requested but sentence-transformers not available. Using local embedder.")
        
        # Vector store selection with enhanced error handling
        self.store = None
        self.store_type = "unknown"
        
        if settings.vector_store == "qdrant":
            try:
                self.store = QdrantStore(collection=settings.collection_name, dim=self.embedder.dim)
                self.store_type = "qdrant"
                logger.info("Successfully initialized QdrantStore")
            except Exception as e:
                logger.error(f"Failed to initialize QdrantStore: {e}. Falling back to InMemoryStore.")
                self.store = InMemoryStore(dim=self.embedder.dim)
                self.store_type = "in_memory_fallback"
        else:
            self.store = InMemoryStore(dim=self.embedder.dim)
            self.store_type = "in_memory"
            logger.info("Using InMemoryStore as configured")

        # LLM selection with enhanced error handling
        if settings.llm_provider == "openai" and settings.openai_api_key:
            try:
                self.llm = OpenAILLM(api_key=settings.openai_api_key)
                self.llm_name = "openai:gpt-4o-mini"
                logger.info("Successfully initialized OpenAI LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI LLM: {e}. Falling back to StubLLM.")
                self.llm = StubLLM()
                self.llm_name = "stub"
        else:
            self.llm = StubLLM()
            self.llm_name = "stub"
            if settings.llm_provider == "openai":
                logger.warning("OpenAI LLM requested but API key not available. Using StubLLM.")

        self.metrics = Metrics()
        self._doc_titles = set()
        self._chunk_count = 0
        self._retrieval_cache = {}
        self._generation_cache = {}
        self._retrieval_order = []
        self._generation_order = []

    def ingest_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        """Ingest chunks with comprehensive error handling and logging."""
        if not chunks:
            logger.warning("No chunks provided for ingestion")
            return (0, 0)
            
        doc_titles_before = len(self._doc_titles)
        vectors = []
        metas = []
        
        successful_chunks = 0
        failed_chunks = 0
        
        for ch in chunks:
            try:
                text = ch["text"]
                if not text or not text.strip():
                    logger.warning(f"Empty text in chunk from {ch.get('title', 'unknown')}")
                    failed_chunks += 1
                    continue
                    
                h = doc_hash(text)
                meta = {
                    "hash": h,  # Keep hash for deduplication but don't use as ID
                    "title": ch["title"],
                    "section": ch.get("section"),
                    "text": text,
                }
                
                # Embed text with error handling
                try:
                    v = self.embedder.embed(text)
                    vectors.append(v)
                    metas.append(meta)
                    self._doc_titles.add(ch["title"])
                    self._chunk_count += 1
                    successful_chunks += 1
                except Exception as e:
                    logger.error(f"Failed to embed chunk from {ch.get('title', 'unknown')}: {e}")
                    failed_chunks += 1
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                failed_chunks += 1
                continue

        if not vectors:
            logger.error("No valid vectors generated from chunks")
            return (0, 0)

        # Upsert vectors with error handling
        try:
            self.store.upsert(vectors, metas)
            logger.info(f"Successfully ingested {successful_chunks} chunks, {failed_chunks} failed")
        except Exception as e:
            logger.error(f"Failed to upsert vectors to store: {e}")
            raise RuntimeError(f"Vector store upsert failed: {e}")

        return (len(self._doc_titles) - doc_titles_before, successful_chunks)

    def _cache_get(self, cache: dict, order: list, key: str, ttl: int):
        v = cache.get(key)
        if not v:
            return None
        if time.time() - v["ts"] > ttl:
            try:
                del cache[key]
                order.remove(key)
            except Exception:
                pass
            return None
        try:
            order.remove(key)
        except ValueError:
            pass
        order.append(key)
        return v["val"]

    def _cache_put(self, cache: dict, order: list, key: str, val, max_size: int):
        cache[key] = {"val": val, "ts": time.time()}
        try:
            order.remove(key)
        except ValueError:
            pass
        order.append(key)
        while len(order) > max_size:
            evict = order.pop(0)
            try:
                del cache[evict]
            except KeyError:
                pass

    def _mmr(self, candidates: List[Dict], lambda_: float, top_k: int) -> List[Dict]:
        if not candidates:
            return []
        sims = np.array([max(0.0, min(1.0, float(c.get("score", 0.0)))) for c in candidates])
        token_sets = []
        for c in candidates:
            toks = set(str(c.get("text", "")).lower().split())
            token_sets.append(toks)
        selected = []
        remaining = list(range(len(candidates)))
        while len(selected) < top_k and remaining:
            if not selected:
                best_local = int(np.argmax(sims[remaining]))
                sel = remaining.pop(best_local)
                selected.append(sel)
                continue
            max_div_scores = []
            for idx in remaining:
                candidate_tokens = token_sets[idx]
                jaccards = []
                for s in selected:
                    inter = len(candidate_tokens & token_sets[s])
                    union = len(candidate_tokens | token_sets[s]) or 1
                    jaccards.append(inter / union)
                diversity = 1.0 - (max(jaccards) if jaccards else 0.0)
                max_div_scores.append(diversity)
            max_div_scores = np.array(max_div_scores)
            relevance = sims[remaining]
            mmr_scores = lambda_ * relevance - (1 - lambda_) * max_div_scores
            pick_local = int(np.argmax(mmr_scores))
            sel = remaining.pop(pick_local)
            selected.append(sel)
        return [candidates[i] for i in selected]

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        """Retrieve relevant chunks with error handling and logging.
        Returns metadata dicts with optional 'score' field preserved.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for retrieval")
            return []
            
        try:
            t0 = time.time()
            cache_key = f"{self.embedding_name}|{self.store_type}|{k}|{query.strip()}"
            if settings.cache_enabled:
                cached = self._cache_get(self._retrieval_cache, self._retrieval_order, cache_key, settings.cache_ttl_seconds)
                if cached is not None:
                    return cached
            qv = self.embedder.embed(query)
            raw_results = self.store.search(qv, k=max(k*3, settings.mmr_candidates))
            retrieval_time = (time.time() - t0) * 1000.0
            
            self.metrics.add_retrieval(retrieval_time)
            candidates = []
            seen_keys = set()
            for score, meta in raw_results:
                key = meta.get("hash") or meta.get("text") or str(meta)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                m = dict(meta)
                m["score"] = float(score)
                candidates.append(m)
            lambda_ = settings.mmr_lambda
            ranked = self._mmr(candidates, lambda_=lambda_, top_k=k)
            retrieved_results = ranked
            logger.debug(f"Retrieved {len(retrieved_results)} results in {retrieval_time:.2f}ms")
            if settings.cache_enabled:
                self._cache_put(self._retrieval_cache, self._retrieval_order, cache_key, retrieved_results, settings.retrieval_cache_size)
            return retrieved_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            # Return empty results rather than crashing
            return []

    def generate(self, query: str, contexts: List[Dict]) -> str:
        """Generate answer with error handling and logging."""
        if not query or not query.strip():
            logger.warning("Empty query provided for generation")
            return "I need a question to provide an answer."
        try:
            t0 = time.time()
            cache_key = None
            if settings.cache_enabled:
                ids = [c.get("hash") or str(c.get("text",""))[:64] for c in contexts]
                cache_key = f"gen|{self.llm_name}|{query.strip()}|{'|'.join(ids)}"
                cached = self._cache_get(self._generation_cache, self._generation_order, cache_key, settings.cache_ttl_seconds)
                if cached is not None:
                    return cached
            answer = self.llm.generate(query, contexts)
            if settings.pdpa_redaction_enabled:
                answer = self._redact_sensitive(answer)
            generation_time = (time.time() - t0) * 1000.0
            self.metrics.add_generation(generation_time)
            logger.debug(f"Generated answer in {generation_time:.2f}ms")
            if settings.cache_enabled and cache_key:
                self._cache_put(self._generation_cache, self._generation_order, cache_key, answer, settings.generation_cache_size)
            return answer
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            raise

    def _redact_sensitive(self, text: str) -> str:
        patterns = [
            (r"\b\d{13,16}\b", "[REDACTED_CARD]"),
            (r"\b\+?\d{7,15}\b", "[REDACTED_PHONE]"),
            (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]")
        ]
        out = text
        for pat, repl in patterns:
            try:
                out = re.sub(pat, repl, out)
            except Exception:
                pass
        return out

    def stats(self) -> Dict:
        """Get comprehensive system statistics."""
        m = self.metrics.summary()
        stats = {
            "total_docs": len(self._doc_titles),
            "total_chunks": self._chunk_count,
            "embedding_model": self.embedding_name,
            "llm_model": self.llm_name,
            "store_type": self.store_type,
            **m
        }
        
        # Add vector store specific stats if available
        if hasattr(self.store, 'get_status'):
            try:
                store_status = self.store.get_status()
                stats["store_status"] = store_status
            except Exception as e:
                logger.warning(f"Could not retrieve store status: {e}")
                stats["store_status"] = {"error": str(e)}
        
        return stats

# ---- Helpers ----
def build_chunks_from_docs(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    """
    Build chunks with improved semantic-aware chunking strategy.
    Preserves document structure while ensuring optimal chunk sizes for retrieval.
    """
    chunks = []
    
    for doc in docs:
        section_text = doc["text"].strip()
        
        # If section is small enough, keep it as one chunk
        if len(section_text.split()) <= chunk_size:
            chunks.append({
                "title": doc["title"],
                "section": doc["section"],
                "text": section_text
            })
        else:
            # Enhanced chunking strategy with multiple splitting approaches
            chunks_from_section = _smart_chunk_section(
                section_text, 
                doc["title"], 
                doc["section"], 
                chunk_size, 
                overlap
            )
            chunks.extend(chunks_from_section)
    
    return chunks


def _smart_chunk_section(text: str, title: str, section: str, chunk_size: int, overlap: int) -> List[Dict]:
    """
    Smart chunking that tries multiple strategies to preserve semantic meaning.
    """
    chunks = []
    
    # Strategy 1: Try to split by semantic boundaries (sentences, then paragraphs)
    sentences = _split_into_sentences(text)
    
    if len(sentences) <= 1:
        # Fallback to paragraph splitting if no sentence boundaries
        return _chunk_by_paragraphs(text, title, section, chunk_size, overlap)
    
    current_chunk = ""
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If single sentence is too large, split it further
        if sentence_words > chunk_size:
            # Finalize current chunk if it has content
            if current_chunk.strip():
                chunks.append({
                    "title": title,
                    "section": section,
                    "text": current_chunk.strip()
                })
            
            # Split large sentence by clauses or phrases
            large_sentence_chunks = _split_large_sentence(sentence, title, section, chunk_size, overlap)
            chunks.extend(large_sentence_chunks)
            
            current_chunk = ""
            current_word_count = 0
            continue
        
        # Check if adding this sentence exceeds chunk size
        if current_word_count + sentence_words > chunk_size and current_chunk:
            chunks.append({
                "title": title,
                "section": section,
                "text": current_chunk.strip()
            })
            
            # Start new chunk with overlap
            overlap_text = _get_overlap_text(current_chunk, overlap)
            current_chunk = overlap_text + (" " if overlap_text else "") + sentence
            current_word_count = len(current_chunk.split())
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_word_count += sentence_words
    
    # Add final chunk if it has content
    if current_chunk.strip():
        chunks.append({
            "title": title,
            "section": section,
            "text": current_chunk.strip()
        })
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using multiple delimiters and heuristics.
    """
    import re
    
    # Enhanced sentence splitting with better handling of abbreviations and edge cases
    sentence_endings = r'[.!?]+(?:\s+|$)'
    
    # Split by sentence endings but be careful with abbreviations
    potential_sentences = re.split(sentence_endings, text)
    
    sentences = []
    for i, sentence in enumerate(potential_sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add back the punctuation (except for the last one)
        if i < len(potential_sentences) - 1:
            # Look for the original ending in the text
            next_pos = text.find(sentence) + len(sentence)
            if next_pos < len(text):
                ending_match = re.match(r'[.!?]+', text[next_pos:])
                if ending_match:
                    sentence += ending_match.group()
        
        sentences.append(sentence)
    
    return [s for s in sentences if s.strip()]


def _chunk_by_paragraphs(text: str, title: str, section: str, chunk_size: int, overlap: int) -> List[Dict]:
    """
    Fallback chunking strategy using paragraph boundaries.
    """
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    current_chunk = ""
    current_word_count = 0
    
    for paragraph in paragraphs:
        para_words = len(paragraph.split())
        
        # If single paragraph is too large, split it by sentences
        if para_words > chunk_size:
            if current_chunk.strip():
                chunks.append({
                    "title": title,
                    "section": section,
                    "text": current_chunk.strip()
                })
                current_chunk = ""
                current_word_count = 0
            
            # Split large paragraph
            para_chunks = _smart_chunk_section(paragraph, title, section, chunk_size, overlap)
            chunks.extend(para_chunks)
            continue
        
        # Check if adding this paragraph exceeds chunk size
        if current_word_count + para_words > chunk_size and current_chunk:
            chunks.append({
                "title": title,
                "section": section,
                "text": current_chunk.strip()
            })
            
            # Start new chunk with overlap
            overlap_text = _get_overlap_text(current_chunk, overlap)
            current_chunk = overlap_text + ("\n\n" if overlap_text else "") + paragraph
            current_word_count = len(current_chunk.split())
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            current_word_count += para_words
    
    # Add final chunk if it has content
    if current_chunk.strip():
        chunks.append({
            "title": title,
            "section": section,
            "text": current_chunk.strip()
        })
    
    return chunks


def _split_large_sentence(sentence: str, title: str, section: str, chunk_size: int, overlap: int) -> List[Dict]:
    """
    Split a sentence that's too large by finding natural break points.
    """
    chunks = []
    
    # Try to split by commas, semicolons, or other natural breaks
    import re
    break_points = re.split(r'([,;:](?:\s+))', sentence)
    
    current_chunk = ""
    current_word_count = 0
    
    for i, part in enumerate(break_points):
        part_words = len(part.split())
        
        if current_word_count + part_words > chunk_size and current_chunk:
            chunks.append({
                "title": title,
                "section": section,
                "text": current_chunk.strip()
            })
            
            overlap_text = _get_overlap_text(current_chunk, overlap)
            current_chunk = overlap_text + (" " if overlap_text else "") + part
            current_word_count = len(current_chunk.split())
        else:
            current_chunk += part
            current_word_count += part_words
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            "title": title,
            "section": section,
            "text": current_chunk.strip()
        })
    
    return chunks


def _get_overlap_text(text: str, overlap: int) -> str:
    """
    Get overlap text from the end of the current chunk, trying to preserve sentence boundaries.
    """
    if overlap <= 0:
        return ""
    
    words = text.split()
    if len(words) <= overlap:
        return text
    
    # Get the last 'overlap' words
    overlap_words = words[-overlap:]
    overlap_text = " ".join(overlap_words)
    
    # Try to start from a sentence boundary if possible
    sentences = _split_into_sentences(overlap_text)
    if len(sentences) > 1:
        # Use the last complete sentence(s) that fit within overlap
        return sentences[-1]
    
    return overlap_text
