"""Central configuration. Every tunable lives here so eval ablations are one place.

Environment overrides (all optional):
  CODERAG_EMBED_MODEL    - sentence-transformers model id for embeddings
  CODERAG_EMBED_TRUST_REMOTE_CODE - "1" to allow custom model code (jina/nomic)
  CODERAG_RERANK_MODEL   - cross-encoder model id for reranking
  CODERAG_INDEX_DIR      - default index directory

  # --- LLM provider (generation + judging) — bring your own key/provider ---
  CODERAG_LLM_PROVIDER   - "anthropic" (default) | "openai". Auto-detected from
                           whichever key is present if unset.
  CODERAG_GEN_MODEL      - model id for answering (per-provider default otherwise)
  CODERAG_JUDGE_MODEL    - model id for faithfulness/correctness judging
  CODERAG_LLM_BASE_URL   - OpenAI-compatible endpoint (OpenRouter, Together, Groq,
                           Ollama/LM Studio/vLLM, Azure, ...). Use with the openai
                           provider to reach any model, including local ones.
  ANTHROPIC_API_KEY      - required when provider is anthropic
  OPENAI_API_KEY         - required when provider is openai (any value for local
                           OpenAI-compatible servers that don't check it)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass
class Settings:
    # --- Models -----------------------------------------------------------
    # Local sentence-transformers model. Default is a code-search model: on the
    # FastAPI golden set it lifts dense recall@5 from 0.70 (all-MiniLM-L6-v2) to
    # 0.95. For an even stronger option use jinaai/jina-embeddings-v2-base-code
    # with CODERAG_EMBED_TRUST_REMOTE_CODE=1; for a tiny/fast general model use
    # sentence-transformers/all-MiniLM-L6-v2. Override via CODERAG_EMBED_MODEL.
    embed_model: str = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # NOTE: the generation/judging model + provider are resolved at *runtime* from
    # the environment via LLMConfig (below), not pinned into the index — so anyone
    # can point the same index at their own provider/key. These two fields are kept
    # only as legacy defaults for indexes saved by older versions.
    gen_model: str = "claude-opus-4-8"
    judge_model: str = "claude-opus-4-8"

    # --- Chunking (§2) ----------------------------------------------------
    max_chunk_tokens: int = 800       # oversized functions get windowed above this
    use_context_header: bool = True   # §2.3 — prepend structural header before embed
    window_chunk: bool = False        # baseline mode: pure line-window chunking (§9.3)
    window_lines: int = 50            # line-window size for baseline/fallback paths

    # --- Retrieval (§3) ---------------------------------------------------
    dense_top_n: int = 40             # candidates from dense search before fusion
    bm25_top_n: int = 40              # candidates from lexical search before fusion
    fuse_top_n: int = 30             # how many fused candidates feed the reranker
    final_k: int = 6                 # final chunks handed to generation
    rrf_k: int = 60                  # RRF constant (§3.3)
    use_dense: bool = True
    use_bm25: bool = True
    use_rerank: bool = True
    expand_graph: bool = False        # pull code-graph neighbors into context
    graph_expand_budget: int = 4      # legacy: max neighbors appended post-rerank
    # Graph fix: feed neighbors into the rerank POOL (so they must earn relevance)
    # instead of appending them post-rerank with a fake score. This makes the graph
    # a recall booster (surface a connected file retrieval missed) without injecting
    # distractors. Set graph_expand_prererank=False for the legacy append behavior.
    graph_expand_prererank: bool = True
    graph_expand_seed: int = 5        # expand neighbors of the top-N fused candidates
    graph_expand_depth: int = 1       # hops to traverse from each seed (>1 = multi-hop)
    graph_pool_budget: int = 10       # max neighbor chunks added to the rerank pool
    graph_rerank_boost: float = 0.0   # >0: boost a chunk's score if graph-connected
                                      # to other high scorers (graph-aware reranking)
    # Personalized-PageRank context selection (Aider-style): rank the connected
    # subgraph from the top retrieval seeds and add the highest-PPR nodes to the
    # rerank pool — graph used to *select* context, not blindly expand neighbors.
    graph_pagerank: bool = False
    graph_pagerank_seeds: int = 5     # retrieval hits used as PPR restart set
    graph_pagerank_add: int = 8       # top-PPR connected nodes to add to the pool
    # Weighted RRF: scale dense vs lexical before rank-fusion (1.0/1.0 = plain RRF).
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    # MMR diversity on the final selection (reduce near-duplicate chunks).
    use_mmr: bool = False
    mmr_lambda: float = 0.7           # 1.0 = pure relevance, 0 = pure diversity
    # HyDE (opt-in): draft a hypothetical code snippet from the query with the LLM
    # and embed THAT for dense search — helps when question vocabulary != code.
    use_hyde: bool = False

    # --- Generation (§4) --------------------------------------------------
    context_token_budget: int = 6000  # cap on source code tokens in the prompt
    gen_max_tokens: int = 4096
    include_graph_context: bool = True  # add a compact neighbor map to the prompt
    # Self-repair (opt-in): if a first answer's faithfulness is below this, retry
    # once with a stricter "cite-or-drop" instruction and keep the better one.
    self_repair_threshold: float = 0.0  # 0 = off; e.g. 0.8 to enable

    # --- Token efficiency -------------------------------------------------
    # Safe savers (default on): they remove redundancy/whitespace/irrelevant tail
    # without cutting relevant code, so answer quality is preserved.
    dedup_sources: bool = True            # drop content-identical source chunks
    merge_adjacent_sources: bool = True   # merge contiguous same-file spans (1 header)
    compact_source_code: bool = True      # collapse blank lines / trailing whitespace
    drop_negative_rerank: bool = True     # drop sources the reranker scores irrelevant (<0)
    min_sources: int = 3                  # ...but never gate below this many
    # Riskier saver (opt-in): trims large chunks to query-relevant lines — can drop
    # code the answer needs, so it's OFF by default.
    trim_sources: bool = False
    max_source_tokens: int = 400          # per-source cap, used only when trim_sources
    # Faithfulness judge cost.
    judge_source_tokens: int = 300        # cap source code sent to the judge
    faithfulness_single_call: bool = True # one judge call (extract+verify) vs two
    faithfulness_skip_when_clean: bool = False  # opt-in: skip judge if structurally clean

    # --- Storage ----------------------------------------------------------
    index_dir: str = ".coderag_index"

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        env_map = {
            "embed_model": os.environ.get("CODERAG_EMBED_MODEL"),
            "rerank_model": os.environ.get("CODERAG_RERANK_MODEL"),
            "gen_model": os.environ.get("CODERAG_GEN_MODEL"),
            "judge_model": os.environ.get("CODERAG_JUDGE_MODEL"),
            "index_dir": os.environ.get("CODERAG_INDEX_DIR"),
        }
        kwargs = {k: v for k, v in env_map.items() if v}
        kwargs.update(overrides)
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (existing vars win).

    Tiny, dependency-free — so `ANTHROPIC_API_KEY` and CODERAG_* can live in a
    local .env instead of being exported every shell. Silently no-ops if absent.
    """
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def _auto_provider() -> str:
    """Pick a provider from whichever key is present (Anthropic wins ties)."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "anthropic"


# Per-provider default models used when CODERAG_GEN_MODEL / _JUDGE_MODEL are unset.
_PROVIDER_DEFAULTS = {
    "anthropic": ("claude-opus-4-8", "claude-opus-4-8"),
    "openai": ("gpt-4o", "gpt-4o-mini"),
}


@dataclass
class LLMConfig:
    """Runtime LLM selection — provider, models, optional base URL — resolved from
    the environment so the same index works with any provider/key. The default is
    Anthropic (Claude); set CODERAG_LLM_PROVIDER=openai (optionally with
    CODERAG_LLM_BASE_URL) to use OpenAI or any OpenAI-compatible endpoint."""
    provider: str = "anthropic"
    gen_model: str = "claude-opus-4-8"
    judge_model: str = "claude-opus-4-8"
    base_url: "str | None" = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = (os.environ.get("CODERAG_LLM_PROVIDER") or _auto_provider()).lower()
        default_gen, default_judge = _PROVIDER_DEFAULTS.get(
            provider, _PROVIDER_DEFAULTS["anthropic"])
        gen = os.environ.get("CODERAG_GEN_MODEL") or default_gen
        judge = os.environ.get("CODERAG_JUDGE_MODEL") or default_judge
        base_url = None
        if provider == "openai":
            base_url = os.environ.get("CODERAG_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        return cls(provider=provider, gen_model=gen, judge_model=judge, base_url=base_url)
