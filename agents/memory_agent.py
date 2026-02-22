# ============================================================
# Memory Agent — Stores & retrieves context via ChromaDB
# First agent to initialise, used by all other agents
# ============================================================

from crewai import LLM, Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai.tools import tool
from pipelines.vector_pipeline import (
    init_vector_pipeline,
    store_eda_summary,
    store_model_results,
    store_agent_insight,
    store_dataframe_sample,
    semantic_search,
    get_full_context,
)
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv(override=True)


# ── 1. Init LLM ─────────────────────────────────────────────

def get_llm() -> LLM:
    return LLM(
        model=f"groq/{os.getenv('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')}",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 512)),
    )


# ── 2. Memory Store Class ────────────────────────────────────

class MemoryStore:
    """
    Singleton-style wrapper around the vector pipeline.
    Shared across all agents for consistent memory access.
    """

    def __init__(self):
        persist_dir      = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        embedding_model  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        self.client, self.collection, self.embedding_model = init_vector_pipeline(
            persist_dir=persist_dir,
            collection_name="ds_crew_memory",
            embedding_model_name=embedding_model,
        )
        logger.success("MemoryStore initialised")

    # ── Store Operations ─────────────────────────────────────

    def remember_eda(self, eda_summary: dict, dataset_name: str = "dataset") -> str:
        """Store EDA summary in vector memory."""
        return store_eda_summary(
            self.collection,
            self.embedding_model,
            eda_summary,
            dataset_name,
        )

    def remember_model_results(self, automl_results: dict) -> str:
        """Store AutoML results in vector memory."""
        return store_model_results(
            self.collection,
            self.embedding_model,
            automl_results,
        )

    def remember_insight(self, agent_name: str, insight: str,
                         metadata: dict = None) -> str:
        """Store a free-text agent insight in vector memory."""
        return store_agent_insight(
            self.collection,
            self.embedding_model,
            agent_name,
            insight,
            metadata,
        )

    def remember_dataframe(self, df, dataset_name: str = "dataset") -> str:
        """Store a DataFrame sample in vector memory."""
        return store_dataframe_sample(
            self.collection,
            self.embedding_model,
            df,
            dataset_name,
        )

    # ── Retrieval Operations ──────────────────────────────────

    def recall(self, query: str, n_results: int = 5,
               filter_type: str = None) -> list[dict]:
        """Semantic search over all stored memory."""
        return semantic_search(
            self.collection,
            self.embedding_model,
            query,
            n_results,
            filter_type,
        )

    def recall_full_context(self, query: str = "dataset analysis summary",
                            n_results: int = 10) -> str:
        """Retrieve full memory context as a single string."""
        return get_full_context(
            self.collection,
            self.embedding_model,
            query,
            n_results,
        )

    def recall_eda(self) -> list[dict]:
        """Retrieve all stored EDA summaries."""
        return self.recall(
            query="dataset EDA profiling statistics",
            filter_type="eda_summary",
        )

    def recall_model_results(self) -> list[dict]:
        """Retrieve all stored model results."""
        return self.recall(
            query="model training results accuracy metrics",
            filter_type="model_results",
        )

    def recall_insights(self, topic: str = "analysis") -> list[dict]:
        """Retrieve agent insights relevant to a topic."""
        return self.recall(
            query=topic,
            filter_type="agent_insight",
        )

    def count(self) -> int:
        """Return total number of documents in memory."""
        return self.collection.count()

    def status(self) -> dict:
        """Return memory store status."""
        return {
            "total_documents": self.count(),
            "collection_name": self.collection.name,
            "persist_dir":     os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        }


# ── 3. CrewAI Memory Agent ───────────────────────────────────

def build_memory_agent(llm: ChatGroq = None) -> Agent:
    """Build the CrewAI Memory Agent."""
    if llm is None:
        llm = get_llm()

    agent = Agent(
        role="Memory & Context Manager",
        goal=(
            "Store, organise, and retrieve all analysis context, insights, "
            "and results using semantic vector memory so that every other "
            "agent has instant access to relevant prior knowledge."
        ),
        backstory=(
            "You are the institutional memory of the data science crew. "
            "You ensure no insight is lost and every agent can recall "
            "relevant context from any prior step in the pipeline. "
            "You maintain a semantic index of everything the crew has learned."
        ),
        llm=llm,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        allow_delegation=False,
        max_iter=int(os.getenv("MAX_ITERATIONS", 10)),
    )

    logger.info("Memory Agent built")
    return agent


# ── 4. Memory Tasks ──────────────────────────────────────────

def build_memory_task(agent: Agent, context: str = "") -> Task:
    """Build a CrewAI Task for the Memory Agent."""
    return Task(
        description=f"""
        Initialise the vector memory store and confirm it is ready.
        Store the following context if provided: {context}

        Then verify:
        1. ChromaDB collection is accessible
        2. Embedding model is loaded
        3. Memory store status is healthy

        Report the memory store status including total documents stored.
        """,
        expected_output=(
            "A confirmation that the memory store is initialised and ready, "
            "including the number of documents currently stored and "
            "the collection name."
        ),
        agent=agent,
    )


# ── 5. Standalone Run ────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Testing Memory Agent standalone...")

    # Init memory store
    memory = MemoryStore()

    # Test store & recall
    memory.remember_insight(
        agent_name="MemoryAgent",
        insight="Memory system initialised and ready for data science pipeline.",
        metadata={"step": "init"},
    )

    results = memory.recall("memory system initialised")
    logger.info(f"Recall test results: {len(results)} documents found")
    logger.info(f"Memory status: {memory.status()}")
    logger.success("Memory Agent test complete")
