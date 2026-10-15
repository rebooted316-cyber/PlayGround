from agentic_os.rag import BM25Index, Document, RAGPipeline

DOCS = [
    Document("The mitochondria is the powerhouse of the cell. It produces ATP through respiration.", {"source": "bio.md"}),
    Document("Photosynthesis converts sunlight, water and carbon dioxide into glucose in chloroplasts.", {"source": "plants.md"}),
    Document("The French Revolution began in 1789 with the storming of the Bastille.", {"source": "history.md"}),
]


def test_bm25_ranks_keyword_matches():
    idx = BM25Index()
    for i, d in enumerate(DOCS):
        idx.add(str(i), d.text)
    assert idx.search("Bastille 1789")[0][0] == "2"
    idx.remove("2")
    assert all(doc_id != "2" for doc_id, _ in idx.search("Bastille"))


def test_hybrid_retrieval_and_citations():
    rag = RAGPipeline()
    assert rag.add_documents(DOCS) == 3
    for mode in ("hybrid", "vector", "keyword"):
        top = rag.retrieve("Where is ATP produced in the cell?", k=1, mode=mode)
        assert top[0].source == "bio.md", mode
    context, used = rag.build_context("photosynthesis glucose")
    assert context.startswith("[1] (source: plants.md)")
    assert used[0].metadata["source"] == "plants.md"


def test_context_budget_and_tool():
    rag = RAGPipeline()
    rag.add_documents(DOCS)
    context, used = rag.build_context("cell", max_tokens=5)
    assert context == "" and used == []
    search = rag.as_tool()
    assert "history.md" in search(query="French Revolution")


def test_empty_pipeline():
    assert RAGPipeline().retrieve("anything") == []
