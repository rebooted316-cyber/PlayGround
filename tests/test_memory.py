from agentic_os.llm.base import Message, ToolCall
from agentic_os.memory import ConversationBuffer, HashingEmbedder, LongTermMemory, VectorStore, chunk_text, cosine


def test_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder(128)
    a, b = e.embed("vector databases store embeddings"), e.embed("vector databases store embeddings")
    assert a == b
    assert abs(sum(x * x for x in a) - 1) < 1e-9
    assert cosine(a, e.embed("databases for vectors")) > cosine(a, e.embed("bananas are yellow"))


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert all(len(c.split()) <= 100 for c in chunks)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]
    assert " ".join(chunks).split()[-1] == "w499"


def test_chunking_keeps_short_paragraphs_together():
    chunks = chunk_text("Para one.\n\nPara two.\n\nPara three.", chunk_size=50, overlap=5)
    assert chunks == ["Para one. Para two. Para three."]


def test_vector_store_search_filter_and_persist(tmp_path):
    path = str(tmp_path / "store.json")
    store = VectorStore(path=path)
    store.add("Paris is the capital of France", {"kind": "geo"})
    store.add("Python is a programming language", {"kind": "tech"})
    hits = store.search("capital city of France", k=1)
    assert "Paris" in hits[0].record.text
    assert store.search("capital of France", where={"kind": "tech"})[0].record.metadata["kind"] == "tech"
    store.save()
    assert len(VectorStore(path=path)) == 2


def test_conversation_buffer_evicts_whole_turns():
    buf = ConversationBuffer(max_tokens=60)
    for i in range(10):
        buf.add_turn(
            [
                Message.user(f"question {i} " + "x " * 20),
                Message.assistant("", [ToolCall(f"c{i}", "t", {})]),
                Message.tool(f"c{i}", "result"),
                Message.assistant(f"answer {i}"),
            ]
        )
    msgs = buf.messages()
    assert msgs[0].role == "user"
    assert len(buf) < 10
    assert msgs[-1].content == "answer 9"


def test_long_term_memory_recall_by_kind(tmp_path):
    mem = LongTermMemory(path=str(tmp_path / "m.json"))
    mem.remember("The user prefers concise answers", kind="fact")
    mem.remember("Always verify arithmetic with the calculator", kind="lesson")
    assert mem.recall("how should answers be written concise", kind="fact")[0].kind == "fact"
    assert mem.recall("arithmetic calculator", kind="lesson")[0].text.startswith("Always")
    assert len(LongTermMemory(path=str(tmp_path / "m.json"))) == 2


def test_memory_tools():
    mem = LongTermMemory()
    remember, recall = mem.as_tools()
    remember(fact="Project Falcon launches in March")
    assert recall(query="When does Project Falcon launch?")[0]["text"].startswith("Project Falcon")
