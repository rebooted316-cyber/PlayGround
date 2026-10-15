"""Stages 6 and 10: short-term chat memory, long-term memory and hybrid RAG."""

from agentic_os import Agent, ConversationBuffer, LongTermMemory, RAGPipeline
from agentic_os.llm import create_llm

rag = RAGPipeline(chunk_size=120, overlap=20)
rag.add_document("BM25 ranks documents by keyword overlap, weighting rare terms higher.", source="bm25.md")
rag.add_document("Dense retrieval embeds text as vectors and ranks by cosine similarity.", source="dense.md")
rag.add_document("Reciprocal Rank Fusion merges several rankings by summing 1/(k + rank).", source="rrf.md")

for chunk in rag.retrieve("how are keyword and vector rankings combined?", k=3):
    print(f"{chunk.score:.4f}  {chunk.source}  {chunk.text[:60]}")

memory = LongTermMemory()
memory.remember("The user is preparing a talk on search engines.", kind="fact")

agent = Agent("tutor", create_llm(), knowledge=rag, long_term=memory, memory=ConversationBuffer())
print(agent.run("Explain hybrid retrieval in two sentences.").output)
print(agent.run("Which source covered rank fusion?").output)
