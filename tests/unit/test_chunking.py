from storm_eval.rag.chunking import chunk_text


def test_chunk_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text("doc1", text, size=40, overlap=10)
    assert len(chunks) >= 3
    assert all(c.doc_id == "doc1" for c in chunks)
    assert chunks[0].text.split()[0] == "w0"
