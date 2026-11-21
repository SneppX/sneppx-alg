from SneppX_ALG import rag

DOCS = [
    "The cat sat on the warm mat in the living room.",
    "Dogs are loyal companions and love to play fetch.",
    "Neural networks learn representations from data via gradient descent.",
    "Gradient descent optimizes a loss function over model parameters.",
    "The quick brown fox jumps over the lazy dog.",
]


def test_text():
    assert rag.tokenize("Hello, World!") == ["hello", "world"]
    assert "the" not in rag.tokenize_no_stop("the cat")
    sents = rag.split_sentences("One. Two! Three?")
    assert sents == ["One.", "Two!", "Three?"]


def test_chunk_by_words():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = rag.chunk_by_words(text, size=4, overlap=0)
    assert chunks[0] == "w0 w1 w2 w3"
    assert len(chunks) == 3
    overlap = rag.chunk_by_words(text, size=4, overlap=2)
    assert overlap[1] == "w2 w3 w4 w5"


def test_chunk_by_sentences():
    text = "Alpha beta. Gamma delta. Epsilon zeta."
    chunks = rag.chunk_by_sentences(text, max_chars=15)
    assert len(chunks) >= 2
    assert all(len(c) <= 20 for c in chunks)


def test_chunk_recursive_and_window():
    text = "a" * 50 + "\n\n" + "b" * 50
    chunks = rag.chunk_recursive(text, max_chars=60)
    assert all(len(c) <= 60 for c in chunks)
    wins = rag.sliding_window("abcdefgh", window=4, stride=2)
    assert wins == ["abcd", "cdef", "efgh"]


def test_bm25_ranks_relevant_doc():
    bm = rag.BM25()
    bm.add_documents(DOCS)
    results = bm.top_k("gradient descent optimization", k=2)
    assert results
    top_idx = results[0][0]
    assert top_idx in (2, 3)


def test_hashing_embedding_and_cosine():
    a = rag.hashing_embed("neural network gradient", dim=128)
    b = rag.hashing_embed("neural network gradient", dim=128)
    c = rag.hashing_embed("totally unrelated words here", dim=128)
    assert abs(rag.cosine_similarity(a, b) - 1.0) < 1e-9
    assert rag.cosine_similarity(a, c) < rag.cosine_similarity(a, b)


def test_embedding_index():
    idx = rag.EmbeddingIndex(dim=128)
    idx.add(DOCS)
    res = idx.query("neural networks gradient learning", k=2)
    assert res
    assert res[0][0] in (2, 3)


def test_retriever_hybrid():
    r = rag.Retriever(mode="hybrid")
    r.add(DOCS)
    out = r.query("loyal dog plays fetch", k=1)
    assert out
    assert out[0][0] == 1


def test_retriever_modes():
    r = rag.Retriever(mode="lexical")
    r.add(DOCS)
    assert r.query("cat mat", k=1)[0][0] == 0
    d = rag.Retriever(mode="dense")
    d.add(DOCS)
    assert d.query("fox jumps lazy", k=1)[0][0] == 4
