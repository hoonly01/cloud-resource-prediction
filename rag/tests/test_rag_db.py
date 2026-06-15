import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Time-LLM"))

import numpy as np
import pytest
from build_rag_db import extract_stats


class TestExtractStats:
    def test_output_shape(self):
        x = np.random.randn(10, 32, 10).astype(np.float32)
        assert extract_stats(x).shape == (10, 50)

    def test_constant_signal_mean_and_zero_trend(self):
        x = np.ones((3, 32, 10), dtype=np.float32) * 2.0
        out = extract_stats(x)
        assert np.allclose(out[:, :10], 2.0)               # mean
        assert np.allclose(out[:, 10:20], 0.0, atol=1e-5)  # std=0
        assert np.allclose(out[:, 40:50], 0.0, atol=1e-5)  # trend=0

    def test_upward_trend_positive(self):
        x = np.zeros((1, 32, 1), dtype=np.float32)
        x[0, :, 0] = np.linspace(0, 1, 32)
        out = extract_stats(x)  # shape (1, 5*1=5)
        assert out[0, 4] > 0  # trend of single channel is at index 4


import json
import faiss
from utils.rag import RAGRetriever, extract_stats as rag_extract_stats


@pytest.fixture
def tiny_db(tmp_path):
    N, T, C = 20, 32, 10
    rng = np.random.default_rng(42)
    x = rng.standard_normal((N, T, C)).astype(np.float32)
    y = rng.standard_normal((N, T, C)).astype(np.float32)
    rag_class = np.array([0, 1, 2, 3, 4] * 4, dtype=np.int8)
    is_fail = (rag_class >= 3).astype(np.int8)

    emb = rag_extract_stats(x)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb = (emb / norms).astype(np.float32)
    index = faiss.IndexFlatL2(50)
    index.add(emb)

    faiss.write_index(index, str(tmp_path / "faiss.index"))
    np.save(tmp_path / "embeddings.npy", emb)
    np.savez(tmp_path / "metadata.npz", y=y, rag_class=rag_class, is_fail=is_fail)
    with open(tmp_path / "db_info.json", "w") as f:
        json.dump({
            "cpu_pressure_idx": 6,
            "mem_pressure_idx": 7,
            "cpu_burst_idx": 8,
        }, f)
    return tmp_path


class TestRAGRetriever:
    def test_query_returns_list_of_strings(self, tiny_db):
        retriever = RAGRetriever(str(tiny_db))
        x = np.random.randn(4, 32, 10).astype(np.float32)
        results = retriever.query(x, top_k=3)
        assert len(results) == 4
        assert all(isinstance(r, str) for r in results)

    def test_pap_text_contains_expected_fields(self, tiny_db):
        retriever = RAGRetriever(str(tiny_db))
        x = np.random.randn(1, 32, 10).astype(np.float32)
        text = retriever.query(x)[0]
        assert "failure rate" in text
        assert "dominant pattern" in text
        assert "cpu_pressure" in text
        assert "cpu_burst" in text

    def test_extract_stats_identical_to_build_script(self):
        x = np.random.randn(5, 32, 10).astype(np.float32)
        assert np.allclose(extract_stats(x), rag_extract_stats(x))

    def test_pap_text_contains_mem_pressure(self, tiny_db):
        retriever = RAGRetriever(str(tiny_db))
        x = np.random.randn(1, 32, 10).astype(np.float32)
        text = retriever.query(x)[0]
        assert "mem_pressure" in text

    def test_pap_text_format(self, tiny_db):
        retriever = RAGRetriever(str(tiny_db))
        x = np.random.randn(1, 32, 10).astype(np.float32)
        text = retriever.query(x)[0]
        assert "Future outlook" in text
        assert "mean" in text
        assert "max" in text

    def test_random_mode_returns_correct_count(self, tiny_db):
        retriever = RAGRetriever(str(tiny_db))
        x = np.random.randn(3, 32, 10).astype(np.float32)
        results = retriever.query(x, top_k=5, mode='random')
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_fixed_mode_returns_same_string_for_all(self, tiny_db):
        retriever = RAGRetriever(str(tiny_db))
        x = np.random.randn(4, 32, 10).astype(np.float32)
        results = retriever.query(x, mode='fixed')
        assert len(results) == 4
        assert len(set(results)) == 1

    def test_normal_mode_is_default(self, tiny_db):
        retriever = RAGRetriever(str(tiny_db))
        x = np.random.randn(2, 32, 10).astype(np.float32)
        assert retriever.query(x) == retriever.query(x, mode='normal')
