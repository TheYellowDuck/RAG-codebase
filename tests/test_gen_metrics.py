"""Generation metrics (coderag/eval/gen_metrics.py): citation precision/recall is
pure set logic (tested directly); answer-correctness grading is exercised with a
stub judge so no LLM/key is needed."""
from coderag.config import Settings
from coderag.eval.gen_metrics import answer_correctness, citation_precision_recall


class _Source:                      # stand-in for generate.Source (needs .n, .file_path)
    def __init__(self, n, file_path):
        self.n = n
        self.file_path = file_path


def test_citation_precision_recall_partial():
    sources = [_Source(1, "a.py"), _Source(2, "b.py"), _Source(3, "c.py")]
    r = citation_precision_recall(sources, cited_ns=[1, 2], relevant_files=["a.py", "x.py"])
    # cited files {a.py, b.py}; only a.py is relevant
    assert r["citation_precision"] == 0.5          # 1 of 2 cited files relevant
    assert r["citation_recall"] == 0.5             # a.py of {a.py, x.py}
    assert r["cited_files"] == ["a.py", "b.py"]


def test_citation_precision_recall_perfect_and_empty():
    sources = [_Source(1, "a.py"), _Source(2, "b.py")]
    perfect = citation_precision_recall(sources, [1], ["a.py"])
    assert perfect["citation_precision"] == 1.0 and perfect["citation_recall"] == 1.0
    none = citation_precision_recall(sources, [], ["a.py"])   # nothing cited
    assert none["citation_precision"] == 0.0 and none["citation_recall"] == 0.0


def test_answer_correctness_grade_mapping():
    class _Judge:
        def __init__(self, grade):
            self.grade = grade
        def judge_json(self, prompt, schema, max_tokens=512):
            return {"grade": self.grade, "reason": "stub"}
    s = Settings()
    assert answer_correctness("q", "a", "ref", s, client=_Judge("correct"))["score"] == 1.0
    assert answer_correctness("q", "a", "ref", s, client=_Judge("partially_correct"))["score"] == 0.5
    assert answer_correctness("q", "a", "ref", s, client=_Judge("wrong"))["score"] == 0.0
    assert answer_correctness("q", "a", "ref", s, client=_Judge("nonsense"))["score"] == 0.0  # unknown → wrong
