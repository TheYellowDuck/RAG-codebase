"""Golden-set scaffolding: construction-verified labels + LLM paraphrase pass."""
from coderag.index import CodeIndex
from coderag.config import Settings
from coderag.eval.scaffold import build_questions, paraphrase_questions


def test_build_questions_labels_are_construction_verified(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    qs = build_questions(idx, n=20, start_id=46, holdout_every=0)

    assert qs and all(q["id"].startswith("q") for q in qs)
    # every relevant_file actually corresponds to an indexed file
    indexed = {c.file_path for c in idx.chunks.values()}
    for q in qs:
        assert q["relevant_files"], q
        assert all(f in indexed for f in q["relevant_files"])

    # the sample repo has a cross-file call (top_level in a.py -> helper in b.py),
    # so a cross-file question spanning both files must appear.
    cross = [q for q in qs if q["type"] == "cross-file"]
    assert any(len(q["relevant_files"]) >= 2 and
               any(f.endswith("a.py") for f in q["relevant_files"]) and
               any(f.endswith("b.py") for f in q["relevant_files"]) for q in cross)


def test_build_questions_excludes_used_symbols(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    all_syms = {s for q in build_questions(idx, n=50, holdout_every=0)
                for s in q["relevant_symbols"]}
    # excluding a symbol must drop every question that references it
    pick = sorted(all_syms)[0]   # deterministic (no set-iteration-order reliance)
    qs = build_questions(idx, n=50, exclude_symbols={pick}, holdout_every=0)
    assert all(pick not in q["relevant_symbols"] for q in qs)


def test_paraphrase_rewrites_text_preserves_labels():
    class FakeLLM:
        provider, gen_model, judge_model = "fake", "g", "j"
        def judge_json(self, prompt, schema, *, max_tokens=2048):
            return {"rewrites": [
                {"id": "q046", "question": "How is a value checked before the handler runs?"},
                # q047 intentionally omitted -> original must be kept
            ]}

    qs = [
        {"id": "q046", "question": "How does `validate` use `check_field`?",
         "type": "cross-file", "relevant_files": ["a.py", "b.py"],
         "relevant_symbols": ["validate", "check_field"], "reference_answer": "..."},
        {"id": "q047", "question": "How does `route` use `add_api_route`?",
         "type": "cross-file", "relevant_files": ["routing.py"],
         "relevant_symbols": ["route", "add_api_route"], "reference_answer": "..."},
    ]
    out = paraphrase_questions(qs, FakeLLM())
    by_id = {q["id"]: q for q in out}
    # rewritten text, no identifiers; labels untouched
    assert by_id["q046"]["question"] == "How is a value checked before the handler runs?"
    assert "`" not in by_id["q046"]["question"]
    assert by_id["q046"]["relevant_symbols"] == ["validate", "check_field"]
    # missing rewrite -> original preserved
    assert by_id["q047"]["question"] == "How does `route` use `add_api_route`?"


def test_paraphrase_survives_judge_failure():
    class BadLLM:
        provider, gen_model, judge_model = "fake", "g", "j"
        def judge_json(self, prompt, schema, *, max_tokens=2048):
            raise ValueError("bad json")

    qs = [{"id": "q046", "question": "orig", "type": "where",
           "relevant_files": ["a.py"], "relevant_symbols": ["x"], "reference_answer": "."}]
    out = paraphrase_questions(qs, BadLLM())
    assert out[0]["question"] == "orig"  # unchanged on failure
