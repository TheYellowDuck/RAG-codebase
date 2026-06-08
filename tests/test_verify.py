from coderag.verify import structural_check, parse_citations


def test_parse_citations():
    assert parse_citations("a [1] b [2] c [1]") == [1, 2, 1]
    assert parse_citations("no citations here") == []


def test_structural_valid_citations():
    ans = "The router registers routes [1]. Validation raises a 422 [2]."
    chk = structural_check(ans, num_sources=2)
    assert chk["valid_citations"] == [1, 2]
    assert chk["invalid_citations"] == []
    assert chk["all_citations_valid"] is True
    assert chk["uncited_claim_sentences"] == []


def test_structural_flags_out_of_range_citation():
    chk = structural_check("Claim with bad cite [3].", num_sources=2)
    assert chk["invalid_citations"] == [3]
    assert chk["all_citations_valid"] is False


def test_structural_flags_uncited_claim():
    ans = "The dispatcher routes requests to handlers based on the path."
    chk = structural_check(ans, num_sources=2)
    assert len(chk["uncited_claim_sentences"]) == 1


def test_structural_does_not_flag_abstention():
    ans = "The sources do not contain information about the database pool size."
    chk = structural_check(ans, num_sources=2)
    assert chk["uncited_claim_sentences"] == []


def test_verify_answer_survives_judge_failure():
    """If the LLM judge errors (e.g. a small model emits invalid JSON), the free
    structural check must still run and report — including flagging a cited source
    number that doesn't exist."""
    from coderag.config import Settings
    from coderag.generate import Source
    from coderag.verify import verify_answer

    class BadJudge:
        provider, gen_model, judge_model = "fake", "g", "j"
        def judge_json(self, prompt, schema, *, max_tokens=2048):
            raise ValueError("invalid json from model")

    sources = [Source(n=1, chunk_id="c", file_path="a.py", start_line=1, end_line=2, code="x")]
    # cites [2] but only 1 source exists -> structural must flag it
    report = verify_answer("Claim A [1]. Claim B [2].", sources, Settings(), client=BadJudge())
    assert report["structural"]["invalid_citations"] == [2]
    assert "faithfulness" not in report
    assert "faithfulness_error" in report


def test_faithfulness_excludes_abstention_claims():
    """An honest 'the sources don't cover X' line must not count against
    faithfulness — otherwise we'd penalize the abstaining behavior we want."""
    from coderag.config import Settings
    from coderag.generate import Source
    from coderag.verify import faithfulness_score
    from coderag.verify.faithfulness import is_abstention

    assert is_abstention("The sources do not contain the low-level mechanism.")
    assert not is_abstention("include_router registers routes via the router.")

    class FakeJudge:
        provider, gen_model, judge_model = "fake", "g", "j"
        def __init__(self):
            self.q = [
                {"claims": [
                    {"text": "include_router registers routes", "sources": [1],
                     "supported": True, "reason": "ok"},
                    {"text": "The sources do not contain the low-level mechanism",
                     "sources": [], "supported": False, "reason": "abstain"},
                ]},
            ]
        def judge_json(self, prompt, schema, *, max_tokens=2048):
            return self.q.pop(0)

    sources = [Source(n=1, chunk_id="c", file_path="a.py", start_line=1, end_line=2,
                      code="def include_router(): ...")]
    out = faithfulness_score(
        "include_router registers routes [1]. The sources do not contain the low-level mechanism.",
        sources, Settings(), client=FakeJudge())
    assert out["n_claims"] == 1          # abstention claim dropped
    assert out["n_supported"] == 1
    assert out["faithfulness"] == 1.0
