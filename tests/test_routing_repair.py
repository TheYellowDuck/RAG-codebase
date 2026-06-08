"""Query routing (heuristic) + self-repair retry on low faithfulness."""
from coderag.config import Settings
from coderag.schema import make_chunk
from coderag.retrieve import RetrievedChunk
from coderag.retrieve.routing import classify_query, route
from coderag.generate import answer_with_repair
from coderag.llm.base import Completion, LLMClient


def test_classify_query():
    assert classify_query("Trace how a request becomes a 422 response") == "multihop"
    assert classify_query("How does the request flow from parsing to the handler") == "multihop"
    assert classify_query("Where is add_api_route defined?") == "where"
    assert classify_query("How do I declare a query parameter?") == "howto"
    assert classify_query("What is APIRouter?") == "lookup"


def test_route_widens_for_multihop():
    assert route("trace the dependency flow end to end").get("k") == 8
    assert route("where is X defined").get("k") == 4
    assert "k" not in route("what is X")  # default effort


def _chunk():
    return make_chunk(repo="r", file_path="a.py", language="python", symbol_name="f",
                      symbol_type="function", start_line=1, end_line=3,
                      code="def f(): return 1", context_header="h", git_sha="s")


class _SeqLLM(LLMClient):
    """Returns a different answer each generate() call; faithfulness scores queued."""
    provider = "fake"
    def __init__(self, answers, faith_scores):
        super().__init__("fake-gen", "fake-judge")
        self._answers = list(answers)
        self._faith = list(faith_scores)
    def generate(self, system, user, *, max_tokens, stream=False):
        return Completion(self._answers.pop(0), {})
    def judge_json(self, prompt, schema, *, max_tokens=2048):
        # one supported/unsupported claim per call to hit the queued faithfulness
        supported = self._faith.pop(0) >= 0.5
        return {"claims": [{"text": "c", "sources": [1], "supported": supported,
                            "reason": "x"}]}


def test_self_repair_off_by_default():
    res = [RetrievedChunk(_chunk(), 1.0, 1)]
    llm = _SeqLLM(answers=["first [1]"], faith_scores=[])
    out = answer_with_repair("q", res, Settings(), client=llm)  # threshold 0 -> no retry
    assert out.answer == "first [1]"


def test_self_repair_retries_and_keeps_better():
    res = [RetrievedChunk(_chunk(), 1.0, 1)]
    s = Settings(self_repair_threshold=0.8)
    # first answer scores 0.0 (below threshold) -> retry scores 1.0 -> keep retry
    llm = _SeqLLM(answers=["weak [1]", "strict [1]"], faith_scores=[0.0, 1.0])
    out = answer_with_repair("q", res, s, client=llm)
    assert out.answer == "strict [1]"
