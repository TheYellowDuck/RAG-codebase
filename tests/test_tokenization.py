from coderag.tokenization import code_tokens, token_len


def test_snake_case_split():
    toks = code_tokens("get_current_user")
    assert "get" in toks and "current" in toks and "user" in toks
    assert "get_current_user" not in toks  # underscores are separators


def test_camel_case_split_keeps_whole_and_parts():
    toks = code_tokens("getCurrentUser")
    assert "getcurrentuser" in toks   # whole identifier, lowercased
    assert "get" in toks and "current" in toks and "user" in toks


def test_camel_and_snake_both_match_query_tokens():
    q = set(code_tokens("get current user"))
    assert q <= set(code_tokens("get_current_user"))
    assert q <= set(code_tokens("getCurrentUser"))


def test_mixed_and_symbols():
    toks = code_tokens("HTTPException(status_code=404)")
    assert "httpexception" in toks
    assert "status" in toks and "code" in toks


def test_token_len_monotonic():
    assert token_len("") == 0
    assert token_len("a" * 4) >= 1
    assert token_len("a" * 400) > token_len("a" * 40)
