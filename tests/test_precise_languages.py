"""Verify each precise LanguageSpec against a real sample.

A language is skipped only if its grammar isn't installed (so this runs broadly
with `pip install 'coderag[dev]'`, which brings tree-sitter-language-pack).
Node types in the specs were derived empirically; these tests guard against a
grammar update (or a typo) breaking extraction.
"""
import pytest

from coderag.ingest.chunker import chunk_file
from coderag.ingest.languages import get_parser, PRECISE

# (language, filename, source, expected_symbols ⊆, expected_calls ⊆)
CASES = [
    ("python", "a.py",
     "import os\nfrom m import helper\n\ndef top(x):\n    return helper(x)\n\nclass C:\n    def m(self):\n        return self.top()\n",
     {"top", "C", "C.m"}, {"helper", "top"}),
    ("javascript", "a.js",
     "import {helper} from './m';\nfunction top(x){ return helper(x); }\nclass C { m(){ return this.top(); } }\n",
     {"top", "C", "C.m"}, {"helper"}),
    ("typescript", "a.ts",
     "import {helper} from './m';\nfunction top(x:number){ return helper(x); }\nclass C { m(){ return this.top(); } }\ninterface I { f():void; }\n",
     {"top", "C", "C.m", "I"}, {"helper"}),
    ("go", "a.go",
     'package main\nimport "fmt"\nfunc Top(x int) int { return helper(x) }\nfunc (c C) M() { fmt.Println() }\n',
     {"Top", "M"}, {"helper"}),
    ("rust", "a.rs",
     "use std::fmt;\nfn top(x:i32)->i32 { helper(x) }\nstruct S{x:i32}\nimpl S { fn m(&self){ self.top(); } }\n",
     {"top", "S", "S.m"}, {"helper"}),
    ("ruby", "a.rb",
     "require 'set'\ndef top(x)\n  helper(x)\nend\nclass C\n  def m\n    helper\n  end\nend\n",
     {"top", "C", "C.m"}, {"helper"}),
    ("java", "A.java",
     "import java.util.List;\nclass C {\n  int m(int x){ return helper(x); }\n}\ninterface I { void f(); }\n",
     {"C", "C.m", "I"}, {"helper"}),
    ("c", "a.c",
     "#include <stdio.h>\nint add(int a){ return mul(a); }\nint mul(int a){ return a; }\n",
     {"add", "mul"}, {"mul"}),
    ("cpp", "a.cpp",
     "#include <vector>\nnamespace N {\nclass C { public: void m(){ helper(); } };\n}\nint f(){ return 1; }\n",
     {"N", "N.C", "N.C.m", "f"}, {"helper"}),
    ("csharp", "A.cs",
     "using System;\nnamespace N {\n class C {\n  int M(int x){ return Helper(x); }\n }\n}\n",
     {"N", "N.C", "N.C.M"}, {"Helper"}),
    ("php", "a.php",
     "<?php\nfunction top($x){ return helper($x); }\nclass C { function m(){ return $this->top(); } }\n",
     {"top", "C", "C.m"}, {"helper"}),
    ("kotlin", "a.kt",
     "import a.b\nfun top(x:Int):Int { return helper(x) }\nclass C { fun m(){ top() } }\n",
     {"top", "C", "C.m"}, {"helper"}),
    ("scala", "a.scala",
     "import a.b\ndef top(x:Int):Int = helper(x)\nclass C { def m() = top() }\nobject O {}\n",
     {"top", "C", "C.m", "O"}, {"helper"}),
    ("swift", "a.swift",
     "import Foundation\nfunc top(x:Int)->Int { return helper(x) }\nclass C { func m(){ top() } }\n",
     {"top", "C", "C.m"}, {"helper"}),
    ("lua", "a.lua",
     'local m = require("m")\nfunction top(x) return helper(x) end\nlocal function helper() end\n',
     {"top", "helper"}, {"helper"}),
    ("bash", "a.sh",
     "source ./lib.sh\ntop() {\n  greet \"$1\"\n}\n",
     {"top"}, set()),
    ("perl", "a.pl",
     "use strict;\nsub top { return helper(); }\n",
     {"top"}, set()),
    ("objc", "a.m",
     "#import <Foundation/Foundation.h>\n@interface C : NSObject\n- (int)m;\n@end\nint add(int a){ return mul(a); }\n",
     {"C", "add"}, {"mul"}),
]


def test_all_cases_cover_precise_specs():
    """Every precise language (except the .tsx alias) has a test case."""
    covered = {lang for lang, *_ in CASES}
    specs = {lang for lang in PRECISE if lang != "tsx"}
    assert specs <= covered, f"untested precise specs: {specs - covered}"


@pytest.mark.parametrize("lang,path,src,want_syms,want_calls", CASES,
                         ids=[c[0] for c in CASES])
def test_precise_extraction(lang, path, src, want_syms, want_calls, make_fileinfo):
    if get_parser(lang) is None:
        pytest.skip(f"no tree-sitter grammar installed for {lang}")
    parse = chunk_file(make_fileinfo(path, src, lang), "repo", "sha")
    assert parse.parsed is True, f"{lang} fell back to window chunking"
    syms = {c.symbol_name for c in parse.chunks}
    assert want_syms <= syms, f"{lang}: missing {want_syms - syms}"
    calls = {name for _, name in parse.calls}
    assert want_calls <= calls, f"{lang}: missing calls {want_calls - calls}"
