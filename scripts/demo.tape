# Animated terminal demo for the README.
# Render with charmbracelet/vhs (https://github.com/charmbracelet/vhs):
#     brew install vhs   # or: go install github.com/charmbracelet/vhs@latest
#     vhs scripts/demo.tape      # -> docs/demo.gif
# Uses retrieval-only (no API key, no token cost) so anyone can reproduce it.

Output docs/demo.gif

Set FontSize 18
Set Width 1200
Set Height 640
Set Theme "Dracula"
Set Padding 18
Set TypingSpeed 45ms

Type "# index this repo (AST chunks + hybrid index + code graph)"
Enter
Type "coderag index coderag --out .coderag_index"
Enter
Sleep 9s

Type "# ask by meaning — retrieval only, no LLM key needed"
Enter
Type `coderag query "reciprocal rank fusion of two ranked lists" --retrieve-only`
Enter
Sleep 5s

Type "# it pinpoints the rrf() function with no shared keywords"
Enter
Sleep 3s
