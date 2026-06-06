# Markdown Rules

## Line Wrapping

Never hardwrap lines. Each paragraph, list item, and table cell is a single line in the source file; rely on the editor's soft-wrap to render it across visual lines.

Why: hardwrapping makes prose edits noisy (every word added forces re-wrapping the rest of the paragraph) and produces diff churn that obscures the actual content change. A one-line-per-paragraph source is stable under edit and clearly attributes each diff line to a single change.

Applies to:

- Prose paragraphs
- List items — each bullet's content is one line, regardless of length
- Table cells

Exceptions (do not unwrap these):

- Code blocks (```` ``` ```` fenced or indented) — preserve their internal newlines
- ASCII art, tree diagrams, and pre-formatted text inside ```` ```text ```` fences
- The blank line *between* paragraphs (mandatory paragraph separator; not "wrapping")
