# Writing Footnotes

Write a note directly where you want its reference to appear:

```markdown
Picking the most likely token\footnote{This is naive; there's a whole rabbit hole here.}
```

The note is numbered automatically and appears in the right margin on wide
screens. Its contents support Markdown formatting, links, and math, including
nested braces such as `$p_{\theta}(x)$`. Code spans and code blocks keep the
command literal. Keep the command on one source line; for multi-paragraph notes,
use the named syntax below. An unclosed command is left as text.

Standard Markdown footnotes also work, and are useful for reusing the same note:

```markdown
The ELBO is a lower bound on log-likelihood.[^elbo]

[^elbo]: ELBO stands for evidence lower bound.
```

Place definitions at the end of your article. Labels such as `elbo` can be any
unique name; numbering is automatic. Notes support links, emphasis, and math.
Indent additional paragraphs in a note by four spaces.

On screens at least 800px wide, notes appear in a reserved right-hand column
alongside their first reference.
Closely spaced notes stack without overlapping. On smaller screens, in print, or
without JavaScript, they appear as a footnote list at the end, with links back to
the text. The same note can be referenced more than once.
