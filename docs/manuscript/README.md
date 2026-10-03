# hapmixQTL manuscript draft

The editable source is [hapmixqtl_draft.md](hapmixqtl_draft.md). The reading copy is
[hapmixqtl_draft.html](hapmixqtl_draft.html), and the Word copy is
[hapmixqtl_draft.docx](hapmixqtl_draft.docx). Both are rendered from the same source.
The HTML embeds its stylesheet and available figure and uses native MathML;
opening it does not require loading a remote math library.

This is a methods-pilot draft based on completed results. Observed-data discovery,
held-out replication, authorship and submission declarations are explicitly pending.
The author working notes at the end identify proposed figures and submission gaps.
Drafting does not lift the project's holds on alignment work or re-quantification.

[evidence.tsv](evidence.tsv) records numerical provenance and limitations;
[evidence_notes.md](evidence_notes.md) records source fingerprints and interpretation
rules. [references.bib](references.bib) contains the bibliography in manuscript order.
The native RASQUAL graphic is copied from the completed pilot report, rather than
generated from a new experiment. Its origin is recorded in the artifact manifest.

To regenerate the reading and Word copies with Pandoc from this directory:

```bash
pandoc hapmixqtl_draft.md --standalone --mathml --embed-resources --css manuscript.css -o hapmixqtl_draft.html
pandoc hapmixqtl_draft.md -o hapmixqtl_draft.docx
```

The Word bibliography is rendered manuscript text, not a Zotero-linked field.
Some bibliography records abbreviate long author lists with `and others`; complete
them for the chosen journal during submission preparation.
