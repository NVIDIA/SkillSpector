# Passive documentation images

An unsupported image format does not by itself establish analysis evasion.
SkillSpector keeps image coverage limitations visible while omitting format-only
AE1 for a positively identified documentation image with corroborated bytes.
Independent findings and genuine analysis failures remain blocking.

Two reference forms are supported:

- A rendered Markdown image, such as `![Overview](assets/overview.png)`.
- An unindented standalone paragraph such as ``Asset: `assets/overview.png`.``
  The supported labels are Asset, Image, Illustration, Diagram, Figure,
  Screenshot, Photo and Picture, case-insensitively. The label must be followed
  by a colon and one single-backtick literal `.png`, `.jpg` or `.jpeg` path.
  A final period is optional. Blank paragraph boundaries are required.

The declaration form has a distinct `documentation_image` reference kind.
Declarations and newly supported rendered JPEG references carry a `source_sha256`
binding. Finalization verifies their exact source bytes, position, role and target
again, with complete bounded reference extraction. The existing rendered-PNG
contract remains unchanged. Frontmatter, list or blockquote continuations, fenced or
indented code, HTML, open code spans, command paragraphs, multiple paths and
mismatched delimiters do not qualify. An independent executable or ambiguous use
of the same target keeps AE1, including when the public reference record is
missing but the canonical source still contains that use. Ordinary quoted paths
and Markdown links do not gain this policy.

PNG support remains the existing bounded non-interlaced format with verified
chunk lengths, CRCs and decoded scanline layout. JPEG support is a conservative
baseline JFIF subset with verified marker, table, frame, scan, Huffman block and
terminal padding structure. Metadata such as COM or EXIF, unsupported coding
modes, malformed data and trailing bytes do not qualify. Structural validation
does not inspect rendered content, perform OCR, or establish that an image's
meaning is safe.

Complete canonical bytes must match the discovered asset size and suffix. Mixed,
fatal, missing or truncated inspection evidence prevents the exception. Source
corroboration is cached per source and bounded by a shared one-million-byte,
16-source and two-second allowance. Format proofs are cached per asset and
bounded by 256 attempts, 128 MiB encoded input and two seconds, in addition to
each format's own deterministic bounds. Failed attempts consume the allowance.
An exhausted proof preserves AE1 and the original incomplete coverage evidence.

This policy does **not** mark image bytes fully inspected or change raw native
`partial` into another status. Consumers may admit a corroborated passive-image
limitation as an advisory; they must still consider other findings and limits.
Native completion and a consumer's admission decision remain separate signals.
