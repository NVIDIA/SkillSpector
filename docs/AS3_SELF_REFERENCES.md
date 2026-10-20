# AS3 current-skill references

AS3 detects access to other installed skills. A literal reference to the
currently selected skill, such as `skills/example-skill/SKILL.md`, is excluded
when the scanner can establish that skill's identity from the selected input.

Input resolution preserves one identity through temporary materialization:

| Selected input | Identity |
| --- | --- |
| Local skill directory | Resolved directory basename |
| Git repository | Repository basename from the successfully cloned input URL |
| Local `SKILL.md` file | Original file's absolute parent directory basename |
| Archive with one outer skill directory | Preserved outer directory basename |
| Flat local ZIP archive | Selected archive filename without `.zip` |

For example, a flat `example-skill.zip` containing `SKILL.md` with
`name: example-skill` retains that identity after extraction to a temporary
`extracted` directory. A `download.zip` containing an `example-skill/` directory
uses the contained directory name. The archive filename does not become a
second identity for a preserved skill directory.

The manifest name must agree with the selected identity when present. It does
not independently authorize suppression: changing `name` to a peer skill's
name must not hide references to that peer. Case and underscore/hyphen variants
remain distinct. Obfuscated paths, explicit enumeration, AS1, and AS2 retain
their existing detection behavior. Filtering happens before finding budgets
and inspection-ledger emission.

An arbitrary externally renamed staging directory, an ambiguous direct-download
URL, or a flat downloaded archive may lack enough provenance to establish the
skill name. Such inputs retain AS3 findings. URL path segments can represent
repository names or slash-containing Git refs, so they are not guessed as
skill aliases. Integrations should preserve the selected skill-directory name
or supply the original supported input to the scanner.
