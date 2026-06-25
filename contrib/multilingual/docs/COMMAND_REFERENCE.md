# Command Reference — Multilingual Batch Scanner

> Every command variant from the documentation, deduplicated.
> Replace `./skills/` with `./tests/fixtures/` to run against built-in test data.

---

## Setup

```bash
pip install -e .
cp contrib/multilingual/.env.example .env
```

## Verify upstream

```bash
skillspector scan ./tests/fixtures/malicious_skill/ --no-llm
```

## Static-only (fast, no API keys)

```bash
# Generic
python -m contrib.multilingual.batch_scan ./skills/ --no-llm

# Fixture test
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f terminal --workers 8 --no-llm
```

## LLM mode

```bash
# Generic
python -m contrib.multilingual.batch_scan ./skills/ -f terminal --workers 4

# Fixture tests
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f terminal --workers 1
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f terminal --workers 7
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f terminal --workers 8
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f terminal --workers 20
```

## Output formats

```bash
# Terminal (default)
python -m contrib.multilingual.batch_scan ./skills/ -f terminal
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f terminal --workers 8

# JSON
python -m contrib.multilingual.batch_scan ./skills/ -f json -o report.json
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f json -o report.json --workers 8

# Markdown
python -m contrib.multilingual.batch_scan ./skills/ -f markdown -o report.md
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f markdown -o report.md --workers 8
```

## Language options

```bash
python -m contrib.multilingual.batch_scan ./skills/ --lang auto --workers 4
python -m contrib.multilingual.batch_scan ./tests/fixtures/ --lang zh -f terminal --workers 4
```

## Debugging

```bash
python -m contrib.multilingual.batch_scan ./skills/ --workers 1 -V
python -m contrib.multilingual.batch_scan ./skills/ --workers 4 -V
python -m contrib.multilingual.batch_scan ./tests/fixtures/ --workers 1 -V
```

## Edge cases

```bash
# Static-only, don't require LLM even for non-English
python -m contrib.multilingual.batch_scan ./skills/ --no-require-llm --no-llm
```

## Compare upstream vs batch

```bash
skillspector scan ./tests/fixtures/malicious_skill/ -f json -o upstream.json
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f json -o batch.json --workers 4
```

## CI

```bash
python -m contrib.multilingual.batch_scan ./tests/fixtures/ -f json -o report.json --workers 8
if [ $? -eq 0 ]; then echo "All clean"; fi
```

## Tests

```bash
# Smoke test — verify ApiKeyPool is wired into ALL LLM paths (PR #100 Issue 1)
python contrib/multilingual/tests/test_pool_wiring.py

# Unit tests — random order (seed=42, 120 tests total)
cd contrib/multilingual/tests/tests-pro && python random_numbered.py

# Unit tests — sequential pytest
pytest contrib/multilingual/tests/tests-pro/ -v

# Mutation test — 30 injected bugs across 4 risk areas
python contrib/multilingual/tests/tests-pro/mutation_max.py
```
