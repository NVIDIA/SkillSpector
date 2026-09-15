// Unit tests for the OpenCode plugin helpers. Stdlib only:
//   node --test tests/opencode/skillspector_scan_lib.test.ts
// Requires Node 22+ (type stripping).

import { describe, it } from "node:test"
import assert from "node:assert/strict"
import path from "node:path"
import {
  MAX_STDERR,
  MAX_STDOUT,
  TIMEOUT_MS,
  buildCliArgs,
  formatExecError,
  formatSuccess,
  isUrlOrAbsolute,
  redact,
  resolveBinary,
  truncate,
} from "../../.opencode/tools/skillspector_scan_lib.ts"

describe("buildCliArgs", () => {
  it("re-applies omitted defaults: json format, LLM off", () => {
    assert.deepEqual(buildCliArgs({ target: "skill" }), [
      "scan",
      "skill",
      "--format",
      "json",
      "--no-llm",
    ])
  })

  it("passes explicit values through, including LLM opt-in", () => {
    assert.deepEqual(
      buildCliArgs({
        target: "skill",
        format: "terminal",
        noLlm: false,
        output: "out.json",
      }),
      ["scan", "skill", "--format", "terminal", "--output", "out.json"],
    )
  })

  it("keeps explicit --no-llm", () => {
    assert.ok(buildCliArgs({ target: "s", noLlm: true }).includes("--no-llm"))
  })
})

describe("truncate", () => {
  it("leaves short text alone", () => {
    assert.equal(truncate("hi", 10), "hi")
  })

  it("caps long text with remaining count", () => {
    const out = truncate("x".repeat(MAX_STDOUT + 5), MAX_STDOUT)
    assert.ok(out.startsWith("x".repeat(MAX_STDOUT)))
    assert.ok(out.endsWith("[truncated 5 chars]"))
  })
})

describe("redact", () => {
  it("redacts keys and tokens, keeps prose", () => {
    const out = redact(
      'sk-ant-secret123 and sk-abcdef OPENAI_API_KEY="hunter2" X_TOKEN: abc plain words',
    )
    assert.ok(!out.includes("secret123"))
    assert.ok(!out.includes("hunter2"))
    assert.ok(!out.includes(" abc"))
    assert.ok(out.includes("plain words"))
    assert.ok(out.includes("[REDACTED]"))
  })
})

describe("resolveBinary", () => {
  it("prefers SKILLSPECTOR_BIN", () => {
    assert.equal(
      resolveBinary("/wt", { env: { SKILLSPECTOR_BIN: " /bin/custom " } }),
      "/bin/custom",
    )
  })

  it("falls back to the checkout venv when present", () => {
    assert.equal(
      resolveBinary("/wt", {
        env: {},
        platform: "win32",
        existsSync: () => true,
      }),
      path.join("/wt", ".venv", "Scripts", "skillspector.exe"),
    )
  })

  it("falls back to PATH lookup", () => {
    assert.equal(
      resolveBinary("/wt", { env: {}, existsSync: () => false }),
      "skillspector",
    )
  })
})

describe("isUrlOrAbsolute", () => {
  it("accepts URLs and absolute paths, rejects relatives", () => {
    assert.equal(isUrlOrAbsolute("https://example.com/skill"), true)
    assert.equal(isUrlOrAbsolute("./relative"), false)
    assert.equal(isUrlOrAbsolute("relative/path"), false)
  })
})

describe("formatExecError", () => {
  it("maps ENOENT to the install hint", () => {
    const out = formatExecError("/bin/missing", { code: "ENOENT" })
    assert.ok(out.includes('tried "/bin/missing"'))
    assert.ok(out.includes("SKILLSPECTOR_BIN"))
  })

  it("maps kills to the timeout message", () => {
    const out = formatExecError("bin", { killed: true, stdout: "part" })
    assert.ok(out.includes(`${TIMEOUT_MS / 1000}s`))
    assert.ok(out.includes("part"))
  })

  it("returns exit-1 stdout as the report", () => {
    assert.equal(formatExecError("bin", { code: 1, stdout: '{"a":1}' }), '{"a":1}')
  })

  it("maps exit 2 to the usage error", () => {
    const out = formatExecError("bin", { code: 2, stderr: "bad flag" })
    assert.ok(out.includes("usage error"))
    assert.ok(out.includes("bad flag"))
  })

  it("redacts secrets in failure output", () => {
    const out = formatExecError("bin", {
      code: 9,
      stderr: "GROQ_API_KEY=hunter2",
    })
    assert.ok(!out.includes("hunter2"))
  })
})

describe("formatSuccess", () => {
  it("reports the saved path when output is silent", () => {
    assert.equal(formatSuccess("r.json", "", ""), "Report saved to: r.json")
  })

  it("returns truncated stdout", () => {
    assert.equal(formatSuccess(undefined, "ok", ""), "ok")
    assert.ok(formatSuccess(undefined, "y".repeat(MAX_STDOUT + 1), "").endsWith("]"))
  })
})

describe("constants", () => {
  it("keeps the documented caps", () => {
    assert.equal(TIMEOUT_MS, 120_000)
    assert.equal(MAX_STDOUT, 12_000)
    assert.equal(MAX_STDERR, 6_000)
  })
})
