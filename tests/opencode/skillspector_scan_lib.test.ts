// Unit tests for the OpenCode plugin helpers. Stdlib only:
//   node --test tests/opencode/skillspector_scan_lib.test.ts
// Requires Node 22+ (type stripping).

import { describe, it } from "node:test"
import assert from "node:assert/strict"
import path from "node:path"
import {
  CREDENTIAL_ENV_NAMES,
  MAX_STDERR,
  MAX_STDOUT,
  TIMEOUT_MS,
  buildCliArgs,
  childProcessEnv,
  executeScan,
  formatExecError,
  formatSuccess,
  isFilesystemRoot,
  isRemoteTarget,
  isScpGitTarget,
  isUrlOrAbsolute,
  llmPermissionMetadata,
  redact,
  resolveBinary,
  truncate,
  type PermissionRequest,
  type RunFile,
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

  it("redacts values for every supported credential environment name", () => {
    const env = Object.fromEntries(
      CREDENTIAL_ENV_NAMES.map((name, index) => [name, `credential-value-${index}`]),
    )
    const out = redact(Object.values(env).join(" "), env)
    for (const value of Object.values(env)) assert.ok(!out.includes(value))
    assert.equal(out.match(/\[REDACTED\]/g)?.length, CREDENTIAL_ENV_NAMES.length)
  })

  it("redacts NVIDIA and AWS assignments even when the environment is unavailable", () => {
    const out = redact(
      "NVIDIA_INFERENCE_KEY=nvapi-secret AWS_SECRET_ACCESS_KEY: aws-secret",
      {},
    )
    assert.ok(!out.includes("nvapi-secret"))
    assert.ok(!out.includes("aws-secret"))
  })

  it("redacts overlapping and regex-bearing credential values in one pass", () => {
    const out = redact("long-secret-value secret-value a+b*c?[x] REDA", {
      OPENAI_API_KEY: "long-secret-value",
      ANTHROPIC_API_KEY: "secret-value",
      NVIDIA_INFERENCE_KEY: "a+b*c?[x]",
      AWS_SECRET_ACCESS_KEY: "REDA",
    })
    assert.equal(out, "[REDACTED] [REDACTED] [REDACTED] [REDACTED]")
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
      path.win32.join("/wt", ".venv", "Scripts", "skillspector.exe"),
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
    assert.equal(isUrlOrAbsolute("custom://example.com/skill"), false)
    assert.equal(isUrlOrAbsolute("./relative"), false)
    assert.equal(isUrlOrAbsolute("relative/path"), false)
  })

  it("accepts supported SCP-style Git targets without accepting arbitrary colons", () => {
    const target = "git@github.com:NVIDIA/SkillSpector.git"
    assert.equal(isScpGitTarget(target), true)
    assert.equal(isRemoteTarget(target), true)
    assert.equal(isUrlOrAbsolute(target), true)
    assert.equal(isScpGitTarget("notes:today"), false)
    assert.equal(isUrlOrAbsolute("notes:today"), false)
    assert.equal(isScpGitTarget("git@github.com:missing-git-suffix"), false)
  })

  it("recognizes POSIX and Windows filesystem-root sentinels", () => {
    assert.equal(isFilesystemRoot("/", "linux"), true)
    assert.equal(isFilesystemRoot("/work", "linux"), false)
    assert.equal(isFilesystemRoot("C:\\", "win32"), true)
    assert.equal(isFilesystemRoot("C:\\work", "win32"), false)
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

  it("labels exit 2 as usage only when CLI usage evidence proves it", () => {
    const out = formatExecError("bin", {
      code: 2,
      stderr: "Usage: skillspector scan [OPTIONS] INPUT\nError: No such option: --bad",
    })
    assert.ok(out.includes("usage error"))
    assert.ok(out.includes("No such option"))
  })

  it("preserves report and diagnostics produced before exit 2", () => {
    const out = formatExecError("bin", {
      code: 2,
      stdout: '{"execution_successful":false,"findings":[{"rule_id":"SC1"}]}',
      stderr: "scan accounting was incomplete",
    })
    assert.ok(out.includes('"execution_successful":false'))
    assert.ok(out.includes('"rule_id":"SC1"'))
    assert.ok(out.includes("stderr:"))
    assert.ok(out.includes("accounting was incomplete"))
    assert.ok(!out.includes("usage error"))
  })

  it("distinguishes session cancellation and preserves partial evidence", () => {
    const out = formatExecError("bin", {
      name: "AbortError",
      code: "ABORT_ERR",
      stdout: "partial report",
      stderr: "provider request canceled",
    })
    assert.ok(out.includes("canceled by the OpenCode session"))
    assert.ok(out.includes("partial report"))
    assert.ok(out.includes("provider request canceled"))
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

  it("returns successful stdout and stderr warnings with independent caps", () => {
    const out = formatSuccess(
      undefined,
      '{"findings":[]}',
      `baseline detected ${"w".repeat(MAX_STDERR + 1)}`,
    )
    assert.ok(out.includes('{"findings":[]}'))
    assert.ok(out.includes("stderr:\nbaseline detected"))
    assert.ok(out.includes("[truncated "))
  })
})

describe("permissioned execution", () => {
  function context(
    ask: (request: PermissionRequest) => Promise<void>,
    abort = new AbortController().signal,
    paths: { directory?: string; worktree?: string } = {},
  ) {
    return {
      directory: paths.directory ?? "/work/project",
      worktree: paths.worktree ?? "/work/project",
      abort,
      ask,
    }
  }

  it("requests scoped host, external-path, process, and LLM permissions", async () => {
    const requests: PermissionRequest[] = []
    let runCalls = 0
    const runFile: RunFile = async () => {
      runCalls += 1
      return { stdout: "ok", stderr: "" }
    }
    const out = await executeScan(
      {
        target: "/outside/skill",
        output: "/outside/report.json",
        noLlm: false,
      },
      context(async (request) => {
        requests.push(request)
      }),
      {
        runFile,
        existsSync: () => false,
        env: {
          SKILLSPECTOR_PROVIDER: "openai",
          SKILLSPECTOR_MODEL: "gpt-test",
          OPENAI_BASE_URL: "https://user:password@example.test/v1?secret=query",
        },
      },
    )

    assert.equal(out, "ok")
    assert.equal(runCalls, 1)
    assert.deepEqual(
      requests.map((request) => request.permission),
      ["external_directory", "read", "external_directory", "edit", "webfetch", "bash"],
    )
    const llm = requests[4]
    assert.equal(llm.metadata.provider, "openai")
    assert.equal(llm.metadata.model, "gpt-test")
    assert.equal(llm.metadata.destination, "https://example.test/v1")
    assert.ok(!JSON.stringify(llm).includes("password"))
    assert.ok(!JSON.stringify(llm).includes("secret=query"))
  })

  it("asks for remote-target network access and preserves SCP input", async () => {
    const requests: PermissionRequest[] = []
    let seenArgs: string[] = []
    await executeScan(
      { target: "git@github.com:NVIDIA/SkillSpector.git" },
      context(async (request) => {
        requests.push(request)
      }),
      {
        runFile: async (_bin, args) => {
          seenArgs = args
          return { stdout: "ok", stderr: "" }
        },
        existsSync: () => false,
        env: {},
      },
    )
    assert.deepEqual(requests.map((request) => request.permission), ["webfetch", "bash"])
    assert.equal(seenArgs[1], "git@github.com:NVIDIA/SkillSpector.git")
  })

  it("never starts the process when any permission is denied", async () => {
    let runCalls = 0
    await assert.rejects(
      executeScan(
        { target: "./skill" },
        context(async () => {
          throw new Error("permission denied")
        }),
        {
          runFile: async () => {
            runCalls += 1
            return { stdout: "unexpected", stderr: "" }
          },
          existsSync: () => false,
          env: {},
        },
      ),
      /permission denied/,
    )
    assert.equal(runCalls, 0)
  })

  it("requests external-directory permission for a relative binary outside the session", async () => {
    const requests: PermissionRequest[] = []
    await executeScan(
      { target: "./skill" },
      context(async (request) => {
        requests.push(request)
      }),
      {
        runFile: async () => ({ stdout: "ok", stderr: "" }),
        existsSync: () => false,
        env: { SKILLSPECTOR_BIN: "../tools/skillspector" },
      },
    )
    assert.deepEqual(
      requests.map((request) => request.permission),
      ["read", "external_directory", "bash"],
    )
    assert.equal(requests[1].metadata.binary, "/work/tools/skillspector")
  })

  it("passes the OpenCode abort signal to the child and reports cancellation", async () => {
    const controller = new AbortController()
    let seenSignal: AbortSignal | undefined
    const out = await executeScan(
      { target: "./skill" },
      context(async () => {}, controller.signal),
      {
        runFile: async (_bin, _args, options) => {
          seenSignal = options.signal
          throw { name: "AbortError", code: "ABORT_ERR" }
        },
        existsSync: () => false,
        env: {},
      },
    )
    assert.equal(seenSignal, controller.signal)
    assert.ok(out.includes("canceled by the OpenCode session"))
  })

  it("strips ambient LangChain and LangSmith tracing from the child environment", async () => {
    let seenEnv: Record<string, string | undefined> = {}
    await executeScan(
      { target: "./skill" },
      context(async () => {}),
      {
        runFile: async (_bin, _args, options) => {
          seenEnv = options.env
          return { stdout: "ok", stderr: "" }
        },
        existsSync: () => false,
        env: {
          PATH: "/usr/bin",
          OPENAI_API_KEY: "provider-secret",
          LANGCHAIN_TRACING: "true",
          LANGCHAIN_TRACING_V2: "true",
          LANGSMITH_TRACING: "true",
          LANGSMITH_TRACING_V2: "true",
          LANGCHAIN_ENDPOINT: "https://trace.example.test",
          langsmith_api_key: "trace-secret",
          LANGSMITH_PROJECT: "sensitive-project",
          LANGCHAIN_TAGS_EXTRA: "sensitive-tag",
        },
      },
    )

    assert.equal(seenEnv.PATH, "/usr/bin")
    assert.equal(seenEnv.OPENAI_API_KEY, "provider-secret")
    assert.deepEqual(
      Object.keys(seenEnv).filter((name) => /^(?:LANGCHAIN|LANGSMITH)_/i.test(name)),
      [],
    )
  })

  it("ignores OpenCode's POSIX root worktree sentinel for external paths", async () => {
    const requests: PermissionRequest[] = []
    await executeScan(
      { target: "/etc/skill", output: "/tmp/report.json" },
      context(
        async (request) => {
          requests.push(request)
        },
        new AbortController().signal,
        { directory: "/work/project", worktree: "/" },
      ),
      {
        runFile: async () => ({ stdout: "ok", stderr: "" }),
        existsSync: () => false,
        env: {},
        platform: "linux",
      },
    )
    assert.deepEqual(
      requests.map((request) => request.permission),
      ["external_directory", "read", "external_directory", "edit", "bash"],
    )
  })

  it("ignores a Windows drive-root worktree sentinel for external paths", async () => {
    const requests: PermissionRequest[] = []
    await executeScan(
      { target: "C:\\Windows\\skill", output: "D:\\tmp\\report.json" },
      context(
        async (request) => {
          requests.push(request)
        },
        new AbortController().signal,
        { directory: "C:\\work\\project", worktree: "C:\\" },
      ),
      {
        runFile: async () => ({ stdout: "ok", stderr: "" }),
        existsSync: () => false,
        env: {},
        platform: "win32",
      },
    )
    assert.deepEqual(
      requests.map((request) => request.permission),
      ["external_directory", "read", "external_directory", "edit", "bash"],
    )
  })

  it("treats a forward-slash Windows drive path as local, not as a URL", async () => {
    const requests: PermissionRequest[] = []
    await executeScan(
      { target: "C://Windows/skill" },
      context(
        async (request) => {
          requests.push(request)
        },
        new AbortController().signal,
        { directory: "C:\\work\\project", worktree: "C:\\" },
      ),
      {
        runFile: async () => ({ stdout: "ok", stderr: "" }),
        existsSync: () => false,
        env: {},
        platform: "win32",
      },
    )
    assert.deepEqual(
      requests.map((request) => request.permission),
      ["external_directory", "read", "bash"],
    )
  })

  it("rejects an existing symlink in the output path before process launch", async () => {
    let runCalls = 0
    await assert.rejects(
      executeScan(
        { target: "./skill", output: "./linked/report.json" },
        context(async () => {}),
        {
          runFile: async () => {
            runCalls += 1
            return { stdout: "unexpected", stderr: "" }
          },
          existsSync: () => false,
          lstatSync: (candidate) => ({
            isSymbolicLink: () => candidate === "/work/project/linked",
          }),
          env: {},
        },
      ),
      /symlinked path/,
    )
    assert.equal(runCalls, 0)
  })

  it("rejects a local target redirected through a symlink before process launch", async () => {
    let runCalls = 0
    await assert.rejects(
      executeScan(
        { target: "./linked-skill" },
        context(async () => {}),
        {
          runFile: async () => {
            runCalls += 1
            return { stdout: "unexpected", stderr: "" }
          },
          existsSync: () => false,
          lstatSync: (candidate) => ({
            isSymbolicLink: () => candidate === "/work/project/linked-skill",
          }),
          env: {},
        },
      ),
      /scan a local target through a symlinked path/,
    )
    assert.equal(runCalls, 0)
  })

  it("allows a non-symlink filesystem-root target", async () => {
    let runCalls = 0
    const out = await executeScan(
      { target: "/" },
      context(async () => {}),
      {
        runFile: async () => {
          runCalls += 1
          return { stdout: "ok", stderr: "" }
        },
        existsSync: () => false,
        lstatSync: () => ({ isSymbolicLink: () => false }),
        env: {},
        platform: "linux",
      },
    )
    assert.equal(out, "ok")
    assert.equal(runCalls, 1)
  })

  it("rejects a configured CLI reached through a symlink before process launch", async () => {
    let runCalls = 0
    await assert.rejects(
      executeScan(
        { target: "./skill" },
        context(async () => {}),
        {
          runFile: async () => {
            runCalls += 1
            return { stdout: "unexpected", stderr: "" }
          },
          existsSync: () => false,
          lstatSync: (candidate) => ({
            isSymbolicLink: () => candidate === "/work/project/bin",
          }),
          env: { SKILLSPECTOR_BIN: "/work/project/bin/skillspector" },
        },
      ),
      /execute the SkillSpector CLI through a symlinked path/,
    )
    assert.equal(runCalls, 0)
  })

  it("rejects a symlinked checkout venv CLI before process launch", async () => {
    let runCalls = 0
    await assert.rejects(
      executeScan(
        { target: "./skill" },
        context(async () => {}),
        {
          runFile: async () => {
            runCalls += 1
            return { stdout: "unexpected", stderr: "" }
          },
          existsSync: (candidate) => candidate.endsWith("/.venv/bin/skillspector"),
          lstatSync: (candidate) => ({
            isSymbolicLink: () => candidate === "/work/project/.venv",
          }),
          env: {},
        },
      ),
      /execute the SkillSpector CLI through a symlinked path/,
    )
    assert.equal(runCalls, 0)
  })

  it("resolves and rejects a POSIX dot-relative symlinked CLI", async () => {
    let runCalls = 0
    await assert.rejects(
      executeScan(
        { target: "./skill" },
        context(async () => {}),
        {
          runFile: async () => {
            runCalls += 1
            return { stdout: "unexpected", stderr: "" }
          },
          existsSync: () => false,
          lstatSync: (candidate) => ({
            isSymbolicLink: () => candidate === "/work/project/skillspector",
          }),
          env: { SKILLSPECTOR_BIN: "./skillspector" },
          platform: "linux",
        },
      ),
      /execute the SkillSpector CLI through a symlinked path/,
    )
    assert.equal(runCalls, 0)
  })

  it("resolves and rejects a Windows dot-relative symlinked CLI", async () => {
    let runCalls = 0
    await assert.rejects(
      executeScan(
        { target: ".\\skill" },
        context(
          async () => {},
          new AbortController().signal,
          { directory: "C:\\work\\project", worktree: "C:\\work\\project" },
        ),
        {
          runFile: async () => {
            runCalls += 1
            return { stdout: "unexpected", stderr: "" }
          },
          existsSync: () => false,
          lstatSync: (candidate) => ({
            isSymbolicLink: () => candidate === "C:\\work\\project\\skillspector.exe",
          }),
          env: { SKILLSPECTOR_BIN: ".\\skillspector.exe" },
          platform: "win32",
        },
      ),
      /execute the SkillSpector CLI through a symlinked path/,
    )
    assert.equal(runCalls, 0)
  })
})

describe("LLM permission metadata", () => {
  it("uses a real supported provider and strips endpoint credentials", () => {
    assert.deepEqual(
      llmPermissionMetadata({
        SKILLSPECTOR_PROVIDER: "anthropic",
        SKILLSPECTOR_MODEL: "claude-test",
        ANTHROPIC_BASE_URL: "https://user:pass@example.test/v1?q=secret",
      }),
      {
        provider: "anthropic",
        model: "claude-test",
        destination: "https://example.test/v1",
      },
    )
  })
})

describe("childProcessEnv", () => {
  it("retains provider settings while removing every tracing namespace alias", () => {
    assert.deepEqual(
      childProcessEnv({
        SKILLSPECTOR_PROVIDER: "openai",
        OPENAI_API_KEY: "provider-secret",
        LANGCHAIN_CALLBACKS_BACKGROUND: "true",
        LangSmith_Sampling_Rate: "1",
      }),
      {
        SKILLSPECTOR_PROVIDER: "openai",
        OPENAI_API_KEY: "provider-secret",
      },
    )
  })
})

describe("constants", () => {
  it("keeps the documented caps", () => {
    assert.equal(TIMEOUT_MS, 120_000)
    assert.equal(MAX_STDOUT, 12_000)
    assert.equal(MAX_STDERR, 6_000)
  })
})
