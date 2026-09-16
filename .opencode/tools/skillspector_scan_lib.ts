// Pure helpers for the skillspector_scan tool. Dependency-free (no
// @opencode-ai/plugin import) so this module runs under plain node --test.

import path from "node:path"

export const TIMEOUT_MS = 120_000
export const MAX_STDOUT = 12_000
export const MAX_STDERR = 6_000
export const INSTALL_HINT =
  "uv tool install git+https://github.com/NVIDIA/skillspector.git"

type Env = Record<string, string | undefined>
type PathApi = typeof path.posix

function pathFor(platform: string): PathApi {
  return platform === "win32" ? path.win32 : path.posix
}

// Credentials that SkillSpector can consume directly, through a provider, or
// through the standard AWS/LangChain credential chains. Keep this explicit:
// reading arbitrary *_KEY variables would widen the host-data boundary.
export const CREDENTIAL_ENV_NAMES = [
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_PROXY_API_KEY",
  "AWS_ACCESS_KEY_ID",
  "AWS_BEARER_TOKEN_BEDROCK",
  "AWS_SECRET_ACCESS_KEY",
  "AWS_SECURITY_TOKEN",
  "AWS_SESSION_TOKEN",
  "AZURE_OPENAI_API_KEY",
  "LANGCHAIN_API_KEY",
  "LANGSMITH_API_KEY",
  "NVIDIA_INFERENCE_KEY",
  "NVIDIA_INFERENCE_METADATA_KEY",
  "OPENAI_API_KEY",
  "SKILLSPECTOR_COMPAT_API_KEY",
] as const

export function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return text.slice(0, max) + `\n...[truncated ${text.length - max} chars]`
}

export function redact(text: string, env: Env = process.env): string {
  const credentialValues = [...new Set(
    CREDENTIAL_ENV_NAMES.map((name) => env[name]?.trim()).filter(
      (value): value is string => Boolean(value && value.length >= 4),
    ),
  )].sort((left, right) => right.length - left.length)
  const credentialPattern = credentialValues.length
    ? new RegExp(
        credentialValues
          .map((value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
          .join("|"),
        "g",
      )
    : undefined
  const redacted = credentialPattern ? text.replace(credentialPattern, "[REDACTED]") : text
  return redacted
    .replace(/sk-ant-[A-Za-z0-9_-]+/g, "[REDACTED]")
    .replace(/\bsk-[A-Za-z0-9_-]{6,}\b/g, "[REDACTED]")
    .replace(
      /\b([A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|ACCESS_KEY_ID|SECRET_ACCESS_KEY|INFERENCE_KEY))(\s*[:=]\s*["']?)[^"'\s,}]+/g,
      "$1$2[REDACTED]",
    )
}

export function isScpGitTarget(target: string): boolean {
  // SkillSpector accepts the conventional git@host:owner/repo.git form. Keep
  // this narrow so strings such as "notes:today" remain ordinary local paths.
  return /^git@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?:[A-Za-z0-9._~/-]+\.git\/?$/.test(
    target,
  )
}

export function isRemoteTarget(
  target: string,
  platform: string = process.platform,
): boolean {
  // A Windows drive path can use forward slashes (C://path) and otherwise
  // looks like a URI scheme. Host-path classification must win first.
  if (pathFor(platform).isAbsolute(target)) return false
  // Match only the remote forms accepted by SkillSpector's input handler.
  // Treating arbitrary schemes as remote can turn a valid local filename into
  // a network-only permission request and bypass the corresponding read gate.
  return target.startsWith("https://") || isScpGitTarget(target)
}

export function isUrlOrAbsolute(
  target: string,
  platform: string = process.platform,
): boolean {
  return pathFor(platform).isAbsolute(target) || isRemoteTarget(target, platform)
}

export function isFilesystemRoot(
  candidate: string,
  platform: string = process.platform,
): boolean {
  const pathApi = pathFor(platform)
  if (!pathApi.isAbsolute(candidate)) return false
  const resolved = pathApi.resolve(candidate)
  return resolved === pathApi.parse(resolved).root
}

export interface ResolveBinaryOpts {
  env?: Env
  platform?: string
  existsSync?: (p: string) => boolean
}

export function resolveBinary(
  worktree: string,
  opts: ResolveBinaryOpts = {},
): string {
  const env = opts.env ?? process.env
  const platform = opts.platform ?? process.platform
  const pathApi = pathFor(platform)
  const existsSync = opts.existsSync
  const fromEnv = env.SKILLSPECTOR_BIN?.trim()
  if (fromEnv) return fromEnv
  // Repo-checkout fallback: <worktree>/.venv (Scripts/skillspector.exe on Windows, bin/skillspector elsewhere).
  const binDir = platform === "win32" ? "Scripts" : "bin"
  const exe = platform === "win32" ? "skillspector.exe" : "skillspector"
  const venvBin = pathApi.join(worktree, ".venv", binDir, exe)
  if (existsSync ? existsSync(venvBin) : false) return venvBin
  return "skillspector"
}

export interface ScanArgs {
  target: string
  format?: string
  noLlm?: boolean
  output?: string
}

export function buildCliArgs(args: ScanArgs): string[] {
  // The host may omit declared defaults, so re-apply them here. noLlm
  // defaults to true: LLM analysis must stay strictly opt-in.
  const format = args.format ?? "json"
  const noLlm = args.noLlm ?? true
  const cliArgs = ["scan", args.target, "--format", format]
  if (noLlm) cliArgs.push("--no-llm")
  if (args.output) cliArgs.push("--output", args.output)
  return cliArgs
}

export interface ExecFailure {
  code?: unknown
  killed?: boolean
  name?: string
  stdout?: unknown
  stderr?: unknown
  message?: string
}

function boundedStreams(stdout: string, stderr: string, env: Env): string {
  const parts: string[] = []
  if (stdout) parts.push(truncate(redact(stdout, env), MAX_STDOUT))
  if (stderr) parts.push(`stderr:\n${truncate(redact(stderr, env), MAX_STDERR)}`)
  return parts.join("\n")
}

function isProvenUsageError(stderr: string): boolean {
  return /Usage:/i.test(stderr) && /(?:Error:|Try .+--help)/i.test(stderr)
}

export function formatExecError(
  bin: string,
  err: unknown,
  env: Env = process.env,
): string {
  const e = err as ExecFailure
  const partialOut = typeof e.stdout === "string" ? e.stdout : ""
  const partialErr = typeof e.stderr === "string" ? e.stderr : ""
  const evidence = boundedStreams(partialOut, partialErr, env)
  if (e.code === "ENOENT") {
    return `SkillSpector CLI not found (tried "${bin}"). Install it with \`${INSTALL_HINT}\`, or point SKILLSPECTOR_BIN at the binary.`
  }
  if (e.name === "AbortError" || e.code === "ABORT_ERR") {
    return (
      "SkillSpector scan canceled by the OpenCode session." +
      (evidence ? ` Partial output:\n${evidence}` : "")
    )
  }
  if (e.killed) {
    return (
      `SkillSpector scan timed out after ${TIMEOUT_MS / 1000}s (killed).` +
      (evidence ? ` Partial output:\n${evidence}` : "")
    )
  }
  if (e.code === 1 && partialOut) {
    // Findings above the risk threshold: the report is the answer, not a crash.
    return evidence
  }
  if (e.code === 2 && !partialOut && isProvenUsageError(partialErr)) {
    return `SkillSpector usage error (exit 2):\n${evidence}`
  }
  const detail = evidence || truncate(redact(e.message || String(err), env), MAX_STDERR)
  return `SkillSpector scan failed${e.code === 2 ? " (exit 2)" : ""}:\n${detail}`
}

export function formatSuccess(
  output: string | undefined,
  stdout: string,
  stderr: string,
  env: Env = process.env,
): string {
  const report = stdout
    ? truncate(redact(stdout, env), MAX_STDOUT)
    : output
      ? `Report saved to: ${output}`
      : ""
  const warning = stderr
    ? `stderr:\n${truncate(redact(stderr, env), MAX_STDERR)}`
    : ""
  return [report, warning].filter(Boolean).join("\n")
}

export interface PermissionRequest {
  permission: string
  patterns: string[]
  always: string[]
  metadata: Record<string, unknown>
}

export interface ScanContext {
  directory?: string
  worktree?: string
  abort: AbortSignal
  ask(input: PermissionRequest): Promise<void>
}

export interface RunFileOptions {
  timeout: number
  maxBuffer: number
  cwd: string
  signal: AbortSignal
  encoding: "utf8"
  env: Env
}

export type RunFile = (
  file: string,
  args: string[],
  options: RunFileOptions,
) => Promise<{ stdout: string; stderr: string }>

export interface ExecuteScanDeps {
  runFile: RunFile
  existsSync?: (p: string) => boolean
  lstatSync?: (p: string) => { isSymbolicLink(): boolean }
  env?: Env
  cwd?: string
  platform?: string
}

interface PreparedScan extends ScanArgs {
  target: string
  output?: string
}

interface PermissionBoundary {
  directory: string
  worktree?: string
  platform: string
}

export function childProcessEnv(env: Env = process.env): Env {
  // LangChain/LangSmith tracing can upload graph inputs and state even when
  // SkillSpector's LLM analysis is disabled. The OpenCode tool has a separate
  // explicit egress gate, so never let ambient tracing controls, endpoints,
  // projects, sampling settings, tags, or tracing credentials reach the CLI.
  return Object.fromEntries(
    Object.entries(env).filter(
      ([name]) => !/^(?:LANGCHAIN|LANGSMITH)_/i.test(name),
    ),
  )
}

function isPathLikeBinary(candidate: string, platform: string): boolean {
  const pathApi = pathFor(platform)
  if (pathApi.isAbsolute(candidate) || pathApi.dirname(candidate) !== ".") {
    return true
  }
  if (candidate === "." || candidate === ".." || candidate.includes(pathApi.sep)) {
    return true
  }
  // Node accepts either separator on Windows, even though path.win32.sep is
  // a backslash. Only a true bare command name should be delegated to PATH.
  return platform === "win32" && candidate.includes("/")
}

function isWithin(root: string, candidate: string, pathApi: PathApi): boolean {
  const relative = pathApi.relative(root, candidate)
  return (
    relative === "" ||
    (relative !== ".." &&
      !relative.startsWith(`..${pathApi.sep}`) &&
      !pathApi.isAbsolute(relative))
  )
}

function usableWorktree(boundary: PermissionBoundary): string | undefined {
  if (!boundary.worktree || isFilesystemRoot(boundary.worktree, boundary.platform)) {
    return undefined
  }
  return boundary.worktree
}

function isInternalPath(boundary: PermissionBoundary, candidate: string): boolean {
  const pathApi = pathFor(boundary.platform)
  if (isWithin(boundary.directory, candidate, pathApi)) return true
  const worktree = usableWorktree(boundary)
  return Boolean(worktree && isWithin(worktree, candidate, pathApi))
}

function displayPath(boundary: PermissionBoundary, candidate: string): string {
  const pathApi = pathFor(boundary.platform)
  const worktree = usableWorktree(boundary)
  const permissionRoot =
    worktree && isWithin(worktree, candidate, pathApi)
      ? worktree
      : isWithin(boundary.directory, candidate, pathApi)
        ? boundary.directory
        : undefined
  if (!permissionRoot) return candidate
  return pathApi.relative(permissionRoot, candidate).replaceAll(pathApi.sep, "/") || "."
}

function localScope(
  boundary: PermissionBoundary,
  candidate: string,
  recursive: boolean,
): string[] {
  const displayed = displayPath(boundary, candidate)
  return recursive ? [displayed, `${displayed.replace(/\/$/, "")}/**`] : [displayed]
}

function assertNoSymlinkedPath(
  candidate: string,
  platform: string,
  lstatSync: (p: string) => { isSymbolicLink(): boolean },
  operation: string,
): void {
  const pathApi = pathFor(platform)
  const normalized = pathApi.resolve(candidate)
  const parsed = pathApi.parse(normalized)
  let current = parsed.root
  for (const component of normalized.slice(parsed.root.length).split(pathApi.sep).filter(Boolean)) {
    current = pathApi.join(current, component)
    try {
      if (lstatSync(current).isSymbolicLink()) {
        throw new Error(`Refusing to ${operation} through a symlinked path: ${current}`)
      }
    } catch (error: unknown) {
      if ((error as { code?: string }).code === "ENOENT") return
      throw error
    }
  }
}

function safeEndpoint(raw: string): string {
  try {
    const parsed = new URL(raw)
    parsed.username = ""
    parsed.password = ""
    parsed.search = ""
    parsed.hash = ""
    return parsed.toString()
  } catch {
    return raw
  }
}

export function llmPermissionMetadata(env: Env = process.env): {
  provider: string
  model: string
  destination: string
} {
  const provider = env.SKILLSPECTOR_PROVIDER?.trim().toLowerCase() || "nv_build"
  const model = env.SKILLSPECTOR_MODEL?.trim() || "provider-default"
  const destinations: Record<string, string> = {
    anthropic: env.ANTHROPIC_BASE_URL?.trim() || "https://api.anthropic.com/",
    anthropic_proxy:
      env.ANTHROPIC_PROXY_ENDPOINT_URL?.trim() || "configured-anthropic-proxy",
    azure_openai: env.AZURE_OPENAI_ENDPOINT?.trim() || "configured-azure-openai",
    bedrock: `aws-bedrock:${env.AWS_REGION?.trim() || "us-west-2"}`,
    claude_cli: "local-claude-cli",
    codex_cli: "local-codex-cli",
    gemini_cli: "local-gemini-cli",
    nv_build: "https://integrate.api.nvidia.com/v1/",
    ollama: env.OLLAMA_BASE_URL?.trim() || "http://localhost:11434/v1/",
    openai: env.OPENAI_BASE_URL?.trim() || "https://api.openai.com/",
    openai_compatible:
      env.SKILLSPECTOR_COMPAT_BASE_URL?.trim() || "configured-openai-compatible",
  }
  return {
    provider,
    model,
    destination: safeEndpoint(destinations[provider] || `configured-provider:${provider}`),
  }
}

export function buildPermissionRequests(
  args: PreparedScan,
  options: PermissionBoundary & { bin: string; cliArgs: string[]; env?: Env },
): PermissionRequest[] {
  const { bin, cliArgs } = options
  const env = options.env ?? process.env
  const requests: PermissionRequest[] = []

  if (isRemoteTarget(args.target, options.platform)) {
    requests.push({
      permission: "webfetch",
      patterns: [args.target],
      always: [args.target],
      metadata: { operation: "scan remote skill", target: args.target },
    })
  } else {
    const patterns = localScope(options, args.target, true)
    if (!isInternalPath(options, args.target)) {
      requests.push({
        permission: "external_directory",
        patterns,
        always: patterns,
        metadata: { operation: "read scan target", target: args.target },
      })
    }
    requests.push({
      permission: "read",
      patterns,
      always: patterns,
      metadata: { operation: "read scan target", target: args.target },
    })
  }

  if (args.output) {
    const patterns = localScope(options, args.output, false)
    if (!isInternalPath(options, args.output)) {
      requests.push({
        permission: "external_directory",
        patterns,
        always: patterns,
        metadata: { operation: "write scan report", output: args.output },
      })
    }
    requests.push({
      permission: "edit",
      patterns,
      always: patterns,
      metadata: { operation: "write scan report", output: args.output },
    })
  }

  if (!(args.noLlm ?? true)) {
    const llm = llmPermissionMetadata(env)
    const pattern = `skillspector-llm:${llm.provider}:${llm.destination}:${llm.model}`
    requests.push({
      permission: "webfetch",
      patterns: [pattern],
      always: [],
      metadata: {
        operation: "send analyzer-eligible skill content for LLM analysis",
        target: args.target,
        ...llm,
      },
    })
  }

  const pathApi = pathFor(options.platform)
  if (pathApi.isAbsolute(bin) && !isInternalPath(options, bin)) {
    const patterns = localScope(options, bin, false)
    requests.push({
      permission: "external_directory",
      patterns,
      always: patterns,
      metadata: { operation: "execute SkillSpector CLI", binary: bin },
    })
  }

  const commandPattern = `${bin} scan *`
  requests.push({
    permission: "bash",
    patterns: [commandPattern],
    always: [commandPattern],
    metadata: { operation: "run SkillSpector CLI", command: [bin, ...cliArgs] },
  })
  return requests
}

export async function executeScan(
  args: ScanArgs,
  context: ScanContext,
  deps: ExecuteScanDeps,
): Promise<string> {
  const env = deps.env ?? process.env
  const platform = deps.platform ?? process.platform
  const pathApi = pathFor(platform)
  const baseDir = context.directory ?? context.worktree ?? deps.cwd ?? process.cwd()
  const worktree = context.worktree
  const binaryRoot = worktree && !isFilesystemRoot(worktree, platform) ? worktree : baseDir
  const target = isUrlOrAbsolute(args.target, platform)
    ? args.target
    : pathApi.resolve(baseDir, args.target)
  const output = args.output
    ? pathApi.isAbsolute(args.output)
      ? args.output
      : pathApi.resolve(baseDir, args.output)
    : undefined
  const prepared = { ...args, target, output }
  const configuredBin = resolveBinary(binaryRoot, {
    env,
    existsSync: deps.existsSync,
    platform,
  })
  const bin = isPathLikeBinary(configuredBin, platform)
    ? pathApi.resolve(baseDir, configuredBin)
    : configuredBin
  const cliArgs = buildCliArgs(prepared)

  for (const request of buildPermissionRequests(prepared, {
    directory: baseDir,
    worktree,
    platform,
    bin,
    cliArgs,
    env,
  })) {
    // A rejection is intentionally not caught: OpenCode owns the denial result,
    // and the process must never start after any denied capability.
    await context.ask(request)
  }

  if (output && deps.lstatSync) {
    assertNoSymlinkedPath(output, platform, deps.lstatSync, "write a report")
  }
  if (!isRemoteTarget(target, platform) && deps.lstatSync) {
    // Core InputHandler also rejects local symlink targets. Enforce the same
    // boundary here before process launch so the OpenCode permission grant
    // cannot be redirected through a symlink or Windows junction.
    assertNoSymlinkedPath(target, platform, deps.lstatSync, "scan a local target")
  }
  if (pathApi.isAbsolute(bin) && deps.lstatSync) {
    assertNoSymlinkedPath(
      bin,
      platform,
      deps.lstatSync,
      "execute the SkillSpector CLI",
    )
  }

  try {
    const result = await deps.runFile(bin, cliArgs, {
      timeout: TIMEOUT_MS,
      maxBuffer: 32 * 1024 * 1024,
      cwd: baseDir,
      signal: context.abort,
      encoding: "utf8",
      env: childProcessEnv(env),
    })
    return formatSuccess(output, result.stdout, result.stderr, env)
  } catch (err: unknown) {
    return formatExecError(bin, err, env)
  }
}
