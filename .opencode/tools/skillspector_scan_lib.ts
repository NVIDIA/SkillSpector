// Pure helpers for the skillspector_scan tool. Dependency-free (no
// @opencode-ai/plugin import) so this module runs under plain node --test.

import path from "node:path"

export const TIMEOUT_MS = 120_000
export const MAX_STDOUT = 12_000
export const MAX_STDERR = 6_000
export const INSTALL_HINT =
  "uv tool install git+https://github.com/NVIDIA/skillspector.git"

export function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return text.slice(0, max) + `\n...[truncated ${text.length - max} chars]`
}

export function redact(text: string): string {
  return text
    .replace(/sk-ant-[A-Za-z0-9_-]+/g, "[REDACTED]")
    .replace(/\bsk-[A-Za-z0-9_-]{6,}\b/g, "[REDACTED]")
    .replace(
      /\b([A-Z][A-Z0-9_]*_(?:API_KEY|TOKEN))(\s*[:=]\s*["']?)[^"'\s,}]+/g,
      "$1$2[REDACTED]",
    )
}

export function isUrlOrAbsolute(target: string): boolean {
  return (
    /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(target) || path.isAbsolute(target)
  )
}

export interface ResolveBinaryOpts {
  env?: Record<string, string | undefined>
  platform?: string
  existsSync?: (p: string) => boolean
}

export function resolveBinary(
  worktree: string,
  opts: ResolveBinaryOpts = {},
): string {
  const env = opts.env ?? process.env
  const platform = opts.platform ?? process.platform
  const existsSync = opts.existsSync
  const fromEnv = env.SKILLSPECTOR_BIN?.trim()
  if (fromEnv) return fromEnv
  // Repo-checkout fallback: <worktree>/.venv (Scripts/skillspector.exe on Windows, bin/skillspector elsewhere).
  const binDir = platform === "win32" ? "Scripts" : "bin"
  const exe = platform === "win32" ? "skillspector.exe" : "skillspector"
  const venvBin = path.join(worktree, ".venv", binDir, exe)
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
  stdout?: unknown
  stderr?: unknown
  message?: string
}

export function formatExecError(bin: string, err: unknown): string {
  const e = err as ExecFailure
  const partialOut = typeof e.stdout === "string" ? e.stdout : ""
  const partialErr = typeof e.stderr === "string" ? e.stderr : ""
  if (e.code === "ENOENT") {
    return `SkillSpector CLI not found (tried "${bin}"). Install it with \`${INSTALL_HINT}\`, or point SKILLSPECTOR_BIN at the binary.`
  }
  if (e.killed) {
    return redact(
      `SkillSpector scan timed out after ${TIMEOUT_MS / 1000}s (killed; partial output below):\n` +
        truncate(partialOut, MAX_STDOUT) +
        (partialErr ? `\nstderr:\n${truncate(partialErr, MAX_STDERR)}` : ""),
    )
  }
  if (e.code === 1 && partialOut) {
    // Findings above the risk threshold: the JSON report is the answer, not a crash.
    return redact(truncate(partialOut, MAX_STDOUT))
  }
  if (e.code === 2) {
    return redact(
      `SkillSpector usage error (exit 2):\n${truncate(partialErr || e.message || "", MAX_STDERR)}`,
    )
  }
  return redact(
    `SkillSpector scan failed: ${truncate(partialErr || e.message || String(err), MAX_STDERR)}`,
  )
}

export function formatSuccess(
  output: string | undefined,
  stdout: string,
  stderr: string,
): string {
  if (output && !stdout) return `Report saved to: ${output}`
  return redact(truncate(stdout || stderr, MAX_STDOUT))
}
