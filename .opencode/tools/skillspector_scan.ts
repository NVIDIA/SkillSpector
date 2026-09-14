import { tool } from "@opencode-ai/plugin"
import { execFile } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { promisify } from "node:util"

const runFile = promisify(execFile)
const TIMEOUT_MS = 120_000
const MAX_STDOUT = 12_000
const MAX_STDERR = 6_000
const INSTALL_HINT = "uv tool install git+https://github.com/NVIDIA/skillspector.git"

function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return text.slice(0, max) + `\n...[truncated ${text.length - max} chars]`
}

function redact(text: string): string {
  return text
    .replace(/sk-ant-[A-Za-z0-9_-]+/g, "[REDACTED]")
    .replace(/\bsk-[A-Za-z0-9_-]{6,}\b/g, "[REDACTED]")
    .replace(/\b([A-Z][A-Z0-9_]*_(?:API_KEY|TOKEN))(\s*[:=]\s*["']?)[^"'\s,}]+/g, "$1$2[REDACTED]")
}

function resolveBinary(worktree: string): string {
  const fromEnv = process.env.SKILLSPECTOR_BIN?.trim()
  if (fromEnv) return fromEnv
  // Repo-checkout fallback: <worktree>/.venv (Scripts/skillspector.exe on Windows, bin/skillspector elsewhere).
  const binDir = process.platform === "win32" ? "Scripts" : "bin"
  const exe = process.platform === "win32" ? "skillspector.exe" : "skillspector"
  const venvBin = path.join(worktree, ".venv", binDir, exe)
  if (fs.existsSync(venvBin)) return venvBin
  return "skillspector"
}

function isUrlOrAbsolute(target: string): boolean {
  return /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(target) || path.isAbsolute(target)
}

export default tool({
  description: "Scan an AI agent skill for security risks with SkillSpector. Static analysis only by default; opt into LLM analysis explicitly.",
  args: {
    target: tool.schema.string().describe("Skill to scan: local path, .md/.zip file, or Git/file URL"),
    format: tool.schema.enum(["terminal", "json", "markdown", "sarif"]).default("json").describe("Report format"),
    noLlm: tool.schema.boolean().default(true).describe("Skip LLM analysis (static checks only). Set false to opt into LLM semantic analysis via SKILLSPECTOR_PROVIDER/SKILLSPECTOR_MODEL"),
    output: tool.schema.string().optional().describe("Write the report to this file instead of returning it (resolved against the session directory if relative)"),
  },
  async execute(args, context) {
    const baseDir = context.directory ?? context.worktree ?? process.cwd()
    const target = isUrlOrAbsolute(args.target) ? args.target : path.resolve(baseDir, args.target)
    const output = args.output
      ? (isUrlOrAbsolute(args.output) ? args.output : path.resolve(baseDir, args.output))
      : undefined
    const bin = resolveBinary(context.worktree ?? baseDir)
    // The host may omit declared defaults, so re-apply them here. noLlm
    // defaults to true: LLM analysis must stay strictly opt-in.
    const format = args.format ?? "json"
    const noLlm = args.noLlm ?? true
    const cliArgs = ["scan", target, "--format", format]
    // LLM runs only on explicit opt-out of --no-llm; provider/model/credentials come from the inherited environment.
    if (noLlm) cliArgs.push("--no-llm")
    if (output) cliArgs.push("--output", output)

    let result: { stdout: string; stderr: string }
    try {
      result = (await runFile(bin, cliArgs, {
        timeout: TIMEOUT_MS,
        maxBuffer: 32 * 1024 * 1024,
        cwd: baseDir,
      })) as { stdout: string; stderr: string }
    } catch (err: unknown) {
      const e = err as { code?: unknown; killed?: boolean; signal?: unknown; stdout?: unknown; stderr?: unknown; message?: string }
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
        return redact(`SkillSpector usage error (exit 2):\n${truncate(partialErr || e.message || "", MAX_STDERR)}`)
      }
      return redact(`SkillSpector scan failed: ${truncate(partialErr || e.message || String(err), MAX_STDERR)}`)
    }

    if (output && !result.stdout) return `Report saved to: ${output}`
    return redact(truncate(result.stdout || result.stderr, MAX_STDOUT))
  },
})
