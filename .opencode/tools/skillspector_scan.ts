import { tool } from "@opencode-ai/plugin"
import { execFile } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { promisify } from "node:util"
import {
  TIMEOUT_MS,
  buildCliArgs,
  formatExecError,
  formatSuccess,
  isUrlOrAbsolute,
  resolveBinary,
} from "./skillspector_scan_lib.ts"

const runFile = promisify(execFile)

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
    const bin = resolveBinary(context.worktree ?? baseDir, {
      existsSync: fs.existsSync,
    })
    const cliArgs = buildCliArgs({ target, format: args.format, noLlm: args.noLlm, output })

    let result: { stdout: string; stderr: string }
    try {
      result = (await runFile(bin, cliArgs, {
        timeout: TIMEOUT_MS,
        maxBuffer: 32 * 1024 * 1024,
        cwd: baseDir,
      })) as { stdout: string; stderr: string }
    } catch (err: unknown) {
      return formatExecError(bin, err)
    }

    return formatSuccess(output, result.stdout, result.stderr)
  },
})
