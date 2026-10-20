import { tool } from "@opencode-ai/plugin"
import { execFile } from "node:child_process"
import fs from "node:fs"
import { promisify } from "node:util"
import { executeScan, type RunFile } from "./skillspector_scan_lib.ts"

const runFile = promisify(execFile) as unknown as RunFile

export default tool({
  description: "Scan an AI agent skill for security risks with SkillSpector. Static analysis only by default; opt into LLM analysis explicitly.",
  args: {
    target: tool.schema.string().describe("Skill to scan: local path, .md/.zip file, or Git/file URL"),
    format: tool.schema.enum(["terminal", "json", "markdown", "sarif"]).default("json").describe("Report format"),
    noLlm: tool.schema.boolean().default(true).describe("Skip LLM analysis (static checks only). Set false to opt into LLM semantic analysis via SKILLSPECTOR_PROVIDER/SKILLSPECTOR_MODEL"),
    output: tool.schema.string().optional().describe("Write the report to this file instead of returning it (resolved against the session directory if relative)"),
  },
  async execute(args, context) {
    return executeScan(args, context, {
      runFile,
      existsSync: fs.existsSync,
      lstatSync: fs.lstatSync,
    })
  },
})
