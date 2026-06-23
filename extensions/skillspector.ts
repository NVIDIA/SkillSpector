import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { StringEnum } from "@earendil-works/pi-ai";
import { Type, type Static } from "typebox";
import { existsSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scanSchema = Type.Object({
  target: Type.String({ description: "Path, URL, zip, Git repo, or SKILL.md to scan." }),
  format: Type.Optional(
    StringEnum(["terminal", "json", "markdown", "sarif"] as const, {
      description: "SkillSpector output format. Defaults to terminal.",
    }),
  ),
  output: Type.Optional(Type.String({ description: "Optional report output path." })),
  noLlm: Type.Optional(Type.Boolean({ description: "Skip LLM analysis. Defaults to true." })),
  provider: Type.Optional(
    StringEnum(["openai", "anthropic", "anthropic_proxy", "nv_build", "nv_inference"] as const, {
      description: "Optional SkillSpector LLM provider when noLlm is false.",
    }),
  ),
  model: Type.Optional(Type.String({ description: "Optional model override." })),
  yaraRulesDir: Type.Optional(Type.String({ description: "Optional extra YARA rules directory." })),
  verbose: Type.Optional(Type.Boolean({ description: "Show detailed progress." })),
});

type SkillSpectorScanParams = Static<typeof scanSchema>;

function isLikelyUrl(value: string): boolean {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(value) || /^[\w.-]+\/[\w.-]+(?:\.git)?(?:@.+)?$/i.test(value);
}

function resolveMaybePath(ctxCwd: string, value?: string): string | undefined {
  if (!value) return undefined;
  if (isLikelyUrl(value)) return value;
  return isAbsolute(value) ? value : resolve(ctxCwd, value);
}

function redactSecrets(value: string): string {
  return value
    .replace(/(sk-ant-[A-Za-z0-9_-]{12,})/g, "[REDACTED_ANTHROPIC_KEY]")
    .replace(/(sk-[A-Za-z0-9_-]{20,})/g, "[REDACTED_OPENAI_KEY]")
    .replace(/([A-Za-z0-9_]*API_KEY[=:]\s*)[^\s]+/gi, "$1[REDACTED]")
    .replace(/([A-Za-z0-9_]*TOKEN[=:]\s*)[^\s]+/gi, "$1[REDACTED]");
}

function truncateText(value: string, maxChars = 12000): { text: string; truncated: boolean } {
  if (value.length <= maxChars) return { text: value, truncated: false };
  return {
    text: `${value.slice(0, maxChars)}\n\n[truncated ${value.length - maxChars} chars]`,
    truncated: true,
  };
}

function packageRoot(): string {
  return resolve(dirname(fileURLToPath(import.meta.url)), "..");
}

function findSkillSpectorBin(): string {
  if (process.env.SKILLSPECTOR_BIN) return process.env.SKILLSPECTOR_BIN;
  const localBin = resolve(packageRoot(), ".venv/bin/skillspector");
  if (existsSync(localBin)) return localBin;
  return "skillspector";
}

function buildScanArgs(params: SkillSpectorScanParams, cwd: string): string[] {
  const args = ["scan", resolveMaybePath(cwd, params.target) ?? params.target];
  args.push("--format", params.format ?? "terminal");

  const noLlm = params.noLlm ?? true;
  if (noLlm) args.push("--no-llm");

  const output = resolveMaybePath(cwd, params.output);
  if (output) args.push("--output", output);

  const yaraRulesDir = resolveMaybePath(cwd, params.yaraRulesDir);
  if (yaraRulesDir) args.push("--yara-rules-dir", yaraRulesDir);

  if (params.verbose) args.push("--verbose");
  return args;
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "skillspector_scan",
    label: "SkillSpector Scan",
    description: "Scan agent skills, directories, zip files, URLs, or Git repos for security risks using the local SkillSpector CLI.",
    promptSnippet: "Scan agent skills for security risks with local SkillSpector CLI.",
    promptGuidelines: [
      "Use skillspector_scan before installing or trusting third-party agent skills.",
      "skillspector_scan defaults to noLlm=true; set noLlm=false only when user wants provider-backed semantic analysis.",
    ],
    parameters: scanSchema,
    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      const bin = findSkillSpectorBin();
      const args = buildScanArgs(params, ctx.cwd);
      const env: Record<string, string> = {};

      if (params.provider) env.SKILLSPECTOR_PROVIDER = params.provider;
      if (params.model) env.SKILLSPECTOR_MODEL = params.model;

      onUpdate?.({ content: [{ type: "text", text: `Running ${bin} ${args.slice(0, 2).join(" ")} ...` }] });

      const result = await pi.exec(bin, args, {
        cwd: ctx.cwd,
        env,
        signal,
        timeout: 120000,
      });

      const stdout = truncateText(redactSecrets(result.stdout ?? ""));
      const stderr = truncateText(redactSecrets(result.stderr ?? ""), 6000);

      if (result.code !== 0) {
        throw new Error(`SkillSpector failed with exit code ${result.code}.\n${stderr.text}`);
      }

      const outputPath = resolveMaybePath(ctx.cwd, params.output);
      const lines = [
        `SkillSpector scan complete: ${params.target}`,
        `format: ${params.format ?? "terminal"}`,
        `noLlm: ${params.noLlm ?? true}`,
      ];
      if (outputPath) lines.push(`output: ${outputPath}`);
      if (stdout.truncated) lines.push("stdout truncated: true");
      if (stderr.text.trim()) lines.push(`stderr:\n${stderr.text}`);
      if (stdout.text.trim()) lines.push(`stdout:\n${stdout.text}`);

      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: {
          code: result.code,
          command: bin,
          args,
          outputPath,
          stdoutTruncated: stdout.truncated,
          stderrTruncated: stderr.truncated,
        },
      };
    },
  });
}
