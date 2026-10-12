import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { StringEnum } from "@earendil-works/pi-ai";
import { Type, type Static } from "typebox";
import { chmodSync, constants, copyFileSync, existsSync, lstatSync, mkdtempSync, realpathSync, renameSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const scanSchema = Type.Object({
  target: Type.String({ description: "Path, URL, zip, Git repo, or SKILL.md to scan. External paths and remote targets require user confirmation." }),
  format: Type.Optional(
    StringEnum(["terminal", "json", "markdown", "sarif"] as const, {
      description: "SkillSpector output format. Defaults to terminal.",
    }),
  ),
  output: Type.Optional(Type.String({ description: "Optional report output path within the current workspace." })),
  noLlm: Type.Optional(Type.Boolean({ description: "Skip LLM analysis. Defaults to true." })),
  provider: Type.Optional(
    StringEnum(["openai", "anthropic", "anthropic_proxy", "nv_build", "nv_inference", "gemini"] as const, {
      description: "Optional SkillSpector LLM provider when noLlm is false.",
    }),
  ),
  model: Type.Optional(Type.String({ description: "Optional model override." })),
  yaraRulesDir: Type.Optional(Type.String({ description: "Optional extra YARA rules directory. External paths require user confirmation." })),
  verbose: Type.Optional(Type.Boolean({ description: "Show detailed progress." })),
});

type SkillSpectorScanParams = Static<typeof scanSchema>;

function redactSecrets(value: string): string {
  return value
    .replace(/(sk-ant-[A-Za-z0-9_-]{12,})/g, "[REDACTED_ANTHROPIC_KEY]")
    .replace(/(sk-[A-Za-z0-9_-]{20,})/g, "[REDACTED_OPENAI_KEY]")
    // Start only at a word boundary; retrying at every character is quadratic.
    .replace(/\b([A-Za-z0-9_]*(?:API_KEY|TOKEN)[=:]\s*)[^\s]+/gi, "$1[REDACTED]");
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
  const bundledBin = process.platform === "win32" ? ".venv/Scripts/skillspector.exe" : ".venv/bin/skillspector";
  const bin = process.env.SKILLSPECTOR_BIN ?? resolve(packageRoot(), bundledBin);
  if (!isAbsolute(bin) || !existsSync(bin)) {
    throw new Error("Install SkillSpector in the extension's .venv or set SKILLSPECTOR_BIN to an absolute executable path.");
  }
  return bin;
}

function isWithin(root: string, path: string): boolean {
  const rel = relative(root, path);
  return rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel);
}

async function approveScanInputs(
  params: SkillSpectorScanParams,
  ctx: ExtensionContext,
  signal?: AbortSignal,
): Promise<SkillSpectorScanParams> {
  signal?.throwIfAborted();
  const workspace = realpathSync(ctx.cwd);
  const prepared = { ...params };
  const localPaths: Array<{ field: "target" | "yaraRulesDir"; input: string; resolved?: string }> = [];
  const requests: string[] = [];
  const readRequest = (field: string, path: string) =>
    `Read external ${field === "target" ? "scan target" : "YARA rules"}: ${JSON.stringify(path)}`;
  async function approve(requests: string[]): Promise<void> {
    if (!requests.length) return;
    signal?.throwIfAborted();
    if (!ctx.hasUI) throw new Error("External scan inputs require user confirmation in an interactive or RPC session.");
    const approved = await ctx.ui.confirm(
      "Allow SkillSpector external access?",
      `${requests.join("\n")}\n\nScanned content and matching rule text can appear in the agent conversation.`,
      { signal },
    );
    signal?.throwIfAborted();
    if (!approved) throw new Error("SkillSpector external access was not approved.");
  }
  for (const field of ["target", "yaraRulesDir"] as const) {
    const value = params[field]?.trim();
    if (field === "yaraRulesDir" && !value) {
      prepared[field] = undefined;
      continue;
    }
    if (!value) throw new Error("A scan target is required.");
    // Match the CLI's remote forms. A local owner/repo path is not a URL.
    const remote = !isAbsolute(value) && (value.startsWith("https://") || (value.startsWith("git@") && value.endsWith(".git")));
    if (remote) {
      if (field !== "target") throw new Error("YARA rules must be a local directory.");
      prepared[field] = value;
      requests.push(`Fetch remote scan target: ${JSON.stringify(value)}`);
    } else {
      const input = resolve(ctx.cwd, value);
      // Ask before resolving external paths, which can probe the host or access
      // a Windows network share even when the file is never opened.
      if (!isWithin(resolve(ctx.cwd), input)) {
        localPaths.push({ field, input });
        requests.push(readRequest(field, input));
      } else {
        let resolved: string;
        try {
          resolved = realpathSync(input);
        } catch {
          throw new Error(`Could not resolve ${field === "target" ? "scan target" : "YARA rules directory"}. Check that it exists and is accessible.`);
        }
        localPaths.push({ field, input, resolved });
        if (!isWithin(workspace, resolved)) requests.push(readRequest(field, resolved));
      }
    }
  }
  await approve(requests);
  const aliasRequests: string[] = [];
  for (const path of localPaths) {
    try {
      path.resolved ??= realpathSync(path.input);
    } catch {
      throw new Error(`Could not resolve ${path.field === "target" ? "scan target" : "YARA rules directory"}. Check that it exists and is accessible.`);
    }
    if (!isWithin(resolve(ctx.cwd), path.input) && !isWithin(workspace, path.resolved) && path.resolved !== path.input) {
      aliasRequests.push(readRequest(path.field, path.resolved));
    }
    // Keep the original target path so the CLI can enforce its no-symlink
    // input policy. YARA directories are canonicalised by the CLI too.
    prepared[path.field] = path.field === "target" ? path.input : path.resolved;
  }
  await approve(aliasRequests);
  signal?.throwIfAborted();
  if (realpathSync(ctx.cwd) !== workspace || localPaths.some(({ input, resolved }) => realpathSync(input) !== resolved)) {
    throw new Error("Scan input path changed while awaiting confirmation.");
  }
  return prepared;
}

function reportOutputPath(cwd: string, value?: string): string | undefined {
  if (!value) return undefined;
  const output = resolve(cwd, value);
  if (output === resolve(cwd) || !isWithin(resolve(cwd), output)) {
    throw new Error("Report output must be a file within the current workspace.");
  }
  const parent = realpathSync(dirname(output));
  if (!isWithin(realpathSync(cwd), parent)) {
    throw new Error("Report output must be a file within the current workspace.");
  }
  const destination = join(parent, basename(output));
  const existing = lstatSync(destination, { throwIfNoEntry: false });
  if (existing && !existing.isFile()) {
    throw new Error("Report output must be a regular file, not a symlink or directory.");
  }
  return destination;
}

function publishReport(source: string, destination: string): void {
  const staging = mkdtempSync(join(dirname(destination), ".skillspector-report-"));
  try {
    const report = join(staging, "report");
    copyFileSync(source, report, constants.COPYFILE_EXCL);
    chmodSync(report, 0o600);
    // Replacing the directory entry never follows an existing hard link.
    renameSync(report, destination);
  } finally {
    rmSync(staging, { recursive: true, force: true });
  }
}

function buildScanArgs(params: SkillSpectorScanParams, output?: string): string[] {
  const args = ["scan", params.target];
  args.push("--format", params.format ?? "terminal");

  const noLlm = params.noLlm ?? true;
  if (noLlm) args.push("--no-llm");

  if (output) args.push("--output", output);

  if (params.yaraRulesDir) args.push("--yara-rules-dir", params.yaraRulesDir);

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
      const outputPath = reportOutputPath(ctx.cwd, params.output);
      const prepared = await approveScanInputs(params, ctx, signal);
      const reportDir = outputPath ? mkdtempSync(join(tmpdir(), "skillspector-report-")) : undefined;
      const reportPath = reportDir ? join(reportDir, "report") : undefined;
      try {
        const args = buildScanArgs(prepared, reportPath);
        const env: Record<string, string> = {};

        if (params.provider) env.SKILLSPECTOR_PROVIDER = params.provider;
        if (params.model) env.SKILLSPECTOR_MODEL = params.model;

        onUpdate?.({ content: [{ type: "text", text: `Running ${bin} ${args.slice(0, 2).join(" ")} ...` }] });

        const result = await pi.exec(bin, args, {
          cwd: ctx.cwd,
          env,
          signal,
          // Match the CLI's 600s workflow budget plus startup/report headroom.
          timeout: (params.noLlm ?? true) ? 120000 : 630000,
        });

        const stdout = truncateText(redactSecrets(result.stdout ?? ""));
        const stderr = truncateText(redactSecrets(result.stderr ?? ""), 6000);

        // Exit 2 can follow a diagnostic report; failures before reporting produce no file.
        const failureReport = result.code === 2 && reportPath
          ? lstatSync(reportPath, { throwIfNoEntry: false }) : undefined;
        const hasFailureReport = failureReport?.isFile() && failureReport.size > 0;
        if ((result.code === 0 || result.code === 1 || hasFailureReport) && outputPath && reportPath) {
          if (reportOutputPath(ctx.cwd, params.output) !== outputPath) {
            throw new Error("Report output directory changed during the scan.");
          }
          publishReport(reportPath, outputPath);
        }
        if (result.code !== 0) {
          throw new Error(`SkillSpector failed with exit code ${result.code}.\n${stderr.text}`);
        }
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
      } finally {
        if (reportDir) rmSync(reportDir, { recursive: true, force: true });
      }
    },
  });
}
