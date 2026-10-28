// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with Node >= 22.18: node --test tests/unit/test_pi_extension.mjs
// The CLI and schema-only peer dependencies are mocked; no subprocess is started.
import assert from "node:assert/strict";
import { copyFileSync, existsSync, linkSync, mkdirSync, mkdtempSync, readFileSync,
  readdirSync, realpathSync, renameSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import test from "node:test";
import { pathToFileURL } from "node:url";

registerHooks({
  resolve(specifier, context, nextResolve) {
    const stub = specifier === "@earendil-works/pi-ai"
      ? "export const StringEnum = () => ({});"
      : specifier === "typebox"
        ? "export const Type = {Object: x => x, Optional: x => x, String: () => ({}), Boolean: () => ({})};"
        : undefined;
    return stub
      ? { url: `data:text/javascript,${encodeURIComponent(stub)}`, shortCircuit: true }
      : nextResolve(specifier, context);
  },
});

async function setup(t, exec, confirm = async () => false) {
  const root = realpathSync(mkdtempSync(join(tmpdir(), "skillspector-pi-test-")));
  const workspace = join(root, "workspace");
  const install = join(root, "install");
  const bin = join(install, process.platform === "win32" ? ".venv/Scripts/skillspector.exe" : ".venv/bin/skillspector");
  const source = process.env.SKILLSPECTOR_EXTENSION_SOURCE
    ?? new URL("../../extensions/skillspector.ts", import.meta.url);
  mkdirSync(workspace);
  writeFileSync(join(workspace, "SKILL.md"), "---\nname: synthetic-skill\ndescription: Local permission test\n---\nA benign skill.");
  mkdirSync(join(install, "extensions"), { recursive: true });
  mkdirSync(dirname(bin), { recursive: true });
  writeFileSync(bin, "unused mocked executable");
  writeFileSync(join(install, "package.json"), '{"type":"module"}');
  copyFileSync(source, join(install, "extensions/skillspector.ts"));
  const originalBin = process.env.SKILLSPECTOR_BIN;
  delete process.env.SKILLSPECTOR_BIN;
  t.after(() => {
    if (originalBin === undefined) delete process.env.SKILLSPECTOR_BIN;
    else process.env.SKILLSPECTOR_BIN = originalBin;
    rmSync(root, { recursive: true, force: true });
  });
  const { default: register } = await import(pathToFileURL(join(install, "extensions/skillspector.ts")));
  let tool;
  const calls = [];
  const prompts = [];
  register({
    registerTool(registered) { tool = registered; },
    async exec(command, args, options) {
      const outputIndex = args.indexOf("--output");
      const output = outputIndex < 0 ? undefined : resolve(options.cwd, args[outputIndex + 1]);
      calls.push({ command, args, options, output });
      if (output) writeFileSync(output, "new report");
      return exec?.({ command, args, options, output, root, workspace })
        ?? { code: 0, stdout: "complete", stderr: "" };
    },
  });
  return {
    root, workspace, bin, calls, prompts,
    scan: (params = {}, context = {}, signal) => tool.execute("scan", { target: "./SKILL.md", ...params }, signal, undefined, {
      cwd: workspace,
      hasUI: true,
      ui: { async confirm(title, message, options) {
        prompts.push({ title, message });
        return new Promise((resolve, reject) => {
          const signal = options?.signal;
          const abort = () => reject(signal.reason);
          if (signal?.aborted) return abort();
          signal?.addEventListener("abort", abort, { once: true });
          Promise.resolve(confirm({ root, workspace })).then(resolve, reject).finally(() => signal?.removeEventListener("abort", abort));
        });
      } },
      ...context,
    }),
  };
}

test("uses installed absolute executable and preserves scan arguments without output", async (t) => {
  const ctx = await setup(t);
  await ctx.scan({ provider: "anthropic", model: "synthetic-model", verbose: true });
  assert.equal(ctx.calls[0].command, ctx.bin);
  assert.deepEqual(ctx.calls[0].args, ["scan", join(ctx.workspace, "SKILL.md"), "--format", "terminal", "--no-llm", "--verbose"]);
  assert.deepEqual(ctx.calls[0].options.env, { SKILLSPECTOR_PROVIDER: "anthropic", SKILLSPECTOR_MODEL: "synthetic-model" });
  assert.equal(ctx.calls[0].options.cwd, ctx.workspace);
});

test("redacts long scanner output without retrying every word character", { timeout: 5000 }, async (t) => {
  const ctx = await setup(t, () => ({
    code: 0,
    stdout: "A".repeat(100_000),
    stderr: "B".repeat(100_000),
  }));
  const started = performance.now();
  const result = await ctx.scan();
  const elapsed = performance.now() - started;
  assert.ok(elapsed < 1000, `redaction took ${elapsed.toFixed(1)} ms`);
  assert.equal(result.details.stdoutTruncated, true);
  assert.equal(result.details.stderrTruncated, true);
  assert.ok(result.content[0].text.length < 19_000);
});

test("redacts complete secrets before truncating at display boundaries", async (t) => {
  const ctx = await setup(t, () => ({
    code: 0,
    stdout: " ".repeat(11_990) + "sk-" + "x".repeat(32),
    stderr: "NPM_TOKEN=synthetic-token\nCUSTOM_API_KEY: synthetic-key\napi_key=lower-key",
  }));
  const result = await ctx.scan();
  const text = result.content[0].text;
  for (const secret of ["sk-", "synthetic-token", "synthetic-key", "lower-key"]) {
    assert.equal(text.includes(secret), false);
  }
  assert.match(text, /NPM_TOKEN=\[REDACTED\]/);
  assert.match(text, /CUSTOM_API_KEY: \[REDACTED\]/);
  assert.match(text, /api_key=\[REDACTED\]/);
});

test("finds the Windows virtualenv executable without a PATH fallback", async (t) => {
  const ctx = await setup(t);
  rmSync(ctx.bin);
  const windowsBin = join(ctx.root, "install/.venv/Scripts/skillspector.exe");
  mkdirSync(dirname(windowsBin), { recursive: true });
  writeFileSync(windowsBin, "unused mocked executable");
  const originalPlatform = Object.getOwnPropertyDescriptor(process, "platform");
  Object.defineProperty(process, "platform", { value: "win32" });
  try {
    await ctx.scan();
    assert.equal(ctx.calls[0].command, windowsBin);
  } finally {
    Object.defineProperty(process, "platform", originalPlatform);
  }
});

for (const [noLlm, timeout] of [[undefined, 120000], [true, 120000], [false, 630000]]) {
  test(`uses ${timeout}ms for noLlm=${noLlm}`, async (t) => {
    const ctx = await setup(t);
    await ctx.scan({ noLlm });
    assert.equal(ctx.calls[0].options.timeout, timeout);
    assert.equal(ctx.calls[0].args.includes("--no-llm"), noLlm ?? true);
  });
}

test("uses an absolute operator override and preserves URL targets", async (t) => {
  const ctx = await setup(t, undefined, async () => true);
  process.env.SKILLSPECTOR_BIN = join(ctx.root, "custom-cli");
  writeFileSync(process.env.SKILLSPECTOR_BIN, "unused");
  await ctx.scan({ target: "https://example.test/skill", noLlm: false, format: "json" });
  assert.equal(ctx.calls[0].command, process.env.SKILLSPECTOR_BIN);
  assert.deepEqual(ctx.calls[0].args, ["scan", "https://example.test/skill", "--format", "json"]);
});

test("keeps nested local paths local and scans workspace inputs without a prompt", async (t) => {
  const ctx = await setup(t);
  mkdirSync(join(ctx.workspace, "owner/repo"), { recursive: true });
  mkdirSync(join(ctx.workspace, "rules"));
  for (const target of ["owner/repo", join(ctx.workspace, "SKILL.md"), "  ./SKILL.md  "]) {
    await ctx.scan({ target, yaraRulesDir: "rules" }, { hasUI: false });
    assert.equal(ctx.calls.at(-1).args[1], realpathSync(resolve(ctx.workspace, target.trim())));
    assert.equal(ctx.calls.at(-1).args.at(-1), join(ctx.workspace, "rules"));
  }
  assert.equal(ctx.prompts.length, 0);
});

test("requires approval before exposing external targets or YARA rules to the CLI", async (t) => {
  const ctx = await setup(t);
  const secret = join(ctx.root, "private.md");
  const rules = join(ctx.root, "rules");
  writeFileSync(secret, "synthetic private content");
  mkdirSync(rules);
  symlinkSync(secret, join(ctx.workspace, "linked.md"));
  symlinkSync(rules, join(ctx.workspace, "rules"));
  for (const params of [
    { target: secret },
    { target: "../private.md" },
    { target: "linked.md" },
    { yaraRulesDir: rules },
    { yaraRulesDir: "../rules" },
    { yaraRulesDir: "rules" },
    { target: "https://example.test/skill.zip" },
    { target: "git@github.com:owner/repo.git" },
  ]) {
    await assert.rejects(ctx.scan(params), /not approved/);
    await assert.rejects(ctx.scan(params, { hasUI: false }), /require user confirmation/);
  }
  assert.equal(ctx.calls.length, 0);
  assert.equal(ctx.prompts.length, 8);
  assert.equal(readFileSync(secret, "utf8"), "synthetic private content");
  assert.ok(ctx.prompts[2].message.includes(JSON.stringify(secret)));
  assert.ok(ctx.prompts[5].message.includes(JSON.stringify(rules)));
});

test("approves canonical external scope and preserves the CLI target symlink policy", async (t) => {
  const ctx = await setup(t, undefined, async () => true);
  writeFileSync(join(ctx.root, "private.md"), "synthetic private content");
  mkdirSync(join(ctx.root, "rules"));
  symlinkSync(ctx.root, join(ctx.workspace, "external"));
  await ctx.scan({ target: "external/private.md", yaraRulesDir: "external/rules" });
  assert.equal(ctx.prompts.length, 1);
  assert.match(ctx.prompts[0].message, /Read external scan target:/);
  assert.match(ctx.prompts[0].message, /Read external YARA rules:/);
  assert.equal(ctx.calls[0].args[1], join(ctx.workspace, "external/private.md"));
  assert.equal(ctx.calls[0].args.at(-1), join(ctx.root, "rules"));
});

test("treats local git@ paths as canonical local reads", async (t) => {
  const ctx = await setup(t, undefined, async () => true);
  mkdirSync(join(ctx.workspace, "git@notes"));
  writeFileSync(join(ctx.workspace, "git@notes", "local.md"), "local notes");
  writeFileSync(join(ctx.root, "private.md"), "synthetic private content");
  await ctx.scan({ target: "git@notes/local.md" }, { hasUI: false });
  assert.equal(ctx.calls[0].args[1], join(ctx.workspace, "git@notes", "local.md"));
  assert.equal(ctx.prompts.length, 0);
  await ctx.scan({ target: "git@notes/../../private.md" });
  assert.equal(ctx.calls[1].args[1], join(ctx.root, "private.md"));
  assert.equal(ctx.prompts.length, 1);
  assert.match(ctx.prompts[0].message, /Read external scan target:/);
  assert.ok(ctx.prompts[0].message.includes(JSON.stringify(join(ctx.root, "private.md"))));
  assert.doesNotMatch(ctx.prompts[0].message, /Fetch remote/);
});

test("does not launch until approval arrives or after a canceled dialog", async (t) => {
  const ctx = await setup(t, undefined, () => new Promise(() => {}));
  const controller = new AbortController();
  const scan = ctx.scan({ target: "https://example.test/skill.zip" }, {}, controller.signal);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(ctx.calls.length, 0);
  controller.abort();
  await assert.rejects(scan, /abort/i);
  assert.equal(ctx.calls.length, 0);
});

test("rejects input aliases retargeted while awaiting approval", async (t) => {
  const ctx = await setup(t, undefined, async ({ workspace, root }) => {
    renameSync(join(workspace, "SKILL.md"), join(workspace, "original.md"));
    symlinkSync(join(root, "private.md"), join(workspace, "SKILL.md"));
    return true;
  });
  writeFileSync(join(ctx.root, "private.md"), "synthetic private content");
  mkdirSync(join(ctx.root, "rules"));
  await assert.rejects(ctx.scan({ yaraRulesDir: "../rules" }), /input path changed/);
  assert.equal(ctx.calls.length, 0);
});

test("rejects remote YARA directories instead of treating them as scan targets", async (t) => {
  const ctx = await setup(t, undefined, async () => true);
  await assert.rejects(ctx.scan({ yaraRulesDir: "https://example.test/rules" }), /local directory/);
  assert.equal(ctx.calls.length, 0);
  assert.equal(ctx.prompts.length, 0);
});

test("never resolves a workspace executable through PATH or a relative override", async (t) => {
  const ctx = await setup(t);
  rmSync(ctx.bin);
  writeFileSync(join(ctx.workspace, "skillspector"), "unused attacker executable");
  for (const value of [undefined, "skillspector", "./skillspector", "../skillspector"]) {
    if (value === undefined) delete process.env.SKILLSPECTOR_BIN;
    else process.env.SKILLSPECTOR_BIN = value;
    await assert.rejects(ctx.scan(), /absolute executable path/);
  }
  assert.equal(ctx.calls.length, 0);
});

test("publishes relative and absolute in-workspace reports and replaces regular files", async (t) => {
  const ctx = await setup(t);
  mkdirSync(join(ctx.workspace, "reports"));
  for (const output of ["reports/result.json", join(ctx.workspace, "reports/result.json")]) {
    const destination = resolve(ctx.workspace, output);
    writeFileSync(destination, "old report");
    const result = await ctx.scan({ output });
    assert.equal(readFileSync(destination, "utf8"), "new report");
    assert.equal(result.details.outputPath, destination);
    assert.notEqual(ctx.calls.at(-1).output, destination);
    assert.equal(existsSync(dirname(ctx.calls.at(-1).output)), false);
    assert.deepEqual(readdirSync(dirname(destination)), ["result.json"]);
  }
});

test("new and replaced reports remain private under a permissive umask", { skip: process.platform === "win32" }, async (t) => {
  const previousUmask = process.umask(0o022);
  t.after(() => process.umask(previousUmask));
  const ctx = await setup(t);
  writeFileSync(join(ctx.workspace, "existing.txt"), "private report", { mode: 0o600 });
  for (const output of ["existing.txt", "new.txt"]) {
    await ctx.scan({ output });
    assert.equal(statSync(join(ctx.workspace, output)).mode & 0o777, 0o600);
    assert.equal(readFileSync(join(ctx.workspace, output), "utf8"), "new report");
  }
});

test("rejects absolute and parent-relative escapes before invoking CLI", async (t) => {
  const ctx = await setup(t);
  const outside = join(ctx.root, "outside.txt");
  writeFileSync(outside, "preserve");
  for (const output of [outside, "../outside.txt", "../workspace-other/report", "."]) {
    await assert.rejects(ctx.scan({ output }), /within the current workspace/);
  }
  assert.equal(readFileSync(outside, "utf8"), "preserve");
  assert.equal(ctx.calls.length, 0);
});

test("rejects symlinked parents, symlink files, and dangling symlinks", async (t) => {
  const ctx = await setup(t);
  const outside = join(ctx.root, "outside.txt");
  writeFileSync(outside, "preserve");
  symlinkSync(ctx.root, join(ctx.workspace, "escape"));
  symlinkSync(outside, join(ctx.workspace, "linked.txt"));
  symlinkSync(join(ctx.root, "missing.txt"), join(ctx.workspace, "dangling.txt"));
  for (const output of ["escape/outside.txt", "linked.txt", "dangling.txt"]) {
    await assert.rejects(ctx.scan({ output }), /within the current workspace|regular file/);
  }
  assert.equal(readFileSync(outside, "utf8"), "preserve");
  assert.equal(existsSync(join(ctx.root, "missing.txt")), false);
  assert.equal(ctx.calls.length, 0);
});

test("replacing a hardlinked report does not alter the outside inode", async (t) => {
  const ctx = await setup(t);
  const outside = join(ctx.root, "outside.txt");
  writeFileSync(outside, "preserve");
  linkSync(outside, join(ctx.workspace, "report.txt"));
  await ctx.scan({ output: "report.txt" });
  assert.equal(readFileSync(outside, "utf8"), "preserve");
  assert.equal(readFileSync(join(ctx.workspace, "report.txt"), "utf8"), "new report");
});

test("rechecks a report parent changed during the scan and cleans private output", async (t) => {
  const ctx = await setup(t, ({ root, workspace }) => {
    renameSync(join(workspace, "reports"), join(workspace, "original-reports"));
    symlinkSync(root, join(workspace, "reports"));
  });
  mkdirSync(join(ctx.workspace, "reports"));
  await assert.rejects(ctx.scan({ output: "reports/result.txt" }), /within the current workspace/);
  assert.equal(existsSync(join(ctx.root, "result.txt")), false);
  assert.equal(existsSync(dirname(ctx.calls[0].output)), false);
});

test("rejects a report replaced by a symlink during the scan", async (t) => {
  const ctx = await setup(t, ({ root, workspace }) => {
    rmSync(join(workspace, "report.txt"), { force: true });
    symlinkSync(join(root, "outside.txt"), join(workspace, "report.txt"));
  });
  writeFileSync(join(ctx.root, "outside.txt"), "preserve");
  await assert.rejects(ctx.scan({ output: "report.txt" }), /regular file/);
  assert.equal(readFileSync(join(ctx.root, "outside.txt"), "utf8"), "preserve");
  assert.equal(existsSync(dirname(ctx.calls[0].output)), false);
});

test("preserves generated reports when the scanner returns exit 1 or 2", async (t) => {
  for (const code of [1, 2]) {
    await t.test(`exit ${code}`, async (t) => {
      const report = JSON.stringify({ execution_successful: code !== 2 });
      const ctx = await setup(t, ({ output }) => {
        writeFileSync(output, report);
        return { code, stderr: "scan failed" };
      });
      writeFileSync(join(ctx.workspace, "existing.json"), "old report");
      for (const output of ["existing.json", "new.json"]) {
        await assert.rejects(ctx.scan({ format: "json", output }), new RegExp(`exit code ${code}`));
        const destination = join(ctx.workspace, output);
        assert.equal(readFileSync(destination, "utf8"), report);
        if (process.platform !== "win32") assert.equal(statSync(destination).mode & 0o777, 0o600);
        assert.equal(existsSync(dirname(ctx.calls.at(-1).output)), false);
      }
      assert.deepEqual(readdirSync(ctx.workspace), ["SKILL.md", "existing.json", "new.json"]);
    });
  }
});

test("cleans staged reports after operational failure, cancellation, and missing output", async (t) => {
  for (const failure of ["status", "empty", "symlink", "cancel", "missing", "killed"]) {
    await t.test(failure, async (t) => {
      const ctx = await setup(t, ({ output, workspace }) => {
        if (failure === "cancel") throw new Error("cancelled");
        if (["status", "missing", "symlink"].includes(failure)) rmSync(output);
        if (failure === "empty") writeFileSync(output, "");
        if (failure === "symlink") symlinkSync(join(workspace, "report.txt"), output);
        return { code: failure === "missing" ? 0 : failure === "killed" ? 137 : 2, stderr: "synthetic failure" };
      });
      writeFileSync(join(ctx.workspace, "report.txt"), "preserve");
      await assert.rejects(ctx.scan({ output: "report.txt" }));
      assert.equal(readFileSync(join(ctx.workspace, "report.txt"), "utf8"), "preserve");
      assert.equal(existsSync(dirname(ctx.calls[0].output)), false);
      assert.deepEqual(readdirSync(ctx.workspace), ["SKILL.md", "report.txt"]);
    });
  }
});


test("rejects external headless paths uniformly before checking their existence", async (t) => {
  const ctx = await setup(t);
  writeFileSync(join(ctx.root, "existing.md"), "private");
  for (const target of ["../existing.md", "../missing.md"]) {
    await assert.rejects(ctx.scan({ target }, { hasUI: false }), /require user confirmation/);
  }
  assert.equal(ctx.calls.length, 0);
});

test("rejects invalid output and pre-canceled calls before showing a dialog", async (t) => {
  const ctx = await setup(t);
  await assert.rejects(ctx.scan({ target: "https://example.test/skill", output: "../report.json" }), /within the current workspace/);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(ctx.scan({ target: "https://example.test/skill" }, {}, controller.signal), /abort/i);
  assert.equal(ctx.prompts.length, 0);
  assert.equal(ctx.calls.length, 0);
});

test("rejects a workspace swapped while a remote target awaits approval", async (t) => {
  const ctx = await setup(t, undefined, async ({ root, workspace }) => {
    renameSync(workspace, join(root, "original-workspace"));
    symlinkSync(root, workspace);
    return true;
  });
  await assert.rejects(ctx.scan({ target: "https://example.test/skill" }), /input path changed/);
  assert.equal(ctx.calls.length, 0);
});

test("gives a clear error for missing workspace targets or rule directories", async (t) => {
  const ctx = await setup(t);
  for (const params of [{ target: "missing.md" }, { yaraRulesDir: "missing-rules" }]) {
    await assert.rejects(ctx.scan(params), /Could not resolve/);
  }
  assert.equal(ctx.calls.length, 0);
  assert.equal(ctx.prompts.length, 0);
});
