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

async function setup(t, exec) {
  const root = realpathSync(mkdtempSync(join(tmpdir(), "skillspector-pi-test-")));
  const workspace = join(root, "workspace");
  const install = join(root, "install");
  const bin = join(install, process.platform === "win32" ? ".venv/Scripts/skillspector.exe" : ".venv/bin/skillspector");
  const source = process.env.SKILLSPECTOR_EXTENSION_SOURCE
    ?? new URL("../../extensions/skillspector.ts", import.meta.url);
  mkdirSync(workspace);
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
    root, workspace, bin, calls,
    scan: (params = {}) => tool.execute("scan", { target: "./SKILL.md", ...params }, undefined, undefined, { cwd: workspace }),
  };
}

test("uses installed absolute executable and preserves scan arguments without output", async (t) => {
  const ctx = await setup(t);
  await ctx.scan({ provider: "anthropic", model: "synthetic-model", verbose: true });
  assert.equal(ctx.calls[0].command, ctx.bin);
  assert.deepEqual(ctx.calls[0].args, ["scan", "./SKILL.md", "--format", "terminal", "--no-llm", "--verbose"]);
  assert.deepEqual(ctx.calls[0].options.env, { SKILLSPECTOR_PROVIDER: "anthropic", SKILLSPECTOR_MODEL: "synthetic-model" });
  assert.equal(ctx.calls[0].options.cwd, ctx.workspace);
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

test("uses an absolute operator override and preserves URL targets", async (t) => {
  const ctx = await setup(t);
  process.env.SKILLSPECTOR_BIN = join(ctx.root, "custom-cli");
  writeFileSync(process.env.SKILLSPECTOR_BIN, "unused");
  await ctx.scan({ target: "https://example.test/skill", noLlm: false, format: "json" });
  assert.equal(ctx.calls[0].command, process.env.SKILLSPECTOR_BIN);
  assert.deepEqual(ctx.calls[0].args, ["scan", "https://example.test/skill", "--format", "json"]);
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
      assert.deepEqual(readdirSync(ctx.workspace), ["existing.json", "new.json"]);
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
      assert.deepEqual(readdirSync(ctx.workspace), ["report.txt"]);
    });
  }
});
