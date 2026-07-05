<div align="center">

# Skill Inspector

> *「别急着安装。先让 skill 解释清楚自己。」*

[![Agent Skills](https://img.shields.io/badge/Agent%20Skills-Compatible-green)](https://agentskills.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Runtime](https://img.shields.io/badge/Runtime-Claude%20Code%20·%20Codex%20·%20WorkBuddy-blueviolet)](#安装本-skill)
[![Powered by SkillSpector](https://img.shields.io/badge/Scanner-NVIDIA%20SkillSpector-76B900)](https://github.com/NVIDIA/SkillSpector)
[![Default Mode](https://img.shields.io/badge/Default-Static%20Scan%20%2B%20Agent%20Review-blue)](#工作原理)

**一个用于审查 AI Agent Skill 的安装前安全工作流。**

Skill Inspector 先调用 NVIDIA SkillSpector 做静态扫描，再让当前 Agent 读取源码做语义复核，最后给出 `APPROVE` / `CAUTION` / `REJECT`。

基于通用 Agent Skills 目录结构，可用于 Claude Code、Codex、WorkBuddy 以及其他支持 `SKILL.md` 的 runtime。

[安装前置](#安装前置) · [安装本 Skill](#安装本-skill) · [使用](#使用) · [工作原理](#工作原理) · [判定标准](#判定标准)

</div>

---

## 它解决什么问题

Agent skill 本质上是一组会影响 Agent 行为的说明、脚本和工具定义。安装一个第三方 skill，等于让它进入你的工作流。

Skill Inspector 用来回答三个问题：

1. 这个 skill 有没有读凭据、发网络请求、执行 shell、写文件或声明过宽权限？
2. 这些敏感行为是否和它的用途一致？
3. 它应该被安装、谨慎使用，还是拒绝？

它不是替代 SkillSpector 的扫描器。它是一个审查编排层：**静态扫描抓证据，Agent 复核判断意图。**

---

## 安装前置

Skill Inspector 依赖 NVIDIA SkillSpector CLI。请先安装它。

### 给人看的入口

- 项目主页：[NVIDIA/SkillSpector](https://github.com/NVIDIA/SkillSpector)

先打开官方 README，按里面的说明安装。常见安装方式：

```bash
pipx install 'git+https://github.com/NVIDIA/SkillSpector.git'
```

确认安装成功：

```bash
skillspector --version
```

### 给 Agent 用的入口

把下面这段发给你的 Agent，让它帮你安装并验证：

```text
请安装 NVIDIA SkillSpector CLI，并确认 `skillspector --version` 可用。
项目地址：https://github.com/NVIDIA/SkillSpector
推荐命令：pipx install 'git+https://github.com/NVIDIA/SkillSpector.git'
安装后运行：skillspector --version
```

Skill Inspector 本身不会自动安装 SkillSpector。没有 `skillspector` 命令时，它只能退化为手工源码审查，并会在报告里说明降级。

---

## 安装本 Skill

把本目录放到你正在使用的 runtime 的 skills 目录即可。Skill Inspector 只依赖标准 `SKILL.md`，不是 Codex 专用 skill。

### 给 Agent 用

把下面这段发给 Claude Code、Codex、WorkBuddy 或其他支持 skills 的 Agent：

```text
请把这个 skill 安装到当前 runtime 的 skills 目录：
/path/to/skill-inspector

安装后请确认能通过 skill 名称 `skill-inspector` 触发。
```

### 手动安装

不同 runtime 的目录可能不同，常见路径如下：

| Runtime | 常见安装目录 |
|---|---|
| Claude Code | `~/.claude/skills/skill-inspector/` |
| Codex | `~/.codex/skills/skill-inspector/` |
| WorkBuddy | 使用 WorkBuddy 的 skills 目录 |
| 其他 runtime | 对应 runtime 的 skills 目录 |

示例：

```bash
# Codex
cp -R skill-inspector ~/.codex/skills/

# Claude Code
cp -R skill-inspector ~/.claude/skills/
```

如果你的 runtime 支持 Agent Skills 风格的 `SKILL.md` 加载，就可以使用；不需要固定安装到 `.codex`。

---

## 使用

装好后，直接让 Agent 审查目标 skill：

```text
用 skill-inspector 审查 /path/to/skill
```

或者：

```text
这个 skill 安全吗？/path/to/skill
```

也可以审查从 GitHub 下载到本地的 skill 目录：

```text
帮我看看 ~/Downloads/some-skill 能不能装
```

---

## 工作原理

Skill Inspector 有两条审查线：一条负责找证据，一条负责判断证据是否合理。最后再把两条线合并成安装建议。

### 1. 证据线：SkillSpector 静态扫描

默认运行：

```bash
skillspector scan "$TARGET" --no-llm --format json --output /tmp/skill-inspector-report.json
```

这一层不理解业务意图，只负责抓硬风险信号。它的输入是目标 skill 目录，输出是结构化 JSON 报告，包括风险分、严重级别、命中规则、文件位置和证据片段。

| 风险面 | 例子 |
|---|---|
| 外部传输 | `curl -d`、`requests.post`、远程 API |
| 凭据访问 | 环境变量、token、密码、本地配置 |
| 执行风险 | shell、subprocess、eval、exec |
| 文件风险 | 写入、删除、持久化、自修改 |
| MCP 风险 | 权限声明和实际行为不一致、tool poisoning |
| 供应链风险 | 下载后执行、混淆、未固定依赖 |

这一线回答的是：

> 代码里出现了哪些值得警惕的行为？

它不负责最终裁决。比如网络请求可能是数据外传，也可能是一个 API skill 的正常能力。

### 2. 判断线：Agent 语义复核

静态扫描之后，Agent 必须读取：

- 目标 `SKILL.md`
- 可执行脚本
- MCP 配置和 tool 描述
- SkillSpector 命中的文件和行号附近源码

这一层会把静态证据放回源码上下文里看：

- skill 声称自己做什么？
- 实际代码是否只做这些事？
- 敏感能力是否被文档明示？
- 权限声明和实际行为是否一致？
- 网络请求的目的地是否可信、必要？
- 读取 env、token、本地配置是否符合用途？
- 是否存在隐藏指令、触发词污染、未知外发、持久化或混淆执行？

这一线回答的是：

> 这些风险行为是否合理、必要、可控？

### 3. 合并线：综合 verdict

最终结论不是简单按分数走，而是按下面的顺序合并：

| 输入 | 作用 |
|---|---|
| SkillSpector 分数 | 给出总体风险姿态 |
| 命中规则和严重级别 | 标出需要优先看的风险点 |
| 源码上下文 | 判断 finding 是真实问题还是合理能力 |
| skill 描述和权限 | 判断是否有描述-行为不一致 |
| Agent 语义复核 | 形成最终安装建议 |

低分不自动通过，因为自然语言里可能有隐藏指令或过宽触发。高分也不机械拒绝，因为某些 skill 天然需要网络、env 或 shell 能力。

最终只输出三个 verdict：

- `APPROVE`：证据干净，或敏感行为很轻且完全符合用途。
- `CAUTION`：有敏感能力，但明示、必要、可控。
- `REJECT`：发现恶意、欺骗、未知外发、凭据风险、混淆执行，或无法解释的 HIGH / CRITICAL。

---

## 报告格式

报告使用 triage 风格，适合快速判断：

```markdown
## 🛡️ Skill Inspector: `<skill-name>`

**来源:** <path-or-url>
**结论:** <APPROVE | CAUTION | REJECT>
**风险:** <score>/100 · <severity> · <recommendation>
**使用姿态:** <适合什么环境，不适合什么环境>

### 🧭 快速判断
<核心判断>

### 📡 信号概览
| 来源 | 结果 | 解读 |
|---|---|---|
| SkillSpector 静态扫描 | <summary> | <meaning> |
| Agent 语义复核 | <summary> | <meaning> |
| 敏感面 | <summary> | <meaning> |

### 🔎 关键证据
| 规则 | 级别 | 位置 | 复核判断 |
|---|---|---|---|
| <rule> | <severity> | <file:line> | <judgment> |

### 🧠 诊断
<综合分析>

### ✅ 建议护栏
1. <condition>
2. <condition>
```

---

## 判定标准

| Verdict | 含义 |
|---|---|
| `APPROVE` | 未发现实质安全问题，行为和描述一致 |
| `CAUTION` | 存在敏感能力，但用途明确、必要、可控 |
| `REJECT` | 存在恶意、欺骗、未知外发、凭据风险、混淆执行，或无法解释的 HIGH / CRITICAL |

SkillSpector 分数只是输入信号，不是最终裁决：

| Score | 默认姿态 |
|---|---|
| `0-20` | 快速复核后通常可接受 |
| `21-35` | 命中项必须能被 skill 目的解释 |
| `36-50` | 必须人工复核，默认谨慎 |
| `51-80` | 默认拒绝，除非来源可信且敏感行为全部必要 |
| `81-100` | 拒绝 |

---

## 设计原则

- **不执行目标 skill 的脚本**：只读取源码和扫描报告。
- **不只看分数**：必须结合 skill 描述和源码语义判断。
- **不静默降级 HIGH / CRITICAL**：无法解释的高危 finding 直接进入拒绝路径。
- **允许合理敏感能力**：网络、env、文件、shell 并非天然恶意，但必须明示、必要、可控。

---

## 为什么不用 SkillSpector LLM 作为必需项

SkillSpector 支持 LLM 语义分析，但需要额外配置 provider。Skill Inspector 默认使用 `--no-llm`，让 SkillSpector 专注静态证据，再由当前 Agent 读源码做语义判断。

这降低了使用门槛，也避免因为 LLM provider 未配置而无法审查。
