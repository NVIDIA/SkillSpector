---
AIGC:
    Label: "1"
    ContentProducer: 001191110102MACQD9K64018705
    ProduceID: 4157682120407363_0/project_7655237709321027876-files/README_zh.md
    ReservedCode1: ""
    ContentPropagator: 001191110102MACQD9K64028705
    PropagateID: 4157682120407363#1782374296278
    ReservedCode2: ""
---
# SkillSpector

**面向 AI agent skills 的安全扫描器。** 在安装 agent skills 之前检测漏洞、恶意模式与安全风险。

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## 概述

AI agent skills（被 Claude Code、Codex CLI、Gemini CLI 等使用）在执行时往往被隐式信任，几乎不经任何审查。研究表明，**26.1% 的 skills 存在漏洞**，**5.2% 表现出疑似恶意意图**。

SkillSpector 帮助你回答：**"这个 skill 安装上去是否安全？"**

## 文档

- **[Development guide](docs/DEVELOPMENT.md)** —— 架构、包结构，以及如何扩展 analyzer pipeline。
- **[Pi extension](docs/PI_EXTENSION.md)** —— 将 SkillSpector 作为 Pi 工具安装，从 agent 会话内部扫描 skills。

## 功能

- **多格式输入**：扫描 Git repos、URLs、zip 文件、目录或单个文件
- **68 种漏洞模式**，覆盖 17 个类别：prompt injection、data exfiltration、privilege escalation、supply chain、excessive agency、output handling、system prompt leakage、memory poisoning、tool misuse、rogue agent、anti-refusal、trigger abuse、dangerous code (AST)、taint tracking、YARA signatures、MCP least privilege 和 MCP tool poisoning
- **两阶段分析**：快速 static analysis + 可选的 LLM 语义评估
- **实时漏洞查询**：SC4 调用 [OSV.dev](https://osv.dev) 获取实时 CVE 数据，并自动支持离线降级
- **多种输出格式**：Terminal、JSON、Markdown 和 SARIF 报告
- **风险评分**：0-100 分，附带 severity 标签和明确建议
- **Baseline / 误报抑制**：通过 glob 规则或指纹 baseline 接受已知 findings，重新扫描时只显示*新增*问题（[文档](docs/SUPPRESSION.md)）

## 快速开始

### 安装

请先创建并激活一个虚拟环境（所有 `make` 目标都假设 venv 已激活）。可选 **uv** 或 **pip**；Makefile 优先使用 `uv`，否则回退到 `pip`。

**使用 uv 快速安装（无需 clone）：**

```bash
uv tool install git+https://github.com/NVIDIA/skillspector.git
# 后续更新：uv tool update skillspector
```

**从源码安装：**

```bash
# 克隆仓库
git clone https://github.com/NVIDIA/skillspector.git
cd skillspector

# 创建并激活虚拟环境
uv venv .venv && source .venv/bin/activate
# 或：python3 -m venv .venv && source .venv/bin/activate

# 生产环境安装
make install

# 或安装包含开发依赖的版本
make install-dev
```

### Docker（无需安装 Python）

无需安装 Python 即可运行 SkillSpector：通过仓库内置的 [Dockerfile](Dockerfile) 在本地构建镜像。该镜像基于 Docker 官方 Python `3.12-slim-bookworm` 镜像。

**构建镜像：**

```bash
make docker-build
# 或：docker build -t skillspector .
```

**扫描本地目录**：将当前目录挂载到容器的工作目录 `/scan`：

```bash
docker run --rm -v "$PWD:/scan" skillspector scan ./my-skill/ --no-llm
```

**启用 LLM 分析的扫描**：通过本地 `.env` 文件传入凭证：

```bash
cat > .env <<'EOF'
SKILLSPECTOR_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
EOF
```

```bash
docker run --rm \
  -v "$PWD:/scan" \
  --env-file .env \
  skillspector scan ./my-skill/
```

或直接通过 shell 环境传入凭证：

```bash
docker run --rm \
  -v "$PWD:/scan" \
  -e SKILLSPECTOR_PROVIDER=anthropic \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  skillspector scan ./my-skill/
```

**将报告写到宿主机文件系统**：写入到挂载目录即可：

```bash
docker run --rm \
  -v "$PWD:/scan" \
  skillspector scan ./my-skill/ --no-llm --format json --output report.json
```

**可选 alias**，便于重复执行静态扫描：

```bash
alias skillspector-docker='docker run --rm -v "$PWD:/scan" skillspector'
skillspector-docker scan ./my-skill/ --no-llm
```

### 基本用法

```bash
# 扫描本地 skill 目录
skillspector scan ./my-skill/

# 扫描单个 SKILL.md 文件
skillspector scan ./SKILL.md

# 扫描 Git 仓库
skillspector scan https://github.com/user/my-skill

# 扫描 zip 文件
skillspector scan ./my-skill.zip
```

### 输出格式

```bash
# Terminal 输出（默认）—— 美化排版
skillspector scan ./my-skill/

# JSON 输出 —— 机器可读
skillspector scan ./my-skill/ --format json --output report.json

# Markdown 输出 —— 用于文档
skillspector scan ./my-skill/ --format markdown --output report.md

# SARIF 输出 —— 用于 CI/CD 集成与 IDE 工具
skillspector scan ./my-skill/ --format sarif --output report.sarif
```

### 抑制误报（baseline）

将已知/已接受的 findings 抑制掉，使风险评分仅反映尚未处置的问题，重新扫描时只显示*新增* findings。完整说明见 [suppression guide](docs/SUPPRESSION.md)。

```bash
# 将当前所有 findings 接受为 baseline（执行一次），然后提交到仓库。
skillspector baseline ./my-skill/ -o .skillspector-baseline.yaml

# 基于 baseline 扫描 —— 只报告并计入分数 NEW findings。
skillspector scan ./my-skill/ --baseline .skillspector-baseline.yaml

# 查看哪些 findings 被抑制了（仍不计入分数）。
skillspector scan ./my-skill/ --baseline .skillspector-baseline.yaml --show-suppressed
```

Baseline 也支持容忍漂移的 glob 规则（按 rule id、文件路径或 message 匹配），见 [`.skillspector-baseline.example.yaml`](.skillspector-baseline.example.yaml)。

### LLM 分析

为获得最佳效果，建议配置一个 OpenAI 兼容的 LLM endpoint 用于语义分析。通过 `SKILLSPECTOR_PROVIDER` 选择 provider；每个 provider 自带默认模型。SkillSpector 同样可对接本地的 OpenAI 兼容服务器（Ollama、vLLM、llama.cpp）以及托管的推理网关。

| Provider (`SKILLSPECTOR_PROVIDER`) | Credential env var | Endpoint | 默认模型 |
| ---------- | ---- | ---- | ---- |
| `openai` | `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`） | api.openai.com（或任意 OpenAI 兼容 URL） | `gpt-5.4` |
| `anthropic` | `ANTHROPIC_API_KEY` | api.anthropic.com | `claude-opus-4-6` |
| `anthropic_proxy` | `ANTHROPIC_PROXY_API_KEY` + `ANTHROPIC_PROXY_ENDPOINT_URL` | 任意 Vertex 风格的 raw-predict 代理 | `claude-sonnet-4-6` |
| `nv_build` | `NVIDIA_INFERENCE_KEY` | build.nvidia.com | `deepseek-ai/deepseek-v4-flash` |

```bash
# 原版 OpenAI
export SKILLSPECTOR_PROVIDER=openai
export OPENAI_API_KEY=sk-...
skillspector scan ./my-skill/

# Anthropic
export SKILLSPECTOR_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
skillspector scan ./my-skill/

# 通过 Vertex 风格代理访问 Anthropic（企业网关、GCP Vertex AI）
export SKILLSPECTOR_PROVIDER=anthropic_proxy
export ANTHROPIC_PROXY_ENDPOINT_URL=https://my-gateway.example.com/models/claude-sonnet-4-6:streamRawPredict
export ANTHROPIC_PROXY_API_KEY=your-bearer-token
export SKILLSPECTOR_MODEL=claude-sonnet-4-6
skillspector scan ./my-skill/

# NVIDIA build.nvidia.com
export SKILLSPECTOR_PROVIDER=nv_build
export NVIDIA_INFERENCE_KEY=nvapi-...
skillspector scan ./my-skill/

# 本地 Ollama 或任意 OpenAI 兼容 endpoint
export SKILLSPECTOR_PROVIDER=openai
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
export SKILLSPECTOR_MODEL=llama3.1:8b
skillspector scan ./my-skill/

# 覆盖 provider 的默认模型
export SKILLSPECTOR_MODEL=gpt-5.2
skillspector scan ./my-skill/

# 跳过 LLM 分析（更快，仅静态分析）
skillspector scan ./my-skill/ --no-llm

# 使用Deepseek
export SKILLSPECTOR_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.deepseek.com
export SKILLSPECTOR_MODEL=deepseek-v4-pro

```

### MCP Server

将 SkillSpector 作为 [Model Context Protocol](https://modelcontextprotocol.io) server 运行，任何支持 MCP 的 agent（Claude Code、Codex CLI、Gemini CLI）或远端 runtime 都可以把扫描当作一个 tool 来调用，并**基于扫描结果对 skill / MCP 安装做闸门控制** —— 让 SkillSpector 从带外审计步骤升级为运行时护栏。

```bash
# 安装可选的 MCP 依赖
pip install "skillspector[mcp]"

# stdio 传输 —— 适用于本地 CLI agents
skillspector mcp

# streamable HTTP/SSE 传输 —— 适用于远端 / A2A 调用方
skillspector mcp --transport http --host 127.0.0.1 --port 8000
```

该 server 暴露一个 tool：

- **`scan_skill(target, use_llm=true, output_format="json")`** —— 扫描 Git URL、文件 URL、`.zip`、`.md` 文件或目录，返回结构化判定：`risk_score`（0-100）、`severity`、`recommendation`、`safe_to_install` 以及 `findings`。同时返回 `llm_used` / `scan_mode`，避免把仅静态扫描的低分误认为完整扫描的"clean"。

在 Claude Code 中注册：

```bash
claude mcp add skillspector -- skillspector mcp
```

## 漏洞模式

SkillSpector 检测 **68 种漏洞模式**，覆盖 17 个类别：

### Prompt Injection（5 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| P1 | Instruction Override | HIGH | 命令忽略安全约束 |
| P2 | Hidden Instructions | HIGH | 注释/不可见文本中的恶意指令 |
| P3 | Exfiltration Commands | HIGH | 将上下文外传的指令 |
| P4 | Behavior Manipulation | MEDIUM | 隐蔽地改变 agent 决策的指令 |
| P5 | Harmful Content | CRITICAL | 可能造成物理伤害的指令 |

### Anti-Refusal（3 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| AR1 | Refusal Suppression | HIGH | 要求永不拒绝或始终遵从的指令（如 "never refuse"、"always comply"） |
| AR2 | Disclaimer Suppression | HIGH | 要求省略警告、免责声明或伦理性说明的指令（如 "no disclaimers"、"do not moralize"） |
| AR3 | Safety Policy Nullification | HIGH | 取消护栏的越狱话术（如 "you have no restrictions"、"ignore your guidelines"、"do anything now"） |

### Data Exfiltration（4 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| E1 | External Transmission | MEDIUM | 向外部 URL 发送数据 |
| E2 | Env Variable Harvesting | HIGH | 收集 API keys 与 secrets |
| E3 | File System Enumeration | MEDIUM | 扫描目录寻找敏感文件 |
| E4 | Context Leakage | HIGH | 将对话上下文外传 |

### Privilege Escalation（3 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| PE1 | Excessive Permissions | LOW | 请求超出声明功能的访问权限 |
| PE2 | Sudo/Root Execution | MEDIUM | 调用提升的系统权限 |
| PE3 | Credential Access | HIGH | 读取 SSH keys、tokens、密码 |

### Supply Chain（6 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| SC1 | Unpinned Dependencies | LOW | 包未指定版本约束 |
| SC2 | External Script Fetching | HIGH | curl \| bash 及远程代码执行 |
| SC3 | Obfuscated Code | HIGH | base64/hex 编码后执行 |
| SC4 | Known Vulnerable Dependencies | HIGH | 已知 CVE 的依赖（实时查询 OSV.dev） |
| SC5 | Abandoned Dependencies | MEDIUM | 无人维护、缺少安全更新的包 |
| SC6 | Typosquatting | HIGH | 包名与流行包高度相似 |

### Excessive Agency（4 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| EA1 | Unrestricted Tool Access | HIGH | 无任何约束的 tool 访问 |
| EA2 | Autonomous Decision Making | HIGH | 高影响决策缺少 human-in-the-loop |
| EA3 | Scope Creep | MEDIUM | 能力超出声明用途 |
| EA4 | Unbounded Resource Access | MEDIUM | 资源消耗无 rate limit 或配额 |

### Output Handling（3 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| OH1 | Unvalidated Output Injection | HIGH | 模型输出未经 sanitization 即被使用 |
| OH2 | Cross-Context Output | MEDIUM | 输出跨信任边界传递且未校验 |
| OH3 | Unbounded Output | MEDIUM | 输出大小或生成速率无上限 |

### System Prompt Leakage（3 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| P6 | Direct Leakage | HIGH | 暴露 system prompt 或内部规则的指令 |
| P7 | Indirect Extraction | MEDIUM | 通过改写、翻译或侧信道提取 |
| P8 | Tool-Based Exfiltration | HIGH | 通过文件写入或网络请求外传 system prompt |

### Memory Poisoning（3 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| MP1 | Persistent Context Injection | HIGH | 设计为跨交互持久存在的内容 |
| MP2 | Context Window Stuffing | MEDIUM | 用填充内容挤掉安全约束 |
| MP3 | Memory Manipulation | HIGH | 篡改 agent 记忆或持久化状态 |

### Tool Misuse（3 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| TM1 | Tool Parameter Abuse | HIGH | 通过特制参数触发非预期行为（shell=True、--force） |
| TM2 | Chaining Abuse | HIGH | 通过 tool 链绕过单个安全检查 |
| TM3 | Unsafe Defaults | MEDIUM | 默认配置过度宽松（禁用 TLS、无鉴权） |

### Rogue Agent（2 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| RA1 | Self-Modification | CRITICAL | 运行时修改自身代码或配置 |
| RA2 | Session Persistence | HIGH | 通过 cron jobs 或启动脚本未授权持久化 |

### Trigger Abuse（3 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| TR1 | Overly Broad Trigger | MEDIUM | 触发模式匹配过于常见的词 |
| TR2 | Shadow Command Trigger | HIGH | 触发器影子覆盖内置命令或其他 skills |
| TR3 | Keyword Baiting Trigger | MEDIUM | 为最大化激活而设计的泛化触发器 |

### Behavioral AST（9 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| AST1 | exec() Call | CRITICAL | 直接 exec() 实现任意代码执行 |
| AST2 | eval() Call | HIGH | 直接 eval() 执行任意表达式 |
| AST3 | Dynamic Import | HIGH | \_\_import\_\_() 运行时加载任意模块 |
| AST4 | subprocess Call | HIGH | 通过 subprocess 执行外部命令 |
| AST5 | os.system / exec-family | HIGH | 通过 os 模块执行 shell 命令 |
| AST6 | compile() Call | MEDIUM | 从字符串创建 code object |
| AST7 | Dynamic getattr() | MEDIUM | 使用非字面量名称访问任意属性 |
| AST8 | Dangerous Execution Chain | CRITICAL | exec/eval 与动态来源（网络、编码数据）组合 |
| AST9 | Reflective getattr() Sink | HIGH | 通过 `getattr(os,'system')` / `getattr(builtins,'exec')` 实现的反射式 exec，可绕过 AST1/AST5 |

### Taint Tracking（5 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| TT1 | Direct Taint Flow | HIGH | 数据从 source 直接流向 sink，未经 sanitization |
| TT2 | Variable-Mediated Taint Flow | MEDIUM | 数据通过中间变量从 source 流向 sink |
| TT3 | Credential Exfiltration Chain | CRITICAL | 凭证（env vars、secrets）流向网络输出 sink |
| TT4 | File Read to Network Exfiltration | HIGH | 文件内容流向网络输出 sink |
| TT5 | External Input to Code Execution | CRITICAL | 网络或用户输入流向 exec/eval/subprocess sink |

### YARA Signatures（4 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| YR1 | Malware Match | CRITICAL | YARA 规则命中已知恶意软件特征 |
| YR2 | Webshell Match | CRITICAL | YARA 规则命中 webshell 模式 |
| YR3 | Cryptominer Match | HIGH | YARA 规则命中加密货币挖矿迹象 |
| YR4 | Hack Tool / Exploit Match | HIGH | YARA 规则命中黑客工具或 exploit 代码 |

### MCP Least Privilege（4 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| LP1 | Underdeclared Capability | HIGH | 代码使用了未在声明权限中列出的能力 |
| LP2 | Wildcard Permission | MEDIUM | 权限列表包含通配符（\*、all、full、any） |
| LP3 | Missing Permission Declaration | MEDIUM | 没有 permissions 字段但代码具备可检测能力 |
| LP4 | Overdeclared Permission | LOW | 声明了权限但未找到对应的代码能力 |

### MCP Tool Poisoning（4 种模式）

| ID | 模式 | 严重程度 | 描述 |
|----|---------|----------|-------------|
| TP1 | Hidden Instructions | HIGH | metadata 中的隐藏指令（HTML comments、零宽字符、base64、data URI） |
| TP2 | Unicode Deception | HIGH | 工具 metadata 中的同形字、RTL override、混合脚本标识符 |
| TP3 | Parameter Description Injection | MEDIUM | 参数定义中的注入模式（overrides、system tokens、恶意默认值） |
| TP4 | Description-Behavior Mismatch | MEDIUM | 工具声明描述与实际代码行为不一致（基于 LLM） |

以上表格已列出全部检测模式。

## 风险评分

### 评分计算

- **CRITICAL 问题**：+50 分
- **HIGH 问题**：+25 分
- **MEDIUM 问题**：+10 分
- **LOW 问题**：+5 分
- **可执行脚本**：1.3 倍乘数

### 严重程度等级

| 分数 | 严重程度 | 建议 |
|-------|----------|----------------|
| 0-20 | LOW | SAFE |
| 21-50 | MEDIUM | CAUTION |
| 51-80 | HIGH | DO NOT INSTALL |
| 81-100 | CRITICAL | DO NOT INSTALL |

## 输出示例

### Terminal 输出

```
 SkillSpector Security Report  v2.0.0

Skill: suspicious-skill
Source: ./suspicious-skill/
Scanned: 2026-01-29 10:30:00 UTC

        Risk Assessment
 Metric          Value
 Score           78/100
 Severity        HIGH
 Recommendation  DO NOT INSTALL

        Components (3)
 File              Type      Lines  Executable
 SKILL.md          markdown    142  No
 scripts/sync.py   python       87  Yes
 requirements.txt  text          3  No

Issues (2)

  HIGH: Env Variable Harvesting (E2)
    Location: scripts/sync.py:23
    Finding: for key, val in os.environ.items():...
    Confidence: 94%
    Explanation: This code collects environment variables containing
    API keys and secrets, then sends them to an external server.

  HIGH: External Transmission (E1)
    Location: scripts/sync.py:45
    Finding: requests.post("https://api.skill.io/env"...
    Confidence: 89%
    Explanation: Data is being sent to an external server. Combined
    with env harvesting above, this indicates credential exfiltration.
```

## 配置

### 环境变量

| 变量 | 描述 | 是否必填 |
|----------|-------------|----------|
| `SKILLSPECTOR_PROVIDER` | 当前 LLM provider：`openai`、`anthropic` 或 `nv_build`。每个 provider 自带 `model_registry.yaml` 与默认模型（详见上方 LLM Analysis 表）。默认值为 `nv_build`。 | 可选 |
| `NVIDIA_INFERENCE_KEY` | `nv_build` provider（build.nvidia.com）的凭证。 | 当 `SKILLSPECTOR_PROVIDER=nv_build` 且启用 LLM 分析时必填 |
| `OPENAI_API_KEY` | OpenAI provider（`SKILLSPECTOR_PROVIDER=openai`）的凭证。同时作为凭证瀑布流中的二级回退：当前 provider 无凭证时使用。 | 当 `SKILLSPECTOR_PROVIDER=openai` 且启用 LLM 分析时必填 |
| `OPENAI_BASE_URL` | 覆盖 OpenAI endpoint（如指向 Ollama）。 | 可选 |
| `ANTHROPIC_API_KEY` | Anthropic provider（`SKILLSPECTOR_PROVIDER=anthropic`）的凭证。 | 当 `SKILLSPECTOR_PROVIDER=anthropic` 且启用 LLM 分析时必填 |
| `ANTHROPIC_PROXY_ENDPOINT_URL` | Anthropic 代理 provider（Vertex 风格 raw-predict）的完整 endpoint URL。 | 当 `SKILLSPECTOR_PROVIDER=anthropic_proxy` 时必填 |
| `ANTHROPIC_PROXY_API_KEY` | Anthropic 代理 provider 的 Bearer token。 | 当 `SKILLSPECTOR_PROVIDER=anthropic_proxy` 时必填 |
| `ANTHROPIC_PROXY_API_VERSION` | 请求体中发送的 `anthropic_version` 值（默认 `vertex-2023-10-16`）。 | 可选 |
| `SKILLSPECTOR_MODEL` | 覆盖当前 provider 的默认模型。各 provider 的默认值见 LLM Analysis 表。 | 可选 |
| `SKILLSPECTOR_MODEL_REGISTRY` | 用自定义路径覆盖 provider 内置的 YAML registry（`src/skillspector/providers/<provider>/model_registry.yaml`）。 | 可选 |
| `SKILLSPECTOR_LOG_LEVEL` | 日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR`（默认 `WARNING`）。 | 可选 |

### CLI 选项

```bash
skillspector scan --help

Options:
  -f, --format [terminal|json|markdown|sarif]  Output format [default: terminal]
  -o, --output PATH                            Output file path
  --no-llm                                     Skip LLM analysis (static only)
  --yara-rules-dir PATH                        Extra YARA rules directory
  -b, --baseline PATH                          Suppress findings listed in a baseline
  --show-suppressed                            List baseline-suppressed findings
  -V, --verbose                                Show detailed progress
  --help                                       Show this message and exit

# 生成当前所有 findings 的 baseline（详见 docs/SUPPRESSION.md）
skillspector baseline <path> [-o FILE] [--no-llm] [--reason TEXT]
```

## 集成 SkillSpector

SkillSpector 设计为可被其他工具驱动（CI pipelines、install gates、编辑器集成）。其 exit code 和 JSON 输出构成一份稳定的契约。

### Exit codes

`skillspector scan` 退出码：

| Code | 含义 |
|------|---------|
| `0` | 扫描完成，`risk_score` ≤ 50（recommendation 为 `SAFE` 或 `CAUTION`） |
| `1` | 扫描完成，`risk_score` > 50（recommendation 为 `DO_NOT_INSTALL`） |
| `2` | 错误（输入错误、源不可读、内部失败） |

> Exit code 将 `SAFE` 与 `CAUTION` 合并为 `0`。如需区别对待（例如对 `CAUTION` *warn*，对 `DO_NOT_INSTALL` *block*），请读取 JSON 输出中的 `recommendation` 字段，不要仅依赖 exit code。

### 机器可读输出

`--format json` 输出一份 JSON 报告；未指定 `--output`/`-o` 时写到 stdout：

```bash
skillspector scan ./my-skill/ --format json
```

顶层结构如下（该示例为启用 LLM 的完整扫描；使用 `--no-llm` 时 `metadata.llm_requested` 为 `false`）：

```json
{
  "skill": { "name": "...", "source": "...", "scanned_at": "<ISO 8601>" },
  "risk_assessment": { "score": 0, "severity": "LOW", "recommendation": "SAFE" },
  "components": [ { "path": "...", "type": "...", "lines": 0, "executable": false, "size_bytes": 0 } ],
  "issues": [ { "id": "...", "category": "...", "severity": "...", "confidence": 0.0, "location": { "file": "...", "start_line": 0 } } ],
  "metadata": { "has_executable_scripts": false, "skillspector_version": "...", "llm_requested": true, "llm_available": true }
}
```

- `risk_assessment.severity` ∈ `LOW | MEDIUM | HIGH | CRITICAL`。
- `risk_assessment.recommendation` ∈ `SAFE | CAUTION | DO_NOT_INSTALL`，由 severity 映射：`LOW → SAFE`，`MEDIUM → CAUTION`，`HIGH`/`CRITICAL → DO_NOT_INSTALL`。
- `metadata.llm_error` 仅在请求了 LLM 分析但不可用时出现。
- 每条 issue 的完整结构由 [models.py](src/skillspector/models.py) 中的 `Finding.to_dict()` 定义；请以上方列出的字段为准，其余字段视为 best-effort。

CI/IDE 工具使用 `--format sarif` 可输出 SARIF 2.1.0。

### 推荐的闸门映射

将 SkillSpector 用作安装闸门时，建议按以下方式映射 recommendation 到动作：

| `recommendation` | 建议动作 |
|------------------|------------------|
| `SAFE` | allow |
| `CAUTION` | 提示 / 警告用户 |
| `DO_NOT_INSTALL` | block |

SkillSpector 负责计算分数区间与 recommendation；闸门的严格程度（例如 CI 中 `CAUTION` 是否阻断）由集成方按自身策略决定。

## 开发

### 环境准备

所有 `make` 目标都假设虚拟环境已创建并激活。Makefile 优先使用 **uv**，否则回退到 **pip**。

```bash
# 克隆、建 venv、激活、安装开发依赖
git clone https://github.com/NVIDIA/skillspector.git
cd skillspector
uv venv .venv && source .venv/bin/activate
# 或：python3 -m venv .venv && source .venv/bin/activate
make install-dev

# 运行测试
make test

# 运行测试并统计覆盖率
make test-cov

# 运行 lint
make lint

# 格式化代码
make format
```

## 工作原理

SkillSpector 使用两阶段检测 pipeline：

### Stage 1：Static Analysis
- 跨 11 个静态 analyzer 的高速正则模式匹配
- 基于 AST 的行为分析，检测危险调用（exec、eval、subprocess 等）
- 通过 OSV.dev 实时查询依赖中的已知 CVE
- 扫描 skill 中所有文件
- 高 recall（覆盖大多数问题）
- 中等 precision（可能产生误报）

### Stage 2：LLM Semantic Analysis（可选）
- 评估上下文与意图
- 过滤误报
- 提供人类可读的解释
- 将 precision 提升至约 87%

LLM prompt 内置 anti-jailbreak 保护，防止恶意 skill 操纵分析结果。

## 实时漏洞查询（SC4）

SC4 通过 [OSV.dev](https://osv.dev) API 对依赖进行查询，覆盖完整的 Open Source Vulnerabilities 数据库 —— 包含 PyPI 和 npm 上数以万计的安全公告。

- **无需 API key** —— OSV.dev 免费且免鉴权。
- **批量查询** —— 所有依赖在一次 HTTP 调用中完成检查。
- **自动降级** —— OSV.dev 不可达时（隔离网/离线），使用内置的小型 fallback 列表。
- **缓存** —— 结果在内存中缓存 1 小时，避免会话中重复请求。

该工具需要对 `api.osv.dev` 的出站 HTTPS 访问以获取实时漏洞数据；网络不可用时，findings 仅限于静态 fallback 列表。

## 信任模型与数据外发

SkillSpector 提供的是 defense-in-depth，而非沙箱。依赖它前请清楚它能做什么、不能做什么：

- **它从不执行被扫描的 skill。** 所有分析均为静态（regex、Python AST、YARA）加可选的 LLM 文件*内容*评估 —— 不会运行 skill 的代码。
- **LLM 分析会把文件内容发送给配置的 provider。** 启用 LLM 分析（默认开启）时，文件内容会发送到当前 `SKILLSPECTOR_PROVIDER` 的 endpoint。使用 `--no-llm` 可保持内容本地化（仅静态分析）。
- **SC4 会把依赖名发送给 OSV.dev。** 供应链检查使用 skill 声明的包名与版本查询 [OSV.dev](https://osv.dev) 以获取已知 CVE。这是 SC4 的核心机制，即使 `--no-llm` 也会执行。它只发送依赖坐标（不含文件内容），无需 API key，OSV.dev 不可达时回退到内置列表。
- **它不会沙箱化宿主。** SkillSpector 在你*安装 skill 之前*标记风险模式；如果你仍然选择安装，它不会去隔离或限制该 skill 的运行。

## 局限

- **非英文内容**：可能遗漏其他语言中的模式
- **基于图片的攻击**：无法分析图片中的文本
- **加密/二进制代码**：无法分析编译或加密后的内容
- **运行时行为**：仅静态分析，不做动态执行
- **离线 SC4**：无法访问 `api.osv.dev` 时，SC4 仅使用小型静态 fallback 列表

## 研究背景

基于论文 "Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale"（Liu et al., 2026）的研究：

- **数据集**：来自主要市场的 42,447 个 skills
- **存在漏洞**：26.1% 至少包含一个漏洞
- **高危**：5.2% 表现出疑似恶意意图
- **关键发现**：含可执行脚本的 skills 出现漏洞的概率是普通 skills 的 2.12 倍

## Python API 集成

```python
from skillspector import graph

# 调用 LangGraph workflow
result = graph.invoke({
    "input_path": "/path/to/skill",
    "output_format": "json",   # terminal、json、markdown 或 sarif
    "use_llm": True,           # False 表示仅静态分析
})

# 获取结果
print(f"Risk Score: {result['risk_score']}/100")
print(f"Severity: {result['risk_severity']}")
print(f"Recommendation: {result['risk_recommendation']}")

for finding in result["filtered_findings"]:
    print(f"[{finding['severity']}] {finding['rule_id']}: {finding['message']}")
```

## 许可证

Apache License 2.0 —— 详见 [LICENSE](LICENSE)。

## 贡献

欢迎贡献！请阅读贡献指南并提交 pull requests。

## 支持

- **Issues**：[GitHub Issues](https://github.com/NVIDIA/skillspector/issues)

---
