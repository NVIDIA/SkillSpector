"""
Main scanner orchestration for SkillSpector.

Coordinates the scanning pipeline:
1. Input handling (URL, zip, file, directory)
2. Component inventory
3. Static analysis
4. LLM analysis (optional)
5. Risk scoring
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from skillspector.input_handler import InputHandler
from skillspector.inventory import InventoryBuilder
from skillspector.static_analyzer import StaticAnalyzer
from skillspector.llm_analyzer import LLMAnalyzer
from skillspector.models import (
    ScanResult,
    RiskAssessment,
    SecurityIssue,
    SkillMetadata,
    Severity,
    StaticFinding,
)


class SkillScanner:
    """
    Main scanner class that orchestrates the full analysis pipeline.

    Usage:
        scanner = SkillScanner()
        result = scanner.scan("./my-skill/")
        print(result.risk_assessment.score)
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_provider: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the scanner.

        Args:
            use_llm: Whether to use LLM for semantic analysis (requires API key)
            llm_provider: LLM provider to use ("anthropic" or "google"). Auto-detected if not specified.
            verbose: Whether to print detailed progress information
        """
        self.use_llm = use_llm
        self.verbose = verbose

        self.input_handler = InputHandler()
        self.inventory_builder = InventoryBuilder()
        self.static_analyzer = StaticAnalyzer()

        if use_llm:
            self.llm_analyzer: Optional[LLMAnalyzer] = LLMAnalyzer(provider=llm_provider)
        else:
            self.llm_analyzer = None

    def scan(self, input_path: str) -> ScanResult:
        """
        Scan a skill for security vulnerabilities.

        Args:
            input_path: Path or URL to scan (supports git URL, file URL, zip, .md, directory)

        Returns:
            ScanResult with all findings and risk assessment
        """
        start_time = time.time()

        # Step 1: Handle input and get normalized file tree
        self._log("Resolving input...")
        skill_dir, source_type = self.input_handler.resolve(input_path)

        # Step 2: Build component inventory
        self._log("Building component inventory...")
        components = self.inventory_builder.build(skill_dir)

        # Step 3: Extract metadata from SKILL.md
        self._log("Extracting metadata...")
        metadata = self.inventory_builder.extract_metadata(skill_dir)

        # Step 4: Run static analysis
        self._log("Running static analysis...")
        static_findings = self.static_analyzer.analyze(skill_dir, components)

        # Step 5: Run LLM analysis (if enabled)
        issues: list[SecurityIssue] = []
        if self.use_llm and self.llm_analyzer and static_findings:
            self._log("Running LLM analysis...")
            issues = self.llm_analyzer.analyze(
                skill_dir=skill_dir,
                static_findings=static_findings,
                metadata=metadata,
            )
        else:
            # Convert static findings directly to issues without LLM
            issues = self._static_to_issues(static_findings)

        # Filter low-confidence findings
        issues = [i for i in issues if i.confidence >= 0.6]

        # Step 6: Calculate risk score
        has_scripts = any(c.executable for c in components)
        risk_score = self._calculate_risk_score(issues, has_scripts)
        risk_assessment = RiskAssessment.from_score(risk_score)

        # Build result
        duration_ms = int((time.time() - start_time) * 1000)

        # Determine skill name
        skill_name = metadata.name or self._infer_skill_name(input_path, skill_dir)

        # Cleanup temporary directory if needed
        self.input_handler.cleanup()

        return ScanResult(
            skill_name=skill_name,
            source=input_path,
            scanned_at=datetime.now(timezone.utc),
            metadata=metadata,
            components=components,
            issues=issues,
            risk_assessment=risk_assessment,
            has_executable_scripts=has_scripts,
            scan_duration_ms=duration_ms,
            llm_used=self.use_llm and self.llm_analyzer is not None,
        )

    def _static_to_issues(self, findings: list[StaticFinding]) -> list[SecurityIssue]:
        """Convert static findings to security issues without LLM analysis."""
        issues = []
        for f in findings:
            issue = SecurityIssue(
                id=f.pattern_id,
                category=f.category,
                pattern=f.pattern_name,
                severity=f.severity,
                location=f.location,
                finding=f.matched_text,
                explanation=self._default_explanation(f.pattern_id),
                confidence=f.confidence,
                code_snippet=f.context[:500] if f.context else None,
            )
            issues.append(issue)
        return issues

    def _default_explanation(self, pattern_id: str) -> str:
        """Provide default explanations for patterns when LLM is not used."""
        explanations = {
            "P1": "This pattern attempts to override system instructions or ignore safety constraints. Without LLM analysis, manual review is recommended.",
            "P2": "Hidden instructions were detected in comments or invisible text. These could contain malicious directives. Manual review is recommended.",
            "P3": "Instructions found that direct the agent to transmit conversation context or user data to external services.",
            "P4": "Subtle instructions detected that may alter agent decision-making or introduce hidden biases.",
            "P5": "This content may contain harmful instructions that could cause physical harm if followed. CRITICAL: Review carefully before use.",
            "E1": "Data is being sent to an external URL. This could be legitimate telemetry or data exfiltration. Manual review is recommended.",
            "E2": "Code accesses environment variables that may contain secrets (API keys, tokens). This is a common pattern for credential theft.",
            "E3": "Code scans file system directories looking for sensitive files. This could be reconnaissance for credential theft.",
            "E4": "Code or instructions that leak agent conversation context to external services, potentially exposing sensitive user interactions.",
            "PE1": "Skill requests more permissions than appear necessary for its stated functionality. Review if elevated access is justified.",
            "PE2": "Commands invoke sudo or root privileges. Verify this elevated access is necessary and justified.",
            "PE3": "Code accesses credential files (SSH keys, AWS credentials, etc.). This could indicate credential theft attempts.",
            "SC1": "Dependencies lack version pinning, allowing potential malicious package updates. Consider pinning versions.",
            "SC2": "Remote code is downloaded and executed. This bypasses code review and could introduce malicious code.",
            "SC3": "Code contains obfuscation (base64, hex encoding with execution). This is often used to hide malicious functionality.",
        }
        return explanations.get(pattern_id, "Potential security issue detected. Manual review is recommended.")

    def _calculate_risk_score(self, issues: list[SecurityIssue], has_scripts: bool) -> int:
        """
        Calculate risk score (0-100) based on issues found.

        Scoring:
        - CRITICAL: +50 points
        - HIGH: +25 points
        - MEDIUM: +10 points
        - LOW: +5 points
        - Executable scripts: 1.3x multiplier
        """
        score = 0

        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                score += 50
            elif issue.severity == Severity.HIGH:
                score += 25
            elif issue.severity == Severity.MEDIUM:
                score += 10
            elif issue.severity == Severity.LOW:
                score += 5

        # Apply multiplier for executable scripts
        if has_scripts:
            score = int(score * 1.3)

        return min(100, score)

    def _infer_skill_name(self, input_path: str, skill_dir: Path) -> str:
        """Infer skill name from input path or directory name."""
        if input_path.startswith("http"):
            # Extract from URL
            parts = input_path.rstrip("/").split("/")
            return parts[-1].replace(".git", "").replace(".zip", "")
        elif input_path.endswith(".zip"):
            return Path(input_path).stem
        elif input_path.endswith(".md"):
            return Path(input_path).stem
        else:
            return skill_dir.name

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[SkillSpector] {message}")
