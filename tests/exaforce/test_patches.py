def test_apply_patches_prunes_model_schemas(run_in_subprocess):
    out = run_in_subprocess(
        """
        import skillspector
        from skillspector.exaforce import apply_patches
        apply_patches()
        from skillspector.llm_analyzer_base import LLMFinding, LLMAnalysisResult
        from skillspector.nodes.meta_analyzer import (
            MetaAnalyzerFinding,
            MetaAnalyzerResult,
        )
        lf = set(LLMFinding.model_json_schema()["properties"])
        assert "explanation" not in lf and "remediation" not in lf, lf
        mf = set(MetaAnalyzerFinding.model_json_schema()["properties"])
        assert "intent" not in mf and "impact" not in mf, mf
        mr = set(MetaAnalyzerResult.model_json_schema()["properties"])
        assert "overall_assessment" not in mr, mr
        # Container schema (what the LLM actually receives) is pruned too:
        nested = LLMAnalysisResult.model_json_schema()["$defs"]["LLMFinding"]["properties"]
        assert "explanation" not in nested, nested
        print("OK")
        """
    )
    assert "OK" in out


def test_apply_patches_prunes_to_finding_and_dump(run_in_subprocess):
    out = run_in_subprocess(
        """
        import skillspector
        from skillspector.exaforce import apply_patches
        apply_patches()
        from skillspector.llm_analyzer_base import LLMFinding
        f = LLMFinding(rule_id="R", message="m", severity="LOW", start_line=3)
        assert "explanation" not in f.model_dump()
        assert "remediation" not in f.model_dump()
        fin = f.to_finding("x.py")
        assert fin.explanation is None
        assert fin.rule_id == "R" and fin.start_line == 3
        print("OK")
        """
    )
    assert "OK" in out


def test_apply_patches_trims_prompts(run_in_subprocess):
    out = run_in_subprocess(
        """
        import skillspector
        from skillspector.exaforce import apply_patches
        apply_patches()
        from skillspector.nodes.analyzers import semantic_developer_intent as d
        from skillspector.nodes.analyzers import semantic_quality_policy as q
        from skillspector.nodes import meta_analyzer as m
        assert "Reference the L-prefixed line numbers" not in d.ANALYZER_PROMPT
        assert "Reference the L-prefixed line numbers" not in q.ANALYZER_PROMPT
        assert "What is the likely intent" not in m.PER_FILE_ANALYSIS_PROMPT
        assert "What is the potential impact" not in m.PER_FILE_ANALYSIS_PROMPT
        assert "Use the rule IDs exactly as listed." in d.ANALYZER_PROMPT
        print("OK")
        """
    )
    assert "OK" in out


def test_apply_patches_is_idempotent(run_in_subprocess):
    out = run_in_subprocess(
        """
        import skillspector
        from skillspector.exaforce import apply_patches
        apply_patches()
        apply_patches()
        apply_patches()
        print("OK")
        """
    )
    assert "OK" in out
