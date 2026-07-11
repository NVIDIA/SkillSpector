def test_bare_import_activates_patches(run_in_subprocess):
    # No explicit apply_patches() call — importing skillspector must patch.
    out = run_in_subprocess(
        """
        import skillspector
        from skillspector.llm_analyzer_base import LLMFinding
        from skillspector.nodes.meta_analyzer import (
            MetaAnalyzerFinding,
            MetaAnalyzerResult,
        )
        from skillspector.nodes import meta_analyzer as m
        assert "explanation" not in LLMFinding.model_json_schema()["properties"]
        assert "intent" not in MetaAnalyzerFinding.model_json_schema()["properties"]
        assert "overall_assessment" not in MetaAnalyzerResult.model_json_schema()["properties"]
        assert "What is the likely intent" not in m.PER_FILE_ANALYSIS_PROMPT
        print("OK")
        """
    )
    assert "OK" in out
