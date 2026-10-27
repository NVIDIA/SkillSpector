from pathlib import Path

p = Path("tests/nodes/test_security_end_to_end.py")
t = p.read_text()
t = t.replace('async @pytest.mark.timeout(300)\n@mock.patch("skillspector.nodes.analyzers.static_runner.MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT", 300.0)\ndef test_rd04_large_file_pair_detects_start_boundary_and_end', '@pytest.mark.timeout(300)\n@mock.patch("skillspector.nodes.analyzers.static_runner.MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT", 300.0)\nasync def test_rd04_large_file_pair_detects_start_boundary_and_end')
t = t.replace('async @pytest.mark.timeout(300)\n@mock.patch("skillspector.nodes.analyzers.static_runner.MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT", 300.0)\ndef test_nine_case_contract_across_public_surfaces', '@pytest.mark.timeout(300)\n@mock.patch("skillspector.nodes.analyzers.static_runner.MAX_STATIC_ANALYSIS_SECONDS_PER_ARTIFACT", 300.0)\nasync def test_nine_case_contract_across_public_surfaces')
p.write_text(t)
