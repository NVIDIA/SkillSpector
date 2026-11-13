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

"""Report-only acquisition metadata, independent of the materialized scan root."""

from urllib.parse import urlsplit, urlunsplit


def input_origin(source: str, source_type: str) -> dict[str, str]:
    """Preserve local names; omit credentials, queries and fragments from URLs.

    The origin describes the requested input, not a content digest or resolved
    Git revision. Keep it separate from analyzer identities and finding paths.
    """
    if source_type in {"git", "url"}:
        source = source.strip()
        if source_type == "git" and source.startswith("git@"):
            # Git's scp-style spelling is not parsed as an ordinary URL.
            source = "ssh://" + source.split("@", 1)[1].replace(":", "/", 1)
            return input_origin(source, source_type)
        else:
            try:
                parsed = urlsplit(source)
                if not parsed.hostname or parsed.scheme not in {"https", "http", "ssh", "git"}:
                    raise ValueError("unsupported origin URL")
                host = parsed.hostname
                if ":" in host:
                    host = f"[{host}]"
                if parsed.port is not None:
                    host += f":{parsed.port}"
                source = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
            except ValueError:
                source = "<redacted-url>"
    return {"type": source_type, "source": source}
