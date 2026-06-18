#!/bin/sh
set -eu

IMAGE="${SKILLSPECTOR_DOCKER_IMAGE:-skillspector}"
REPO_DIR="${SKILLSPECTOR_REPO_DIR:-$(pwd)}"
LOCAL_REPORT="${SKILLSPECTOR_DOCKER_LOCAL_REPORT:-.skillspector-docker-smoke.json}"
GITHUB_REPORT="${SKILLSPECTOR_DOCKER_GITHUB_REPORT:-.skillspector-docker-github-smoke.json}"
GITHUB_URL="${SKILLSPECTOR_DOCKER_GITHUB_URL:-https://github.com/octocat/Hello-World}"
GITHUB_EXPECTED_COMPONENT="${SKILLSPECTOR_DOCKER_GITHUB_EXPECTED_COMPONENT:-README}"

run() {
  printf "\n>> %s\n" "$*"
  "$@"
  return
}

validate_json_report() {
  report_path="$1"

  test -s "${REPO_DIR}/${report_path}"
  run docker run --rm --entrypoint python -v "${REPO_DIR}:/scan" "${IMAGE}" \
    -m json.tool "/scan/${report_path}" >/dev/null
  return
}

assert_report_contains_component() {
  report_path="$1"
  expected_component="$2"

  run docker run --rm --entrypoint python -v "${REPO_DIR}:/scan" "${IMAGE}" \
    -c 'import json, sys; data = json.load(open("/scan/" + sys.argv[1])); expected = sys.argv[2]; assert any(c.get("path") == expected for c in data.get("components", [])), f"missing component: {expected}"' \
    "${report_path}" "${expected_component}"
  return
}

scan_github_url() {
  printf "\n>> docker run --rm -v %s:/scan %s scan %s --no-llm --format json --output /scan/%s\n" \
    "${REPO_DIR}" "${IMAGE}" "${GITHUB_URL}" "${GITHUB_REPORT}"

  set +e
  docker run --rm -v "${REPO_DIR}:/scan" "${IMAGE}" scan "${GITHUB_URL}" \
    --no-llm --format json --output "/scan/${GITHUB_REPORT}"
  github_scan_status="$?"
  set -e

  if [ "${github_scan_status}" -ne 0 ] && [ "${github_scan_status}" -ne 1 ]; then
    echo "GitHub URL scan failed with exit code ${github_scan_status}"
    exit "${github_scan_status}"
  fi

  validate_json_report "${GITHUB_REPORT}"
  assert_report_contains_component "${GITHUB_REPORT}" "${GITHUB_EXPECTED_COMPONENT}"
  echo "GitHub URL scan completed with accepted exit code ${github_scan_status}"
  return
}

run docker run --rm "${IMAGE}" --version
run docker run --rm --entrypoint git "${IMAGE}" --version

run docker run --rm -v "${REPO_DIR}:/scan" "${IMAGE}" scan tests/fixtures/safe_skill \
  --no-llm --format json --output "/scan/${LOCAL_REPORT}"
validate_json_report "${LOCAL_REPORT}"

scan_github_url
