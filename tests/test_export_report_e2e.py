import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from playwright.sync_api import sync_playwright


REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_ok(url: str, timeout_s: float = 30) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            res = requests.get(url, timeout=1)
            if res.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Server not ready: {url}")


@pytest.fixture(scope="session")
def analysis_full() -> dict:
    path = REPO_ROOT / "tests" / "fixtures" / "analysis_full.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def analysis_empty() -> dict:
    path = REPO_ROOT / "tests" / "fixtures" / "analysis_empty.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def server_url(tmp_path_factory) -> str:
    novels_root = tmp_path_factory.mktemp("novels")
    folder = novels_root / "sample-folder"
    folder.mkdir(parents=True, exist_ok=True)

    content = "第1章\n" + ("测试内容" * 2500)
    (folder / "sample.txt").write_text(content, encoding="utf-8")

    port = _find_free_port()
    env = os.environ.copy()
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "NOVEL_PATH": str(novels_root),
            "LOG_LEVEL": "error",
        }
    )

    proc = subprocess.Popen(
        [sys.executable, "backend.py"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_http_ok(f"{base_url}/api/config", timeout_s=30)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _select_first_novel(page) -> None:
    page.locator("[data-testid='novel-dropdown']").click()
    page.locator(".file-item").first.wait_for(state="visible", timeout=10_000)
    page.locator(".file-item").first.click()


def _stub_analyze(page, analysis: dict) -> None:
    body = json.dumps({"analysis": analysis}, ensure_ascii=False)

    def handler(route, request):
        if request.method != "POST":
            return route.fallback()
        return route.fulfill(status=200, headers={"Content-Type": "application/json"}, body=body)

    page.route("**/api/analyze", handler)


def _run_analysis(page) -> None:
    page.locator("[data-testid='analyze-button']").click()
    page.locator("#toastContainer div:has-text('分析完成')").wait_for(state="visible", timeout=30_000)


def _capture_modules_inner_html(page) -> dict:
    return page.evaluate(
        """() => ({
        quickStats: document.getElementById('quickStats')?.innerHTML ?? '',
        relationshipSummary: document.getElementById('relationshipSummary')?.innerHTML ?? '',
        relationshipDetails: document.getElementById('relationshipDetails')?.innerHTML ?? '',
        mainCharacters: document.getElementById('mainCharacters')?.innerHTML ?? '',
        firstSexScene: document.getElementById('firstSexScene')?.innerHTML ?? '',
        sexSceneCount: document.getElementById('sexSceneCount')?.innerHTML ?? '',
        relationshipProgress: document.getElementById('relationshipProgress')?.innerHTML ?? ''
    })"""
    )


def _assert_export_cdn_links(export_html_text: str) -> None:
    assert "https://cdn.jsdelivr.net/npm/daisyui@4/dist/full.min.css" in export_html_text
    assert "https://cdn.tailwindcss.com" in export_html_text
    assert "https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js" in export_html_text


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_export_report_matches_web_ui(server_url, analysis_full, tmp_path, theme: str):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(accept_downloads=True)

        page = context.new_page()
        _stub_analyze(page, analysis_full)

        page.goto(server_url, wait_until="domcontentloaded")

        if theme == "light":
            page.locator("button[title='切换主题']").click()
            page.wait_for_function("() => document.documentElement.getAttribute('data-theme') === 'light'")

        _select_first_novel(page)
        _run_analysis(page)

        page.locator("#relationshipSummary .summary-section").wait_for(state="attached", timeout=10_000)
        web_modules = _capture_modules_inner_html(page)

        with page.expect_download() as download_info:
            page.locator("text=导出").click()

        download = download_info.value
        export_path = tmp_path / download.suggested_filename
        download.save_as(str(export_path))

        export_html = export_path.read_text(encoding="utf-8")
        _assert_export_cdn_links(export_html)

        export_page = context.new_page()
        export_page.goto(export_path.as_uri(), wait_until="domcontentloaded")
        export_page.locator("#quickStats").wait_for(state="visible", timeout=30_000)

        export_modules = _capture_modules_inner_html(export_page)
        assert export_modules == web_modules

        tab_expectations = [
            ("总结", "#quickStats"),
            ("角色", "#mainCharacters"),
            ("关系图", "#relationshipChart"),
            ("首次", "#firstSexScene"),
            ("统计", "#sexSceneCount"),
            ("发展", "#relationshipProgress"),
        ]
        for tab_name, selector in tab_expectations:
            export_page.locator(f"button:has-text('{tab_name}')").click()
            export_page.locator(selector).wait_for(state="visible", timeout=10_000)

        export_page.locator("button:has-text('关系图')").click()
        svg = export_page.locator("#relationshipChart svg")
        svg.wait_for(state="attached", timeout=10_000)
        assert svg.get_attribute("viewBox") == "0 0 1200 800"
        assert svg.get_attribute("width") == "1200"
        assert svg.get_attribute("height") == "800"

        rect_fill = export_page.locator("#relationshipChart svg rect").first.get_attribute("fill")
        if theme == "dark":
            assert rect_fill == "#1c1c1e"
        else:
            assert rect_fill == "#f2f2f7"

        context.close()
        browser.close()


def test_empty_state_is_safe_to_export(server_url, analysis_empty, tmp_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(accept_downloads=True)

        page = context.new_page()
        _stub_analyze(page, analysis_empty)
        page.goto(server_url, wait_until="domcontentloaded")

        _select_first_novel(page)
        _run_analysis(page)

        with page.expect_download() as download_info:
            page.locator("text=导出").click()

        download = download_info.value
        export_path = tmp_path / download.suggested_filename
        download.save_as(str(export_path))

        export_page = context.new_page()
        export_page.goto(export_path.as_uri(), wait_until="domcontentloaded")
        export_page.locator("button:has-text('角色')").click()
        export_page.locator("#mainCharacters .empty-state").first.wait_for(state="visible", timeout=10_000)

        context.close()
        browser.close()


def test_lewdness_color_mapping_and_filename_sanitize():
    js_text = (REPO_ROOT / "static" / "chart-view.js").read_text(encoding="utf-8")
    assert "function getLewdnessColor" in js_text
    assert "function sanitizeFilename" in js_text
    assert "a.download = sanitizeFilename(novelName)" in js_text

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("about:blank")
        page.add_script_tag(path=str(REPO_ROOT / "static" / "chart-view.js"))

        assert page.evaluate("() => getLewdnessColor(95)") == "#ef4444"
        assert page.evaluate("() => getLewdnessColor(75)") == "#f97316"
        assert page.evaluate("() => getLewdnessColor(55)") == "#eab308"
        assert page.evaluate("() => getLewdnessColor(35)") == "#22c55e"
        assert page.evaluate("() => getLewdnessColor(10)") == "#6366f1"

        assert page.evaluate("() => sanitizeFilename('a:b?c.txt')") == "a_b_c.txt"

        browser.close()
