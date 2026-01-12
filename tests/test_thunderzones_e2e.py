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
def analysis_empty() -> dict:
    path = REPO_ROOT / "tests" / "fixtures" / "analysis_empty.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def analysis_with_thunderzones(analysis_empty: dict) -> dict:
    analysis = json.loads(json.dumps(analysis_empty, ensure_ascii=False))
    analysis["thunderzones"] = [
        {
            "type": "NTR",
            "severity": "高",
            "description": "主角恋人被夺走",
            "involved_characters": ["主角", "恋人", "反派"],
            "chapter_location": "第15章",
            "relationship_context": "恋爱期间遭遇背叛",
        }
    ]
    analysis["thunderzone_summary"] = "包含高危NTR情节"
    analysis["lewd_elements"] = [
        {
            "type": "恋足",
            "example": "出现足交描写",
            "involved_characters": ["主角"],
            "chapter_location": "第3章",
        }
    ]
    analysis["lewd_elements_summary"] = "包含恋足元素"
    return analysis


@pytest.fixture(scope="session")
def sample_novel_path(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("novels")
    path = root / "sample.txt"
    content = "第1章\n" + ("测试内容" * 2500)
    path.write_text(content, encoding="utf-8")
    return path



@pytest.fixture(scope="session")
def server_url(tmp_path_factory) -> str:
    port = _find_free_port()
    env = os.environ.copy()
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
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


def _upload_novel(page, novel_path: Path) -> None:
    page.locator("[data-testid='novel-file-input']").set_input_files(str(novel_path))
    page.wait_for_function(
        """() => {
        const btn = document.querySelector("[data-testid='analyze-button']");
        return btn && !btn.disabled;
    }"""
    )


def _stub_analyze(page, analysis: dict) -> None:
    def handler(route, request):
        if request.method != "POST":
            return route.fallback()

        url = request.url
        if url.endswith("/api/analyze/meta"):
            payload = {
                "analysis": {
                    "novel_info": analysis.get("novel_info", {}),
                    "summary": analysis.get("summary", ""),
                }
            }
        elif url.endswith("/api/analyze/core"):
            payload = {
                "analysis": {
                    "characters": analysis.get("characters", []),
                    "relationships": analysis.get("relationships", []),
                }
            }
        elif url.endswith("/api/analyze/scenes"):
            payload = {
                "analysis": {
                    "first_sex_scenes": analysis.get("first_sex_scenes", []),
                    "sex_scenes": analysis.get("sex_scenes", {"total_count": 0, "scenes": []}),
                    "evolution": analysis.get("evolution", []),
                }
            }
        elif url.endswith("/api/analyze/lewd-elements"):
            payload = {
                "analysis": {
                    "lewd_elements": analysis.get("lewd_elements", []),
                    "lewd_elements_summary": analysis.get("lewd_elements_summary", ""),
                }
            }
        elif url.endswith("/api/analyze/thunderzones"):
            payload = {
                "analysis": {
                    "thunderzones": analysis.get("thunderzones", []),
                    "thunderzone_summary": analysis.get("thunderzone_summary", ""),
                }
            }
        else:
            return route.fallback()

        body = json.dumps(payload, ensure_ascii=False)
        return route.fulfill(status=200, headers={"Content-Type": "application/json"}, body=body)

    page.route("**/api/analyze/**", handler)


def _run_analysis(page) -> None:
    page.locator("[data-testid='analyze-button']").click()
    page.locator("#toastContainer div:has-text('分析完成')").wait_for(state="visible", timeout=30_000)


def test_thunderzone_tab_visible(server_url, analysis_empty, sample_novel_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _stub_analyze(page, analysis_empty)

        page.goto(server_url, wait_until="domcontentloaded")
        _upload_novel(page, sample_novel_path)
        _run_analysis(page)

        page.locator("button:has-text('雷点')").wait_for(state="visible", timeout=10_000)
        page.locator("button:has-text('涩情元素')").wait_for(state="visible", timeout=10_000)
        browser.close()


def test_thunderzone_empty_state(server_url, analysis_empty, sample_novel_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _stub_analyze(page, analysis_empty)

        page.goto(server_url, wait_until="domcontentloaded")
        _upload_novel(page, sample_novel_path)
        _run_analysis(page)

        page.locator("button:has-text('雷点')").click()
        page.locator("#thunderzoneSection").wait_for(state="visible", timeout=10_000)
        page.locator("#thunderzoneSection").locator("text=未检测到雷点").first.wait_for(state="visible", timeout=10_000)
        browser.close()


def test_thunderzone_display(server_url, analysis_with_thunderzones, sample_novel_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _stub_analyze(page, analysis_with_thunderzones)

        page.goto(server_url, wait_until="domcontentloaded")
        _upload_novel(page, sample_novel_path)
        _run_analysis(page)

        page.locator("button:has-text('雷点')").click()
        page.locator("#thunderzoneSection .thunderzone-card.thunderzone-high").wait_for(state="visible", timeout=10_000)
        page.locator("#thunderzoneSection .badge.badge-error:has-text('高')").wait_for(state="visible", timeout=10_000)
        page.locator("#thunderzoneSection .thunderzone-type:has-text('NTR')").wait_for(state="visible", timeout=10_000)
        browser.close()


def test_lewd_elements_empty_state(server_url, analysis_empty, sample_novel_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _stub_analyze(page, analysis_empty)

        page.goto(server_url, wait_until="domcontentloaded")
        _upload_novel(page, sample_novel_path)
        _run_analysis(page)

        page.locator("button:has-text('涩情元素')").click()
        page.locator("#lewdElementsSection").wait_for(state="visible", timeout=10_000)
        page.locator("#lewdElementsSection").locator("text=未检测到相关元素").first.wait_for(
            state="visible", timeout=10_000
        )
        browser.close()


def test_thunderzone_export(server_url, analysis_with_thunderzones, tmp_path, sample_novel_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        _stub_analyze(page, analysis_with_thunderzones)

        page.goto(server_url, wait_until="domcontentloaded")
        _upload_novel(page, sample_novel_path)
        _run_analysis(page)

        with page.expect_download() as download_info:
            page.locator("text=导出").click()

        download = download_info.value
        export_path = tmp_path / download.suggested_filename
        download.save_as(str(export_path))

        export_html = export_path.read_text(encoding="utf-8")
        assert "id: 'thunderzones'" in export_html
        assert "id: 'lewd-elements'" in export_html
        assert "雷点" in export_html
        assert "涩情元素" in export_html
        assert "NTR" in export_html
        assert "恋足" in export_html
        assert "thunderzone-high" in export_html

        context.close()
        browser.close()
