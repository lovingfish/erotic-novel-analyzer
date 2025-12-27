from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("about:blank")
        page.add_script_tag(path=str(REPO_ROOT / "static" / "chart-view.js"))
        yield page
        browser.close()


def _build_thunderzones_html(page, analysis: dict) -> str:
    return page.evaluate("analysis => buildThunderzonesHtml(analysis)", analysis)


def test_build_thunderzones_html_empty(page):
    html = _build_thunderzones_html(page, {})
    assert "未检测到雷点" in html
    assert 'data-lucide="shield-check"' in html


def test_build_thunderzones_html_with_data(page):
    analysis = {
        "thunderzones": [
            {
                "type": "NTR",
                "severity": "高",
                "description": "测试描述",
                "involved_characters": ["角色A"],
                "chapter_location": "第1章",
                "relationship_context": "测试关系",
            }
        ],
        "thunderzone_summary": "测试总结",
    }
    html = _build_thunderzones_html(page, analysis)
    assert "NTR" in html
    assert "badge-error" in html
    assert "thunderzone-high" in html
    assert "测试描述" in html
    assert "测试总结" in html


def test_thunderzone_sorting(page):
    analysis = {
        "thunderzones": [
            {"type": "低", "severity": "低", "description": "lowdesc"},
            {"type": "高", "severity": "高", "description": "highdesc"},
            {"type": "中", "severity": "中", "description": "middesc"},
        ],
        "thunderzone_summary": "排序测试",
    }
    html = _build_thunderzones_html(page, analysis)
    assert html.index("highdesc") < html.index("middesc") < html.index("lowdesc")


def test_escape_html_in_thunderzones(page):
    analysis = {
        "thunderzones": [
            {
                "type": "<script>alert('xss')</script>",
                "severity": "高",
                "description": "<img src=x onerror=alert(1)>",
                "involved_characters": ['<a href="javascript:alert(2)">x</a>'],
                "chapter_location": "<svg onload=alert(3)>",
                "relationship_context": "<script>alert(4)</script>",
            }
        ],
        "thunderzone_summary": "<img src=x onerror=alert(5)>",
    }
    html = _build_thunderzones_html(page, analysis)
    assert "<script" not in html
    assert "<img" not in html
    assert "<svg" not in html
    assert "<a" not in html
    assert "&lt;script" in html
    assert "&lt;img" in html
