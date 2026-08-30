from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8123
BASE_URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="session")
def live_server():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/info", timeout=2) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.5)
    else:
        logs = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"Server failed to start on {BASE_URL}\n{logs}")

    yield

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def page(live_server):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        yield page
        browser.close()


def test_homepage_renders(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    assert page.locator("h1").text_content().strip() == "Technical Document Q&A"
    assert page.locator("#q").is_visible()
    assert page.locator("#source-panel").is_visible()
    assert page.locator("#doc-count").is_visible()
    assert page.locator("#status-chip").is_visible()
    assert page.locator("#messages").get_attribute("aria-live") == "polite"


def test_chat_answer_appears(page: Page):
    page.goto(BASE_URL)
    page.fill("#q", "What is the purpose of the embedded C coding standard?")
    page.click("#send")

    page.wait_for_selector("text=Confidence")
    assert page.locator("text=Confidence").first.is_visible()
    assert page.locator("#source-panel").is_visible()
    assert page.locator("#source-content").text_content().strip() != ""


def test_fallback_message_is_shown(page: Page):
    page.goto(BASE_URL)
    page.fill("#q", "What is the capital of Mars?")
    page.click("#send")

    page.wait_for_selector("text=No relevant information was found in the document to answer this question.")
    assert page.locator("text=No relevant information was found in the document to answer this question.").is_visible()
    assert page.locator("#source-panel").is_visible()


def test_upload_pdf_and_show_source_panel(page: Page):
    pdf_path = ROOT / "data" / "barr_c_coding_standard_2018.pdf"
    if not pdf_path.exists():
        pytest.skip(f"PDF fixture not found: {pdf_path}")

    page.goto(BASE_URL)
    page.set_input_files("#file", str(pdf_path))
    page.fill("#q", "What is the purpose of the Embedded C coding standard?")
    page.click("#send")

    page.wait_for_selector("text=Confidence")
    assert page.locator("#source-panel").is_visible()
    assert page.locator("#source-toggle").is_visible()
    assert page.locator("#source-content").text_content().strip() != ""
    assert page.locator("text=Snippet gợi ý").first.is_visible()

    page.click("#source-toggle")
    assert page.locator("#source-panel").evaluate("el => el.classList.contains('collapsed')")
    page.click("#source-toggle")
    assert not page.locator("#source-panel").evaluate("el => el.classList.contains('collapsed')")
