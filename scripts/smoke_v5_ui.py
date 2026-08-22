"""Exercise the PacketFlowAI v5 Assurance War Room in Chromium."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


def exercise(page: Page, base_url: str, screenshot: Path) -> None:
    errors: list[str] = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(base_url, wait_until="domcontentloaded")
    page.locator("#system-state").wait_for(state="visible")
    page.wait_for_function("document.querySelector('#system-state')?.textContent === 'System live'")
    page.locator("button[data-view='assurance']").click()
    page.locator("#assurance-level").wait_for(state="visible")
    page.wait_for_function("document.querySelector('#assurance-level')?.textContent === 'A4'")
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=screenshot.with_name(f"{screenshot.stem}-hero{screenshot.suffix}"))
    page.get_by_role("button", name="Witness receipt").click()
    page.wait_for_function("document.querySelector('#proof-active')?.textContent === 'WITNESS RECEIPT'")
    dark_cells = page.locator(".heat-cell.dark")
    if dark_cells.count() < 1:
        raise RuntimeError("assurance heatmap has no dark-period controls")
    dark_cells.first.click()
    page.wait_for_function("document.querySelector('#dark-period-detail')?.textContent.includes('SENSOR GAP')")
    page.screenshot(path=screenshot)
    if errors:
        raise RuntimeError("browser errors: " + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path("artifacts/presentation"))
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        desktop = browser.new_page(viewport={"width": 1600, "height": 900})
        exercise(desktop, args.url, args.output / "v5-assurance-desktop.png")
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        exercise(mobile, args.url, args.output / "v5-assurance-mobile.png")
        browser.close()
    print("Assurance War Room passed desktop and mobile browser smoke tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
