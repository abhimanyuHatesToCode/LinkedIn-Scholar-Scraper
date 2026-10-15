"""
linkedin_browser.py
====================
Handles ALL browser lifecycle and LinkedIn authentication.

STRATEGY: Persistent Browser Profile
--------------------------------------
Instead of injecting cookies or automating login on every run, we use a
dedicated Chrome user-data directory. The first time you run the scraper
you log in manually in the visible browser window. Chrome saves the full
session (cookies, localStorage, IndexedDB, fingerprint seed) to disk.
Every subsequent run re-opens that same profile — LinkedIn sees the same
"human" browser it has always seen, so it stays logged in without any
additional login flow.

Why this beats cookie injection:
  - Cookies alone are not enough; LinkedIn also checks localStorage tokens.
  - A persistent profile preserves the entire browser state, not just cookies.
  - The browser fingerprint (canvas hash, WebGL, fonts) stays consistent.
  - No need to re-authenticate when li_at / JSESSIONID rotate.

Patchright vs plain Playwright:
  - Patchright is a patched fork of Playwright that removes CDP leak signals,
    passes the Puppeteer Extra stealth test suite, and works with real Chrome
    via channel="chrome".
  - channel="chrome" means we use the user's installed Chrome binary, which
    has a real Chrome build id, extension list, and renderer flags — much
    harder to fingerprint as a bot than Chromium.
"""

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Optional

from patchright.async_api import (
    async_playwright,
    BrowserContext,
    Page,
    Playwright,
)

# ── Config ────────────────────────────────────────────────────────────────────

LINKEDIN_HOME = "https://www.linkedin.com"
LINKEDIN_LOGIN = "https://www.linkedin.com/login"

# We store our dedicated Chrome profile here.
# Using a project-local directory avoids conflicts with the user's own Chrome.
PROFILE_DIR = Path(os.getenv("PROFILE_DIR", "./chrome_profile")).resolve()

# Viewport that looks like a normal laptop screen
VIEWPORT = {"width": 1440, "height": 900}

# How long (ms) to wait for navigation / network events
NAV_TIMEOUT = int(os.getenv("NAV_TIMEOUT_MS", "60000"))

# Set to "true" to hide the browser window during scraping (not first-run)
HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"

# Realistic Chrome user-agent for Windows — keep in sync with channel="chrome"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ── Stealth init script ───────────────────────────────────────────────────────

# Injected into every page before any JS runs.
# Removes the most common bot-detection signals.
_STEALTH_SCRIPT = """
() => {
    // 1. Hide navigator.webdriver
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

    // 2. Restore plugins array (headless Chrome has 0 plugins)
    Object.defineProperty(navigator, 'plugins', {
        get: () => [
            { name: 'Chrome PDF Plugin' },
            { name: 'Chrome PDF Viewer' },
            { name: 'Native Client' },
        ],
    });

    // 3. Realistic language list
    Object.defineProperty(navigator, 'languages', {
        get: () => ['en-US', 'en'],
    });

    // 4. Hardware concurrency / memory that matches a normal laptop
    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
    Object.defineProperty(navigator, 'deviceMemory',        { get: () => 8 });

    // 5. WebGL renderer — spoof Intel integrated GPU (common on laptops)
    const origGetParam = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(param) {
        if (param === 37445) return 'Intel Inc.';               // VENDOR
        if (param === 37446) return 'Intel Iris OpenGL Engine'; // RENDERER
        return origGetParam.apply(this, arguments);
    };

    // 6. Add window.chrome so pages think we're a real Chrome install
    if (!window.chrome) {
        window.chrome = {
            runtime: {},
            loadTimes: function() {},
            csi: function() {},
            app: {},
        };
    }

    // 7. Notification permission — real browsers default to 'default'
    if (window.Notification) {
        Object.defineProperty(Notification, 'permission', {
            get: () => 'default',
        });
    }
}
"""


# ── LinkedInBrowser ───────────────────────────────────────────────────────────

class LinkedInBrowser:
    """
    Wraps a persistent Patchright/Chrome browser context.

    Usage (typical):
        browser = LinkedInBrowser()
        await browser.start()          # opens Chrome with saved profile
        await browser.ensure_logged_in()
        page = await browser.new_page()
        html = await browser.fetch_page(page, "https://linkedin.com/in/someone")
        await browser.close()

    On the very first run (empty profile directory), ensure_logged_in() will
    open a visible browser window and wait up to 5 minutes for you to log in
    manually. After that the session is persisted automatically.
    """

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch Patchright with a persistent Chrome profile."""
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[browser] Profile directory: {PROFILE_DIR}")
        print(f"[browser] Headless: {HEADLESS}")

        self._playwright = await async_playwright().start()

        # launch_persistent_context keeps the full browser state on disk.
        # channel="chrome" uses the real installed Chrome binary — not Chromium.
        # This is critical: LinkedIn's fingerprinting checks the Chrome version
        # string, extension markers, and renderer quirks that only real Chrome has.
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            channel="chrome",          # ← real installed Chrome
            headless=HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-popup-blocking",
                # Prevent "Chrome is being controlled by automated software" banner
                "--disable-infobars",
            ],
            user_agent=USER_AGENT,
            viewport=VIEWPORT,
            locale="en-US",
            timezone_id="America/New_York",
            # Mimic a normal laptop screen
            device_scale_factor=1,
            is_mobile=False,
            has_touch=False,
        )

        # Inject stealth overrides into every new page
        await self._context.add_init_script(_STEALTH_SCRIPT)
        self._context.set_default_timeout(NAV_TIMEOUT)

        print("[browser] Browser started successfully.")

    async def close(self) -> None:
        """Gracefully close browser and Playwright."""
        if self._context:
            await self._context.close()
        if self._playwright:
            await self._playwright.stop()
        print("[browser] Browser closed.")

    async def new_page(self) -> Page:
        """Open a fresh tab in the persistent context."""
        if not self._context:
            raise RuntimeError("Browser not started. Call await browser.start() first.")
        return await self._context.new_page()

    # ── Authentication ────────────────────────────────────────────────────────

    async def ensure_logged_in(self) -> bool:
        """
        Guarantee we have an active LinkedIn session.

        Flow:
          1. Open LinkedIn home with the persisted profile.
          2. If already logged in  → return True immediately.
          3. If NOT logged in      → open a visible browser, navigate to /login,
                                     and wait up to 5 minutes for the user to
                                     complete the login (including any 2FA or
                                     CAPTCHA challenges).
          4. Once login is confirmed, the persistent profile is automatically
             saved to disk — no explicit cookie export needed.

        Returns True if a valid session exists after the call.
        """
        page = await self.new_page()
        try:
            print("[auth] Checking session status…")
            await page.goto(LINKEDIN_HOME, wait_until="domcontentloaded")
            await _human_pause(1.5, 2.5)

            if await _is_logged_in(page):
                print("[auth] ✓ Already logged in (persistent profile).")
                return True

            # ── First-run / expired session: manual login ──────────────────
            print("\n" + "=" * 60)
            print("[auth] NOT logged in. Opening LinkedIn login page…")
            print("[auth] Please log in manually in the browser window.")
            print("[auth] The scraper will wait up to 5 minutes.")
            print("=" * 60 + "\n")

            await page.goto(LINKEDIN_LOGIN, wait_until="domcontentloaded")

            # Poll until logged in or timeout (300 s = 5 minutes)
            deadline = asyncio.get_event_loop().time() + 300
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(3)
                if await _is_logged_in(page):
                    print("\n[auth] ✓ Login detected! Session saved to profile.")
                    # Persist cookies explicitly as well (belt-and-suspenders)
                    await _save_cookies(self._context)
                    return True

            print("[auth] ✗ Login timeout — did not detect a valid session.")
            return False

        finally:
            await page.close()

    # ── Page fetching ─────────────────────────────────────────────────────────

    async def fetch_page(self, page: Page, url: str) -> str:
        """
        Navigate to URL and return the rendered inner text of the page body.

        We use inner_text() rather than page.content() (raw HTML) because:
          - It strips all tags, scripts, styles — sending clean text to the LLM.
          - Token cost drops by ~70 % compared to raw HTML.
          - No BeautifulSoup / regex cleaning needed.
          - The visible text is exactly what a human reads; it's sufficient for
            structured data extraction by an LLM.

        A light scroll pass is also performed so lazy-loaded sections
        (experience, education) are fully rendered before we capture text.
        """
        print(f"[browser] Fetching: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        await _human_pause(2, 3)

        # Scroll through the page to trigger lazy-loaded sections
        await _scroll_page(page)
        await _human_pause(1, 2)

        # Grab fully-rendered visible text
        text = await page.inner_text("body")
        print(f"[browser] Captured {len(text):,} characters of visible text.")
        return text


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _is_logged_in(page: Page) -> bool:
    """
    Check multiple reliable signals that indicate an active LinkedIn session.
    We avoid any single fragile selector by checking several independently.
    """
    url = page.url

    # URL-based check is the most reliable signal
    if any(path in url for path in ("/feed", "/mynetwork", "/jobs", "/messaging")):
        return True

    # DOM-based fallbacks
    selectors = [
        "div.global-nav__me",            # nav profile bubble
        "img.global-nav__me-photo",      # profile photo in nav
        "[data-control-name='nav.settings']",  # settings icon
    ]
    for sel in selectors:
        try:
            if await page.locator(sel).count() > 0:
                return True
        except Exception:
            continue

    # Absence of the login form is also a positive signal
    try:
        login_form = await page.locator("input#username").count()
        if login_form == 0 and "linkedin.com" in url and "authwall" not in url:
            # We're on LinkedIn and there's no login form — likely logged in
            title = await page.title()
            if "Sign In" not in title and "Log In" not in title:
                return True
    except Exception:
        pass

    return False


async def _save_cookies(context: BrowserContext) -> None:
    """Belt-and-suspenders: also write cookies to a JSON file."""
    cookies_file = PROFILE_DIR / "linkedin_cookies.json"
    cookies = await context.cookies()
    with open(cookies_file, "w") as f:
        json.dump(cookies, f, indent=2)
    print(f"[auth] Cookies backed up to {cookies_file}")


async def _human_pause(min_s: float = 1.0, max_s: float = 3.0) -> None:
    """Random sleep that mimics human reading/thinking time."""
    await asyncio.sleep(random.uniform(min_s, max_s))


async def _scroll_page(page: Page) -> None:
    """
    Gradually scroll through the full page height so lazy-loaded React
    components (LinkedIn renders sections on scroll) are fully visible
    before we capture the text.
    """
    total_height: int = await page.evaluate("document.body.scrollHeight")
    viewport_h: int = await page.evaluate("window.innerHeight")
    current_y = 0
    step = viewport_h // 2  # scroll half a viewport at a time

    while current_y < total_height:
        current_y = min(current_y + step, total_height)
        await page.evaluate(f"window.scrollTo(0, {current_y})")
        await asyncio.sleep(random.uniform(0.4, 0.8))

    # Scroll back to top so the page looks natural
    await page.evaluate("window.scrollTo(0, 0)")
