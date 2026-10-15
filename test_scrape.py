"""
test_scrape.py
===============
Quick end-to-end smoke test you can run from the terminal to verify the
full pipeline works before starting the FastAPI server.

Usage:
    python test_scrape.py williamhgates
    python test_scrape.py https://www.linkedin.com/in/williamhgates
"""

import asyncio
import sys
import json

from scraper.linkedin_browser import LinkedInBrowser
from scraper.linkedin_scraper import LinkedInScraper
from extraction.ai_extractor import AIExtractor


async def main(profile_identifier: str):
    print(f"\n[test] Scraping profile: {profile_identifier!r}\n")

    browser   = LinkedInBrowser()
    extractor = AIExtractor()

    await browser.start()
    logged_in = await browser.ensure_logged_in()

    if not logged_in:
        print("[test] ✗ Not logged in. Run python first_login.py first.")
        await browser.close()
        sys.exit(1)

    scraper = LinkedInScraper(browser, extractor)

    try:
        profile = await scraper.scrape(profile_identifier)
        print("\n" + "=" * 60)
        print("  Extracted Profile")
        print("=" * 60)
        print(profile.model_dump_json(indent=2))
        print()
        print(f"  Summary: {profile.display()}")
        print(f"  Complete: {profile.is_complete}")
        print("=" * 60 + "\n")
    except Exception as exc:
        print(f"[test] ✗ Error: {exc}")
    finally:
        await browser.close()


if __name__ == "__main__":
    identifier = sys.argv[1] if len(sys.argv) > 1 else "williamhgates"
    asyncio.run(main(identifier))
