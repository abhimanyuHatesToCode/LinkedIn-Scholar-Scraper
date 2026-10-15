"""
first_login.py
==============
Run this ONCE before starting the FastAPI server.

It opens a visible Chrome window at the LinkedIn login page, waits for
you to log in manually (including any 2FA / CAPTCHA), then saves the full
session to the persistent profile directory.

After this script exits successfully you can set HEADLESS=true in .env
and start the server — no further logins are needed until LinkedIn
invalidates the session (usually 30-90 days of inactivity).

Usage:
    python first_login.py
"""

import asyncio
import sys
from scraper.linkedin_browser import LinkedInBrowser, HEADLESS


async def main():
    if HEADLESS:
        print(
            "ERROR: HEADLESS=true is set in .env.\n"
            "Set HEADLESS=false before running first_login.py so you can\n"
            "see the browser and log in manually."
        )
        sys.exit(1)

    print("=" * 60)
    print("  LinkedIn First-Run Login")
    print("=" * 60)
    print()
    print("A Chrome window will open at linkedin.com/login.")
    print("Please log in manually — including any 2FA prompts.")
    print("The script will detect successful login and exit.")
    print()
    print("DO NOT close the terminal while logging in.")
    print()

    browser = LinkedInBrowser()
    await browser.start()

    success = await browser.ensure_logged_in()

    if success:
        print()
        print("=" * 60)
        print("  ✓ Login successful!")
        print("  Session saved to:", browser._context)
        print()
        print("  You can now:")
        print("  1. Set HEADLESS=true in .env")
        print("  2. Run: uvicorn main:app --port 8000")
        print("=" * 60)
    else:
        print()
        print("✗ Login was not detected. Please try again.")

    await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
