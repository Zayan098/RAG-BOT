"""Scrape The AI Agent Factory book (agentfactory.panaversity.org) into markdown.

Downloads the sitemap, fetches every English docs page (server-rendered HTML),
extracts the article content, converts it to markdown, and saves it into
book_content/ with a manifest.json describing each page.

Usage:
    python scrape_book.py            # fetch anything missing
    python scrape_book.py --refresh  # re-download every page
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://agentfactory.panaversity.org"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
OUT_DIR = Path(__file__).parent / "book_content"
MANIFEST_PATH = OUT_DIR / "manifest.json"

CONTENT_SELECTORS = [
    "div.theme-doc-markdown",
    "article",
]

SKIP_PREFIXES = ("/docs/roman/", "/docs/ur/")
REQUEST_DELAY_SECONDS = 1.0


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (compatible; BookRagBot/1.0; educational research bot)"
    )
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_doc_urls(session: requests.Session) -> list[str]:
    resp = session.get(SITEMAP_URL, timeout=30)
    resp.raise_for_status()
    urls = sorted(
        set(re.findall(r"<loc>([^<]+)</loc>", resp.text))
    )
    doc_urls = []
    for url in urls:
        path = url.split(BASE_URL, 1)[-1]
        if path.startswith("/docs/") and not path.startswith(SKIP_PREFIXES):
            doc_urls.append(url)
    return doc_urls


def extract_content(soup: BeautifulSoup) -> BeautifulSoup | None:
    for selector in CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node is not None:
            return node
    return None


def extract_title(soup: BeautifulSoup, content: BeautifulSoup | None) -> str:
    if content is not None:
        h1 = content.find("h1")
        if h1 is not None:
            return h1.get_text(strip=True)
    title_tag = soup.find("title")
    if title_tag is not None:
        return title_tag.get_text(strip=True)
    return "Untitled"


def extract_breadcrumb(soup: BeautifulSoup) -> str:
    parts = [
        a.get_text(strip=True)
        for a in soup.select("nav.theme-doc-breadcrumbs a, nav.theme-doc-breadcrumbs li span")
        if a.get_text(strip=True)
    ]
    parts = [p for p in parts if p.lower() != "home"]
    collapsed = []
    for part in parts:
        if not collapsed or part != collapsed[-1]:
            collapsed.append(part)
    return " > ".join(collapsed)


ANCHOR_LINK_RE = re.compile(r"\[\u200b\]\(#[^)]* \"Direct link to [^\"]*\"\)")
ANCHOR_LINK_RE_PLAIN = re.compile(r"\[\u200b\]\(#[^)]*\)")


def clean_markdown(markdown: str) -> str:
    markdown = ANCHOR_LINK_RE.sub("", markdown)
    markdown = ANCHOR_LINK_RE_PLAIN.sub("", markdown)
    return markdown.strip() + "\n"


def slug_from_url(url: str) -> str:
    path = url.split(BASE_URL, 1)[-1]
    path = path.removeprefix("/docs/")
    return path.strip("/")


def fetch_page(session: requests.Session, url: str) -> tuple[str, str, str] | None:
    resp = session.get(url, timeout=60)
    if resp.status_code != 200:
        print(f"  ! {resp.status_code} for {url}")
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    content = extract_content(soup)
    if content is None:
        print(f"  ! no article content found for {url}")
        return None
    body_html = str(content)
    markdown = md(
        body_html,
        heading_style="ATX",
        strip=["img", "figure", "script", "style"],
    )
    title = extract_title(soup, content)
    section = extract_breadcrumb(soup)
    return clean_markdown(markdown), title, section


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape the book site into markdown.")
    parser.add_argument("--refresh", action="store_true", help="Re-download every page")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    session = make_session()
    urls = get_doc_urls(session)
    print(f"Found {len(urls)} English docs pages in sitemap.")

    skipped = 0
    for url in urls:
        slug = slug_from_url(url)
        out_path = OUT_DIR / f"{slug}.md"
        if not args.refresh and out_path.exists() and slug in manifest:
            skipped += 1
            continue

        print(f"  Fetching {slug}")
        result = fetch_page(session, url)
        if result is None:
            continue
        markdown, title, section = result
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        manifest[slug] = {
            "url": url,
            "title": title,
            "section": section,
        }
        time.sleep(REQUEST_DELAY_SECONDS)

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done. {len(manifest)} pages cached ({skipped} skipped as up to date).")


if __name__ == "__main__":
    main()