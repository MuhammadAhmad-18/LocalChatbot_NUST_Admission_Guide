"""
NUST Admissions Full Scraper
============================
Scrapes https://nust.edu.pk/admissions/ and all sublinks.
Handles 403 bot-protection using rotating headers + delays.
Saves all data to nust_data/ folder as .txt files.

Run:
    pip install requests beautifulsoup4 html2text playwright tqdm
    playwright install chromium
    python nust_scraper.py
"""

import requests
from bs4 import BeautifulSoup
import html2text
import os
import time
import json
import random
from urllib.parse import urljoin, urlparse
from collections import deque
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BASE_DOMAIN   = "nust.edu.pk"
START_URL     = "https://nust.edu.pk/admissions/"
OUTPUT_DIR    = "nust_data"
MAX_PAGES     = 150          # safety cap
DELAY_MIN     = 2.0          # seconds between requests (be polite)
DELAY_MAX     = 4.0

# Rotating User-Agents to avoid 403
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# ALL known NUST admissions sublinks (discovered from site structure + search)
KNOWN_URLS = {
    # ── Undergraduate ──────────────────────────────────────
    "ug_overview":              "https://nust.edu.pk/admissions/undergraduates/",
    "ug_eligibility":           "https://nust.edu.pk/admissions/undergraduates/eligibility-criteria-for-ug-programmes/",
    "ug_academic_qualification":"https://nust.edu.pk/admissions/undergraduates/academic-qualification-required-for-different-ug-progammes/",
    "ug_dae_eligibility":       "https://nust.edu.pk/admissions/undergraduates/dae-eligibility-criteria/",
    "ug_programmes_list":       "https://nust.edu.pk/admissions/undergraduates/list-of-ug-programmes/",
    "ug_merit_formula":         "https://nust.edu.pk/admissions/undergraduates/merit-formula/",
    "ug_procedure":             "https://nust.edu.pk/admissions/undergraduates/procedure-of-admission/",
    "ug_net_test":              "https://nust.edu.pk/admissions/undergraduates/net/",
    "ug_act_sat":               "https://nust.edu.pk/admissions/undergraduates/act-sat/",
    "ug_expatriate":            "https://nust.edu.pk/admissions/undergraduates/expatriate-students/",
    "ug_dates":                 "https://nust.edu.pk/admissions/undergraduates/dates-to-remember/",
    "ug_helpdesk":              "https://nust.edu.pk/admissions/undergraduates/ug-admissions-help-desk/",
    "ug_cancellation":          "https://nust.edu.pk/admissions/undergraduates/cancellation-of-admission/",

    # ── Masters / Postgraduate ─────────────────────────────
    "pg_overview":              "https://nust.edu.pk/admissions/masters/",
    "pg_announcement":          "https://nust.edu.pk/admissions/masters/admission-announcement/",
    "pg_eligibility":           "https://nust.edu.pk/admissions/masters/eligibility-criteria/",
    "pg_programmes":            "https://nust.edu.pk/admissions/masters/programmes-offered/",
    "pg_dates":                 "https://nust.edu.pk/admissions/masters/dates-to-remember/",
    "pg_faqs":                  "https://nust.edu.pk/admissions/masters/faqs/",
    "pg_how_to_apply":          "https://nust.edu.pk/admissions/masters/how-to-apply/",
    "pg_filling_form":          "https://nust.edu.pk/admissions/masters/instructions-for-filling-admission-form/",

    # ── PhD ────────────────────────────────────────────────
    "phd_overview":             "https://nust.edu.pk/admissions/phd/",
    "phd_programmes":           "https://nust.edu.pk/admissions/phd/phd-programmes-of-study/",
    "phd_eligibility":          "https://nust.edu.pk/admissions/phd/eligibility-criteria/",
    "phd_dates":                "https://nust.edu.pk/admissions/phd/dates-to-remember/",
    "phd_faqs":                 "https://nust.edu.pk/admissions/phd/faqs/",
    "phd_how_to_apply":         "https://nust.edu.pk/admissions/phd/how-to-apply/",

    # ── Fee Structure ──────────────────────────────────────
    "fee_ug":                   "https://nust.edu.pk/admissions/fee-structure/undergraduate-financial-matters/",
    "fee_pg":                   "https://nust.edu.pk/admissions/fee-structure/postgraduate-financial-matters/",
    "fee_refund":               "https://nust.edu.pk/admissions/fee-structure/refund-policy/",
    "fee_hostel":               "https://nust.edu.pk/admissions/fee-structure/hostel-accommodation/",
    "fee_other":                "https://nust.edu.pk/admissions/fee-structure/other-fee-charges/",
    "fee_mode_of_payment":      "https://nust.edu.pk/admissions/fee-structure/mode-of-payment/",
    "fee_advance_tax":          "https://nust.edu.pk/admissions/fee-structure/advance-tax-on-payment-of-fee/",

    # ── Scholarships ───────────────────────────────────────
    "scholarship_overview":     "https://nust.edu.pk/admissions/scholarships/",
    "scholarship_merit":        "https://nust.edu.pk/admissions/scholarships/merit-based-scholarships-for-masters-and-phd/",
    "scholarship_need_based":   "https://nust.edu.pk/admissions/scholarships/need-based-financial-aid/",
    "scholarship_nfaaf":        "https://nust.edu.pk/admissions/scholarships/nfaaf-documentation-requirements/",
    "scholarship_peef":         "https://nust.edu.pk/admissions/scholarships/punjab-educational-endowment-fund-peef-scholarships/",
    "scholarship_ihsan":        "https://nust.edu.pk/admissions/scholarships/interest-free-loan-ihsan-trust/",
    "scholarship_ug_financial": "https://nust.edu.pk/admissions/scholarships/financial-assistance-for-undergraduate-students/",
    "scholarship_deferment":    "https://nust.edu.pk/admissions/scholarships/deferment-of-tuition-fee-subsistence-allowance/",
    "scholarship_other":        "https://nust.edu.pk/admissions/scholarships/other-opportunities/",

    # ── NSHS (Medical) ─────────────────────────────────────
    "nshs_overview":            "https://nust.edu.pk/admissions/NSHS/",
    "nshs_merit_mbbs":          "https://nust.edu.pk/admissions/NSHS/merit-generation-criteria-mbbs/",
    "nshs_fee":                 "https://nust.edu.pk/admissions/NSHS/fee-structure/",
    "nshs_how_to_apply":        "https://nust.edu.pk/admissions/NSHS/how-to-apply/",

    # ── Allied / BScHND ────────────────────────────────────
    "allied_fee":               "https://nust.edu.pk/admissions/alliedprogram/fee-structure-bshnd/",

    # ── General FAQs ───────────────────────────────────────
    "faqs_main":                "https://nust.edu.pk/faqs/",
    "faqs_admissions":          "https://nust.edu.pk/admissions/faqs/",

    # ── Contact ────────────────────────────────────────────
    "contact":                  "https://nust.edu.pk/admissions/contact-us/",
}

# Keywords that make a URL worth following during crawl
RELEVANT_KEYWORDS = [
    "admission", "undergraduate", "postgraduate", "masters", "phd",
    "fee", "scholarship", "hostel", "net-test", "merit", "program",
    "eligibility", "faq", "apply", "dates", "criteria", "nshs",
    "allied", "financial", "refund", "payment", "dae", "expatriate",
    "cancellation", "procedure", "helpdesk", "announcement",
]

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://nust.edu.pk/",
    }

def url_to_filename(url):
    path = urlparse(url).path.strip("/").replace("/", "__")
    return (path[:80] or "index") + ".txt"

def is_relevant_url(url):
    return any(kw in url.lower() for kw in RELEVANT_KEYWORDS)

def is_same_domain(url):
    return BASE_DOMAIN in urlparse(url).netloc

converter = html2text.HTML2Text()
converter.ignore_links = False
converter.ignore_images = True
converter.body_width = 0

def extract_clean_text(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # Remove clutter
    for tag in soup(["nav", "footer", "script", "style", "header",
                      "aside", "iframe", ".wp-block-navigation",
                      ".site-header", ".site-footer", ".cookie-notice"]):
        tag.decompose()

    # Target main content
    main = (
        soup.find("main") or
        soup.find("div", class_="entry-content") or
        soup.find("div", class_="page-content") or
        soup.find("article") or
        soup.find("div", id="content") or
        soup.find("div", class_="container") or
        soup.find("body")
    )

    text = converter.handle(str(main))

    # Clean blank lines
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def extract_faqs(html):
    """Try to extract structured Q&A pairs from FAQ pages."""
    soup = BeautifulSoup(html, "html.parser")
    faqs = []

    # Pattern 1: accordion items
    for item in soup.select(".accordion-item, .faq-item, .wp-block-faq"):
        q = item.select_one("h3, h4, .accordion-header, .faq-question, button")
        a = item.select_one("p, .accordion-body, .faq-answer, .answer")
        if q and a:
            faqs.append({"Q": q.get_text(strip=True), "A": a.get_text(strip=True)})

    # Pattern 2: dt/dd definition lists
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            faqs.append({"Q": dt.get_text(strip=True), "A": dd.get_text(strip=True)})

    # Pattern 3: heading followed by subsequent siblings
    if not faqs:
        for h in soup.find_all(["h3", "h4"]):
            q_text = h.get_text(strip=True)
            if not q_text or len(q_text) < 5 or not q_text.strip().endswith("?"):
                continue

            # Collect all subsequent siblings until next heading
            ans_parts = []
            for nxt in h.find_next_siblings():
                if nxt.name in ["h1", "h2", "h3", "h4"]:
                    break
                txt = nxt.get_text(strip=True)
                if txt:
                    ans_parts.append(txt)

            if ans_parts:
                faqs.append({"Q": q_text, "A": "\n".join(ans_parts)})

    return faqs

def save_page(label, url, text, faqs=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, url_to_filename(url))

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"TOPIC   : {label}\n")
        f.write(f"URL     : {url}\n")
        f.write(f"SCRAPED : {time.strftime('%Y-%m-%d')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)

        if faqs:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("STRUCTURED FAQs\n")
            f.write("=" * 60 + "\n\n")
            for i, faq in enumerate(faqs, 1):
                f.write(f"Q{i}: {faq['Q']}\n")
                f.write(f"A{i}: {faq['A']}\n\n")

    return filename

# ─────────────────────────────────────────────
#  PLAYWRIGHT FALLBACK (for JS-heavy / 403 pages)
# ─────────────────────────────────────────────

def scrape_with_playwright(url, label):
    """Use headless browser for pages that block requests."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [SKIP] playwright not installed. Run: playwright install chromium")
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1280, "height": 800},
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"}
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000)

            # Expand all accordions / FAQ toggles
            for selector in ['[aria-expanded="false"]', ".accordion-button",
                              ".faq-toggle", ".toggle-btn"]:
                try:
                    btns = page.query_selector_all(selector)
                    for btn in btns:
                        btn.click()
                        page.wait_for_timeout(200)
                except:
                    pass

            html = page.content()
            browser.close()

        text = extract_clean_text(html, url)
        faqs = extract_faqs(html) if "faq" in url.lower() else []
        return html, text, faqs

    except Exception as e:
        print(f"  [Playwright ERROR] {e}")
        return None

# ─────────────────────────────────────────────
#  MAIN SCRAPER
# ─────────────────────────────────────────────

def scrape_url(url, label="page"):
    """Fetch a URL, return (html, text, faqs) or None on failure."""
    try:
        session = requests.Session()
        # Warm-up: visit homepage first (sets cookies, looks human)
        session.get("https://nust.edu.pk/", headers=get_headers(), timeout=15)
        time.sleep(random.uniform(1, 2))

        r = session.get(url, headers=get_headers(), timeout=20)

        if r.status_code == 403:
            print(f"  [403] Trying Playwright fallback for: {label}")
            result = scrape_with_playwright(url, label)
            if result:
                return result
            print(f"  [FAIL] Could not scrape: {url}")
            return None

        r.raise_for_status()

        if "text/html" not in r.headers.get("Content-Type", ""):
            return None

        html = r.text
        text = extract_clean_text(html, url)
        faqs = extract_faqs(html) if "faq" in url.lower() else []
        return html, text, faqs

    except requests.exceptions.RequestException as e:
        print(f"  [REQUEST ERROR] {label}: {e}")
        # Try playwright as last resort
        result = scrape_with_playwright(url, label)
        return result if result else None


def discover_sublinks(html, base_url):
    """Find new admission-related links on a page."""
    soup = BeautifulSoup(html, "html.parser")
    found = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"]).split("#")[0].split("?")[0]
        if is_same_domain(full) and is_relevant_url(full) and full.startswith("http"):
            found.add(full.rstrip("/") + "/")
    return found


def run_scraper():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    visited      = set()
    saved_files  = []
    all_faqs     = []
    failed_urls  = []

    # Queue: (url, label)
    # Start with all known URLs, then auto-discover more
    queue = deque([(url, label) for label, url in KNOWN_URLS.items()])
    queue.appendleft((START_URL, "admissions_home"))

    print(f"\n{'='*60}")
    print(f"  NUST Admissions Scraper")
    print(f"  Output folder: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Starting with {len(queue)} known URLs + auto-discovery")
    print(f"{'='*60}\n")

    with tqdm(total=len(queue), desc="Scraping", unit="page") as pbar:
        while queue and len(visited) < MAX_PAGES:
            url, label = queue.popleft()
            url = url.rstrip("/") + "/"

            if url in visited:
                continue
            visited.add(url)

            pbar.set_description(f"Scraping: {label[:40]}")

            result = scrape_url(url, label)

            if result:
                html, text, faqs = result

                if len(text) < 100:
                    print(f"  [SKIP] Too short: {label}")
                    pbar.update(1)
                    continue

                filepath = save_page(label, url, text, faqs)
                saved_files.append(filepath)

                if faqs:
                    all_faqs.extend(faqs)
                    print(f"  ✓ {label} ({len(text)} chars, {len(faqs)} FAQs)")
                else:
                    print(f"  ✓ {label} ({len(text)} chars)")

                # Auto-discover new links from this page
                try:
                    new_links = discover_sublinks(html, url)
                    for link in new_links:
                        if link not in visited:
                            link_label = urlparse(link).path.strip("/").replace("/", "_")
                            queue.append((link, link_label))
                            pbar.total += 1
                            pbar.refresh()
                except:
                    pass

            else:
                failed_urls.append(url)
                print(f"  ✗ FAILED: {label}")

            pbar.update(1)
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    # ── Save master FAQ file ──────────────────────────────
    if all_faqs:
        faq_txt = os.path.join(OUTPUT_DIR, "_all_faqs.txt")
        faq_json = os.path.join(OUTPUT_DIR, "_all_faqs.json")

        with open(faq_txt, "w", encoding="utf-8") as f:
            f.write("NUST ADMISSIONS — ALL FAQs\n")
            f.write(f"Collected: {time.strftime('%Y-%m-%d')}\n")
            f.write("=" * 60 + "\n\n")
            for i, faq in enumerate(all_faqs, 1):
                f.write(f"Q{i}: {faq['Q']}\n")
                f.write(f"A{i}: {faq['A']}\n\n")

        with open(faq_json, "w", encoding="utf-8") as f:
            json.dump(all_faqs, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Saved {len(all_faqs)} FAQs → {faq_txt}")

    # ── Save scrape report ────────────────────────────────
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "total_pages_saved": len(saved_files),
        "total_faqs": len(all_faqs),
        "failed_urls": failed_urls,
        "saved_files": [os.path.basename(f) for f in saved_files],
    }
    with open(os.path.join(OUTPUT_DIR, "_scrape_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Pages saved  : {len(saved_files)}")
    print(f"  FAQs found   : {len(all_faqs)}")
    print(f"  Failed URLs  : {len(failed_urls)}")
    print(f"  Output folder: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}\n")

    if failed_urls:
        print("Failed URLs (try running again or use Playwright):")
        for u in failed_urls:
            print(f"  - {u}")

if __name__ == "__main__":
    run_scraper()