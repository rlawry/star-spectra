#!/usr/bin/env python3
# download_fits.py
import os
import time
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/230/16/fits/"
OUT_DIR = Path("ApJS230_16_FITS")
MAX_WORKERS = 8
RETRY = 3
TIMEOUT = 60

def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def list_fits():
    """Scrape directory index for *.fits links."""
    with requests.Session() as s:
        s.headers.update({"User-Agent": "star-spectra-downloader/1.0"})
        r = s.get(BASE_URL, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        files = []
        for a in soup.find_all("a"):
            href = a.get("href") or ""
            if href.lower().endswith(".fits"):
                files.append(urllib.parse.urljoin(BASE_URL, href))
        return sorted(set(files))

def download_one(url):
    """Download a single file with retries and .part temp file."""
    fname = OUT_DIR / os.path.basename(urllib.parse.urlparse(url).path)
    if fname.exists() and fname.stat().st_size > 0:
        return fname.name, "skip"
    for attempt in range(1, RETRY + 1):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT, headers={"User-Agent": "star-spectra-downloader/1.0"}) as r:
                r.raise_for_status()
                tmp = str(fname) + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, fname)
            return fname.name, "ok"
        except Exception as e:
            if attempt == RETRY:
                return fname.name, f"fail: {e}"
            time.sleep(2 * attempt)

def main():
    ensure_outdir()
    urls = list_fits()
    print(f"Found {len(urls)} FITS files in remote folder")
    if not urls:
        return
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(download_one, u) for u in urls]
        done = 0
        for fut in as_completed(futs):
            name, status = fut.result()
            done += 1
            print(f"[{done}/{len(urls)}] {name} -> {status}")

if __name__ == "__main__":
    main()
