#!/usr/bin/env python3
# make_spectra_giants_robust.py
import argparse, json, re, sys
from pathlib import Path
import numpy as np

SRC_DIRS = [Path("ApjS230_16_FITS"), Path("ApJS230_16_FITS"),
            Path("StarSpectra-FITS"), Path("StarSpectra_FITS")]
DEFAULT_OUT = Path("spectra_giants.json")

try:
    from astropy.io import fits
except Exception:
    sys.stderr.write("Astropy required (inside venv): python -m pip install 'numpy<2' astropy==5.3.4\n")
    sys.exit(1)

# -------- wavelength builders --------
def wl_from_coeff01(hdr, n):
    if "COEFF0" in hdr and "COEFF1" in hdr:
        i = np.arange(n, dtype=np.float64)
        return 10.0 ** (hdr["COEFF0"] + hdr["COEFF1"] * i)  # Å
    return None

def wl_from_linear(hdr, n):
    # Support CRVAL1/CDELT1 with optional CRPIX1; CD1_1 alternative
    if "CRVAL1" in hdr and ("CDELT1" in hdr or "CD1_1" in hdr):
        crval = float(hdr["CRVAL1"])
        cdelt = float(hdr.get("CDELT1", hdr.get("CD1_1", 0.0)))
        crpix = float(hdr.get("CRPIX1", 1.0))
        # pixel index in FITS is 1-based, numpy is 0-based -> i+1
        i = np.arange(n, dtype=np.float64) + 1.0
        lam = crval + (i - crpix) * cdelt
        # Rare: log sampling hint
        if str(hdr.get("DC-FLAG", "")).strip() == "1" and lam.min() < 50:
            lam = 10.0 ** lam
        return lam
    return None

def wl_from_start_step(hdr, n):
    # A few files use custom keywords
    keys = {k.upper(): k for k in hdr.keys()}
    if "WSTART" in keys and "WSTEP" in keys:
        w0 = float(hdr[keys["WSTART"]]); dw = float(hdr[keys["WSTEP"]])
        i = np.arange(n, dtype=np.float64)
        return w0 + i * dw
    return None

def wavelengths_from_header(hdr, n):
    for fn in (wl_from_coeff01, wl_from_linear, wl_from_start_step):
        lam = fn(hdr, n)
        if lam is not None:
            return lam
    return None

# -------- readers --------
def first_table_wl_flux(ext):
    cols = {c.name.upper(): c.name for c in ext.columns}
    wname = next((cols[k] for k in ("WAVE","WAVELENGTH","LAMBDA","WL","LOGLAM") if k in cols), None)
    fname = next((cols[k] for k in ("FLUX","FLAM","F_LAMBDA","FNU","SPEC") if k in cols), None)
    if not (wname and fname):
        return None
    wl = np.asarray(ext.data[wname], dtype=np.float64)
    fx = np.asarray(ext.data[fname], dtype=np.float64)
    # If LOGLAM provided (log10 Å), convert
    if wname.upper() == "LOGLAM":
        wl = 10.0 ** wl
    return wl, fx

def read_one(path: Path, to_nm: bool, debug: bool = False):
    try:
        with fits.open(path, memmap=True) as hdul:
            # 1) Try any IMAGE HDU (including primary) that contains 1-D data
            image_candidates = []
            for h in hdul:
                if getattr(h, "data", None) is None:
                    continue
                arr = np.asarray(h.data)
                if arr.size >= 10 and arr.ndim >= 1:
                    arr = arr.reshape(-1)
                    image_candidates.append((h, arr))
            for h, arr in image_candidates:
                wl = wavelengths_from_header(h.header, arr.size)
                if wl is None:
                    # try primary header as source of WCS if not the same
                    wl = wavelengths_from_header(hdul[0].header, arr.size)
                if wl is None:
                    continue
                wl = wl.astype(np.float64); fx = arr.astype(np.float64)
                good = np.isfinite(wl) & np.isfinite(fx)
                wl, fx = wl[good], fx[good]
                if wl.size < 10:  # too short
                    continue
                order = np.argsort(wl)
                wl, fx = wl[order], fx[order]
                if to_nm:
                    wl = wl / 10.0  # Å -> nm
                return wl, fx

            # 2) Fall back: first TABLE HDU with wavelength/flux columns
            for ext in hdul[1:]:
                if getattr(ext, "data", None) is not None and hasattr(ext, "columns"):
                    res = first_table_wl_flux(ext)
                    if res:
                        wl, fx = res
                        wl = wl.astype(np.float64); fx = fx.astype(np.float64)
                        good = np.isfinite(wl) & np.isfinite(fx)
                        wl, fx = wl[good], fx[good]
                        if wl.size < 10:
                            continue
                        order = np.argsort(wl)
                        wl, fx = wl[order], fx[order]
                        if to_nm:
                            wl = wl / 10.0
                        return wl, fx
    except Exception as e:
        if debug:
            sys.stderr.write(f"Read error {path.name}: {e}\n")
        return None

    if debug:
        sys.stderr.write(f"No wavelength mapping found in headers/tables: {path.name}\n")
    return None

# -------- filename parsing / selection --------
STEM_RE = re.compile(r"""
    ^
    (?P<class>[OBAFGKMLTY])
    (?P<sub>\d)?
    (?:_(?P<metal>[+\-]\d(?:\.\d)?))?
    _?(?P<lum>Dwarf|Giant|Subgiant|Supergiant)?
    $
""", re.IGNORECASE | re.VERBOSE)

def parse_stem(stem: str):
    m = STEM_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    d["class"] = (d["class"] or "").upper()
    d["sub"]   = d["sub"] or ""
    d["lum"]   = d["lum"] or ""
    d["metal"] = d["metal"] or ""
    return d

METAL_PRIORITY = ["+0.0", "-0.5", "+0.5", "-1.0", "+0.3", "-0.3"]
def prefer_a_over_b(ma, mb):
    ia = METAL_PRIORITY.index(ma) if ma in METAL_PRIORITY else 999
    ib = METAL_PRIORITY.index(mb) if mb in METAL_PRIORITY else 999
    return ia < ib

# -------- main --------
def main():
    ap = argparse.ArgumentParser(
        description="Build JSON for GIANT spectra (FGKM default), robust header handling."
    )
    ap.add_argument("--classes", default="FGKM", help="Spectral classes to include. Default: FGKM")
    ap.add_argument("--lum", default="Giant", help="Luminosity class to include. Default: Giant")
    ap.add_argument("--keep-all-metal", action="store_true",
                    help="Keep all metallicities for each subtype.")
    ap.add_argument("--visible-only", action="store_true",
                    help="Crop to 380–780 nm (or 3800–7800 Å if --angstrom).")
    ap.add_argument("--angstrom", action="store_true",
                    help="Output wavelengths in Å instead of nm.")
    ap.add_argument("--debug", action="store_true", help="Verbose skip reasons.")
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT), help=f"Output JSON path. Default: {DEFAULT_OUT}")
    args = ap.parse_args()

    src = next((d for d in SRC_DIRS if d.exists()), None)
    if not src:
        sys.stderr.write("No FITS folder found (looked for ApjS230_16_FITS, ApJS230_16_FITS, StarSpectra_*).\n")
        sys.exit(1)

    wanted = set(c.upper() for c in args.classes)
    files = sorted(src.glob("*.fits"))
    if not files:
        sys.stderr.write(f"No FITS files in {src}\n"); sys.exit(1)

    # Select giants and dedupe metallicity per subtype unless keep-all
    buckets = {}  # (class, sub, lum) -> (metal, Path)
    selected = {}

    for p in files:
        stem = p.stem
        meta = parse_stem(stem)
        if not meta: 
            if args.debug: sys.stderr.write(f"Name not parsed: {p.name}\n")
            continue
        if meta["lum"].lower() != args.lum.lower(): 
            continue
        if meta["class"] not in wanted: 
            continue

        cls, sub, lum = meta["class"], meta["sub"], meta["lum"]
        metal = meta["metal"] or "UNK"

        if args.keep_all_metal:
            selected[stem] = p
        else:
            root = (cls, sub, lum)
            prev = buckets.get(root)
            if prev is None or prefer_a_over_b(metal, prev[0]):
                buckets[root] = (metal, p)

    if not args.keep_all_metal:
        for (cls, sub, lum), (metal, path) in buckets.items():
            key = f"{cls}{sub}"
            if metal != "UNK":
                key += f"_{metal}"
            key += f"_{lum}"
            selected[key] = path

    if not selected:
        sys.stderr.write("No matching spectra after filters.\n"); sys.exit(1)

    to_nm = not args.angstrom
    out = {}
    for key, path in selected.items():
        res = read_one(path, to_nm=to_nm, debug=args.debug)
        if not res:
            if args.debug:
                sys.stderr.write(f"Skipped (unreadable): {path.name}\n")
            continue
        wl, fx = res
        if args.visible_only:
            lo, hi = (380.0, 780.0) if to_nm else (3800.0, 7800.0)
            m = (wl >= lo) & (wl <= hi)
            wl, fx = wl[m], fx[m]
            if wl.size < 10:
                if args.debug: sys.stderr.write(f"Skipped (no visible-band data): {path.name}\n")
                continue
        out[key] = {"wavelength": wl.tolist(), "flux": fx.tolist()}

    if not out:
        sys.stderr.write("No spectra extracted after reading/cropping.\n"); sys.exit(1)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out_path} with {len(out)} entries")
    print("Included keys:")
    for k in sorted(out.keys()):
        print(" ", k)

if __name__ == "__main__":
    main()
