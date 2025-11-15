#!/usr/bin/env python3
import json, math, sys
from pathlib import Path
import numpy as np

SRC_DIRS = [Path("StarSpectra-FITS"), Path("StarSpectra_FITS")]
OUT_JSON = Path("spectra.json")

try:
    from astropy.io import fits
except Exception:
    sys.stderr.write("Astropy required. Example: pip install 'numpy<2' astropy==5.3.4\n")
    sys.exit(1)

def wavelengths_from_header(hdr, n):
    # SDSS/BOSS style: log10(Å) = COEFF0 + COEFF1 * pix
    if "COEFF0" in hdr and "COEFF1" in hdr:
        i = np.arange(n, dtype=np.float64)
        lam = 10.0 ** (hdr["COEFF0"] + hdr["COEFF1"] * i)  # Å
        return lam
    # Linear dispersion: λ = CRVAL1 + CDELT1 * pix (unit often Å)
    if "CRVAL1" in hdr and "CDELT1" in hdr:
        i = np.arange(n, dtype=np.float64)
        lam = hdr["CRVAL1"] + hdr["CDELT1"] * i
        # If header says values are log bins (rare), exponentiate
        if str(hdr.get("DC-FLAG", "")).strip() == "1" and lam.min() < 50:
            lam = 10.0 ** lam
        return lam
    return None

def read_one(path: Path):
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[0]
        data = np.asarray(hdu.data, dtype=np.float64) if hdu.data is not None else None
        if data is None or data.ndim == 0:
            return None
        if data.ndim > 1:
            # pick first row if 2-D [npix] or [1,npix]
            data = data.reshape(-1)
        wl = wavelengths_from_header(hdu.header, data.size)
        if wl is None:
            # try first table HDU as last resort
            for ext in hdul[1:]:
                if getattr(ext, "data", None) is not None and hasattr(ext.data, "columns"):
                    cols = {c.name.upper(): c.name for c in ext.columns}
                    wname = next((cols[k] for k in ("WAVE","WAVELENGTH","LAMBDA","WL") if k in cols), None)
                    fname = next((cols[k] for k in ("FLUX","FLAM","F_LAMBDA","FNU","SPEC") if k in cols), None)
                    if wname and fname:
                        wl = np.asarray(ext.data[wname], dtype=np.float64)
                        data = np.asarray(ext.data[fname], dtype=np.float64)
                        break
        if wl is None:
            return None

        # sanitize
        good = np.isfinite(wl) & np.isfinite(data)
        wl = wl[good]; fx = data[good]
        # keep only sensible range
        order = np.argsort(wl)
        wl = wl[order]; fx = fx[order]
        if wl.size < 10:
            return None
        return {"wavelength": wl.tolist(), "flux": fx.tolist()}

def main():
    src = next((d for d in SRC_DIRS if d.exists()), None)
    if not src:
        sys.stderr.write("No FITS folder found: StarSpectra-FITS or StarSpectra_FITS\n")
        sys.exit(1)
    out = {}
    for p in sorted(src.glob("*.fits")):
        rec = read_one(p)
        if rec:
            out[p.stem] = rec
        else:
            sys.stderr.write(f"Skipped (no usable primary/headers): {p.name}\n")
    if not out:
        sys.stderr.write("No spectra extracted.\n")
        sys.exit(1)
    OUT_JSON.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT_JSON} with {len(out)} entries")

if __name__ == "__main__":
    main()
