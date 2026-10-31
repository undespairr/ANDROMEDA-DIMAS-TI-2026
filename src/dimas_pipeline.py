# # DIMAS-TI 2026 — Divisi Penambangan Data
# **Identifikasi Ketidaksesuaian Kesiapan Digital Sekolah dan Capaian Numerasi untuk Pemetaan Prioritas Pendidikan di Indonesia Berbasis Data Mining**

import os, re, sys, json, hashlib, itertools, warnings, textwrap, subprocess, shutil, unicodedata, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from scipy import stats
from scipy.cluster import hierarchy as sch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.decomposition import PCA

try:
    from IPython.display import display
except Exception:          
    display = print

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42                                 
np.random.seed(SEED)
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:                               
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "geopandas", "folium"], check=False)

try:
    import geopandas as gpd
    import folium
    HAS_GEO = True
except Exception:
    HAS_GEO = False

ROOT = Path(os.environ.get("DIMAS_OUTPUT_ROOT", "."))
OUT = ROOT / "output_dimas_ti"
FIG = OUT / "figures"
GEO = OUT / "data_geo"
INPUT_DIR = Path(os.environ.get("DIMAS_INPUT_DIR", "data_input"))
CACHE_DIR = os.environ.get("DIMAS_CACHE")  
for d in (OUT, FIG, GEO):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.labelsize": 10, "font.family": "DejaVu Sans"})

SUMBER_URL = {
    "A": "https://data.kemendikdasmen.go.id/dataset/p/prioritas-data/persentase-sekolah-dengan-akses-komputer-untuk-pengajaran-2025",
    "B": "https://data.kemendikdasmen.go.id/dataset/p/prioritas-data/persentase-sekolah-dengan-akses-internet-untuk-pengajaran-2025-2",
    "C": "https://data.kemendikdasmen.go.id/dataset/p/prioritas-data/persentase-satuan-pendidikan-yang-mencapai-standar-kompetensi-minimum-dalam-asesmen-kompetensi-tingkat-nasional-a-literasi-membaca-dan-b-numerasi-2025-indonesia",
    "D": "https://data.kemendikdasmen.go.id/dataset/p/rapor-pendidikan-indonesia/data-rapor-pendidikan-indonesia-2025",
}
NAMA_DATASET = {"A": "Akses komputer untuk pengajaran", "B": "Akses internet untuk pengajaran",
                "C": "Satdik mencapai kompetensi minimum (Asesmen Nasional)", "D": "Rapor Pendidikan Indonesia 2025"}

def save_fig(fig, name):
    """Simpan grafik ke output_dimas_ti/figures dan tampilkan."""
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    return path

def save_excel(path, sheets, index=False, notes=None):
    """Tulis beberapa DataFrame ke satu XLSX dengan format rapi (header tebal, lebar kolom otomatis, freeze panes)."""
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        if notes:
            pd.DataFrame({"Catatan": notes}).to_excel(xw, sheet_name="README", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=index)
        for ws in xw.book.worksheets:
            for c in ws[1]:
                c.font = Font(name="Arial", bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F4E79")
                c.alignment = Alignment(wrap_text=True, vertical="center")
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max((len(str(c.value)) if c.value is not None else 0) for c in col[:200])
                ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(10, width + 2), 70)
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.font = Font(name="Arial", size=10)
    return path

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

print("Python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__,
      "| GeoPandas tersedia:", HAS_GEO, "| Colab:", IN_COLAB)

if IN_COLAB and not list(INPUT_DIR.glob("*.xlsx")):
    from google.colab import files
    INPUT_DIR.mkdir(exist_ok=True)
    _up = files.upload()
    for n, b in _up.items():
        (INPUT_DIR / n).write_bytes(b)

def identify_datasets(input_dir):
    """Mengenali Dataset A/B/C/D dari judul pada sheet pertama (bukan dari nama berkas)."""
    found = {}
    for p in sorted(Path(input_dir).glob("*.xlsx")):
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        title = " ".join(str(c) for row in ws.iter_rows(min_row=1, max_row=3, values_only=True)
                         for c in row if isinstance(c, str)).lower()
        wb.close()
        if "akses komputer" in title: key = "A"
        elif "akses internet" in title: key = "B"
        elif "standar kompetensi minimum" in title: key = "C"
        elif "metadata seluruh indikator" in title or "rapor pendidikan" in title: key = "D"
        else: key = None
        if key is None:
            print(f"[PERINGATAN] Berkas tidak dikenali dan dilewati: {p.name}")
        elif key in found:
            print(f"[PERINGATAN] Dataset {key} ganda ({found[key].name} vs {p.name}); dipakai yang pertama.")
        else:
            found[key] = p
    missing = [k for k in "ABCD" if k not in found]
    if missing:
        raise FileNotFoundError(
            f"Dataset {missing} belum ditemukan di '{input_dir}'. Unggah berkas XLSX yang sesuai (lihat tautan sumber) "
            "lalu jalankan ulang sel ini. Pipeline tidak membuat data sintetis sebagai pengganti.")
    return found

FILES = identify_datasets(INPUT_DIR)
for k, p in FILES.items():
    print(f"Dataset {k}: {p.name}  ({p.stat().st_size/1024:.0f} KB)")
CHECKSUM = {k: sha256(p) for k, p in FILES.items()}   

# %% Inventaris berkas A, B, C 
def inventory_small(files):
    rows, previews = [], {}
    for k in ("A", "B", "C"):
        wb = openpyxl.load_workbook(files[k], data_only=True)
        for ws in wb.worksheets:
            previews[(k, ws.title)] = [list(r[:8]) for r in ws.iter_rows(min_row=1, max_row=12, values_only=True)]
            rows.append(dict(dataset=k, file=files[k].name, sheet=ws.title, n_baris=ws.max_row, n_kolom=ws.max_column,
                             n_kolom_nilai=None, tipe_isi="tabel data / metadata"))
    return pd.DataFrame(rows), previews

INV_ABC, PREV = inventory_small(FILES)

# %% Dataset D (Rapor Pendidikan): parse per-sheet, cek struktur, ekstrak label indikator pendukung
RAPOR_TARGET = ["A.1", "A.2", "D.1", "D.1.3", "D.2.3", "D.21.4", "E.7.4"]   # kandidat indikator pendukung
RAPOR_JENIS = {"SD Umum": "SD", "SMP Umum": "SMP", "SMA Umum": "SMA", "SMK Umum": "SMK"}
RAPOR_CODE_RE = re.compile(r"^([A-E]\.\d+(?:\.\d+|\.skor)?)\s")

def _pct_to_float(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    m = re.match(r"^\s*(-?\d+(?:[.,]\d+)?)\s*%?\s*$", str(x))
    return float(m.group(1).replace(",", ".")) if m else np.nan

def parse_rapor(path):
    """Membaca seluruh sheet Rapor Pendidikan satu per satu (hemat memori). Tidak mengasumsikan jumlah sheet."""
    xl = pd.ExcelFile(path)
    inv, lab, prev, nas_rows = [], [], {}, []
    meta_df = xl.parse("Metadata", header=3) if "Metadata" in xl.sheet_names else pd.DataFrame()
    ref_sig = None
    for sh in xl.sheet_names:
        raw = xl.parse(sh, header=None)
        row = dict(dataset="D", file=path.name, sheet=sh, n_baris=raw.shape[0], n_kolom=raw.shape[1])
        if sh == "Metadata":
            row.update(tipe_isi="metadata indikator", n_kolom_nilai=0)
            prev[sh] = raw.iloc[:12, :8].values.tolist()
            inv.append(row); continue
        names, sub = raw.iloc[6].ffill(), raw.iloc[7]
        sig = hashlib.md5("|".join(names.fillna("").astype(str) + "/" + sub.fillna("").astype(str)).encode()).hexdigest()
        n_nilai = int(sub.astype(str).str.startswith("Nilai Capaian").sum())
        n_label = int(sub.astype(str).str.startswith("Label Capaian").sum())
        is_nas = (sh == "Nasional")
        if not is_nas and ref_sig is None:
            ref_sig = sig
        row.update(tipe_isi="nasional (nilai+label)" if is_nas else "provinsi + kab/kota (label saja)",
                   n_kolom_nilai=n_nilai, n_kolom_label=n_label,
                   struktur_sama_dgn_sheet_provinsi_pertama=(None if is_nas else sig == ref_sig),
                   n_kab_kota=(None if is_nas else int(raw.iloc[8:, 1].replace("-", np.nan).nunique())))
        inv.append(row)
        if sh in ("Nasional",) or (not is_nas and len(prev) < 3):
            prev[sh] = raw.iloc[:12, :8].values.tolist()
        colmap, valmap = {}, {}
        for j in range(raw.shape[1]):
            m = RAPOR_CODE_RE.match(str(names[j]))
            if not m or m.group(1) not in RAPOR_TARGET:
                continue
            t = str(sub[j])
            if t.startswith("Label") and m.group(1) not in colmap:      # blok pertama = pendidikan dasar-menengah
                colmap[m.group(1)] = j
            if t.startswith("Nilai") and m.group(1) not in valmap:
                valmap[m.group(1)] = j
        d = raw.iloc[8:]
        jen_col = 1 if is_nas else 2
        d = d[d[jen_col].isin(RAPOR_JENIS)]
        for k, j in colmap.items():
            tmp = pd.DataFrame({"sheet": sh, "prov_raw": d[0].values, "kab": ("-" if is_nas else d[1].values),
                                "jenis": d[jen_col].values, "kode_ind": k, "label": d[j].values})
            lab.append(tmp)
        if is_nas:
            for k, j in valmap.items():
                for _, r in d.iterrows():
                    nas_rows.append(dict(kode_ind=k, jenis=r[1], nilai=_pct_to_float(r[j]), label=r[colmap[k]] if k in colmap else None))
    labels = pd.concat(lab, ignore_index=True)
    return dict(inv=pd.DataFrame(inv), labels=labels, nasional=pd.DataFrame(nas_rows), prev=prev, meta=meta_df)

_cache = Path(CACHE_DIR) / "rapor.pkl" if CACHE_DIR else None
if _cache is not None and _cache.exists():
    RAPOR = pd.read_pickle(_cache)
else:
    t0 = time.time()
    RAPOR = parse_rapor(FILES["D"])
    print(f"Parse Dataset D selesai dalam {time.time()-t0:.0f} detik")
    if _cache is not None:
        pd.to_pickle(RAPOR, _cache)

INV = pd.concat([INV_ABC, RAPOR["inv"]], ignore_index=True)
for k, v in RAPOR["prev"].items():
    PREV[("D", k)] = v

print("Jumlah sheet per dataset:\n", INV.groupby("dataset").sheet.count().to_string())
for (k, sh), prev in PREV.items():
    r = INV[(INV.dataset == k) & (INV.sheet == sh)].iloc[0]
    print(f"\n=== Dataset {k} | sheet '{sh}' | {r.n_baris} baris x {r.n_kolom} kolom ===")
    shown = 0
    for row in prev:
        if any(c is not None and not (isinstance(c, float) and np.isnan(c)) for c in row):
            print("  ", [("" if (c is None or (isinstance(c, float) and np.isnan(c))) else str(c)[:26]) for c in row])
            shown += 1
        if shown >= 6: break
prov_sheets = INV[(INV.dataset == "D") & INV.tipe_isi.str.startswith("provinsi")]
print(f"\nDataset D: {len(prov_sheets)} sheet provinsi; struktur identik dengan sheet pertama: {bool(prov_sheets.struktur_sama_dgn_sheet_provinsi_pertama.all())}; "
      f"kolom 'Nilai Capaian' pada sheet provinsi: {int(prov_sheets.n_kolom_nilai.sum())}; pada sheet Nasional: {int(INV[(INV.dataset=='D')&(INV.sheet=='Nasional')].n_kolom_nilai.iloc[0])}")

JENJANG_RE = re.compile(r"\(([A-Za-z]+)\)")

def read_digital_access(path, indikator):
    """Format panjang apa adanya (tanpa mengubah skala): kode, wilayah, jenjang, status, nilai_raw."""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Data"] if "Data" in wb.sheetnames else wb.worksheets[0]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    title = next((str(c) for r in rows[:2] for c in r if isinstance(c, str)), "")
    h = next(i for i, r in enumerate(rows) if any(isinstance(c, str) and "kode wilayah" in c.lower() for c in r))
    hdr = rows[h]
    c_kode = next(i for i, c in enumerate(hdr) if isinstance(c, str) and "kode" in c.lower())
    c_wil = next(i for i, c in enumerate(hdr) if isinstance(c, str) and c.strip().lower() == "wilayah")
    jen = pd.Series(hdr).ffill()
    sta = pd.Series(rows[h + 1])
    cols = [i for i in range(len(hdr)) if sta[i] in ("Negeri", "Swasta", "Total")]
    out = []
    for r in rows[h + 2:]:
        if r[c_wil] is None:
            continue
        for c in cols:
            m = JENJANG_RE.search(str(jen[c]))
            out.append(dict(indikator=indikator, kode_raw=r[c_kode], wilayah_raw=r[c_wil], jenjang_raw=str(jen[c]),
                            jenjang=m.group(1) if m else str(jen[c]), status=sta[c], nilai_raw=r[c]))
    df = pd.DataFrame(out)
    per = re.search(r"Tahun Ajaran\s+([\d/]+)", title)
    meta = {}
    if "Metadata" in wb.sheetnames:
        for r in wb["Metadata"].iter_rows(values_only=True):
            if r[0] and r[1]:
                meta[str(r[0]).strip()] = str(r[1]).strip()
    return df, dict(judul=title, periode="TA " + per.group(1) if per else "tidak terdeteksi", metadata=meta)

def read_an(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Data"] if "Data" in wb.sheetnames else wb.worksheets[0]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    title = next((str(c) for r in rows[:2] for c in r if isinstance(c, str)), "")
    h = next(i for i, r in enumerate(rows) if any(isinstance(c, str) and c.strip().lower() == "periode" for c in r))
    hdr, sub = rows[h], rows[h + 1]
    ci = {str(c).strip().lower(): i for i, c in enumerate(hdr) if c is not None}
    c_per, c_kode = ci["periode"], ci["kode_prov"]
    c_prov = next(i for i, c in enumerate(hdr) if isinstance(c, str) and c.strip().lower() == "provinsi")
    c_jen = next(i for i, c in enumerate(hdr) if isinstance(c, str) and "pendidikan" in c.lower())
    c_lit = next(i for i, c in enumerate(sub) if isinstance(c, str) and c.strip().lower() == "literasi")
    c_num = next(i for i, c in enumerate(sub) if isinstance(c, str) and c.strip().lower() == "numerasi")
    head_text = " ".join(str(c) for c in hdr if isinstance(c, str))
    thr = re.search(r">=\s*(\d+)\s*%", head_text)
    out = []
    for r in rows[h + 2:]:
        if r[c_prov] is None:
            continue
        out.append(dict(periode=r[c_per], kode_raw=r[c_kode], wilayah_raw=r[c_prov], jenjang=r[c_jen],
                        literasi_raw=r[c_lit], numerasi_raw=r[c_num]))
    meta = {}
    if "Metadata" in wb.sheetnames:
        for r in wb["Metadata"].iter_rows(values_only=True):
            if r[0] and r[1]:
                meta[str(r[0]).strip()] = str(r[1]).strip()
    return pd.DataFrame(out), dict(judul=title, ambang_header=int(thr.group(1)) if thr else None, metadata=meta)

A_raw, A_meta = read_digital_access(FILES["A"], "akses_komputer")
B_raw, B_meta = read_digital_access(FILES["B"], "akses_internet")
C_raw, C_meta = read_an(FILES["C"])
print(f"A: {A_raw.shape} | periode: {A_meta['periode']}\nB: {B_raw.shape} | periode: {B_meta['periode']}\nC: {C_raw.shape} | periode: {sorted(C_raw.periode.dropna().unique())} | ambang pada header: >= {C_meta['ambang_header']}%")
print("\nTipe data (setelah dibaca):")
for nm, d in [("A", A_raw), ("B", B_raw), ("C", C_raw)]:
    print(f"--- {nm}"); print(d.dtypes.astype(str).to_string())
print("\nLima baris pertama masing-masing dataset (format panjang):")
for nm, d in [("A", A_raw), ("B", B_raw), ("C", C_raw)]:
    print(f"--- {nm}"); display(d.head(5))

# %% Normalisasi nama/kode wilayah dan klasifikasi baris (provinsi / agregat nasional / luar negeri)
def norm_name(s):
    """Nama wilayah dinormalkan agar variasi penulisan (Prov./Provinsi, D.K.I., Kep., Daerah Istimewa) tidak menghalangi pencocokan."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bprovinsi\b|\bprov\b\.?", " ", s)
    s = s.replace("daerah istimewa", "di").replace("d.k.i.", "dki").replace("d.i.", "di").replace("kep.", "kepulauan")
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def classify_row(kode, wilayah):
    w = str(wilayah).strip().lower()
    if w in ("indonesia", "nasional"):
        return "agregat_nasional"
    if "luar negeri" in w:
        return "luar_negeri"
    return "provinsi"

def to_int_code(x):
    try:
        return int(float(x))
    except Exception:
        return np.nan

for _d in (A_raw, B_raw, C_raw):
    _d["tipe_baris"] = [classify_row(k, w) for k, w in zip(_d.kode_raw, _d.wilayah_raw)]
    _d["kode"] = _d.kode_raw.map(to_int_code)
    _d["nama_norm"] = _d.wilayah_raw.map(norm_name)

# %% Audit kualitas data (duplikasi, nilai hilang, tipe, skala, rentang, konsistensi internal, metadata)
QC = []
def qc(dataset, check, result, n_flag="", detail=""):
    QC.append(dict(dataset=dataset, pemeriksaan=check, hasil=result, jumlah_temuan=n_flag, rincian=detail))

def type_profile(s):
    t = s.map(lambda v: type(v).__name__).value_counts()
    return ", ".join(f"{k}:{v}" for k, v in t.items())

def parse_thresholds(text):
    """Cari ambang persen siswa ('NN% peserta didiknya') per butir bernomor pada teks metadata."""
    out = {}
    for item in re.split(r"\n\s*\n", text or ""):
        m = re.search(r"(\d+)\s*%\s*peserta didik", item)
        key = "literasi" if "literasi" in item.lower() else ("numerasi" if "numerasi" in item.lower() else None)
        if m and key:
            out[key] = int(m.group(1))
    return out

# --- A dan B
for nm, d, meta in (("A", A_raw, A_meta), ("B", B_raw, B_meta)):
    qc(nm, "Profil tipe nilai", "OK" if d.nilai_raw.map(lambda v: isinstance(v, (int, float))).all() else "ADA NILAI NON-NUMERIK",
       "", type_profile(d.nilai_raw))
    dup = d.duplicated(["kode_raw", "wilayah_raw", "jenjang", "status"]).sum()
    qc(nm, "Duplikasi kunci (wilayah, jenjang, status)", "OK" if dup == 0 else "ADA DUPLIKASI", int(dup))
    miss = int(d.nilai_raw.isna().sum())
    qc(nm, "Nilai hilang", "OK" if miss == 0 else "ADA NILAI HILANG", miss, f"dari {len(d)} sel")
    mx = d.nilai_raw.max()
    skala = "fraksi 0–1" if mx <= 1.0 + 1e-9 else "persen 0–100"
    qc(nm, "Skala nilai terdeteksi", skala, "", f"min={d.nilai_raw.min():.4f}, maks={mx:.4f}")
    claim = meta["metadata"].get("Mekanisme perhitungan", "")
    if skala.startswith("fraksi") and "dikali 100" in claim:
        qc(nm, "Konsistensi metadata vs skala", "TIDAK SESUAI", 1,
           "Metadata menyatakan hasil 'dikali 100' (persen), tetapi nilai tersimpan sebagai fraksi 0–1 → dikonversi ×100 saat preprocessing.")
    lo = (d.nilai_raw * (100 if skala.startswith("fraksi") else 1) < 0).sum()
    hi = (d.nilai_raw * (100 if skala.startswith("fraksi") else 1) > 100).sum()
    qc(nm, "Nilai di luar rentang 0–100 (setelah penskalaan)", "OK" if lo + hi == 0 else "ADA", int(lo + hi))
    w = d[(d.tipe_baris == "provinsi")].pivot_table(index=["kode", "jenjang"], columns="status", values="nilai_raw", aggfunc="first")
    bad = w[(w.Total < w[["Negeri", "Swasta"]].min(axis=1) - 1e-9) | (w.Total > w[["Negeri", "Swasta"]].max(axis=1) + 1e-9)]
    qc(nm, "Total berada di antara Negeri dan Swasta (sifat rata-rata tertimbang)", "OK" if len(bad) == 0 else "ADA ANOMALI", len(bad),
       f"dari {len(w)} kombinasi provinsi×jenjang")
    non_prov = d[d.tipe_baris != "provinsi"].wilayah_raw.unique().tolist()
    qc(nm, "Baris bukan provinsi", "DIKECUALIKAN dari analisis", len(non_prov), ", ".join(non_prov))
    nz = (d[(d.status == "Total") & (d.tipe_baris == "provinsi")].nilai_raw == 0).sum()
    qc(nm, "Nilai nol pada 'Total' provinsi", "PERLU DICATAT", int(nz),
       "Nol/satu murni biasanya berasal dari jumlah sekolah kecil (persentase tidak stabil), terutama TK/SLB dan provinsi baru.")

# --- C
dupC = C_raw.duplicated(["kode_raw", "wilayah_raw", "jenjang"]).sum()
qc("C", "Profil tipe nilai", "OK", "", f"literasi: {type_profile(C_raw.literasi_raw)} | numerasi: {type_profile(C_raw.numerasi_raw)}")
qc("C", "Duplikasi kunci (wilayah, jenjang)", "OK" if dupC == 0 else "ADA DUPLIKASI", int(dupC))
qc("C", "Nilai hilang (literasi/numerasi)", "OK" if C_raw[["literasi_raw", "numerasi_raw"]].isna().sum().sum() == 0 else "ADA",
   int(C_raw[["literasi_raw", "numerasi_raw"]].isna().sum().sum()))
mxC = max(C_raw.literasi_raw.max(), C_raw.numerasi_raw.max())
qc("C", "Skala nilai terdeteksi", "persen 0–100" if mxC > 1 else "fraksi 0–1", "", f"maks={mxC:.2f}")
qc("C", "Nilai di luar 0–100", "OK" if ((C_raw[["literasi_raw", "numerasi_raw"]] < 0) | (C_raw[["literasi_raw", "numerasi_raw"]] > 100)).sum().sum() == 0 else "ADA",
   int(((C_raw[["literasi_raw", "numerasi_raw"]] < 0) | (C_raw[["literasi_raw", "numerasi_raw"]] > 100)).sum().sum()))
Cp = C_raw[C_raw.tipe_baris == "provinsi"]
JC = {"SD": "SD/sederajat", "SMP": "SMP/sederajat", "SMA_SMK": "SMA/SMK/sederajat"}
for ind in ("literasi_raw", "numerasi_raw"):
    wC = Cp.pivot(index="kode", columns="jenjang", values=ind)
    three = wC[list(JC.values())]
    out = wC[(wC.SEMUA < three.min(axis=1) - 1e-9) | (wC.SEMUA > three.max(axis=1) + 1e-9)]
    qc("C", f"'SEMUA' berada dalam rentang SD–SMP–SMA/SMK ({ind[:-4]})", "OK" if len(out) == 0 else "ADA ANOMALI", len(out),
       "Konsisten dengan SEMUA sebagai rata-rata tertimbang antar jenjang.")
# bobot implisit SEMUA (diagnostik): menyelesaikan SEMUA = w1*SD + w2*SMP + w3*SMA/SMK dari literasi & numerasi dengan sum(w)=1
wl = Cp.pivot(index="kode", columns="jenjang", values="literasi_raw"); wn = Cp.pivot(index="kode", columns="jenjang", values="numerasi_raw")
_W = []
for k in wn.index:
    M = np.array([[wl.loc[k, c] for c in JC.values()], [wn.loc[k, c] for c in JC.values()], [1, 1, 1]])
    y = np.array([wl.loc[k, "SEMUA"], wn.loc[k, "SEMUA"], 1.0])
    try: _W.append([k, *np.linalg.solve(M, y)])
    except np.linalg.LinAlgError: _W.append([k, np.nan, np.nan, np.nan])
W_IMPL = pd.DataFrame(_W, columns=["kode", "w_SD", "w_SMP", "w_SMA_SMK"]).set_index("kode")
neg = int((W_IMPL < 0).any(axis=1).sum())
qc("C", "Bobot implisit jenjang pada 'SEMUA' (diagnostik)", "VALID" if neg == 0 else "ADA BOBOT NEGATIF", neg,
   "Rata-rata bobot SD/SMP/SMA-SMK = " + "/".join(f"{W_IMPL[c].mean():.2f}" for c in W_IMPL.columns) +
   ". Dipakai hanya sebagai uji sensitivitas, bukan bobot utama.")
thr = parse_thresholds(C_meta["metadata"].get("Metode/Rumus Perhitungan", ""))
thr_def = parse_thresholds(C_meta["metadata"].get("Definisi", ""))
qc("C", "Konsistensi ambang kompetensi minimum (header vs metadata)",
   "TIDAK KONSISTEN" if (len(set(thr.values()) | set(thr_def.values()) | {C_meta['ambang_header']}) > 1) else "OK", "",
   f"Header: ≥{C_meta['ambang_header']}% siswa; Definisi: {thr_def}; Rumus: {thr}. Ambang numerasi pada rumus (70%) berbeda dari header/definisi (75%) → "
   "definisi operasional numerasi perlu dikonfirmasi ke penerbit data; dampaknya diuji lewat sensitivitas peringkat (angka tidak dapat dikoreksi dari berkas).")
nat = C_raw[C_raw.tipe_baris == "agregat_nasional"]
qc("C", "Baris agregat nasional", "DIKECUALIKAN dari analisis", len(nat), "Dipakai hanya sebagai pembanding.")

# --- D
RL = RAPOR["labels"]
RL_prov = RL[RL.kab == "-"]
qc("D", "Nilai numerik tersedia pada level provinsi/kab-kota?", "TIDAK (hanya label ordinal)",
   int(prov_sheets.n_kolom_nilai.sum()), f"Sheet Nasional: {int(INV[(INV.dataset=='D')&(INV.sheet=='Nasional')].n_kolom_nilai.iloc[0])} kolom nilai; 38 sheet provinsi: 0 kolom nilai.")
vocab = RL.groupby("kode_ind").label.value_counts().unstack(fill_value=0)
for k in RAPOR_TARGET:
    if k in vocab.index:
        tot = int(vocab.loc[k].sum()); tt = int(vocab.loc[k].get("Tidak Tersedia", 0))
        top = vocab.loc[k].drop("Tidak Tersedia", errors="ignore")
        share = top.max() / max(top.sum(), 1)
        status = "TIDAK INFORMATIF" if tt == tot else ("EFEK LANGIT-LANGIT (>90% satu label)" if share > 0.9 else "DAPAT DIPAKAI (ordinal)")
        qc("D", f"Ketersediaan & variasi label indikator {k}", status, tt, f"{tt}/{tot} 'Tidak Tersedia'; sebaran label: " + top.to_dict().__str__())

# %% Pemeriksaan kompatibilitas A–B–C: wilayah, periode, jenjang, definisi, populasi
def prov_table(d):
    return (d[d.tipe_baris == "provinsi"][["kode", "wilayah_raw", "nama_norm"]].drop_duplicates().sort_values("kode").reset_index(drop=True))
PA, PB, PC = prov_table(A_raw), prov_table(B_raw), prov_table(C_raw)
ka, kb, kc = set(PA.kode), set(PB.kode), set(PC.kode)
name_ab = PA.merge(PB, on="kode", suffixes=("_A", "_B")); name_ac = PA.merge(PC, on="kode", suffixes=("_A", "_C"))
n_name_mis = int((name_ab.nama_norm_A != name_ab.nama_norm_B).sum() + (name_ac.nama_norm_A != name_ac.nama_norm_C).sum())
KODE_CHECK = pd.DataFrame({
    "pemeriksaan": ["Jumlah kode provinsi A", "Jumlah kode provinsi B", "Jumlah kode provinsi C", "Irisan kode A∩B∩C",
                    "Kode di A tetapi tidak di C", "Kode di C tetapi tidak di A", "Nama (ternormalisasi) berbeda untuk kode yang sama"],
    "nilai": [len(ka), len(kb), len(kc), len(ka & kb & kc), sorted(ka - kc), sorted(kc - ka), n_name_mis]})
display(KODE_CHECK)

JENJANG_XWALK = pd.DataFrame([
    dict(jenjang_analisis="SD", jenjang_A_B="SD", jenjang_C="SD/sederajat", padanan="1:1", catatan="Kemungkinan perbedaan cakupan 'sederajat' (madrasah) — tidak dapat diverifikasi dari berkas."),
    dict(jenjang_analisis="SMP", jenjang_A_B="SMP", jenjang_C="SMP/sederajat", padanan="1:1", catatan="Idem."),
    dict(jenjang_analisis="SMA_SMK", jenjang_A_B="SMA dan SMK (terpisah)", jenjang_C="SMA/SMK/sederajat (gabungan)", padanan="2:1",
         catatan="Tanpa jumlah sekolah, SMA dan SMK dirata-ratakan tanpa bobot (primer); sensitivitas: SMA saja dan SMK saja."),
    dict(jenjang_analisis="(tidak dianalisis)", jenjang_A_B="TK", jenjang_C="—", padanan="tanpa padanan", catatan="Asesmen Nasional tidak mencakup TK."),
    dict(jenjang_analisis="(tidak dianalisis)", jenjang_A_B="SLB", jenjang_C="—", padanan="tanpa padanan", catatan="Tidak ada kategori SLB pada Dataset C."),
    dict(jenjang_analisis="(tidak dianalisis)", jenjang_A_B="— (tidak ada agregat semua jenjang)", jenjang_C="SEMUA", padanan="tanpa padanan langsung",
         catatan="A/B tidak menyediakan agregat provinsi; dikonstruksi dari jenjang (bobot sama; sensitivitas: bobot implisit C)."),
])
display(JENJANG_XWALK)

ISU = pd.DataFrame([
    dict(no=1, isu="Periode referensi tidak identik", dataset="A,B vs C",
         detail=f"A/B: {A_meta['periode']} ('Tahun Ajaran 2025/2026'); C: tahun '2025' (Asesmen Nasional). Berkas tidak menyatakan bulan/tahun ajaran pelaksanaan Asesmen Nasional sehingga kesesuaian waktu tidak dapat dipastikan.",
         dampak="Hubungan digital–numerasi dapat dipengaruhi selisih waktu pengukuran.", mitigasi="Diperlakukan sebagai potret lintas-seksi (cross-section) 2025; tidak ada klaim urutan waktu/kausal."),
    dict(no=2, isu="Penyebut (populasi) berbeda", dataset="A,B vs C",
         detail="A/B: persentase atas seluruh satuan pendidikan; C: persentase atas satdik peserta Asesmen Nasional. Cakupan madrasah ('sederajat') pada A/B tidak dinyatakan.",
         dampak="Provinsi dengan partisipasi/cakupan berbeda dapat tampak 'tidak sesuai' karena perbedaan populasi.", mitigasi="Dicatat sebagai keterbatasan; tidak dapat dikoreksi dari data yang ada."),
    dict(no=3, isu="Ambang kompetensi minimum numerasi tidak konsisten", dataset="C",
         detail=f"Header & definisi: ≥75% siswa; rumus numerasi: 70% (hasil parse: {thr}).", dampak="Definisi operasional numerasi belum pasti.",
         mitigasi="Diuji lewat sensitivitas (literasi sebagai pembanding, SEMUA vs komposit jenjang); konfirmasi ke penerbit data direkomendasikan."),
    dict(no=4, isu="Indikator C adalah proporsi sekolah (ambang), bukan rata-rata skor", dataset="C",
         detail="Nilai = % satdik dengan ≥ambang siswa mencapai kompetensi minimum; sensitif terhadap ambang dan banyak provinsi bernilai rendah (efek lantai).",
         dampak="Perbedaan kecil di dekat dasar distribusi dapat membesar saat distandarkan.", mitigasi="Standardisasi alternatif (rank, robust) dan bootstrap."),
    dict(no=5, isu="Definisi akses komputer = ≥5 komputer; akses internet = tersedia untuk pengajaran", dataset="A,B",
         detail="Ambang komputer ≥5 unit tidak mempertimbangkan ukuran sekolah/rasio murid-komputer; internet cenderung jenuh (dekat 100%) di Jawa-Bali.",
         dampak="Skala dua indikator berbeda (lantai vs langit-langit).", mitigasi="Standardisasi per jenjang; analisis terpisah dan gabungan dibandingkan."),
    dict(no=6, isu="Metadata A/B menyatakan 'dikali 100' tetapi nilai berskala 0–1", dataset="A,B",
         detail="Nilai tersimpan sebagai fraksi.", dampak="Salah skala jika tidak dideteksi.", mitigasi="Skala dideteksi otomatis dan dikonversi ×100."),
    dict(no=7, isu="Dataset D tanpa nilai numerik di provinsi/kab-kota", dataset="D",
         detail="Hanya label ordinal; nilai numerik hanya nasional.", dampak="Tidak dapat dipakai sebagai variabel kontinu inti.",
         mitigasi="Dipakai untuk validasi konvergen (label A.2 vs numerasi C) dan profil pendukung."),
    dict(no=8, isu="Indikator E.7.4 (Indeks Fasilitas TIK) hampir seluruhnya berlabel 'Baik'", dataset="D",
         detail="Tidak membedakan wilayah dan berbeda ketat dari 'akses ≥5 komputer'.", dampak="Tidak informatif sebagai pembeda.", mitigasi="Dikeluarkan dari analisis; dilaporkan."),
    dict(no=9, isu="SMA dan SMK terpisah pada A/B, tergabung pada C", dataset="A,B vs C", detail="Tidak ada jumlah sekolah untuk pembobotan.",
         dampak="Komposit SMA/SMK bergantung pada pembobotan.", mitigasi="Rata-rata tanpa bobot + sensitivitas SMA/SMK terpisah."),
    dict(no=10, isu="Persentase pada provinsi/jenjang kecil tidak stabil", dataset="A,B,C", detail="Nilai persis 0/1/0,5/0,333 muncul pada kategori kecil (mis. SLB, TK Papua).",
         dampak="Kebisingan sampling.", mitigasi="Hanya jenjang SD/SMP/SMA-SMK ('Total') yang dianalisis; bootstrap."),
])
display(ISU)

# %% Tabel audit ringkas dan laporan kelayakan integrasi
def n_miss(d, col): return int(d[col].isna().sum())
AUDIT = pd.DataFrame([
    dict(dataset="A", nama=NAMA_DATASET["A"], sumber=SUMBER_URL["A"], file=FILES["A"].name, unit_observasi="Provinsi × jenjang × status (Negeri/Swasta/Total)",
         periode=A_meta["periode"], indikator="% sekolah dengan akses komputer (≥5 unit) untuk pengajaran", skala_asli="fraksi 0–1",
         n_baris_data=len(A_raw), n_provinsi=len(ka), nilai_hilang=n_miss(A_raw, "nilai_raw"),
         status_integrasi="LAYAK (kunci: kode provinsi; jenjang SD/SMP/SMA/SMK)"),
    dict(dataset="B", nama=NAMA_DATASET["B"], sumber=SUMBER_URL["B"], file=FILES["B"].name, unit_observasi="Provinsi × jenjang × status (Negeri/Swasta/Total)",
         periode=B_meta["periode"], indikator="% sekolah dengan akses internet untuk pengajaran", skala_asli="fraksi 0–1",
         n_baris_data=len(B_raw), n_provinsi=len(kb), nilai_hilang=n_miss(B_raw, "nilai_raw"),
         status_integrasi="LAYAK (kunci: kode provinsi; jenjang SD/SMP/SMA/SMK)"),
    dict(dataset="C", nama=NAMA_DATASET["C"], sumber=SUMBER_URL["C"], file=FILES["C"].name, unit_observasi="Provinsi × jenjang (SD, SMP, SMA/SMK, SEMUA)",
         periode="Asesmen Nasional 2025 (bulan pelaksanaan tidak dinyatakan)", indikator=f"% satdik dengan ≥{C_meta['ambang_header']}% siswa mencapai kompetensi minimum (literasi, numerasi)",
         skala_asli="persen 0–100", n_baris_data=len(C_raw), n_provinsi=len(kc), nilai_hilang=int(C_raw[["literasi_raw", "numerasi_raw"]].isna().sum().sum()),
         status_integrasi="LAYAK DENGAN CATATAN (ambang 75% vs 70%; populasi peserta AN; periode tidak identik)"),
    dict(dataset="D", nama=NAMA_DATASET["D"], sumber=SUMBER_URL["D"], file=FILES["D"].name, unit_observasi="Provinsi/kab-kota × jenis satuan pendidikan (label ordinal); nilai numerik hanya nasional",
         periode="Data 2025 (diperbarui 8 Juni 2026 menurut berkas)", indikator="Ratusan indikator; kandidat pendukung: A.1, A.2, D.1, D.21.4, E.7.4",
         skala_asli="label ordinal (Kurang/Sedang/Baik; Rendah/Sedang/Tinggi)", n_baris_data=len(RL), n_provinsi=int(RL.sheet.nunique()),
         nilai_hilang=int((RL.label == "Tidak Tersedia").sum()),
         status_integrasi="TERBATAS: hanya pendukung/validasi (A.2, D.1, D.21.4); E.7.4 tidak informatif; D.1.3 dan D.2.3 tidak tersedia"),
])
display(AUDIT)

n_common = len(ka & kb & kc)
jenjang_ok = 3
blocking = []
if n_common < 30: blocking.append("irisan kode provinsi A∩B∩C < 30")
if any(r["nilai_hilang"] > 0 for r in AUDIT.to_dict("records")[:3]): blocking.append("ada nilai hilang pada dataset inti")
VERDICT = ("LAYAK DENGAN CATATAN" if not blocking else "TIDAK LAYAK: " + "; ".join(blocking))
FEAS_TEXT = f"""LAPORAN KELAYAKAN INTEGRASI DATASET
===================================
Keputusan: {VERDICT}
- Kunci gabung: kode provinsi; irisan A∩B∩C = {n_common} provinsi (A={len(ka)}, B={len(kb)}, C={len(kc)}); selisih nama setelah normalisasi = {n_name_mis}.
- Jenjang yang dapat disejajarkan: {jenjang_ok} (SD, SMP, SMA/SMK) → unit analisis = {n_common} provinsi × {jenjang_ok} jenjang = {n_common*jenjang_ok} observasi (non-independen: berulang per provinsi).
- Nilai hilang pada A/B/C: {int(AUDIT.nilai_hilang.iloc[:3].sum())}. Tidak ada imputasi yang diperlukan.
- Dataset D hanya berisi label ordinal pada tingkat provinsi/kab-kota → digunakan sebagai validasi/pendukung, bukan variabel inti.
- Catatan utama: (1) periode referensi A/B (TA 2025/2026) vs C (Asesmen Nasional 2025) tidak dapat dipastikan identik; (2) populasi penyebut berbeda; (3) ambang numerasi 75% vs 70% pada metadata C tidak konsisten; (4) SMA/SMK tergabung pada C; (5) n=38 membatasi pemodelan kompleks.
Kesimpulan: analisis mismatch dapat dilakukan pada level provinsi dengan kesadaran atas keterbatasan di atas; tidak ada hambatan mendasar yang mengharuskan penghentian sebelum pemodelan.
Alternatif jika kelak tersedia data kab/kota: A/B/C pada level kab/kota akan menaikkan n dari 38 menjadi ± 500 dan memungkinkan clustering yang lebih bermakna."""
print(FEAS_TEXT)

QCDF = pd.DataFrame(QC)
display(QCDF[["dataset", "pemeriksaan", "hasil", "jumlah_temuan"]])
AUDIT_SHEETS = {"Ringkasan_Dataset": AUDIT, "Inventaris_Sheet": INV, "Pemeriksaan_Kualitas": QCDF, "Kecocokan_Kode": KODE_CHECK,
                "Padanan_Jenjang": JENJANG_XWALK, "Isu_Kompatibilitas": ISU,
                "Checksum_Berkas": pd.DataFrame({"dataset": list(CHECKSUM), "file": [FILES[k].name for k in CHECKSUM], "sha256": list(CHECKSUM.values())})}
save_excel(OUT / "dataset_audit.xlsx", AUDIT_SHEETS, notes=["Laporan audit dataset DIMAS-TI 2026. Seluruh isi dihasilkan otomatis oleh pipeline dari berkas yang diunggah.",
                                                            "Berkas mentah tidak pernah dimodifikasi (lihat Checksum_Berkas)."])
(OUT / "laporan_kelayakan_integrasi.txt").write_text(FEAS_TEXT, encoding="utf-8")


# %% Standardisasi format: skala persen, nama kanonik, wilayah, jenjang
MERGE_LOG = []
def mlog(msg):
    MERGE_LOG.append(msg); print(msg)

def island_group(kode):
    """Kelompok pulau/wilayah berdasarkan dua digit kode provinsi resmi (dipakai hanya untuk perbandingan antarwilayah)."""
    k = int(kode)
    if 11 <= k <= 21: return "Sumatera"
    if 31 <= k <= 36: return "Jawa"
    if 51 <= k <= 53: return "Bali–Nusa Tenggara"
    if 61 <= k <= 65: return "Kalimantan"
    if 71 <= k <= 76: return "Sulawesi"
    if 81 <= k <= 82: return "Maluku"
    if 91 <= k <= 96: return "Papua"
    return "Lainnya"

def clean_name(s):
    return re.sub(r"^(Provinsi|Prov\.)\s+", "", str(s)).strip()

def to_percent(series, scale_label):
    return series * 100.0 if str(scale_label).startswith("fraksi") else series.astype(float)

SCALE = {r["dataset"]: r["hasil"] for r in QC if r["pemeriksaan"] == "Skala nilai terdeteksi"}
print("Skala terdeteksi:", SCALE)

MASTER = (A_raw[A_raw.tipe_baris == "provinsi"][["kode", "wilayah_raw"]].drop_duplicates().assign(provinsi=lambda d: d.wilayah_raw.map(clean_name))
          .drop(columns="wilayah_raw").sort_values("kode").reset_index(drop=True))
MASTER["kode"] = MASTER.kode.astype(int)
MASTER["nama_norm"] = MASTER.provinsi.map(norm_name)
MASTER["wilayah_pulau"] = MASTER.kode.map(island_group)
KODE2NAMA = MASTER.set_index("kode").provinsi.to_dict()

def clean_access(d, label):
    x = d[d.tipe_baris == "provinsi"].copy()
    x["kode"] = x.kode.astype(int)
    x["provinsi"] = x.kode.map(KODE2NAMA)
    x["nilai_pct"] = to_percent(x.nilai_raw, SCALE[label])
    return x[["kode", "provinsi", "jenjang", "status", "nilai_pct"]].reset_index(drop=True)

A_clean, B_clean = clean_access(A_raw, "A"), clean_access(B_raw, "B")
JMAP_C = {"SD/sederajat": "SD", "SMP/sederajat": "SMP", "SMA/SMK/sederajat": "SMA_SMK", "SEMUA": "SEMUA"}
C_clean = C_raw[C_raw.tipe_baris == "provinsi"].copy()
C_clean["kode"] = C_clean.kode.astype(int)
C_clean["provinsi"] = C_clean.kode.map(KODE2NAMA)
C_clean["jenjang"] = C_clean.jenjang.map(JMAP_C)
C_clean["literasi_pct"] = to_percent(C_clean.literasi_raw, SCALE["C"])
C_clean["numerasi_pct"] = to_percent(C_clean.numerasi_raw, SCALE["C"])
C_clean = C_clean[["kode", "provinsi", "jenjang", "periode", "literasi_pct", "numerasi_pct"]].reset_index(drop=True)
assert C_clean.provinsi.notna().all() and A_clean.provinsi.notna().all(), "Ada kode provinsi yang tidak ada pada master — periksa Kecocokan_Kode."
mlog(f"Master provinsi: {len(MASTER)} baris (dari Dataset A; kunci = kode resmi). Wilayah pulau: {MASTER.wilayah_pulau.value_counts().to_dict()}")

# %% Penggabungan bertahap dengan validasi relasi dan pencatatan baris tidak cocok
def merge_checked(left, right, on, name, expect="one_to_one", how="outer"):
    ul = not left.duplicated(on).any(); ur = not right.duplicated(on).any()
    rel = ("one" if ul else "many") + "_to_" + ("one" if ur else "many")
    m = left.merge(right, on=on, how=how, indicator=True, suffixes=("", "_r"))
    only_l, only_r = int((m._merge == "left_only").sum()), int((m._merge == "right_only").sum())
    mlog(f"[{name}] kiri={len(left)} | kanan={len(right)} | hasil={len(m)} | relasi terdeteksi={rel} (diharapkan {expect}) "
         f"| hanya-kiri={only_l} | hanya-kanan={only_r}")
    if rel != expect:
        raise ValueError(f"Relasi gabung '{name}' bukan {expect}: periksa duplikasi kunci {on}.")
    return m.drop(columns="_merge"), m[m._merge != "both"]

JEN_AB = ["SD", "SMP", "SMA", "SMK"]                                   # jenjang yang punya padanan di Dataset C
A_tot = A_clean[A_clean.status == "Total"].pivot(index=["kode", "provinsi"], columns="jenjang", values="nilai_pct").reset_index()
B_tot = B_clean[B_clean.status == "Total"].pivot(index=["kode", "provinsi"], columns="jenjang", values="nilai_pct").reset_index()
A_l = A_tot.melt(id_vars=["kode", "provinsi"], var_name="jenjang", value_name="komputer_pct")
B_l = B_tot.melt(id_vars=["kode", "provinsi"], var_name="jenjang", value_name="internet_pct")
AB, un1 = merge_checked(A_l, B_l.drop(columns="provinsi"), ["kode", "jenjang"], "Langkah 1: A (komputer) ⋈ B (internet)")
AB = AB.drop(columns=[c for c in AB.columns if c.endswith("_r")])
mlog(f"  Jenjang pada A/B: {sorted(AB.jenjang.unique())}; jenjang TK dan SLB tidak punya padanan di Dataset C → dikeluarkan dari analisis inti.")

# Susun jenjang analisis: SD, SMP, dan SMA_SMK (= rata-rata tak berbobot SMA & SMK; SMA/SMK terpisah disimpan untuk sensitivitas)
ABw = AB[AB.jenjang.isin(JEN_AB)].pivot(index=["kode", "provinsi"], columns="jenjang", values=["komputer_pct", "internet_pct"])
ABw.columns = [f"{a}_{b}" for a, b in ABw.columns]; ABw = ABw.reset_index()
rows = []
for _, r in ABw.iterrows():
    rows.append(dict(kode=r.kode, provinsi=r.provinsi, jenjang="SD", komputer_pct=r.komputer_pct_SD, internet_pct=r.internet_pct_SD))
    rows.append(dict(kode=r.kode, provinsi=r.provinsi, jenjang="SMP", komputer_pct=r.komputer_pct_SMP, internet_pct=r.internet_pct_SMP))
    rows.append(dict(kode=r.kode, provinsi=r.provinsi, jenjang="SMA_SMK",
                     komputer_pct=(r.komputer_pct_SMA + r.komputer_pct_SMK) / 2, internet_pct=(r.internet_pct_SMA + r.internet_pct_SMK) / 2,
                     komputer_SMA_pct=r.komputer_pct_SMA, komputer_SMK_pct=r.komputer_pct_SMK,
                     internet_SMA_pct=r.internet_pct_SMA, internet_SMK_pct=r.internet_pct_SMK))
DIG = pd.DataFrame(rows)
mlog(f"Langkah 2: komposit SMA_SMK = rata-rata tak berbobot SMA & SMK (tidak ada jumlah sekolah untuk pembobotan) → {len(DIG)} baris provinsi×jenjang")

C_j = C_clean[C_clean.jenjang.isin(["SD", "SMP", "SMA_SMK"])]
PJ, un2 = merge_checked(DIG, C_j[["kode", "jenjang", "literasi_pct", "numerasi_pct"]], ["kode", "jenjang"], "Langkah 3: (A⋈B) ⋈ C pada kode×jenjang", how="outer")
PJ = PJ.merge(MASTER[["kode", "wilayah_pulau"]], on="kode", how="left")
mlog(f"Hasil integrasi inti: {len(PJ)} baris = {PJ.kode.nunique()} provinsi × {PJ.jenjang.nunique()} jenjang")

# Baris yang tidak masuk analisis beserta alasannya
EXCL = []
for nm, d in (("A", A_raw), ("B", B_raw)):
    for w in d[d.tipe_baris != "provinsi"].wilayah_raw.unique():
        EXCL.append(dict(dataset=nm, wilayah=w, alasan="Luar Negeri: tidak ada padanan di Dataset C/provinsi" if "luar" in w.lower() else "Agregat nasional: bukan unit provinsi (hanya pembanding)"))
for w in C_raw[C_raw.tipe_baris != "provinsi"].wilayah_raw.unique():
    EXCL.append(dict(dataset="C", wilayah=w, alasan="Agregat nasional: bukan unit provinsi (hanya pembanding)"))
for j, why in (("TK", "Tidak ada padanan pada Asesmen Nasional"), ("SLB", "Tidak ada padanan pada Dataset C")):
    EXCL.append(dict(dataset="A,B", wilayah=f"Jenjang {j} (38 provinsi)", alasan=why))
EXCL.append(dict(dataset="C", wilayah="Jenjang SEMUA (38 provinsi)", alasan="Tidak ada agregat setara di A/B; dipakai untuk validasi/sensitivitas"))
EXCL = pd.DataFrame(EXCL)

# %% Nilai hilang, pencilan, dan wilayah/provinsi pendukung dari Dataset D
CORE = ["komputer_pct", "internet_pct", "numerasi_pct", "literasi_pct"]
MISS = PJ[CORE].isna().sum().rename("jumlah_hilang").to_frame().assign(persen=lambda d: 100 * d.jumlah_hilang / len(PJ))
display(MISS)
if MISS.jumlah_hilang.sum() > 0:
    mlog(f"[PERINGATAN] Ada {int(MISS.jumlah_hilang.sum())} nilai hilang pada kolom inti → listwise deletion dipakai dan dilaporkan (tidak ada imputasi otomatis).")
    PJ = PJ.dropna(subset=CORE).reset_index(drop=True)
else:
    mlog("Nilai hilang pada kolom inti: 0 → tidak diperlukan imputasi (perbandingan metode imputasi tidak relevan).")

def iqr_flags(df, cols, by="jenjang"):
    out = []
    for j, g in df.groupby(by):
        for c in cols:
            q1, q3 = g[c].quantile([.25, .75]); i = q3 - q1
            lo, hi = q1 - 1.5 * i, q3 + 1.5 * i
            for _, r in g[(g[c] < lo) | (g[c] > hi)].iterrows():
                out.append(dict(jenjang=j, indikator=c, provinsi=r.provinsi, nilai=round(r[c], 2), batas_bawah=round(lo, 2), batas_atas=round(hi, 2),
                                arah="rendah ekstrem" if r[c] < lo else "tinggi ekstrem"))
    return pd.DataFrame(out)
OUTL = iqr_flags(PJ, CORE)
mlog(f"Pencilan (aturan 1,5×IQR per jenjang×indikator): {len(OUTL)} titik (dipertahankan; dampaknya diuji pada analisis sensitivitas)")

# Dataset D → variabel pendukung ordinal level provinsi
ORD = {"Kurang": 1, "Sedang": 2, "Baik": 3, "Rendah": 1, "Tinggi": 3}                 # A.1, A.2, D.1, E.7.4: 1=Kurang..3=Baik ; D.21.4: 1=kesenjangan rendah..3=tinggi
MASTER_NN = MASTER.set_index("nama_norm").kode.to_dict()
RL = RAPOR["labels"].copy()
RL = RL[RL.sheet != "Nasional"].copy()
RL["kode"] = RL.sheet.map(norm_name).map(MASTER_NN)
unm = RL[RL.kode.isna()].sheet.unique().tolist()
mlog(f"D: sheet provinsi yang tidak cocok dengan master (via nama): {unm if unm else 'tidak ada'}; {RL.sheet.nunique()} sheet → {RL.kode.nunique()} kode")
RL["jenjang"] = RL.jenis.map(RAPOR_JENIS)
RL["skor_ord"] = RL.label.map(ORD)
RLp = RL[RL.kab == "-"]
DSUP = RLp.pivot_table(index="kode", columns=["kode_ind", "jenjang"], values="skor_ord", aggfunc="first")
DSUP.columns = [f"D_{a.replace('.', '_')}_{b}" for a, b in DSUP.columns]
DSUP = DSUP.reset_index()
agg = lambda ind: RLp[RLp.kode_ind == ind].groupby("kode").skor_ord.mean()
DPROV = pd.DataFrame({"D_A2_ord_mean": agg("A.2"), "D_A1_ord_mean": agg("A.1"), "D_D1_ord_mean": agg("D.1"), "D_D21_4_gap_mean": agg("D.21.4")})
kab = RL[(RL.kab != "-") & (RL.kode_ind == "A.2")].dropna(subset=["skor_ord"])
DPROV["D_A2_pct_kabkota_kurang"] = kab.groupby("kode").apply(lambda g: 100 * (g.label == "Kurang").mean(), include_groups=False)
DPROV["D_n_kabkota"] = kab.groupby("kode").kab.nunique()
DPROV = DPROV.reset_index()
mlog(f"D: variabel pendukung level provinsi = {len(DPROV)} baris; kab/kota total pada label A.2 = {int(DPROV.D_n_kabkota.sum())}")

# %% Tabel provinsi lebar (38 baris) + SEMUA dari C + pendukung D; simpan seluruh keluaran Tahap II
PW = PJ.pivot(index=["kode", "provinsi", "wilayah_pulau"], columns="jenjang",
              values=["komputer_pct", "internet_pct", "numerasi_pct", "literasi_pct"])
PW.columns = [f"{a}_{b}" for a, b in PW.columns]; PW = PW.reset_index()
for ind in ("komputer", "internet", "numerasi", "literasi"):
    PW[f"{ind}_rata3"] = PW[[f"{ind}_pct_{j}" for j in ("SD", "SMP", "SMA_SMK")]].mean(axis=1)
semua = C_clean[C_clean.jenjang == "SEMUA"][["kode", "literasi_pct", "numerasi_pct"]].rename(columns={"literasi_pct": "literasi_SEMUA_C", "numerasi_pct": "numerasi_SEMUA_C"})
PW, un3 = merge_checked(PW, semua, ["kode"], "Langkah 4: provinsi (A⋈B⋈C jenjang) ⋈ C(SEMUA)")
PW, un4 = merge_checked(PW, DPROV, ["kode"], "Langkah 5: provinsi ⋈ D (label pendukung)")
mlog(f"Dataset final provinsi: {PW.shape[0]} baris × {PW.shape[1]} kolom")

merge_report = "LAPORAN VALIDASI PENGGABUNGAN DATA\n" + "=" * 40 + "\n" + "\n".join(MERGE_LOG) + "\n\nBaris/jenjang yang tidak masuk analisis dan alasannya:\n" + EXCL.to_string(index=False)
(OUT / "merge_validation_report.txt").write_text(merge_report, encoding="utf-8")

save_excel(OUT / "data_raw_extracted.xlsx", {"A_komputer_apa_adanya": A_raw.drop(columns=["nama_norm"]), "B_internet_apa_adanya": B_raw.drop(columns=["nama_norm"]),
                                            "C_AN_apa_adanya": C_raw.drop(columns=["nama_norm"])},
           notes=["Hasil pembacaan berkas mentah TANPA perubahan nilai/skala (A, B berskala 0–1; C berskala 0–100). Untuk jejak audit."])
save_excel(OUT / "dataset_cleaned.xlsx", {"A_komputer_persen": A_clean, "B_internet_persen": B_clean, "C_AN_persen": C_clean, "master_provinsi": MASTER,
                                         "pencilan_IQR": OUTL if len(OUTL) else pd.DataFrame({"info": ["tidak ada"]}), "nilai_hilang": MISS.reset_index(names="kolom"),
                                         "D_label_provinsi": RL[["kode", "sheet", "kab", "jenis", "kode_ind", "label", "skor_ord"]]},
           notes=["Seluruh nilai persentase pada skala 0–100. Nama provinsi kanonik dari Dataset A; kunci = kode provinsi.",
                  "Jenjang A/B: SD, SMP, SMA, SMK (Total negeri+swasta). Dataset C: SD, SMP, SMA_SMK, SEMUA."])
save_excel(OUT / "dataset_merged.xlsx", {"provinsi_x_jenjang": PJ, "provinsi_lebar": PW, "baris_dikecualikan": EXCL},
           notes=["provinsi_x_jenjang: 38 provinsi × 3 jenjang (SD, SMP, SMA_SMK) = unit analisis utama.",
                  "SMA_SMK pada komputer/internet = rata-rata tak berbobot SMA & SMK; kolom SMA/SMK terpisah disertakan untuk sensitivitas.",
                  "provinsi_lebar: 1 baris per provinsi; *_rata3 = rata-rata tak berbobot SD, SMP, SMA_SMK; *_SEMUA_C = nilai 'SEMUA' dari Dataset C."])
display(PJ.head(6)); display(PW.head(3))

# %% Kamus data
DD = pd.DataFrame([
    ("kode", "Kode provinsi (Kemendagri/BPS 2 digit)", "—", "A/B/C", "integer; kunci gabung", "semua"),
    ("provinsi", "Nama provinsi kanonik (tanpa awalan 'Provinsi')", "—", "A", "dari Dataset A", "semua"),
    ("wilayah_pulau", "Kelompok pulau berdasarkan 2 digit kode provinsi", "kategori", "turunan", "pemetaan awalan kode", "semua"),
    ("jenjang", "SD, SMP, SMA_SMK (SMA_SMK = gabungan)", "kategori", "A/B/C", "penyelarasan jenjang", "provinsi_x_jenjang"),
    ("komputer_pct", "% sekolah dengan akses komputer (≥5 unit) untuk pengajaran, Total negeri+swasta", "persen 0–100", "A", "fraksi×100; SMA_SMK=rata-rata SMA & SMK", "provinsi_x_jenjang"),
    ("internet_pct", "% sekolah dengan akses internet untuk pengajaran, Total negeri+swasta", "persen 0–100", "B", "fraksi×100; SMA_SMK=rata-rata SMA & SMK", "provinsi_x_jenjang"),
    ("numerasi_pct", "% satdik dengan ≥ambang siswa mencapai kompetensi minimum numerasi (AN 2025)", "persen 0–100", "C", "tanpa transformasi", "provinsi_x_jenjang"),
    ("literasi_pct", "Idem untuk literasi membaca (pembanding/sensitivitas)", "persen 0–100", "C", "tanpa transformasi", "provinsi_x_jenjang"),
    ("*_SMA_pct / *_SMK_pct", "Komputer/internet khusus SMA atau SMK", "persen 0–100", "A/B", "fraksi×100", "provinsi_x_jenjang"),
    ("*_rata3", "Rata-rata tak berbobot dari SD, SMP, SMA_SMK", "persen 0–100", "turunan", "mean 3 jenjang", "provinsi_lebar"),
    ("literasi_SEMUA_C / numerasi_SEMUA_C", "Nilai 'SEMUA jenjang' pada Dataset C (rata-rata tertimbang oleh penerbit)", "persen 0–100", "C", "tanpa transformasi", "provinsi_lebar"),
    ("D_A2_ord_mean", "Rerata skor ordinal label Rapor A.2 (Kemampuan numerasi) pada SD,SMP,SMA,SMK umum (1=Kurang,2=Sedang,3=Baik)", "ordinal 1–3", "D", "encoding label", "provinsi_lebar"),
    ("D_A1_ord_mean", "Idem untuk A.1 (Kemampuan literasi)", "ordinal 1–3", "D", "encoding label", "provinsi_lebar"),
    ("D_D1_ord_mean", "Idem untuk D.1 (Kualitas pembelajaran)", "ordinal 1–3", "D", "encoding label", "provinsi_lebar"),
    ("D_D21_4_gap_mean", "Rerata skor kesenjangan fasilitas TIK D.21.4 (1=rendah, 3=tinggi)", "ordinal 1–3", "D", "encoding label", "provinsi_lebar"),
    ("D_A2_pct_kabkota_kurang", "% kab/kota di provinsi berlabel 'Kurang' pada A.2 (gabungan jenjang)", "persen 0–100", "D", "dihitung dari label kab/kota", "provinsi_lebar"),
    ("D_n_kabkota", "Jumlah kab/kota yang memiliki label A.2 tersedia", "jumlah", "D", "hitungan", "provinsi_lebar"),
], columns=["variabel", "deskripsi", "satuan", "sumber", "transformasi", "tabel"])
save_excel(OUT / "data_dictionary.xlsx", {"kamus_data": DD}, notes=["Variabel analisis tambahan (z-score, mismatch, cluster) didefinisikan pada mismatch_analysis.xlsx dan clustering_results.xlsx."])

# %% Statistik deskriptif, normalitas, dan kelengkapan data
JEN3 = ["SD", "SMP", "SMA_SMK"]
INDS = ["komputer_pct", "internet_pct", "numerasi_pct", "literasi_pct"]
LAB = {"komputer_pct": "Akses komputer (%)", "internet_pct": "Akses internet (%)",
       "numerasi_pct": "Numerasi: % satdik ≥ ambang (%)", "literasi_pct": "Literasi: % satdik ≥ ambang (%)"}
JCOL = {"SD": "#1b9e77", "SMP": "#d95f02", "SMA_SMK": "#7570b3"}
RNG = np.random.default_rng(SEED)

def describe_df(df, cols, by=None):
    rows = []
    groups = [("Semua", df)] if by is None else list(df.groupby(by))
    for g, d in groups:
        for c in cols:
            x = d[c].dropna()
            q1, q3 = x.quantile([.25, .75])
            sw = stats.shapiro(x).pvalue if len(x) >= 3 else np.nan
            rows.append(dict(kelompok=g, indikator=c, n=len(x), mean=x.mean(), median=x.median(), std=x.std(), min=x.min(), q1=q1, q3=q3, max=x.max(),
                             rentang=x.max() - x.min(), IQR=q3 - q1, skewness=stats.skew(x), kurtosis=stats.kurtosis(x), shapiro_p=sw,
                             normal_pada_5persen=("ya" if sw >= 0.05 else "tidak")))
    return pd.DataFrame(rows)

DESC_J = describe_df(PJ, INDS, by="jenjang")
PW3 = PW.rename(columns={f"{i}_rata3": f"{i}_pct" for i in ("komputer", "internet", "numerasi", "literasi")}).copy()
DESC_P = describe_df(PW3, INDS)
display(DESC_J.round(2)[["kelompok", "indikator", "n", "mean", "median", "std", "min", "q1", "q3", "max", "skewness", "shapiro_p"]])
n_nonnormal = int((DESC_J.normal_pada_5persen == "tidak").sum())
print(f"Uji Shapiro–Wilk: {n_nonnormal} dari {len(DESC_J)} kombinasi jenjang×indikator tidak normal pada α=5% → Spearman dipakai sebagai ukuran korelasi utama, Pearson sebagai pembanding.")

# Kelengkapan data dan potensi bias
zero_edges = []
for _, r in A_clean[(A_clean.status == "Total") & A_clean.jenjang.isin(JEN_AB)].iterrows():
    if r.nilai_pct in (0.0, 100.0): zero_edges.append(("komputer", r.provinsi, r.jenjang, r.nilai_pct))
for _, r in B_clean[(B_clean.status == "Total") & B_clean.jenjang.isin(JEN_AB)].iterrows():
    if r.nilai_pct in (0.0, 100.0): zero_edges.append(("internet", r.provinsi, r.jenjang, r.nilai_pct))
ZERO_EDGES = pd.DataFrame(zero_edges, columns=["indikator", "provinsi", "jenjang", "nilai"])
KELENGKAPAN = pd.DataFrame([
    dict(aspek="Provinsi tercakup A/B/C", hasil=f"{PW.shape[0]} dari 38 (100%)"),
    dict(aspek="Nilai hilang pada indikator inti", hasil=f"{int(PJ[INDS].isna().sum().sum())}"),
    dict(aspek="Nilai persis 0 atau 100 pada Total SD/SMP/SMA/SMK (indikasi sekolah sedikit)", hasil=f"{len(ZERO_EDGES)} sel" + (": " + "; ".join(f"{a} {b} {c}={d:.0f}" for a, b, c, d in zero_edges) if zero_edges else "")),
    dict(aspek="Provinsi baru (kode 93–96) — potensi data tidak stabil", hasil=", ".join(MASTER[MASTER.kode.isin([93, 94, 95, 96])].provinsi)),
    dict(aspek="Potensi bias seleksi", hasil="C hanya mencakup satdik peserta Asesmen Nasional sedangkan A/B mencakup seluruh satdik; provinsi dengan partisipasi/cakupan berbeda dapat tampak berbeda semata karena populasi."),
])
display(KELENGKAPAN)

# %% Histogram, boxplot
def fig_hist():
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for i, ind in enumerate(["komputer_pct", "internet_pct", "numerasi_pct"]):
        for j, jn in enumerate(JEN3):
            ax = axes[i, j]; x = PJ[PJ.jenjang == jn][ind]
            sns.histplot(x, bins=10, kde=True, color=JCOL[jn], ax=ax, edgecolor="white")
            ax.axvline(x.median(), color="k", ls="--", lw=1)
            ax.set_xlabel(LAB[ind] if i == 2 or True else ""); ax.set_ylabel("Jumlah provinsi")
            ax.set_title(f"{jn.replace('_','/')} | median={x.median():.1f}, skew={stats.skew(x):.2f}", fontsize=9)
    fig.suptitle("Distribusi indikator per jenjang (38 provinsi; garis putus = median)", fontweight="bold", y=1.01)
    fig.tight_layout()
    return save_fig(fig, "eda_histogram.png")
fig_hist()

def fig_box():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, ind in zip(axes, INDS):
        sns.boxplot(data=PJ, x="jenjang", y=ind, order=JEN3, hue="jenjang", palette=JCOL, legend=False, ax=ax, width=.55, fliersize=0)
        sns.stripplot(data=PJ, x="jenjang", y=ind, order=JEN3, color="k", size=3, alpha=.55, ax=ax)
        o = OUTL[OUTL.indikator == ind]
        for _, r in o.iterrows():
            ax.annotate(r.provinsi, (JEN3.index(r.jenjang), r.nilai), xytext=(6, 0), textcoords="offset points", fontsize=7, color="firebrick")
        ax.set_title(LAB[ind], fontsize=9); ax.set_xlabel(""); ax.set_ylabel("%")
        ax.set_xticks(range(3)); ax.set_xticklabels(["SD", "SMP", "SMA/SMK"])
    fig.suptitle("Boxplot per jenjang (label merah = pencilan aturan 1,5×IQR)", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "eda_boxplot.png")
fig_box()

# %% Korelasi: Spearman & Pearson, interval kepercayaan bootstrap, leave-one-out, ketahanan terhadap wilayah
def boot_spearman(x, y, B=2000, rng=RNG):
    n = len(x); r = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        r[b] = stats.spearmanr(x[i], y[i]).correlation
    r = r[~np.isnan(r)]
    return np.percentile(r, [2.5, 97.5])

def loo_range(x, y):
    n = len(x); r = [stats.spearmanr(np.delete(x, i), np.delete(y, i)).correlation for i in range(n)]
    return min(r), max(r)

PAIRS = [("komputer_pct", "numerasi_pct"), ("internet_pct", "numerasi_pct"), ("komputer_pct", "internet_pct"), ("literasi_pct", "numerasi_pct")]
CORR_ROWS = []
sets = [(jn, PJ[PJ.jenjang == jn]) for jn in JEN3] + [("Provinsi (rata 3 jenjang)", PW3)]
for nm, d in sets:
    for a, b in PAIRS:
        x, y = d[a].values, d[b].values
        rs, ps = stats.spearmanr(x, y); rp, pp = stats.pearsonr(x, y)
        lo, hi = boot_spearman(x, y); l1, l2 = loo_range(x, y)
        CORR_ROWS.append(dict(kelompok=nm, variabel_1=a, variabel_2=b, n=len(d), spearman_rho=rs, p_spearman=ps, CI95_bawah=lo, CI95_atas=hi,
                              pearson_r=rp, p_pearson=pp, LOO_rho_min=l1, LOO_rho_maks=l2))
CORR = pd.DataFrame(CORR_ROWS)
display(CORR.round(3))

# Matriks korelasi
mats = {}
for nm, d in sets:
    mats[f"spearman | {nm}"] = d[INDS].corr(method="spearman")
    mats[f"pearson | {nm}"] = d[INDS].corr(method="pearson")
def fig_corr():
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6))
    for r_, meth in enumerate(("spearman", "pearson")):
        for c_, (nm, d) in enumerate(sets):
            ax = axes[r_, c_]; m = d[INDS].corr(method=meth)
            m.index = m.columns = ["Komputer", "Internet", "Numerasi", "Literasi"]
            sns.heatmap(m, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, cbar=False, square=True, annot_kws={"fontsize": 9})
            ax.set_title(f"{meth.capitalize()} — {nm.replace('_','/')}", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, "eda_matriks_korelasi.png")
fig_corr()

# Ketahanan korelasi pada tingkat provinsi terhadap pengecualian kelompok wilayah (cek pola Simpson/pengaruh klaster geografis)
ROBUST_ROWS = []
for ex in [None] + sorted(PW3.wilayah_pulau.unique()):
    d = PW3 if ex is None else PW3[PW3.wilayah_pulau != ex]
    row = dict(dikecualikan=("tidak ada" if ex is None else ex), n=len(d))
    for a in ("komputer_pct", "internet_pct"):
        row[f"rho_{a[:-4]}_numerasi"] = stats.spearmanr(d[a], d.numerasi_pct).correlation
    ROBUST_ROWS.append(row)
ROBUST_CORR = pd.DataFrame(ROBUST_ROWS)
display(ROBUST_CORR.round(3))

# %% Scatterplot digital vs numerasi per jenjang + perbandingan antarwilayah
def fig_scatter():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for i, dig in enumerate(["komputer_pct", "internet_pct"]):
        for j, jn in enumerate(JEN3):
            ax = axes[i, j]; d = PJ[PJ.jenjang == jn]
            ax.scatter(d[dig], d.numerasi_pct, c=JCOL[jn], s=34, alpha=.85, edgecolor="white")
            b1, b0 = np.polyfit(d[dig], d.numerasi_pct, 1); xs = np.linspace(d[dig].min(), d[dig].max(), 50)
            ax.plot(xs, b0 + b1 * xs, color="k", lw=1.2, ls="--")
            res = d.numerasi_pct - (b0 + b1 * d[dig])
            for k_ in list(res.abs().sort_values(ascending=False).index[:3]):
                ax.annotate(d.loc[k_, "provinsi"], (d.loc[k_, dig], d.loc[k_, "numerasi_pct"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
            rho = stats.spearmanr(d[dig], d.numerasi_pct).correlation
            ax.set_title(f"{jn.replace('_','/')} | Spearman ρ = {rho:.2f}", fontsize=9)
            ax.set_xlabel(LAB[dig]); ax.set_ylabel("Numerasi (% satdik ≥ ambang)"); ax.set_ylim(bottom=-2)
    fig.suptitle("Kesiapan digital vs capaian numerasi per jenjang (anotasi: 3 residual terbesar terhadap garis OLS)", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "eda_scatter_digital_numerasi.png")
fig_scatter()

KW = []
for ind in ["komputer_pct", "internet_pct", "numerasi_pct"]:
    groups = [g[ind].values for _, g in PW3.groupby("wilayah_pulau")]
    H, p = stats.kruskal(*groups)
    KW.append(dict(indikator=ind, H=H, p=p, epsilon2=(H - len(groups) + 1) / (len(PW3) - len(groups)), n_kelompok=len(groups)))
KW = pd.DataFrame(KW)
REG = PW3.groupby("wilayah_pulau")[["komputer_pct", "internet_pct", "numerasi_pct"]].agg(["count", "median", "min", "max"]).round(1)
REG.columns = ["_".join(c) for c in REG.columns]; REG = REG.reset_index()
display(KW.round(4)); display(REG)
def fig_reg():
    order = PW3.groupby("wilayah_pulau").numerasi_pct.median().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, ind in zip(axes, ["komputer_pct", "internet_pct", "numerasi_pct"]):
        sns.boxplot(data=PW3, y="wilayah_pulau", x=ind, order=order, ax=ax, color="#c6dbef", fliersize=0)
        sns.stripplot(data=PW3, y="wilayah_pulau", x=ind, order=order, ax=ax, color="k", size=3.5)
        p = KW.set_index("indikator").loc[ind, "p"]
        ax.set_title(f"{LAB[ind]}\nKruskal–Wallis p={p:.4f}", fontsize=9); ax.set_ylabel(""); ax.set_xlabel("Rata-rata 3 jenjang (%)")
    fig.suptitle("Perbandingan distribusi antarkelompok wilayah (kelompok kecil, mis. Maluku n=2, bersifat indikatif)", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "eda_perbandingan_wilayah.png")
fig_reg()

# Identifikasi wilayah ekstrem (|z| > 2 dalam jenjang) dan peringkat ujung
EXT = []
for jn, d in PJ.groupby("jenjang"):
    for c in ["komputer_pct", "internet_pct", "numerasi_pct"]:
        z = (d[c] - d[c].mean()) / d[c].std()
        for k_ in z[z.abs() > 2].index:
            EXT.append(dict(jenjang=jn, indikator=c, provinsi=d.loc[k_, "provinsi"], nilai=round(d.loc[k_, c], 2), z=round(z[k_], 2)))
EXTREME = pd.DataFrame(EXT).sort_values(["indikator", "jenjang", "z"]) if EXT else pd.DataFrame(columns=["jenjang", "indikator", "provinsi", "nilai", "z"])
TOPBOT = []
for c in ["komputer_pct", "internet_pct", "numerasi_pct"]:
    s = PW3.set_index("provinsi")[c].sort_values()
    TOPBOT.append(dict(indikator=c, terendah_3=", ".join(f"{i} ({v:.1f})" for i, v in s.head(3).items()), tertinggi_3=", ".join(f"{i} ({v:.1f})" for i, v in s.tail(3)[::-1].items())))
TOPBOT = pd.DataFrame(TOPBOT)
display(EXTREME); display(TOPBOT)

# %% Kecukupan hubungan antarindikator untuk analisis lanjutan dan ringkasan EDA
def strength_row(g, a):
    r = CORR[(CORR.kelompok == g) & (CORR.variabel_1 == a) & (CORR.variabel_2 == "numerasi_pct")].iloc[0]
    return r
ADEQ = []
for nm, _ in sets:
    for a in ("komputer_pct", "internet_pct"):
        r = strength_row(nm, a)
        ok = (r.spearman_rho >= 0.5) and (r.CI95_bawah > 0.3) and (r.LOO_rho_min > 0.3)
        ADEQ.append(dict(kelompok=nm, indikator_digital=a, rho=round(r.spearman_rho, 3), CI95=f"[{r.CI95_bawah:.2f}, {r.CI95_atas:.2f}]",
                         LOO=f"[{r.LOO_rho_min:.2f}, {r.LOO_rho_maks:.2f}]", cukup_kuat="ya" if ok else "tidak"))
ADEQ = pd.DataFrame(ADEQ)
display(ADEQ)
r_ki = CORR[(CORR.kelompok == "Provinsi (rata 3 jenjang)") & (CORR.variabel_1 == "komputer_pct") & (CORR.variabel_2 == "internet_pct")].iloc[0]
r_ln = CORR[(CORR.kelompok == "Provinsi (rata 3 jenjang)") & (CORR.variabel_1 == "literasi_pct")].iloc[0]
EDA_TEXT = f"""RINGKASAN TEMUAN EDA (dihasilkan otomatis dari data aktual)
1. Cakupan: {PW.shape[0]} provinsi × 3 jenjang; tidak ada nilai hilang pada indikator inti; {len(OUTL)} titik pencilan (1,5×IQR) dipertahankan.
2. Sebaran: {n_nonnormal}/{len(DESC_J)} kombinasi jenjang×indikator tidak normal (Shapiro α=5%); karena itu Spearman dijadikan ukuran utama.
3. Hubungan digital–numerasi (Spearman, tingkat provinsi rata 3 jenjang): komputer–numerasi ρ={strength_row('Provinsi (rata 3 jenjang)','komputer_pct').spearman_rho:.2f}, internet–numerasi ρ={strength_row('Provinsi (rata 3 jenjang)','internet_pct').spearman_rho:.2f}. Hubungan {'cukup kuat' if (ADEQ[ADEQ.kelompok=='Provinsi (rata 3 jenjang)'].cukup_kuat=='ya').all() else 'tidak sepenuhnya kuat'} berdasarkan kriteria ρ≥0,5, batas bawah CI95>0,3, dan LOO>0,3.
4. Komputer–internet berkorelasi ρ={r_ki.spearman_rho:.2f} → dua indikator berkaitan namun tidak identik (mendukung analisis terpisah dan gabungan).
5. Literasi–numerasi ρ={r_ln.spearman_rho:.2f} → numerasi dan literasi hampir berurutan sama antarprovinsi; mismatch 'numerasi' kemungkinan mencerminkan mismatch hasil belajar umum, bukan hanya numerasi.
6. Antarwilayah: perbedaan kelompok pulau signifikan untuk {', '.join(KW[KW.p<0.05].indikator.str.replace('_pct','')) or 'tidak ada indikator'} (Kruskal–Wallis p<0,05); korelasi dapat sebagian mencerminkan pola geografis (lihat ketahanan korelasi saat satu wilayah dikeluarkan).
7. Keterbatasan: korelasi bersifat asosiatif; tidak menunjukkan sebab-akibat; n=38; populasi penyebut C berbeda dari A/B.
"""
print(EDA_TEXT)
(OUT / "ringkasan_eda.txt").write_text(EDA_TEXT, encoding="utf-8")
save_excel(OUT / "eda_results.xlsx", {"deskriptif_per_jenjang": DESC_J, "deskriptif_provinsi_rata3": DESC_P, "korelasi_lengkap": CORR,
                                      "kecukupan_hubungan": ADEQ, "ketahanan_korelasi_wilayah": ROBUST_CORR, "kruskal_wallis_wilayah": KW,
                                      "ringkasan_wilayah_pulau": REG, "wilayah_ekstrem_z2": EXTREME, "peringkat_ujung": TOPBOT, "kelengkapan_data": KELENGKAPAN,
                                      **{k[:31].replace("|", "-"): v.reset_index(names="indikator") for k, v in mats.items() if k.startswith("spearman")}},
           notes=["Statistik deskriptif, korelasi, dan kecukupan hubungan antarindikator. Nilai dihasilkan oleh pipeline."])

# %% Fungsi standardisasi dan pembangun skor (mendukung variasi untuk uji sensitivitas dan bootstrap)
def zs(x, fit=None, method="z"):
    """Standardisasi x memakai parameter dari 'fit' (default: x sendiri). method: z | robust | rank | minmax."""
    f = x if fit is None else fit
    if method == "z":
        return (x - f.mean()) / f.std(ddof=1)
    if method == "robust":
        med = f.median(); mad = np.median(np.abs(f - med)) * 1.4826
        return (x - med) / mad
    if method == "rank":                                   # rank-normal (van der Waerden); monoton, tahan terhadap pencilan
        return pd.Series(stats.norm.ppf((stats.rankdata(x) - 0.5) / len(x)), index=x.index)
    if method == "minmax":
        return (x - f.min()) / (f.max() - f.min())
    raise ValueError(method)

def build_scores(pj, digital=("komputer_pct", "internet_pct"), dig_w=None, num="numerasi_pct", method="z",
                 jenjang=("SD", "SMP", "SMA_SMK"), jenjang_w=None, fit_codes=None, num_province=None):
    """Mengembalikan DataFrame (indeks=kode) berisi ZD, ZN, M per jenjang dan tingkat provinsi.
    fit_codes: daftar kode (boleh duplikat → bootstrap) untuk menghitung parameter standardisasi; skor tetap dihitung untuk semua provinsi.
    jenjang_w: None (bobot sama) atau DataFrame (indeks=kode, kolom=jenjang) berbobot khusus per provinsi.
    num_province: Series (indeks=kode) untuk menggantikan komposit numerasi provinsi (mis. 'SEMUA' dari C)."""
    dig_w = dig_w or [1 / len(digital)] * len(digital)
    ZDj, ZNj = {}, {}
    for jn in jenjang:
        d = pj[pj.jenjang == jn].set_index("kode")
        fit = d if fit_codes is None else d.loc[fit_codes]
        D = sum(w * zs(d[c], fit[c], method) for c, w in zip(digital, dig_w))
        Dfit = D if fit_codes is None else D.loc[fit_codes]
        ZDj[jn] = zs(D, Dfit, method)
        ZNj[jn] = zs(d[num], fit[num], method)
    ZDm, ZNm = pd.DataFrame(ZDj), pd.DataFrame(ZNj)
    if jenjang_w is None:
        Dp, Np = ZDm.mean(axis=1), ZNm.mean(axis=1)
    else:
        w = jenjang_w.loc[ZDm.index, list(jenjang)]; w = w.div(w.sum(axis=1), axis=0)
        Dp, Np = (ZDm * w).sum(axis=1), (ZNm * w).sum(axis=1)
    fitp = Dp.index if fit_codes is None else fit_codes
    ZD = zs(Dp, Dp.loc[fitp], method)
    if num_province is not None:
        Np = num_province.loc[Dp.index]
    ZN = zs(Np, Np.loc[fitp], method)
    out = pd.DataFrame({"ZD": ZD, "ZN": ZN, "M": ZD - ZN})
    for jn in jenjang:
        out[f"ZD_{jn}"], out[f"ZN_{jn}"], out[f"M_{jn}"] = ZDm[jn], ZNm[jn], ZDm[jn] - ZNm[jn]
    return out

PJi = PJ.copy()
PRIM = build_scores(PJi)                                    # skor utama (z-score, bobot sama)
PRIM_K = build_scores(PJi, digital=("komputer_pct",))       # hanya komputer
PRIM_I = build_scores(PJi, digital=("internet_pct",))       # hanya internet
r_zdzn = PRIM.ZD.corr(PRIM.ZN)
s_M = PRIM.M.std(ddof=1)
print(f"Korelasi Z_digital–Z_numerasi = {r_zdzn:.3f}; SD(Mismatch) = {s_M:.3f} (secara matematis SD(M)=√(2−2r)={np.sqrt(2-2*r_zdzn):.3f}). "
      "Makin kuat korelasi, makin kecil dispersi mismatch: M pada dasarnya menangkap penyimpangan dari tren umum digital–numerasi.")

# %% Kategori: kuadran (median), tingkat kesesuaian, dan level gabungan
def quadrant(ZD, ZN):
    hd, hn = ZD >= ZD.median(), ZN >= ZN.median()
    lab = np.select([hd & ~hn, ~hd & hn, ~hd & ~hn, hd & hn],
                    ["Digital tinggi – Numerasi rendah", "Digital rendah – Numerasi tinggi", "Digital rendah – Numerasi rendah", "Digital tinggi – Numerasi tinggi"])
    return pd.Series(lab, index=ZD.index)

def mismatch_class(M, s):
    return pd.Series(np.select([M >= s, M <= -s], ["Digital melampaui numerasi", "Numerasi melampaui digital"], "Relatif seimbang"), index=M.index)

def level_class(ZD, ZN, cut=0.5):
    L = (ZD + ZN) / 2
    return pd.Series(np.select([L >= cut, L <= -cut], ["tinggi", "rendah"], "menengah"), index=L.index)

SC = PRIM.copy()
SC = SC.join(MASTER.set_index("kode")[["provinsi", "wilayah_pulau"]])
SC["kuadran"] = quadrant(SC.ZD, SC.ZN)
SC["kelas_mismatch"] = mismatch_class(SC.M, s_M)
SC["level_gabungan"] = level_class(SC.ZD, SC.ZN)
SC["absM"] = SC.M.abs()
SC["M_z"] = SC.M / s_M                                   # mismatch dalam satuan SD(M)
SC["M_komputer_saja"] = PRIM_K.M; SC["M_internet_saja"] = PRIM_I.M
SC["ZK"], SC["ZI"] = PRIM_K.ZD, PRIM_I.ZD
SC["peringkat_M"] = SC.M.rank(ascending=False).astype(int)                   # 1 = digital paling melampaui numerasi
SC["peringkat_absM"] = SC.absM.rank(ascending=False).astype(int)
SC["konsistensi_jenjang"] = [int(sum(np.sign(r[f"M_{j}"]) == np.sign(r.M) for j in JEN3)) for _, r in SC.iterrows()]   # berapa jenjang searah dgn M provinsi
SC = SC.join(PW.set_index("kode")[["komputer_rata3", "internet_rata3", "numerasi_rata3", "literasi_rata3", "numerasi_SEMUA_C"]])
SC.index.name = "kode"
# kategori seimbang dibedakan level (bukan 'seimbang = bagus')
SC["keterangan"] = np.where(SC.kelas_mismatch == "Relatif seimbang", "Seimbang-" + SC.level_gabungan, SC.kelas_mismatch)
display(SC[["provinsi", "ZD", "ZN", "M", "kuadran", "kelas_mismatch", "level_gabungan", "konsistensi_jenjang"]].sort_values("M", ascending=False).round(2))

# %% Stabilitas bootstrap: parameter standardisasi dihitung ulang pada resampel provinsi
B = 1000
codes = PRIM.index.values
rng = np.random.default_rng(SEED)
BM = np.empty((B, len(codes)))
for b in range(B):
    fit_codes = rng.choice(codes, size=len(codes), replace=True).tolist()
    BM[b] = build_scores(PJi, fit_codes=fit_codes).M.reindex(codes).values
BM = pd.DataFrame(BM, columns=codes)
BOOT = pd.DataFrame({"M_bootstrap_median": BM.median(), "M_CI95_bawah": BM.quantile(.025), "M_CI95_atas": BM.quantile(.975)})
top5p = BM.apply(lambda r: r.nlargest(5).index.values, axis=1, result_type="expand")
top5n = BM.apply(lambda r: r.nsmallest(5).index.values, axis=1, result_type="expand")
BOOT["frek_top5_positif"] = pd.Series(top5p.values.ravel()).value_counts().reindex(codes).fillna(0) / B
BOOT["frek_top5_negatif"] = pd.Series(top5n.values.ravel()).value_counts().reindex(codes).fillna(0) / B
BOOT["tanda_stabil_95"] = np.where(BOOT.M_CI95_bawah > 0, "positif", np.where(BOOT.M_CI95_atas < 0, "negatif", "tidak pasti"))
BOOT.index.name = "kode"
SC = SC.join(BOOT)
print(f"Bootstrap B={B}: provinsi dengan tanda M stabil pada CI95: positif={int((SC.tanda_stabil_95=='positif').sum())}, negatif={int((SC.tanda_stabil_95=='negatif').sum())}, tidak pasti={int((SC.tanda_stabil_95=='tidak pasti').sum())}")
print("Catatan: bootstrap ini hanya mengukur ketidakpastian parameter standardisasi (sampling provinsi), bukan galat pengukuran persentase.")

# %% Uji sensitivitas definisi indikator, standardisasi, bobot, dan penanganan pencilan
def pj_variant(mode):
    x = PJi.copy()
    m = x.jenjang == "SMA_SMK"
    if mode in ("SMA", "SMK"):
        x.loc[m, "komputer_pct"] = x.loc[m, f"komputer_{mode}_pct"]; x.loc[m, "internet_pct"] = x.loc[m, f"internet_{mode}_pct"]
    return x
def pj_winsor(p=0.05):
    x = PJi.copy()
    for jn in JEN3:
        for c in ["komputer_pct", "internet_pct", "numerasi_pct"]:
            idx = x.jenjang == jn; lo, hi = x.loc[idx, c].quantile([p, 1 - p])
            x.loc[idx, c] = x.loc[idx, c].clip(lo, hi)
    return x
def pj_numvar(kind):
    x = PJi.copy()
    x["numerasi_pct"] = x.literasi_pct if kind == "literasi" else (x.literasi_pct + x.numerasi_pct) / 2
    return x
outlier_provs = MASTER[MASTER.provinsi.isin(OUTL.provinsi.unique())].kode.tolist()
nonout = [k for k in codes if k not in outlier_provs]
W_equal_none = None
SEM = PW.set_index("kode").numerasi_SEMUA_C
VARIANTS = {
    "Primer: z-score, komputer+internet 50:50, 3 jenjang sama bobot": PRIM,
    "Standardisasi: robust z (median/MAD)": build_scores(PJi, method="robust"),
    "Standardisasi: rank-normal": build_scores(PJi, method="rank"),
    "Standardisasi: min–max": build_scores(PJi, method="minmax"),
    "Digital: komputer saja": PRIM_K,
    "Digital: internet saja": PRIM_I,
    "Bobot digital 25% komputer : 75% internet": build_scores(PJi, dig_w=[.25, .75]),
    "Bobot digital 75% komputer : 25% internet": build_scores(PJi, dig_w=[.75, .25]),
    "SMA/SMK: hanya SMA": build_scores(pj_variant("SMA")),
    "SMA/SMK: hanya SMK": build_scores(pj_variant("SMK")),
    "Bobot jenjang = bobot implisit 'SEMUA' (C)": build_scores(PJi, jenjang_w=W_IMPL.rename(columns={"w_SD": "SD", "w_SMP": "SMP", "w_SMA_SMK": "SMA_SMK"})),
    "Numerasi provinsi = 'SEMUA' dari C": build_scores(PJi, num_province=SEM),
    "Numerasi → literasi (ambang berbeda)": build_scores(pj_numvar("literasi")),
    "Numerasi → rata-rata literasi & numerasi": build_scores(pj_numvar("gabung")),
    "Pencilan: winsorisasi 5%–95%": build_scores(pj_winsor()),
    "Pencilan: parameter z tanpa provinsi pencilan": build_scores(PJi, fit_codes=nonout),
}
def topk(s, k, largest=True):
    return set((s.nlargest(k) if largest else s.nsmallest(k)).index)
SENS = []
for nm, v in VARIANTS.items():
    M2 = v.M.reindex(codes)
    row = dict(varian=nm, spearman_vs_primer=stats.spearmanr(PRIM.M, M2).correlation,
               tanda_sama_persen=100 * (np.sign(PRIM.M) == np.sign(M2)).mean(),
               overlap_top5_positif=len(topk(PRIM.M, 5) & topk(M2, 5)), overlap_top5_negatif=len(topk(PRIM.M, 5, False) & topk(M2, 5, False)),
               overlap_top10_positif=len(topk(PRIM.M, 10) & topk(M2, 10)), overlap_top10_negatif=len(topk(PRIM.M, 10, False) & topk(M2, 10, False)))
    # kestabilan kuadran
    row["kuadran_sama_persen"] = 100 * (quadrant(v.ZD, v.ZN) == SC.kuadran).mean()
    SENS.append(row)
SENS = pd.DataFrame(SENS)
display(SENS.round(2))
RANKS = pd.DataFrame({nm: v.M.reindex(codes).rank(ascending=False) for nm, v in VARIANTS.items()}).join(MASTER.set_index("kode").provinsi)
SC["peringkat_rata_varian"] = RANKS.drop(columns="provinsi").mean(axis=1)
SC["peringkat_min_varian"] = RANKS.drop(columns="provinsi").min(axis=1); SC["peringkat_maks_varian"] = RANKS.drop(columns="provinsi").max(axis=1)

# Komputer vs internet terpisah: provinsi yang tandanya berbeda
SC["tanda_komputer"] = np.sign(SC.M_komputer_saja); SC["tanda_internet"] = np.sign(SC.M_internet_saja)
SPLIT = SC[SC.tanda_komputer != SC.tanda_internet][["provinsi", "M", "M_komputer_saja", "M_internet_saja"]].round(2)
print(f"Provinsi yang tanda mismatch-nya berbeda antara analisis komputer-saja vs internet-saja: {len(SPLIT)}")
display(SPLIT)
rho_ki_M = stats.spearmanr(SC.M_komputer_saja, SC.M_internet_saja).correlation
print(f"Korelasi Spearman antara M(komputer saja) dan M(internet saja) = {rho_ki_M:.2f}")

# %% Selisih bertanda vs selisih absolut + daftar teratas
TOP_POS = SC.sort_values("M", ascending=False).head(8)[["provinsi", "wilayah_pulau", "komputer_rata3", "internet_rata3", "numerasi_rata3", "ZD", "ZN", "M", "tanda_stabil_95", "frek_top5_positif", "konsistensi_jenjang", "kuadran"]]
TOP_NEG = SC.sort_values("M").head(8)[["provinsi", "wilayah_pulau", "komputer_rata3", "internet_rata3", "numerasi_rata3", "ZD", "ZN", "M", "tanda_stabil_95", "frek_top5_negatif", "konsistensi_jenjang", "kuadran"]]
ABSV = SC.sort_values("absM", ascending=False).head(12)[["provinsi", "M", "absM", "peringkat_M", "peringkat_absM", "kelas_mismatch", "kuadran"]]
print("\nTop mismatch POSITIF (digital relatif lebih tinggi daripada numerasi):"); display(TOP_POS.round(2))
print("Top mismatch NEGATIF (numerasi relatif lebih tinggi daripada digital):"); display(TOP_NEG.round(2))
print("Selisih absolut (|M|) — menggabungkan kedua arah; arah hilang sehingga hanya cocok untuk 'seberapa tidak sejalan':"); display(ABSV.round(2))

# %% Grafik: scatter dengan garis referensi, distribusi, peringkat ber-CI, heatmap jenjang, sensitivitas
QCOL = {"Digital tinggi – Numerasi rendah": "#d95f02", "Digital rendah – Numerasi tinggi": "#1b9e77",
        "Digital rendah – Numerasi rendah": "#7570b3", "Digital tinggi – Numerasi tinggi": "#e7298a"}
def fig_mismatch_scatter(df, name="mismatch_scatter.png", title="Kesiapan digital vs numerasi (skor-z relatif antar 38 provinsi)"):
    fig, ax = plt.subplots(figsize=(9.2, 8))
    lim = float(np.ceil(max(df.ZD.abs().max(), df.ZN.abs().max()) + .3))
    xs = np.array([-lim, lim])
    ax.fill_between(xs, xs - s_M, xs + s_M, color="#bdbdbd", alpha=.25, label=f"|Mismatch| < 1 SD(M) = {s_M:.2f} (relatif seimbang)")
    ax.plot(xs, xs, "k--", lw=1.1, label="Mismatch = 0 (garis 45°)")
    ax.axvline(df.ZD.median(), color="gray", lw=.8, ls=":"); ax.axhline(df.ZN.median(), color="gray", lw=.8, ls=":")
    for q, c in QCOL.items():
        d = df[df.kuadran == q]
        ax.scatter(d.ZD, d.ZN, s=58, c=c, edgecolor="white", label=f"{q} (n={len(d)})", zorder=3)
    lab = pd.concat([df.nlargest(5, "M"), df.nsmallest(5, "M")]).index
    for k in lab:
        ax.annotate(df.loc[k, "provinsi"], (df.loc[k, "ZD"], df.loc[k, "ZN"]), xytext=(5, 5), textcoords="offset points", fontsize=8, fontweight="bold")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("Z kesiapan digital (komputer+internet, 3 jenjang)"); ax.set_ylabel("Z numerasi")
    ax.text(lim * .97, -lim * .97, "Di bawah garis 45°: digital melampaui numerasi\n(mismatch positif)", ha="right", va="bottom", fontsize=8, color="#555")
    ax.text(-lim * .97, lim * .97, "Di atas garis 45°: numerasi melampaui digital\n(mismatch negatif)", ha="left", va="top", fontsize=8, color="#555")
    ax.set_title(title + "\nGaris putus-putus abu-abu = median; mismatch≈0 dapat berarti sama-sama rendah ATAU sama-sama tinggi", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.09), ncol=2, fontsize=8, frameon=False)
    return save_fig(fig, name)
fig_mismatch_scatter(SC)

def fig_mismatch_dist():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    sns.histplot(SC.M, bins=12, kde=True, color="#4c72b0", ax=ax, edgecolor="white", label="Mismatch bertanda (M)")
    sns.histplot(SC.absM, bins=12, color="#dd8452", ax=ax, alpha=.35, edgecolor="white", label="|M| (selisih absolut)")
    ax.axvline(0, color="k", ls="--", lw=1); ax.axvline(s_M, color="gray", ls=":"); ax.axvline(-s_M, color="gray", ls=":")
    ax.set_xlabel("Skor mismatch (satuan SD z)"); ax.set_ylabel("Jumlah provinsi"); ax.legend(fontsize=8)
    ax.set_title(f"Distribusi mismatch (SD={s_M:.2f}; skewness={stats.skew(SC.M):.2f})", fontsize=10)
    ax = axes[1]
    d = SC.sort_values("M")
    y = np.arange(len(d)); colors = np.where(d.M >= 0, "#d95f02", "#1b9e77")
    ax.hlines(y, d.M_CI95_bawah, d.M_CI95_atas, color="gray", lw=1, alpha=.7)
    ax.scatter(d.M, y, c=colors, s=26, zorder=3)
    ax.axvline(0, color="k", lw=1); ax.set_yticks(y); ax.set_yticklabels(d.provinsi, fontsize=7)
    ax.set_xlabel("Mismatch (garis abu-abu = interval bootstrap 95% atas parameter standardisasi)")
    ax.set_title("Peringkat mismatch provinsi (oranye: digital > numerasi; hijau: numerasi > digital)", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, "mismatch_distribusi_peringkat.png")
fig_mismatch_dist()

def fig_mismatch_jenjang():
    d = SC.sort_values("M", ascending=False)
    m = d[[f"M_{j}" for j in JEN3] + ["M"]]; m.columns = ["SD", "SMP", "SMA/SMK", "Provinsi (komposit)"]; m.index = d.provinsi
    fig, ax = plt.subplots(figsize=(6.4, 11))
    sns.heatmap(m, cmap="PuOr_r", center=0, annot=True, fmt=".1f", annot_kws={"fontsize": 7}, ax=ax, cbar_kws={"label": "Mismatch (+ digital > numerasi)", "shrink": .6})
    ax.set_title("Mismatch per jenjang dan komposit provinsi", fontsize=10); ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    return save_fig(fig, "mismatch_per_jenjang_heatmap.png")
fig_mismatch_jenjang()

def fig_sens():
    m = pd.DataFrame({nm[:44]: v.M.reindex(codes) for nm, v in VARIANTS.items()}).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 8.6))
    sns.heatmap(m, annot=True, fmt=".2f", cmap="YlGnBu", vmin=.3, vmax=1, ax=ax, annot_kws={"fontsize": 7}, cbar_kws={"shrink": .7})
    ax.set_title("Korelasi peringkat (Spearman) skor mismatch antarvarian definisi/metode", fontsize=10)
    ax.tick_params(labelsize=7)
    return save_fig(fig, "mismatch_sensitivitas_heatmap.png")
fig_sens()

SC_OUT = SC.reset_index().rename(columns={"M": "mismatch", "ZD": "Z_digital", "ZN": "Z_numerasi", "absM": "mismatch_absolut"})
cols_first = ["kode", "provinsi", "wilayah_pulau", "komputer_rata3", "internet_rata3", "numerasi_rata3", "Z_digital", "Z_numerasi", "mismatch", "M_z", "mismatch_absolut",
              "kuadran", "kelas_mismatch", "level_gabungan", "keterangan", "peringkat_M", "peringkat_absM"]
SC_OUT = SC_OUT[cols_first + [c for c in SC_OUT.columns if c not in cols_first]]
JENJANG_SC = SC.reset_index()[["kode", "provinsi"] + [f"{p}_{j}" for j in JEN3 for p in ("ZD", "ZN", "M")]]
save_excel(OUT / "mismatch_analysis.xlsx", {
    "skor_provinsi": SC_OUT, "skor_per_jenjang": JENJANG_SC, "top_positif": TOP_POS.reset_index(), "top_negatif": TOP_NEG.reset_index(),
    "selisih_absolut_vs_bertanda": ABSV.reset_index(), "sensitivitas_ringkas": SENS, "sensitivitas_peringkat": RANKS.reset_index(),
    "tanda_beda_komputer_vs_internet": SPLIT.reset_index(), "bootstrap_stabilitas": BOOT.join(MASTER.set_index("kode").provinsi).reset_index()},
    notes=["Mismatch = Z_digital − Z_numerasi (skor-z relatif antar 38 provinsi). Positif: digital relatif > numerasi; negatif: numerasi relatif > digital.",
           "Z_digital: rata-rata z(komputer), z(internet) per jenjang → distandarkan ulang → rata-rata 3 jenjang (bobot sama) → distandarkan ulang.",
           "kuadran: pemisahan median; kelas_mismatch memakai ambang 1 SD(M); 'Relatif seimbang' dapat berarti rendah-rendah atau tinggi-tinggi (lihat level_gabungan).",
           "Bootstrap hanya merefleksikan ketidakpastian parameter standardisasi; bukan galat pengukuran."])
