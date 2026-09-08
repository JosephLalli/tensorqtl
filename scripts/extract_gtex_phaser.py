"""Stream a subset of the GTEx v8 phASER haplotype-expression matrix.

Produces the real-data inputs behind docs/ase_validation.md sec 7d. Public
data, no dbGaP needed. Requires the sample->tissue map, which comes from
GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt in the same bucket:

  curl -sO https://storage.googleapis.com/adult-gtex/annotations/v8/\
metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
  # then build samp2tissue.json mapping SAMPID -> SMTSD for SMAFRZE == RNASEQ

Usage:
  python3 scripts/extract_gtex_phaser.py \
      phASER_GTEx_v8_matrix.gw_phased.txt.gz "Muscle - Skeletal" 400 out.npz

Extracts, for one tissue, genes with enough allele-specific coverage to be
usable, writing yL / yR integer matrices as .npz. Streams over HTTP so the
528 MB gzip is never stored.
"""
import sys, gzip, json, subprocess, numpy as np

URL_BASE = ("https://storage.googleapis.com/adult-gtex/haplotype-expression/v8/"
            "haplotype-expression-matrices/")
fname = sys.argv[1]          # phASER_GTEx_v8_matrix.gw_phased.txt.gz | phASER_WASP_...
tissue = sys.argv[2]         # e.g. "Muscle - Skeletal"
n_genes = int(sys.argv[3])
out = sys.argv[4]
# mixQTL-style usability filters (R/mixqtl.R): asc_cutoff=5 per haplotype
MIN_AS_PER_HAP, MIN_SAMPLES = 5, 100

s2t = json.load(open('samp2tissue.json'))
p = subprocess.Popen(['bash', '-c', f'curl -sS --max-time 3000 "{URL_BASE}{fname}"'],
                     stdout=subprocess.PIPE)
gz = gzip.GzipFile(fileobj=p.stdout)

hdr = gz.readline().decode().rstrip('\n').split('\t')
cols = [i for i, s in enumerate(hdr[4:], start=4) if s2t.get(s) == tissue]
samples = [hdr[i] for i in cols]
print(f"{len(samples)} samples in tissue {tissue!r}", flush=True)

genes, YL, YR = [], [], []
scanned = 0
for raw in gz:
    scanned += 1
    f = raw.decode().rstrip('\n').split('\t')
    if len(f) < 5:
        continue
    yl = np.empty(len(cols), np.int32); yr = np.empty(len(cols), np.int32)
    ok = True
    for k, i in enumerate(cols):
        v = f[i]
        j = v.find('|')
        if j < 0:
            ok = False; break
        try:
            yl[k] = int(v[:j]); yr[k] = int(v[j + 1:])
        except ValueError:
            ok = False; break
    if not ok:
        continue
    usable = int(np.sum((yl >= MIN_AS_PER_HAP) & (yr >= MIN_AS_PER_HAP)))
    if usable < MIN_SAMPLES:
        continue
    genes.append(f[1]); YL.append(yl); YR.append(yr)
    if len(genes) % 50 == 0:
        print(f"  kept {len(genes)} / scanned {scanned}", flush=True)
    if len(genes) >= n_genes:
        break

np.savez_compressed(out, yL=np.array(YL), yR=np.array(YR),
                    genes=np.array(genes), samples=np.array(samples))
print(f"wrote {out}: {len(genes)} genes x {len(samples)} samples "
      f"(scanned {scanned} rows)", flush=True)
try:
    p.kill()
except Exception:
    pass
