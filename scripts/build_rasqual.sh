#!/usr/bin/env bash
# Build the REAL RASQUAL (Kumasaka et al. 2016) from source.
#
# The upstream Makefile targets the authors' cluster: it expects CLAPACK headers
# (blaswrap.h, f2c.h, clapack.h) that are not packaged on modern Linux, and
# predates GCC 10's -fno-common default. RASQUAL calls only four LAPACK routines
# (dsytrf_, dsytri_, dsytrs_, dgetri_), all present in system liblapack, so the
# fix is minimal compatibility headers -- the numerics are unchanged.
#
# Verified against the authors' bundled example: recovers C11orf21 / rs2521269
# with phi_hat = 0.525 and delta_hat = 3.3e-5, the magnitudes the paper reports.
set -euo pipefail
DEST="${1:-rasqual_src}"

# Install the build dependencies only if they are actually missing. This ran
# apt-get unconditionally, which made the script unusable for anyone without
# root -- including on the controlled-access server this is meant for, where
# the libraries are typically already present and sudo typically is not.
have_deps() {
    { pkg-config --exists gsl 2>/dev/null || [ -e /usr/include/gsl/gsl_version.h ]; } \
      && ls /usr/lib/*/liblapack* >/dev/null 2>&1 \
      && ls /usr/lib/*/libblas* >/dev/null 2>&1 \
      && { [ -e /usr/include/zlib.h ] || ls /usr/include/*/zlib.h >/dev/null 2>&1; }
}
if have_deps; then
    echo "build deps already present (gsl, lapack, blas, zlib) -- skipping apt-get"
else
    sudo=""; [ "$(id -u)" -eq 0 ] || sudo=sudo
    if ! $sudo apt-get install -y libgsl-dev liblapack-dev libblas-dev zlib1g-dev; then
        echo "could not install build deps automatically." >&2
        echo "Install libgsl-dev liblapack-dev libblas-dev zlib1g-dev and re-run." >&2
        exit 1
    fi
fi

[ -d "$DEST" ] || git clone --depth 1 https://github.com/natsuhiko/rasqual.git "$DEST"
cd "$DEST/src"
mkdir -p compat

cat > compat/blaswrap.h <<'EOF'
/* Stand-in for CLAPACK's blaswrap.h. RASQUAL calls the Fortran-suffixed LAPACK
   symbols directly, so no name remapping is needed against system LAPACK. */
#ifndef BLASWRAP_H
#define BLASWRAP_H
#endif
EOF

cat > compat/f2c.h <<'EOF'
/* Stand-in for CLAPACK's f2c.h: RASQUAL uses only `integer`, plus the min/max
   macros that f2c.h supplies to sort.c and nbem.c. */
#ifndef F2C_H
#define F2C_H
typedef int integer;
typedef double doublereal;
typedef int logical;
typedef int ftnlen;
#endif
#ifndef min
#define min(a,b) ((a) <= (b) ? (a) : (b))
#endif
#ifndef max
#define max(a,b) ((a) >= (b) ? (a) : (b))
#endif
EOF

cat > compat/clapack.h <<'EOF'
/* Prototypes for the four LAPACK routines RASQUAL calls; these resolve to
   system liblapack -- the same reference implementation CLAPACK wraps. */
#ifndef CLAPACK_H
#define CLAPACK_H
#include "f2c.h"
int dsytrf_(char*, integer*, doublereal*, integer*, integer*, doublereal*, integer*, integer*);
int dsytri_(char*, integer*, doublereal*, integer*, integer*, doublereal*, integer*);
int dsytrs_(char*, integer*, integer*, doublereal*, integer*, integer*, doublereal*, integer*, integer*);
int dgetri_(integer*, doublereal*, integer*, integer*, doublereal*, integer*, integer*);
int dgetrf_(integer*, integer*, doublereal*, integer*, integer*, integer*);
#endif
EOF

# -fcommon: upstream relies on pre-GCC-10 tentative definitions (rng, rngT).
# -lf2c dropped: only CLAPACK needs it; system LAPACK does not.
sed -i 's|^CFLAGS := .*|CFLAGS := $(CFLAGS) -std=gnu99 -fcommon -Icompat -I/usr/include -I/usr/include/gsl -fpic -g -O2|' Makefile
sed -i 's| -lf2c||' Makefile
rm -f ./*.o rasqual
make
echo "built: $(pwd)/rasqual"

# smoke test on the authors' own example (region fed via zcat, no tabix needed)
cd ..
zcat data/chr11.gz | awk -F'\t' '$1=="11" && $2>=2315000 && $2<=2340000' \
  | ./src/rasqual -y data/Y.bin -k data/K.bin -n 24 -j 1 -l 378 -m 62 \
      -s 2316875,2320655,2321750,2321914,2324112 \
      -e 2319151,2320937,2321843,2323290,2324279 -t -f C11orf21 -z \
  | head -1 | cut -f1,2,14,13
echo "(expect: C11orf21  rs2521269  phi~0.525  delta~3.3e-5)"
