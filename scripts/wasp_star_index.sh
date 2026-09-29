#!/usr/bin/env bash
# Rebuild the STAR index the T2T RNA BAMs were aligned with, for WASP's remapping step. Parameters from the BAMs'
# STAR Log.out (reference_comparison_results_RNA/T2T_NCBI110/star_salmon/log/HSB100Log.out, "### STAR --runMode
# genomeGenerate"): chm13v2.0_maskedY_rCRS FASTA with RefSeq contig names, RefSeq RS_2023_03 GTF, sjdbOverhang 149,
# genomeSAindexNbases 14. The original GTF also carried chrM lines (unspliced, so no junctions); NCBI's has none.
# Known answer from that log: 344196 collapsed junctions from the GTF.
set -euo pipefail
D=/mnt/ssd/lalli/brainvar_hapmix_deploy
OUT=$D/wasp_20260928
STAR=/mnt/ssd/lalli/usr/local/bin/STAR-2.7.10a
FA_UCSC=/mnt/ssd/lalli/nf_stage/genome_refs/T2T-CHM13_v2_ncbi110/chm13v2.0_maskedY_rCRS.fasta
BAM=/mnt/data/lalli/nf_stage/reference_comparison_results_RNA/T2T_NCBI110/star_salmon/HSB100.markdup.sorted.bam
GTF_GZ=$OUT/annot/GCF_009914755.1_T2T-CHM13v2.0_genomic.RS_2023_03.gtf.gz
FA=$OUT/annot/chm13v2.0_maskedY_rCRS_RefSeq_names.fasta
GTF=$OUT/annot/GCF_009914755.1_T2T-CHM13v2.0_genomic.RS_2023_03.gtf
IDX=$OUT/star_index

awk 'NR == FNR { nc[">" $1] = ">" $2; next } /^>/ { h = $1; if (!(h in nc)) { print "unmapped contig " h > "/dev/stderr"; exit 1 } print nc[h]; next } { print }' \
  "$D/vcf/chr2nc.tsv" "$FA_UCSC" > "$FA.tmp"
mv "$FA.tmp" "$FA"
samtools faidx "$FA"
samtools view -H "$BAM" | awk -F'\t' '$1 == "@SQ" { sub("SN:", "", $2); sub("LN:", "", $3); print $2 "\t" $3 }' > "$OUT/annot/bam_contigs.tsv"
cut -f1,2 "$FA.fai" | diff - "$OUT/annot/bam_contigs.tsv"
echo "FASTA contigs, order and lengths identical to the BAM header ($(wc -l < "$OUT/annot/bam_contigs.tsv") contigs)"

gunzip -c "$GTF_GZ" > "$GTF.tmp"
mv "$GTF.tmp" "$GTF"

mkdir -p "$IDX"
"$STAR" --runMode genomeGenerate --runThreadN 32 --genomeDir "$IDX" --genomeFastaFiles "$FA" --genomeSAindexNbases 14 \
  --limitGenomeGenerateRAM 91168055040 --sjdbGTFfile "$GTF" --sjdbOverhang 149 --outFileNamePrefix "$IDX/"
grep -A4 "Processing pGe.sjdbGTFfile" "$IDX/Log.out"
grep -E "^genomeFileSizes|^versionGenome" "$IDX/genomeParameters.txt"
