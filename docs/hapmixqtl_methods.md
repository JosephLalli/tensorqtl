# Statistical methods of hapmixQTL

## cis-eQTL mapping from haplotype-resolved expression, with the quantifier's Gibbs variance as the allelic measurement error

This document specifies the statistics of hapmixQTL at the level needed to reproduce them: every quantity, its estimator, the reference distribution used for inference, and the defaults. It describes the shipped code in `tensorqtl/hapmixqtl.py` (default mode) and `tensorqtl/mixqtl_replication.py` (mixQTL mode). Equation numbers are referenced throughout.

Companion documents: `docs/pipeline_rules.md` states the rules that govern input values, units, the gene filter and the movement of covariates under permutation; `docs/outputs.md` defines every output column.

### The two modes

hapmixQTL ships exactly two configurations.

**Default mode** is the hapmixQTL estimator. It measures one cis effect through two channels built from Salmon point estimates against a personalized diploid transcriptome. The allelic channel regresses each donor's log2 ratio of haplotype expression on the signed heterozygosity of the variant; the total channel regresses the donor's log2 expression on half the variant's dosage. The allelic channel's per-donor measurement variance is the across-draw variance of Salmon's Gibbs samples (the "Gibbs variance"), used as the shape of the error variance with a scale fitted from the residuals; the total channel is unweighted. The two channel estimates are combined by inverse-variance weighting with Meier's correction for weights estimated from the same residuals, and the gene-level call is an empirical p-value from a donor-record permutation null. Sections 1 to 6 specify it.

**mixQTL mode** is a NumPy port of the published mixQTL estimator (Liang et al. 2021). It reads the Salmon point estimates and never the Gibbs draws, so it is the comparator that isolates what the draws contribute. Section 7 specifies it.

The variance configurations that earlier versions of the code offered are deprecated and quarantined (Section 10). Section 8 lists the entry points that refuse default mode, and Section 9 states the assumptions and limitations.

**Units.** Default mode works in log2 throughout: expression, allelic ratios, effect sizes and their variances. A slope of 1 is a twofold allelic fold change. mixQTL mode keeps its published natural-log response, so a mixQTL-mode slope equals the corresponding default-mode slope times $\ln 2$; convert before comparing raw slopes or standard errors.

## 1. Notation and inputs

One gene is analysed at a time. Donors are indexed $i = 1,\dots,N$ and cis variants $v = 1,\dots,V$.

**Quantification.** Salmon is run against each donor's personalized diploid transcriptome, which holds one copy of every transcript per haplotype, $L$ and $R$. Where a donor is homozygous across a transcript the two copies are identical; Salmon's indexer can collapse one copy, leaving no haplotype pair for that transcript in that donor. For donor $i$ and the gene, $p_{L,i}$ and $p_{R,i}$ are Salmon's point estimates (`quant.sf` NumReads) summed over transcripts that have both haplotype copies; $p_{T,i}$ is the point estimate summed over all transcripts of the gene; and $y^{(k)}_{L,i}$, $y^{(k)}_{R,i}$, $k = 1,\dots,K$, are Salmon's Gibbs draws summed the same way. $K$ is the number of Gibbs samples requested during quantification. $L_i$ is the donor's effective library size: edgeR's library size times its TMM normalization factor, computed on the eQTL gene set.

**Why the total is not $p_L + p_R$.** The second haplotype copy of a transcript exists only where the donor is heterozygous across it, so $p_L + p_R$ is a subtotal over heterozygous transcripts. Its level is set by local heterozygosity, which is in linkage disequilibrium with the variants being tested. The total must be summed over every transcript of the gene, and a donor who is homozygous across every transcript has $p_L = p_R = 0$ while still being expressed: such a donor carries no allelic information and stays in the total channel.

**Phased genotypes.** For variant $v$, $x_{L,v,i}, x_{R,v,i} \in \{0,1\}$ indicate the ALT allele on haplotypes $L$ and $R$, the same haplotype labels the quantification used. The dosage is $g_{v,i} = x_{L,v,i} + x_{R,v,i}$ and the signed heterozygosity is
$$ s_{v,i} = x_{L,v,i} - x_{R,v,i} \in \{-1, 0, +1\}, \tag{1} $$
so $s = +1$ when the ALT allele sits on haplotype $L$, $-1$ when it sits on $R$, and $0$ for homozygotes.

**Covariates.** $C_t$ ($N \times p_t$) is the total channel's covariate matrix. Its columns are of two kinds that enter the regression identically and differ only under permutation (Section 5.3): covariates tied to the RNA record (age, sex, RIN, expression principal components) and covariates tied to the genotypes (genotype principal components). $C_a$ ($N \times p_a$) is the allelic channel's covariate matrix; by default it is empty and the allelic regression runs through the origin (Section 3.3).

**Parameter of interest.** $\beta = \log_2(e_{\mathrm{ALT}}/e_{\mathrm{REF}})$, the log2 allelic fold change: if the REF haplotype of a heterozygote is expressed at level $e$, the ALT haplotype is expressed at $e\,2^{\beta}$.

## 2. Default-mode inputs

### 2.1 Phenotypes and working variances

`prepare_default_inputs(pL, pR, pT, eff_lib_size, yL, yR, kappa=0.5, count_noise=True)` returns the four per-donor matrices $A$, $T$, $V_a$, $V_t$ (genes by donors). With $\kappa = 0.5$,
$$ a_i = \log_2\frac{p_{L,i} + \kappa}{p_{R,i} + \kappa}, \qquad t_i = \log_2\!\Big(\frac{p_{T,i} + 0.5}{L_i + 1} \times 10^6\Big), \tag{2} $$
$$ v_{a,i} = \frac{1}{K}\sum_{k}\big(a^{(k)}_i - \bar a_i\big)^2 + q_i, \qquad a^{(k)}_i = \log_2\frac{y^{(k)}_{L,i} + \kappa}{y^{(k)}_{R,i} + \kappa}, \qquad q_i = \frac{1}{(\ln 2)^2}\Big(\frac{1}{p_{L,i} + \kappa} + \frac{1}{p_{R,i} + \kappa}\Big), \tag{3} $$
$$ v_{t,i} = 1, \tag{4} $$
where $\bar a_i$ is the mean of the $a^{(k)}_i$ over draws and the variance has divisor $K$.

**Allelic admission.** A donor-gene pair enters the allelic channel only if $v_{a,i} > 10^{-12}$, $p_{L,i} + p_{R,i} > 0$, and it is not the case that exactly one of $p_{L,i}$, $p_{R,i}$ is below 0.5 reads. Otherwise $v_{a,i}$ is set to 0, which removes the pair from the allelic channel (Section 4.1); $a_i$ keeps its finite value. Every donor enters the total channel, including donors with zero total count, whose $t_i = \log_2(0.5/(L_i+1)\times 10^6)$ is a real low-expression observation.

**What each piece is.**

- *Values from point estimates, variance from the draws.* Every analyzed value (the allelic ratio, total expression, effective library sizes and expression principal components) comes from Salmon's point estimates. The Gibbs draws are used only for the allelic channel's working measurement variance.
- *The half-read total.* The offset in (2) is half a read added before normalization. Unlike $\log_2(\mathrm{CPM}+1)$, its count-scale offset does not change with library size. The default total channel uses this transform with unit working variance.
- *The counting term $q_i$.* $q_i$ is the delta-method variance of the empirical log2 ratio of two Poisson counts with the half-count (Haldane-Anscombe) correction, evaluated at the point estimate. Gibbs draws may already carry some counting variation, so the term can overlap with their variance. It is on by default (`count_noise=True`); the Salmon runner's `--no-count-noise` removes it.
- *The Gibbs prior.* Gibbs variance at low information can depend materially on the quantifier's prior and sampling settings. Record the Salmon version and options with each run, and use a sensitivity analysis when low-count donor-gene pairs drive a result.

### 2.2 Covariates

The expression principal components are built on the same half-read transform as $t_i$, on the same eQTL gene set and with the same effective library sizes (`docs/pipeline_rules.md`, rules 2 and 3). The Salmon runner refuses covariates whose recorded build (`covariate_build.json`, written by `scripts/build_covariates.py --point-estimates`) names another unit, gene set or set of library sizes, unless `--covariates-unverified` is passed. Genotype principal components are passed separately (`genotype_covariates_df`; in the runner, the columns listed in `genotype_covariates.txt` beside `--covariates`), because they move with the genotypes rather than with the RNA record under permutation (Section 5.3).

### 2.3 Input contract

`map_nominal` and `map_cis` read every input frame by position, so they check the inputs once per call (`_validate_inputs`) and raise `ValueError` on any violation. $A$, $T$, $V_a$ and $V_t$ must have the same phenotype rows and donor columns, in the same order; every entry must be finite; $V_a$ and $V_t$ must be nonnegative; phenotype, donor and variant identifiers must be unique. Phase must be supplied as both $x_L$ and $x_R$ or as neither; when supplied, their columns must equal the genotype frame's columns in order (`_assert_phase_columns`), their rows must equal the genotype frame's rows in identity and order, and every phase value must be finite. Without these checks such inputs were accepted, with consequences that left every output column finite and plausible: a missing value in $A$ dropped the allelic channel for that gene and moved the lead, a negative or missing $V_a$ was read as "no information", reordered $V_a$ columns or phase rows changed the result, and one phase frame alone ran a total-only analysis. An excluded donor-gene pair is represented as $V_a = 0$ with a finite $A$, which is what `prepare_default_inputs` writes. Genotype dosages are not checked beyond unique variant identifiers; a missing dosage coded $-9$, as tensorQTL's genotype readers write it, is imputed to the variant's mean dosage (Section 5.1).

### 2.4 Other input routes

**Raw BED input through the command line.** The `hapmixqtl` and `hapmixqtl_nominal` modes of `tensorqtl` require `--hap_A`, `--hap_T` and `--hap_Va`. `--hap_Vt` is optional: omitting it supplies unit total working variance, and a supplied file is used exactly as a custom override. BED files carry no counts or library sizes, so this route cannot verify that $T$ is the half-read transform or that the allelic admission rule was applied; the caller must guarantee both.

**Optional count cutoffs.** `count_cutoff_masks` builds per-channel admission masks from point-estimate counts with mixQTL-style thresholds (a floor and a ceiling on both haplotypes for the allelic channel, a floor on the total over all transcripts for the total channel), and the mappers accept them as `keep_a_df` and `keep_t_df`. An excluded donor is given zero working variance in that channel, the state of a donor with no coverage. They are off by default and exist for matched-donor comparisons with mixQTL mode (runner `--asc-cutoff`, `--asc-cap`, `--trc-cutoff`, `--mixqtl-cutoffs`).

## 3. Model

### 3.1 Two channels for one parameter

For a causal variant $v$ with log2 allelic fold change $\beta$,
$$ a_i = \beta\, s_{v,i} + C_{a,i}\, b_a + e_{a,i}, \tag{5} $$
$$ t_i = \alpha_t + \beta\, \tfrac{1}{2} g_{v,i} + C_{t,i}\, b_t + e_{t,i}. \tag{6} $$

Equation (5) is exact under the definition of $\beta$: a heterozygote with the ALT allele on $L$ has expected $\log_2(y_L/y_R) = \beta$, one with the ALT allele on $R$ has $-\beta$, and a homozygote has 0, which is $\beta s_{v,i}$. Equation (6) is first order. Relative to a homozygous-REF donor the expected total is $(2 - g + g\,2^{\beta})/2$, so the expected log2 total shifts by $\log_2(1 + g(2^{\beta} - 1)/2)$, which equals $\beta g/2$ exactly at $g \in \{0, 2\}$. At $g = 1$ it is
$$ \log_2\frac{1 + 2^{\beta}}{2} = \frac{\beta}{2} + \log_2\cosh\!\Big(\frac{\beta \ln 2}{2}\Big), $$
since $(1 + 2^{\beta})/2 = 2^{\beta/2}\cosh(\beta\ln 2/2)$. The heterozygote therefore sits above the line by $\log_2\cosh(\beta \ln 2/2) \approx \beta^2 \ln 2/8$, an amount even in $\beta$: 0.014 at $\beta = 0.4$, 0.055 at $\beta = 0.8$ and 0.085 at $\beta = 1$. Adding a heterozygote indicator to the total-channel design would make (6) exact; neither hapmixQTL nor the parent method does so.

### 3.2 Error variance

$$ \mathrm{Var}(e_{a,i}) = \sigma_a^2\, v_{a,i}, \qquad \mathrm{Var}(e_{t,i}) = \sigma_t^2\, v_{t,i} = \sigma_t^2, \tag{7} $$
with errors independent across donors. The working variances $v_{a,i}$ set the relative weights of the donors; the scales $\sigma_a^2$ and $\sigma_t^2$ are unknown and are fitted from the residuals at each variant (Section 4.3). The working variances are therefore a shape only: multiplying every $v_{a,i}$ of a gene by a constant leaves every slope, standard error and p-value unchanged. In the code this is `tau_mode='zero'` (no additive variance term) together with `se_mode='fitted'` (the residual scale is estimated).

**Relation to mixQTL.** mixQTL's allelic error variance is $\sigma^2(1/Y_1 + 1/Y_2)$: the counts set its shape and $\sigma^2$ is a free multiplicative scale. Equation (7) has the same structure with the Gibbs variance (plus the counting term) as the shape, and without mixQTL's cap on how far one donor's weight may exceed another's (Section 7). mixQTL's total channel is an unweighted least-squares regression with one flat variance per gene, which is what $v_{t,i} = 1$ gives here, on a different transform.

**Why these weights.** The allelic working variance represents donor-specific uncertainty in a within-donor contrast, so inverse-$v_a$ weighting gives more weight to more precise contrasts. The total channel uses unit working variance and estimates one residual scale from its regression. In both channels the working variance sets relative weights only; the fitted scale in (7) supplies the absolute residual variance.

There is no additive between-donor variance term. The deprecated models that estimated one from each gene's own residuals are described, with the two structural reasons they were withdrawn, in Section 10.

### 3.3 Covariates per channel

$a_i$ is a within-donor contrast, so a covariate acting equally on both haplotypes (library size, expression principal components, sex, age) cancels in it. The default allelic design has no covariate columns and no intercept (`ase_covariates_df=None`). This preserves the fit when the haplotype labels of any donor are swapped, since $a_i$ and $s_{v,i}$ both change sign. `ase_covariates_df=SAME_COVARIATES` applies the total channel's covariates to the allelic channel as well, and a custom DataFrame supplies its own; custom allelic covariates must have a meaningful orientation under a swap of haplotype labels, which the code does not check. The total channel always has an intercept plus $C_t$.

## 4. Estimation at one variant

### 4.1 Weights and whitening

The informative set of channel $c \in \{a, t\}$ is $I_c = \{i : v_{c,i} > \varepsilon\}$ with $\varepsilon = 10^{-12}$; under (4), absent count cutoffs, every donor is informative for the total channel. The weights and square-root weights are
$$ w_{c,i} = \begin{cases} 1/\max(v_{c,i}, 10^{-8}) & i \in I_c \\ 0 & \text{otherwise,} \end{cases} \qquad r_{c,i} = \sqrt{w_{c,i}}. \tag{8} $$
Whitened responses and predictors are $y^{*}_c = r_c \circ y_c$ and $x^{*}_{c,v} = r_c \circ x_{c,v}$, with $y_a = a$, $y_t = t$, $x_{a,v} = s_v$ and $x_{t,v} = g_v/2$. The whitened null designs are
$$ D_a = \mathrm{diag}(r_a)\,C_a, \qquad D_t = \big[\, r_t \;\; \mathrm{diag}(r_t)\,C_t \,\big], \tag{9} $$
with $q_a = p_a$ and $q_t = 1 + p_t$ columns. With the thin QR factorization $D_c = Q_c R_c$, residualization is the projection
$$ P_c z = z - Q_c\,(Q_c^{\top} z). \tag{10} $$
With no allelic covariates $D_a$ has no columns and $P_a$ is the identity.

### 4.2 Design checks

**Sparse-channel rule.** A channel with fewer informative donors than its design has columns plus two, $|I_c| < q_c + 2$, is switched off: every weight is set to zero and the channel contributes nothing to the gene. The default allelic channel therefore stays on from 2 informative donors and the total channel from $p_t + 3$. This rule is separate from the allelic admission floor of Section 4.5, which decides whether an allelic channel that is on enters the combination.

**Rank of the covariate design.** `_combine_covariates` joins the RNA-tied and genotype-tied covariates, with the genotype-tied columns last, and requires the matrix $[\mathbf{1}, C_t]$ to be finite and of full column rank; otherwise it raises `ValueError` naming the rank. A QR factorization of a rank-deficient design returns a basis direction the covariates do not span, which changes the projection (10) and charges a degree of freedom for a column that carries none.

**Rank of the weighted design.** A full-rank covariate matrix can still give one gene a rank-deficient weighted design when zero weights remove the donors that distinguish two columns, for example a categorical covariate that is constant over the donors carrying weight in an allelic channel with `SAME_COVARIATES` or under count cutoffs. `WeightedResidualizer` declares column $j$ of the weighted design dependent when the diagonal entry of $R$ satisfies $|R_{jj}| \le \max(N, q_c)\,\epsilon_{\mathrm{mach}}\,\lVert D_{c,j}\rVert$, where $\epsilon_{\mathrm{mach}}$ is the floating-point machine epsilon (a tolerance relative to each column's own norm), and raises `ValueError` if any column is dependent. A switched-off channel has nothing to project and is not checked.

### 4.3 Per-channel slope and standard error

$$ xy_{c,v} = \langle P_c x^{*}_{c,v},\, P_c y^{*}_c \rangle, \qquad xx_{c,v} = \lVert P_c x^{*}_{c,v} \rVert^2, \qquad \hat\beta_{c,v} = \frac{xy_{c,v}}{xx_{c,v}}, \tag{11} $$
$$ e_{c,v} = P_c y^{*}_c - \hat\beta_{c,v}\, P_c x^{*}_{c,v}, \qquad \mathrm{SE}_{c,v} = \sqrt{\frac{\lVert e_{c,v} \rVert^2}{\nu_c\, xx_{c,v}}}, \qquad \nu_c = |I_c| - 1 - q_c. \tag{12} $$
This is weighted least squares with an estimated dispersion: $\lVert e_{c,v}\rVert^2/\nu_c$ is the fitted $\sigma_c^2$ of (7) at that variant. So $\nu_a = n_a - 1$ for the default through-origin allelic design and $\nu_t = n_t - 2 - p_t$, where $n_c = |I_c|$. The degrees of freedom count informative donors only, because a zero-weight donor adds nothing to the residual sum of squares and charging it a degree of freedom would shrink the scale (`_residual_dof`). A predictor is valid only if $xx_{c,v} > 10^{-12}\max(\lVert x^{*}_{c,v}\rVert^2, 10^{-30})$; otherwise $\hat\beta_{c,v} = 0$, $\mathrm{SE}_{c,v} = \infty$, and the channel contributes nothing at that variant (no heterozygous informative donor, or a predictor in the span of the design).

In the scan and the permutations (Section 5) the same quantities are computed from dot-product summaries: with $yy_c = \lVert P_c y^{*}_c\rVert^2$, $\mathrm{RSS}_{c,v} = yy_c - xy_{c,v}^2/xx_{c,v}$ (formed in double precision because the subtraction cancels when the fit is good) and $\mathrm{SE}^2_{c,v} = \mathrm{RSS}_{c,v}/(\nu_c\, xx_{c,v})$, which is (12).

### 4.4 Combining the channels

With $w_{c,v} = 1/\mathrm{SE}^2_{c,v}$ (zero for an invalid predictor, a switched-off channel, or an allelic channel below the admission floor of Section 4.5),
$$ \hat\beta_v = \frac{w_{a,v}\hat\beta_{a,v} + w_{t,v}\hat\beta_{t,v}}{w_{a,v} + w_{t,v}}, \qquad \mathrm{SE}^{\circ}_v = \big(w_{a,v} + w_{t,v}\big)^{-1/2}. \tag{13} $$
This is the inverse-variance weighted mean of the two channel estimates. $\mathrm{SE}^{\circ}_v$ is the plug-in standard error before the correction of Section 4.5.

The two channel estimators are treated as independent, and this is not an approximation of convenience: under random phase $s_v$ and $g_v/2$ are orthogonal predictors ($\mathbb{E}[s_v \mid g_v = 1] = 0$), so any noise that $a_i$ and $t_i$ share within a donor is projected onto orthogonal directions by the two regressions and the correlation of $\hat\beta_a$ and $\hat\beta_t$ is zero even when $a$ and $t$ are strongly correlated. `tests/test_hapmixqtl_calibration.py` asserts this at an inferential correlation of 0.9. The inferential covariance of $a$ and $t$ across draws is therefore not needed: the default inputs do not compute it, and no mapping function uses it (the BED reader loads an optional covariance file for inspection only).

### 4.5 Nominal p-values: per-channel references, Welch-Satterthwaite, Meier's correction and the allelic admission floor

**Per-channel references.** Each channel's p-value is referred to the degrees of freedom of the scale its standard error was fitted with:
$$ p_{a,v} = 2\Pr\big(t_{\nu_a} > |\hat\beta_{a,v}/\mathrm{SE}_{a,v}|\big), \qquad p_{t,v} = 2\Pr\big(t_{\nu_t} > |\hat\beta_{t,v}/\mathrm{SE}_{t,v}|\big). \tag{14} $$
Under (7) with normal errors each is exact. $\nu_a$ and $\nu_t$ are reported as `dof_a` and `dof_t`, and both they and the p-value are NaN for a channel that is switched off.

**The combined reference.** The combined statistic is referred to the Welch-Satterthwaite degrees of freedom of its variance estimate:
$$ \nu_v = \frac{(w_{a,v} + w_{t,v})^2}{w_{a,v}^2/\nu_a + w_{t,v}^2/\nu_t}. \tag{15} $$
The Welch-Satterthwaite approximation treats a fixed linear combination $\sum_k c_k s_k^2$ of independent variance estimates on $\nu_k$ degrees of freedom as a scaled $\chi^2$ whose first two moments match the combination's, which gives $(\sum_k c_k s_k^2)^2 / \sum_k (c_k s_k^2)^2/\nu_k$ degrees of freedom. The combined slope is $\sum_k c_k \hat\beta_k$ with $c_k = w_k/(w_a + w_t)$, its variance estimate is $\sum_k c_k^2 \mathrm{SE}_k^2 = 1/(w_a + w_t)$, and substituting $c_k^2\mathrm{SE}_k^2 = w_k/(w_a + w_t)^2$ gives (15) (`_satterthwaite_dof`). $\nu_v$ lies between $\min(\nu_a, \nu_t)$ and $\nu_a + \nu_t$, varies per variant because the weights do, equals $\nu_a$ where the total channel carries no weight and $\nu_t$ where the allelic channel carries none, and is NaN where neither carries weight, so that pair has no p-value. It is reported per pair as `dof_nominal`.

**Weights estimated from the same residuals.** The weights of (13) are estimated from the residuals whose slopes they combine: a chance-small $\mathrm{SE}_a$ both enlarges $|\hat\beta_a/\mathrm{SE}_a|$ and raises the allelic share of the combination. This is the Graybill-Deal effect: the common-mean estimator that weights each estimate by its estimated inverse variance (Graybill and Deal 1959) has a true variance larger than its plug-in $1/\sum_k w_k$, because chance-small variance estimates receive chance-large weight.

**Meier's correction.** Meier (1953) gives the variance of such a weighted mean to first order in $1/\nu_k$. Write the estimated weight as $\hat w_k = w_k(1 + d_k)$, where $\hat w_k$ is $w_k$ times $\nu_k/\chi^2_{\nu_k}$ under normality, so that to first order $\mathbb{E}[d_k] = 2/\nu_k$ and $\mathrm{Var}(d_k) = 2/\nu_k$, independent of the channel slopes. With shares $f_k = w_k/W$, $W = w_a + w_t$, and $S = \sum_k f_k(1 - f_k)/\nu_k$, the true variance of the weighted mean is $(1/W)(1 + 2S)$ while the reported $1/\hat W$ has expectation $(1/W)(1 - 2S)$; their ratio is $1 + 4S$, and with two channels $f_a(1 - f_a) = f_t(1 - f_t) = f_a f_t$. The shipped statistic is therefore
$$ m_v = 1 + 4 f_{a,v} f_{t,v}\Big(\frac{1}{\nu_a} + \frac{1}{\nu_t}\Big), \qquad \mathrm{SE}_v = \mathrm{SE}^{\circ}_v\sqrt{m_v}, \qquad T_v = \hat\beta_v/\mathrm{SE}_v, \qquad p_{\mathrm{nom},v} = 2\Pr\big(t_{\nu_v} > |T_v|\big), \tag{16} $$
with $m_v$ evaluated at the estimated shares (`_meier_factor`) and $\nu_v$ from (15) unchanged. $m_v = 1$ exactly wherever fewer than two channels carry weight, since the combination is then one channel whose $t$ is exact on its own $\nu_c$. The correction is part of the statistic everywhere default mode computes it: in `map_nominal`, in the observed scan and every permutation of `map_cis` (with $m_v$ recomputed from each permutation's refitted weights, Section 5.3), at the reported lead, and in the leave-one-donor-out refits (Section 5.6). Because $m_v$ varies per variant it is not a monotone map of the uncorrected statistic and can change which variant is the lead.

**The allelic admission floor.** The allelic channel enters (13), (15) and (16) only for a gene with at least 15 informative allelic donors (`MIN_ALLELIC_DONORS`, `_allelic_admitted`, reported as `allelic_admitted`). Below the floor $w_{a,v} = 0$: the combined slope, standard error and p-value are the total channel's, taken verbatim, and $p_a$ is still reported on its own $\nu_a$. The count is per gene; it does not check that a given variant has heterozygous informative donors. The floor follows the parent method's minimum for combining its two channels (`META_N_CUTOFF`, Section 7) and avoids combining a channel whose residual scale has very few degrees of freedom. It is waived when the gene has no total channel, where there is no combination to protect and the allelic channel is the statistic whenever it is on. It is applied identically to the observed scan, every permutation, the lead and the leave-one-donor-out refits; the informative allelic donors are the same set under every permutation because each donor's weight moves with its record.

**Other standard errors.** Two non-default standard errors keep a single shared reference $\nu = N - 2 - \max(p_t, p_a)$, with no admission floor and no Meier factor, and the dof columns then report that $\nu$. The HC1 sandwich (`se_mode='robust'`, `map_nominal` only) estimates the slope variance as $\sum_i x_i^2 e_i^2/(\sum_i x_i^2)^2$ on the whitened, residualized data, times the small-sample factor $N/(N - 1 - q_c)$. The known-variance standard error $1/\sqrt{xx}$ (`se_mode='model'`) is deprecated (Section 10).

## 5. Cis scan and gene-level inference

### 5.1 Window and variant filters

The window is the transcription start site $\pm W$, or the gene span extended by $W$ on each side when a start and end are given, with $W = 10^6$ by default. Missing dosages are imputed to the variant's mean dosage. Optionally, variants with in-sample minor allele frequency below $m$ are removed. `map_cis` additionally drops every variant whose imputed dosage is constant across donors; `map_nominal` keeps such variants and reports an infinite standard error for the total channel and a NaN combined p wherever the allelic channel carries no weight either (`dof_nominal` NaN, Section 4.5). The constant-dosage filter is not quite monomorphism: an all-heterozygous variant has constant dosage and is dropped by `map_cis`, although $s$ still varies there. Phase values must be finite (Section 2.3); the Salmon runner's VCF reader keeps biallelic SNPs only and discards any record with an unphased or missing call in any donor.

### 5.2 Scan

The weights (8) and projections (10) are computed once per gene. With $S$ the $V \times N$ matrix of $s_{v,i}$ and $G$ that of $g_{v,i}$,
$$ \tilde S = P_a\big(S \circ r_a^{\top}\big), \quad \tilde G = P_t\big(\tfrac{1}{2} G \circ r_t^{\top}\big), \quad \tilde y_a = P_a y^{*}_a, \quad \tilde y_t = P_t y^{*}_t, $$
$$ xy_a = \tilde S\, \tilde y_a, \quad xx_a = \mathrm{rowsum}(\tilde S \circ \tilde S), \quad xy_t = \tilde G\, \tilde y_t, \quad xx_t = \mathrm{rowsum}(\tilde G \circ \tilde G), $$
and the per-variant statistic $T^2_v$ follows from (12), (13) and (16) (`_combined_tstat2`). The lead is $\arg\max_v T^2_v$, with undefined statistics ranked last.

### 5.3 Permutation null

**Donor records with haplotype-label swaps (the default, `perm_scheme='records_signflip'`).** Draw $P$ permutations $\pi_1, \dots, \pi_P$ of $\{1, \dots, N\}$ from a seeded generator, shared across genes and across the two channels. Under $\pi$, donor $i$ receives donor $\pi(i)$'s record: its whitened phenotype value, its weight and its row of RNA-tied covariates move together, while the genotypes and the genotype-tied covariate columns stay in place. In addition, each permuted record's haplotype labels are swapped with probability one half: signs $\epsilon_{\pi,i} \in \{-1, +1\}$ are drawn from the same generator immediately after the permutations, and donor $i$ receives $\epsilon_{\pi,i}\,y^{*}_{a,\pi(i)}$ in place of $y^{*}_{a,\pi(i)}$, which negates its allelic log ratio and leaves its weight and covariate row unchanged. The total channel is never swapped, since a swap leaves total expression unchanged.

The whitened null design is rebuilt for each permutation from the permuted weights and covariate rows,
$$ D^{\pi}_t = \big[\sqrt{w^{\pi}_t},\ \sqrt{w^{\pi}_t} \odot C^{\pi}_{\mathrm{RNA}},\ \sqrt{w^{\pi}_t} \odot C_{\mathrm{geno}}\big], $$
with orthonormal basis $Q^{\pi}_t$; the permuted phenotype and the weighted predictors are residualized on it, and each channel's $xy^{\pi}_{c,v}$, $xx^{\pi}_{c,v}$ and $yy^{\pi}_c$ give that permutation's fitted scales, weights, Meier factor and combined statistic $T^{2,\pi}_v$ exactly as in Section 5.2 (`_record_permutation_channel`, `_combined_tstat2`). Then
$$ M_\pi = \max_v T^{2,\pi}_v. \tag{17} $$
$\nu_a$ and $\nu_t$ do not change under permutation, because each donor's weight travels with its record. The default allelic design has no covariates, so the allelic channel costs two matrix products per batch of permutations; the total channel is re-residualized with a batched QR per chunk of permutations.

By relabeling the donors, $T^{2,\pi}$ equals the observed statistic computed with the genotype columns, and the genotype-tied covariate rows, permuted together by $\pi^{-1}$ while the records stay fixed: without the swap this is the genotype-permutation null of FastQTL and tensorQTL, with each donor's weight carried along (`perm_scheme='records'`, checked by `tests/test_hapmixqtl_perm_scheme.py`). The swap adds a symmetry of the allelic null that the permutation alone does not use: the labels $L$ and $R$ are arbitrary phase order, so under the null a record's allelic ratio is as likely to carry either sign. Without the swap, the through-origin allelic slope has a permutation mean equal to the gene's net allelic imbalance times the variant's phase lopsidedness; with it, the permutation distribution of the allelic numerator is symmetric about zero for every gene and phase pattern. The swap changes the null only; the observed statistic keeps whatever chance offset its own records carry, which under arbitrary phase labels is exchangeable with the swapped draws.

**Genotype-tied covariates.** `map_cis(genotype_covariates_df=...)` and `_combine_covariates` place the genotype principal components last in $C_t$, and `WeightedResidualizer.n_fixed_cov` tells `_record_permutation_channel` how many trailing columns to hold in genotype order. This reaches the allelic channel only under `SAME_COVARIATES`; the default allelic design has no covariates (`test_genotype_tied_covariates_permute_with_the_genotypes` pins the relabeling identity).

**Retained alternatives.** `perm_scheme='records'` is the same permutation without the swap. `perm_scheme='residuals'` is the Freedman-Lane scheme in whitened space: the whitened residuals of the covariate-only fit are standardized by $\sqrt{1 - h_{c,i}}$, $h_{c,i}$ the leverage of the null design, permuted among each channel's informative donors with predictors, weights and covariates held in place, and the statistic recomputed. It requires those standardized residuals to be exchangeable across donors, an assumption that can fail when residual size remains related to the working variance. Neither alternative is the default.

### 5.4 Empirical p and Beta approximation

The scanned statistic is mapped to a correlation scale with the constant $\nu = N - 2 - \max(p_t, p_a)$: $r^2_{\mathrm{nom}} = T^2_{\mathrm{lead}}/(T^2_{\mathrm{lead}} + \nu)$ and $r^2_{\pi} = M_\pi/(M_\pi + \nu)$, and
$$ p_{\mathrm{perm}} = \frac{\#\{\pi : r^2_{\pi} \ge r^2_{\mathrm{nom}}\} + 1}{P + 1}. \tag{18} $$
The map is monotone and the same for the observed lead and every permutation, so it leaves the lead and $p_{\mathrm{perm}}$ exactly as the statistic determines them; it is not the lead's t reference, which is $\nu_v$ of (15). Because Meier's factor is part of both the observed and the permuted statistics, they are one statistic and (18) keeps its exchangeability argument; a correction applied to the observed scan alone would bias it.

As in FastQTL and tensorQTL, an effective degrees of freedom $\nu^{*}$ is found by Newton iteration on $\log\nu$, starting at $\nu$, such that the method-of-moments first shape parameter of a Beta distribution fitted to $\{p_t(r^2_\pi; \nu^{*})\}$ equals 1, where $p_t(r^2; \nu) = 2\Pr(t_\nu > \sqrt{\nu r^2/(1 - r^2)})$; a $\mathrm{Beta}(k, n)$ is then fitted to those values by maximum likelihood from the moment estimates, and
$$ p_{\mathrm{beta}} = F_{\mathrm{Beta}(k,n)}\big(p_t(r^2_{\mathrm{nom}}; \nu^{*})\big). \tag{19} $$
$p_{\mathrm{perm}}$ and $p_{\mathrm{beta}}$ are the gene-level p-values. A gene in which neither channel carries weight was not tested: its $p_{\mathrm{nom}}$, $p_{\mathrm{perm}}$ and `dof_nominal` are NaN and no Beta fit is made.

### 5.5 Reported lead quantities

The lead's `slope` is $\hat\beta_v$ of (13), its `slope_se` is $\mathrm{SE}_v$ of (16) including Meier's factor, and its `pval_nominal` is $p_{\mathrm{nom},v}$ on $\nu_v$, all at the lead. Its per-channel slopes and standard errors, $\alpha_{\mathrm{cis}}$ and the cis/trans p (Section 6.2) are on the same fitted scale. Default mode has no additive variance term to re-estimate, so the lead is never refitted: `map_cis(tau_refit=True)` has no effect unless the deprecated `tau_mode='estimate'` is selected, the Salmon runner does not pass it, and the command line's `--tau_refit` is accepted and inert.

The lead is the variant with the largest combined $|T|$; with a per-variant reference its `pval_nominal` need not be the gene's smallest `map_nominal` p. A lead's nominal p is the best of the window and is never a gene-level p. The detection call is $p_{\mathrm{beta}}$ when the Beta approximation is fitted (`beta_approx=True`, the default) and $p_{\mathrm{perm}}$ otherwise; a failed Beta fit leaves $p_{\mathrm{beta}}$ missing without raising, so a run should be checked for missing values in that column.

### 5.6 Leave-one-donor-out influence at the lead

A single donor's record can carry a gene-level call, and the empirical p does not protect against it (Section 9, item 10). `map_cis` therefore reports, for each gene's lead in default mode, the donor whose exclusion moves the lead's combined statistic furthest toward zero (`loo_donor`) and the lead's nominal p without that donor (`loo_pval_nominal`).

**What it computes.** For every donor that is informative in at least one channel (every donor when $v_{t,i} = 1$ and no count cutoffs are applied), the lead variant is refitted with that donor's weight set to zero in both channels. Each refit recomputes the per-channel slopes and fitted scales (11)-(12); $\nu_a$ and $\nu_t$ lose one degree of freedom in each channel the donor was informative in; the sparse-channel rule and the allelic admission floor are re-applied to the reduced counts; and the combination (13), the Welch-Satterthwaite reference (15) and Meier's factor (16) are recomputed. The reported donor minimizes the refit's $|T|$, and `loo_pval_nominal` is $2\Pr(t_{\nu'} > |T'|)$ with $T'$ and $\nu'$ that refit's statistic and its own Welch-Satterthwaite degrees of freedom. Because the reported refit is the minimum over single exclusions, its $|T'|$ can still exceed the lead's own $|T|$ when every exclusion strengthens the lead, and `loo_pval_nominal` is then smaller than `pval_nominal`.

**Exactness.** In default mode a donor's weight depends only on its own working variance, and no variance parameter is estimated from the gene as a whole, so excluding one donor changes no other donor's weight. Each refit is therefore exactly the statistic `map_nominal` reports at the lead variant with that donor masked out of both channels (`keep_a_df` and `keep_t_df` False for that donor); `tests/test_hapmixqtl.py::TestLeadInfluence` pins this agreement at a relative tolerance of $10^{-4}$. Against an explicit loop that rebuilt the channels and refitted once per donor, on 40 simulated genes with 92 donors, 17 covariates and 14 to 92 informative allelic donors, the vectorized computation named the same donor in 40 of 40 genes, with $|T|$ and the degrees of freedom within $7.8 \times 10^{-5}$ relative (single precision).

**Computation.** All refits for a gene run as one batch through the record-permutation machinery: refit $k$ is the identity order with donor $k$'s weight zeroed (`_record_permutation_channel` with a mask), so one call per channel gives every refit's dot products. The degrees of freedom, the sparse-channel rule and the admission floor depend only on which channels the excluded donor was informative in, so the combined statistic is evaluated once per such class (`_lead_influence`). Measured cost: 7.5 ms per lead at 92 donors and 17 covariates on an NVIDIA L4 GPU.

**Limits.** First, the lead is held fixed. Excluding a donor can move the lead to another variant, and `loo_pval_nominal` is the original lead's nominal p, not the gene's best p without that donor. Second, $p_{\mathrm{perm}}$ and $p_{\mathrm{beta}}$ are not recomputed, so the columns say how much the lead's nominal statistic rests on one donor, not whether the gene-level call survives that donor's removal. Third, a donor with leverage 1 in a channel's covariate design, for example the only donor at one level of a categorical covariate, would leave that design rank-deficient when excluded and is skipped. When no donor can be evaluated, `loo_donor` is empty and `loo_pval_nominal` is NaN; `loo_pval_nominal` is also NaN when the selected exclusion leaves no channel carrying weight. The columns are computed only under `tau_mode='zero'` with `se_mode='fitted'` and a finite lead statistic. They are a diagnostic and are never used as a filter.

### 5.7 Algorithm

Algorithm 1 (one gene; `map_cis` in default mode).

1. Inputs: $a_i$, $t_i$, $v_{a,i}$, $v_{t,i}$ from (2)-(4), checked by the input contract (Section 2.3); covariates checked for rank (Section 4.2).
2. Genotypes: $G$, $S$ for the window (Section 5.1); mean imputation, the optional MAF filter and the constant-dosage filter, applied to $G$ and $S$ together.
3. Per channel: informative set $I_c$; the sparse-channel rule; weights (8); weighted design (9), its rank check and $Q_c$; the allelic admission floor.
4. Scan: $xy$, $xx$, $yy$ (Section 5.2); per-variant fitted scales, combination and Meier factor; $T^2_v$; lead $= \arg\max_v T^2_v$.
5. Null: $P$ seeded permutations of the donor records with haplotype-label swaps; $M_\pi$ by (17); $p_{\mathrm{perm}}$ by (18); $p_{\mathrm{beta}}$ by (19).
6. Lead: per-channel $\hat\beta_c$ and $\mathrm{SE}_c$, combined $\hat\beta$, $\mathrm{SE}$, $p_{\mathrm{nom}}$ on $\nu_v$; $\alpha_{\mathrm{cis}}$ and the cis/trans p (Section 6.2); leave-one-donor-out influence (Section 5.6).
7. Output one row (`docs/outputs.md`, mode `hapmixqtl`).

The scan costs $O(VN)$ and the permutations $O(VNP)$ as dense matrix products; arithmetic is single precision except where noted, and runs on a GPU when one is available.

## 6. Effect sizes and diagnostics

### 6.1 The allelic fold change

$\hat\beta$ estimates $\log_2(e_{\mathrm{ALT}}/e_{\mathrm{REF}})$, so the allelic fold change is $\mathrm{aFC} = 2^{\hat\beta}$, and a nominal 95% interval at a prespecified variant is $2^{\hat\beta \pm c\,\mathrm{SE}}$ with $c$ the 0.975 quantile of $t$ on $\nu_v$ (`dof_nominal`). At a selected lead that coverage is not guaranteed, because the lead was chosen for its large statistic. With the default through-origin allelic design, homozygotes have $s = 0$ and contribute nothing to the allelic $xy_a$ or $xx_a$, but informative homozygotes do contribute to the allelic residual scale through their residuals $a_i$. $\hat\beta_t$ uses every informative total-channel donor. The effect allele is the VCF's ALT allele.

### 6.2 The cis/trans diagnostic

The combination (13) assumes both channels estimate the same $\beta$, that is, a purely cis effect. A trans component acting on total expression only, reference mapping bias attenuating the allelic channel, phasing error or feature-level misquantification violate it, and the combined slope then averages two different quantities. Because the channel estimators are uncorrelated (Section 4.4), the difference has variance $\mathrm{SE}_a^2 + \mathrm{SE}_t^2$ with no covariance term, and
$$ z = \frac{\hat\beta_a - \hat\beta_t}{\sqrt{\mathrm{SE}_a^2 + \mathrm{SE}_t^2}}, \qquad p_{\mathrm{cis/trans}} = 2\Pr(t_{\nu_d} > |z|), \qquad \nu_d = \frac{(\mathrm{SE}_a^2 + \mathrm{SE}_t^2)^2}{\mathrm{SE}_a^4/\nu_a + \mathrm{SE}_t^4/\nu_t}, \qquad \alpha_{\mathrm{cis}} = \frac{\hat\beta_a}{\hat\beta_t}. \tag{20} $$
$\nu_d$ is the Welch-Satterthwaite construction of (15) with coefficients $+1$ and $-1$ (`cis_trans_diagnostic`). A small $p_{\mathrm{cis/trans}}$ flags a variant whose combined slope should not be read as a cis log2 aFC; it is a diagnostic, not a filter, and is undefined when either channel has no finite standard error. The allelic slope used here is the channel's own estimate even for a gene below the admission floor.

### 6.3 The reference-bias gate

hapmixQTL does not model reference mapping bias and is anticonservative in its presence, so the precondition is tested rather than assumed. For gene $g$ and donor $i$ the haplotype orientation is
$$ o_{g,i} = \operatorname{sign}\Big(\sum_{v \in F_g} d_{v,i}\, s_{v,i}\Big), \tag{21} $$
over the gene's feature sites $F_g$ (the heterozygous sites inside the exon union when exons are supplied, else inside the gene body, else the heterozygous site nearest the TSS), with $d_{v,i}$ the allele-specific read depth at the site when per-site counts are supplied and 1 otherwise (`orient_haplotypes`). Mapping bias favours REF at every site carrying reads, so the bias it adds to the haplotype totals is proportional to this depth-weighted sum. The reference count of a donor-gene pair is $p_{R,i}$ when $o = +1$ (ALT on $L$) and $p_{L,i}$ when $o = -1$, the alternate count the other; a pair is usable if $o \ne 0$ and its allele-specific total is at least 10. For each gene with at least 3 usable donors, $f_g$ is the pooled reference fraction; over $G \ge 5$ such genes, with at least 20 usable pairs in all,
$$ m = \frac{1}{G}\sum_g f_g, \qquad z = \frac{m - 0.5}{\mathrm{sd}(f_g)/\sqrt{G}}, \qquad p = 2\Phi(-|z|), \tag{22} $$
and bias is declared at $p < 10^{-3}$ (`reference_bias_diagnostic`). Genuine allele-specific expression raises REF or ALT with a sign that is arbitrary per gene and cancels in $m$, while mapping bias accumulates, which is why the test is clustered at the gene. The Salmon runner applies the gate to the point-estimate haplotype counts and refuses to map when it fires, unless `--force` is given.

## 7. mixQTL mode

`tensorqtl/mixqtl_replication.py` is a NumPy port of the published mixQTL estimator, `hakyimlab/mixqtl` at commit `624ae4421e43fe152f3851f795b6326901dfeb93`. The module docstring lists the implementation differences between that estimator and hapmixQTL. It is the no-draws comparator: holding the data and gene set fixed, the pair of modes isolates whether propagating the Gibbs draws changes the result relative to the count-based precision model.

**Inputs.** `inputs_from_point_estimates(pL, pR, pT)` returns the point-estimate counts $y_1 = p_L$, $y_2 = p_R$ and $y_T = p_T$ and refuses an array with a draws axis: mixQTL mode never uses Gibbs draws. The library size $L_i$ is the same edgeR effective library size as in default mode. The haplotype dosages are $h_1 = x_L$ and $h_2 = x_R$, with missing values imputed to 0.5 as the reference does. Responses are in natural log.

**Allelic channel (`asc_channel`).** A donor is admitted when both haplotype counts lie in $[c_a, c_{\max}]$. The response is $\log(y_1/y_2)$ with no pseudocount, regressed through the origin on $h_1 - h_2$ with the harmonic weights
$$ w_i = \Big(\frac{1}{y_{1,i}} + \frac{1}{y_{2,i}}\Big)^{-1}, \tag{23} $$
the reciprocal of the delta-method Poisson variance of $\log(y_1/y_2)$. The weights are capped as a fold limit: with $n$ admitted donors, $\kappa_w = \min(\texttt{weight\_cap}, \lfloor n/10 \rfloor)$, and every weight above $\kappa_w \min_i w_i$ is set to that value. The standard error is the fitted residual scale over $\sqrt{\sum_i w_i x_i^2}$ with $n - 1$ degrees of freedom. The channel needs more than 2 admitted donors, and variants with zero variance among the admitted donors are dropped.

**Total channel (`trc_channel`).** The response is $\log(y_T/2/L_i)$ minus a covariate offset, over donors with $y_T \ge c_t$ and a finite response, regressed without weights on $(h_1 + h_2)/2$ with an intercept; the standard error has $n - 2$ degrees of freedom. The covariate offset is mixQTL's two-step procedure (`covariate_offset`): regress $\log(y_T/L_i/2)$ on every covariate with an intercept, keep the covariates whose $|t| > 2$, refit on the kept set, and take their fitted contribution without the intercept as the offset.

**Combination (`meta_analyze`).** Each channel's p-value is referred to the standard normal when its sample size exceeds 15 and to $t$ on $n$ degrees of freedom otherwise. When both channels have at least 15 admitted donors (`META_N_CUTOFF`), the two estimates are combined by inverse-variance weighting, $w_k = 1/\mathrm{SE}_k^2$, $\hat\beta = \sum_k w_k\hat\beta_k/\sum_k w_k$, $\mathrm{SE} = (\sum_k w_k)^{-1/2}$, with the p-value on the normal reference for $n_{\mathrm{trc}} + n_{\mathrm{asc}}$ samples. Otherwise the result is the channel with the larger sample size, with missing variants filled from the other. Note that this 15 is a count of donors for combining the channels, not a per-donor read floor.

**Cutoffs.** `PUBLISHED_CUTOFFS` are those of the GTEx v8 driver that produced the paper and are the default: $c_t = 100$, $c_a = 50$, $c_{\max} = 1000$ and `weight_cap` 10. `PACKAGE_DEFAULT_CUTOFFS` are the R function's signature defaults ($c_t = 20$, $c_a = 5$, $c_{\max} = 5000$ and `weight_cap` 100), kept as a sensitivity setting. The upper cap was designed as a guard against alignment artifacts; callers using abundance estimates should report how many donor-gene pairs each cutoff removes.

**Gene-level inference (`mixqtl_permutation_scan`).** The reference permutes the phenotype bundle: the allelic response, weight and admission mask, and the total response and its mask, move together under one index while the genotype design stays in place. The RNA-tied covariates and the library size move with the bundle. When genotype-tied covariates are given they stay with the genotypes and the two-step offset is refitted on each permuted dataset, the rule default mode also follows. The function returns, for each permutation, the maximum $|$meta statistic$|$ over the window; the gene-level empirical p is the number of permutation maxima at least as large as the observed maximum, plus one, over the number of finite permutation maxima plus one, the analogue of (18) (`scripts/mixqtl_gene_level_typeI.py` computes it this way). Two defects of the distributed reference are documented in the module: its permutation path zeroes cutoff-failing weights before taking the minimum for the fold cap, so any cutoff failure zeroes every weight (reproduced only under `strict_reference_cap=True`; the default takes the minimum over admitted donors, as the reference's non-permutation path does); and $\lfloor n/10 \rfloor$ makes the cap 0 for 3 to 9 admitted donors and 1 for 10 to 19, where the allelic channel becomes unweighted.

**How to run it.** `mixqtl_scan` runs one gene's nominal pass and `mixqtl_permutation_scan` its null. The port's algebra is checked per variant against `numpy.linalg.lstsq`, and the cutoff, cap and degrees-of-freedom rules are pinned to the R source lines they encode in `tests/test_mixqtl_replication.py`.

## 8. Entry points that refuse default mode

Default mode's statistics rest on a residual scale fitted per channel. Three entry points have no such scale and refuse `tau_mode='zero'` with a `ValueError` rather than return uncalibrated results.

**Fine-mapping (`map_susie`).** `map_susie` stacks the two whitened, covariate-residualized channels into one $2N$-row design with a shared effect vector and passes it to tensorQTL's SuSiE implementation. SuSiE (the sum of single effects) is a Bayesian variable-selection regression that models a window's signal as a sum of up to $L$ single-effect regressions, each placing one effect on one variant, and reports per-variant posterior inclusion probabilities and credible sets. The stacked design has no per-channel residual scale: by default it fixes the residual variance at 1, which treats the working variances as the entire error variance, and with `estimate_residual_variance=True` it fits one scale shared by both channels. With `tau_mode='zero'` neither matches the default model, so it is refused. The function's `tau_mode='estimate'` path (deprecated, Section 10) remains only for compatibility; the command line does not offer a fine-mapping mode, and credible sets and posterior inclusion probabilities are outside the validated default surface. `fine_mapping_provenance` classifies a stored fine-mapping summary by its recorded `tau_mode`.

**The STR and multi-allelic second pass (`map_str_curvature`, `map_multiallelic`).** These fit a joint model per locus after the scan: a linear-plus-quadratic model in repeat length per haplotype for short tandem repeats, and a categorical model with one indicator per non-reference allele for multi-allelic sites. Both use a stacked known-variance generalized least-squares fit (`_joint_gls`) with no fitted residual scale, so with `tau_mode='zero'` their standard errors would take the Gibbs variance and the unit total working variance as the entire error variance. `_second_pass` therefore refuses this pairing. The Salmon runner does not run the second pass. STR and multi-ALT rows can still enter the `map_cis` scan as ordinary one-column rows (runner `--str-vcf` and `--multiallelic`; `scripts/str_integrate.py`): an STR as per-haplotype repeat length, so its slope is the log2 aFC per repeat unit, and a multi-ALT site as one split row per ALT allele.

**Standard errors without a permutation counterpart.** `map_cis` accepts only `se_mode='fitted'` (and the deprecated `'model'`); the HC1 sandwich has no permutation counterpart and is available in `map_nominal` only. The command line offers `--se_mode fitted` (default) and `robust`, so `--mode hapmixqtl --se_mode robust` stops with an error.

## 9. Assumptions and limitations

1. **Phase.** $s$ is taken from the phased VCF. A phase switch between the tested variant and the gene flips one donor's allelic contribution and attenuates $\hat\beta_a$. Verify the phase convention and haplotype labels before mapping.
2. **Reference mapping bias** is not modelled. It is tested for (Section 6.3), and the method is not valid when the gate fails.
3. **Log-normal errors and no default read floor.** The allelic response is a pseudocounted log2 ratio treated as normal with variance proportional to $v_{a,i}$. The approximation is coarse at low counts, and the pseudocount shrinks ratios toward zero. The one-sided admission rule removes the most extreme case; `--asc-cutoff` can impose a stronger floor for a sensitivity analysis.
4. **The quantifier's prior shapes low-information variance.** Gibbs variance can depend on Salmon's prior and sampling settings when few reads distinguish the haplotypes (Section 2.1).
5. **One residual scale per channel.** Equation (7) assumes that within a gene the residual variance is proportional to the working variance. Misspecification affects nominal standard errors and power. The gene-level detection call is the empirical permutation p-value.
6. **First-order total channel and one marginal variant.** Equation (6) omits the heterozygote excess of Section 3.1, and the combined statistic is a marginal single-variant test. Default mode does not provide a multi-variant fit (Section 8).
7. **Approximate combined reference.** Each channel uses its fitted-scale t reference. The combined nominal p-value uses Welch-Satterthwaite degrees of freedom and Meier's first-order correction; the empirical p-value in (18) is the gene-level detection quantity.
8. **Independence across donors.** Relatedness and repeated measures are not modelled. They can violate the assumptions of both the regressions and donor-record permutations.
9. **Sparse channels.** The allelic channel is reported on its own degrees of freedom when it can be fitted, but it enters a two-channel combination only from 15 informative donors (Section 4.5).
10. **Single-record influence.** One donor can materially affect a selected lead. The leave-one-donor-out columns in Section 5.6 expose that influence; they do not remove the donor or recompute the gene-level permutation p-value.
11. **Exchangeability.** The retained residual permutation (`perm_scheme='residuals'`) requires exchangeable standardized residuals. The default record permutation instead tests association after reassigning complete RNA records while genotype-tied covariates remain with the genotypes.
12. **Different transcript sets.** $p_L$ and $p_R$ cover transcripts quantified per haplotype, while $p_T$ covers every transcript of the gene. For a multi-isoform gene with transcript-specific effects, $\beta_a$ and $\beta_t$ may differ; equation (20) is the diagnostic.
13. **Beta approximation.** `pval_beta` is an approximation fitted to the permutation maxima. Inspect the fit and use `pval_perm` when enough permutations are available for the required tail resolution.

## 10. Deprecated configurations

These paths remain for compatibility and reproduction. They are not reachable from the Salmon runner or the command line.

**Variance models.** The additive model $\mathrm{Var}(e_i) = v_i + \tau_g$, the two-component and library-scaled forms $c_g v_i + \tau_g$ and $d_i(c_g v_i + \tau_g)$, empirical-Bayes shrinkage of $(c_g, \tau_g)$, and the known-variance standard error $1/\sqrt{xx}$ are quarantined in `tensorqtl/fitted_variance.py`. They fit a gene's variance function from that gene's squared residuals and then use that fit to weight the same residuals. With both $c_g$ and $\tau_g$ free, rescaling every $v_{ig}$ by $k$ returns $c_g/k$ and leaves the weights unchanged, so the absolute calibration of the Gibbs variance cannot affect the result. Default mode never imports this module (`tests/test_fitted_variance_quarantine.py`). The legacy lead refit (`tau_refit`) belongs to this path and never runs in default mode.

**Legacy phenotypes.** `compute_summaries_from_gibbs` builds natural-log means over Gibbs draws with Gibbs variances in both channels. `summaries_from_point_estimates` builds point-estimate values with a $\log_2(\mathrm{CPM}+1)$ total and Gibbs variance in both channels. These functions remain for compatibility; neither defines the default input contract of Section 2.

**Alternative permutation schemes.** `perm_scheme='records'` and `perm_scheme='residuals'` remain available as described in Section 5.3. The default is `records_signflip`.

## 11. Symbols, defaults and implementation

| Symbol or rule | Meaning | Default | Code |
|---|---|---|---|
| $\kappa$ | pseudocount of the allelic ratio (2), (3) | 0.5 | `prepare_default_inputs(kappa)` |
| half-read offset | 0.5 read added to $p_T$ and 1 to $L_i$ in (2) | fixed | `half_read_log_cpm` |
| $q_i$ | counting term of (3) | on | `prepare_default_inputs(count_noise)`; runner `--count-noise` / `--no-count-noise` |
| allelic admission | $v_a > 10^{-12}$, $p_L + p_R > 0$, not exactly one haplotype below 0.5 | fixed | `prepare_default_inputs` |
| $v_t$ | total working variance (4) | 1 | `prepare_default_inputs`; CLI default when `--hap_Vt` is omitted |
| input contract | shared rows and columns, finite values, nonnegative variances, unique ids, both phase frames or neither | enforced | `_validate_inputs`, `_assert_phase_columns` |
| $\varepsilon$ | informative threshold | $10^{-12}$ | `_channel_weights(eps)` |
| variance floor in (8) | | $10^{-8}$ | `_channel_weights` |
| sparse-channel rule | minimum informative donors for a channel to be on | $q_c + 2$ ($q_a = p_a$, $q_t = 1 + p_t$) | `_min_informative` |
| rank checks | covariate design with intercept full rank; weighted design full rank | enforced | `_combine_covariates`, `WeightedResidualizer` |
| $C_a$ | allelic covariate design | none, through the origin | `ase_covariates_df` (`None`, `SAME_COVARIATES` or a DataFrame) |
| error model (7) | $\sigma_c^2 v_{c,i}$, scale fitted per variant | shipped | `tau_mode='zero'`, `se_mode='fitted'`; `_wls_regression(fitted=True)`, `_combined_tstat2(fitted=True)` |
| $\nu_a$, $\nu_t$ | per-channel residual degrees of freedom, references of $p_a$, $p_t$ (14) | $\lvert I_a\rvert - 1 - q_a$, $\lvert I_t\rvert - 1 - q_t$ | `_residual_dof`, `_channel_dof`, `_reported_dof`; columns `dof_a`, `dof_t` |
| $\nu_v$ | Welch-Satterthwaite reference of the combined p (15) | per variant | `_satterthwaite_dof`; column `dof_nominal` |
| $m_v$ | Meier's factor (16) | per variant | `_meier_factor` |
| allelic admission floor | informative allelic donors for the allelic channel to enter the combination; waived without a total channel | 15 | `MIN_ALLELIC_DONORS`, `_allelic_admitted`, `min_allelic_donors`; column `allelic_admitted` |
| $\nu$ | constant correlation-scale map of (18) and starting value of the Beta fit; reference of the non-default standard errors | $N - 2 - \max(p_t, p_a)$ | `map_cis`, `map_nominal` |
| $W$ | cis window | $10^6$ | `window` |
| $m$ | in-sample MAF filter | 0 (off) | `maf_threshold` |
| $P$ | permutations | 10,000 | `nperm` |
| permutation null | donor records with haplotype-label swaps | `records_signflip` | `perm_scheme`; `_record_permutation_channel` |
| genotype-tied covariates | columns of $C_t$ held in genotype order under permutation | none | `genotype_covariates_df`, `WeightedResidualizer.n_fixed_cov` |
| Beta approximation | fitted, else $p_{\mathrm{beta}}$ is missing | on | `beta_approx`; `calculate_beta_approx_pval` in `core.py` |
| seed | permutation generator | none in the API and command line; 42 in the Salmon runner | `seed`; runner `SEED` |
| predictor validity | $xx > 10^{-12}\max(\lVert x^{*}\rVert^2, 10^{-30})$ | | `_wls_regression` |
| leave-one-donor-out | donor minimizing the lead's refit $\lvert T\rvert$ and that refit's p | computed in default mode | `_lead_influence`; columns `loo_donor`, `loo_pval_nominal` |
| gate thresholds | min allele-specific depth 10, min usable pairs 20, $p < 10^{-3}$ | | `reference_bias_diagnostic` |

Functions, by section: `prepare_default_inputs` and `half_read_log_cpm` (2.1); `count_cutoff_masks` (2.4); `_validate_inputs` and `_assert_phase_columns` (2.3); `_prepare_channels`, `_channel_weights` and `_zero_degenerate_ase_weights` (4.1); `_combine_covariates` and `WeightedResidualizer` (4.2); `_wls_regression` (4.3); `calculate_hapmixqtl_nominal` (4.4, 4.5); `_satterthwaite_dof`, `_meier_factor` and `_allelic_admitted` (4.5); `calculate_hapmixqtl_permutations`, `_combined_tstat2` and `_record_permutation_channel` (5.2, 5.3); `_leverage_standardized` and `_permute_within_informative` (the retained residual scheme, 5.3); `_lead_influence` (5.6); `map_nominal` and `map_cis` (5); `cis_trans_diagnostic` (6.2); `orient_haplotypes` and `reference_bias_diagnostic` (6.3); `inputs_from_point_estimates`, `harmonic_weights`, `apply_weight_cap`, `covariate_offset`, `asc_channel`, `trc_channel`, `meta_analyze`, `mixqtl_scan` and `mixqtl_permutation_scan` in `mixqtl_replication.py` (7); `map_susie`, `fine_mapping_provenance`, `map_multiallelic`, `map_str_curvature` and `_second_pass` (8).

## References

- Kumasaka N, Knights AJ, Gaffney DJ. Fine-mapping cellular QTLs with RASQUAL and ATAC-seq. Nature Genetics 2016.
- Liang Y, Aguet F, Barbeira AN, Ardlie K, Im HK. A scalable unified framework of total and allele-specific counts for cis-QTL, fine-mapping, and prediction (mixQTL). Nature Communications 2021.
- Mohammadi P, Castel SE, Brown AA, Lappalainen T. Quantifying the regulatory effect size of cis-acting genetic variation using allelic fold change. Genome Research 2017.
- Patro R, Duggal G, Love MI, Irizarry RA, Kingsford C. Salmon provides fast and bias-aware quantification of transcript expression. Nature Methods 2017.
- Robinson MD, Oshlack A. A scaling normalization method for differential expression analysis of RNA-seq data (TMM). Genome Biology 2010.
- Ongen H, Buil A, Brown AA, Dermitzakis ET, Delaneau O. Fast and efficient QTL mapper for thousands of molecular phenotypes (FastQTL). Bioinformatics 2016.
- Taylor-Weiner A, Aguet F, et al. Scaling computational genomics to millions of individuals with GPUs (tensorQTL). Genome Biology 2019.
- Wang G, Sarkar A, Carbonetto P, Stephens M. A simple new approach to variable selection in regression, with application to genetic fine mapping (SuSiE). Journal of the Royal Statistical Society B 2020.
- Freedman D, Lane D. A nonstochastic interpretation of reported significance levels. Journal of Business and Economic Statistics 1983. Winkler AM, Ridgway GR, Webster MA, Smith SM, Nichols TE. Permutation inference for the general linear model. NeuroImage 2014. (The retained residual permutation of Section 5.3.)
- Satterthwaite FE. An approximate distribution of estimates of variance components. Biometrics Bulletin 1946. Welch BL. The generalization of 'Student's' problem when several different population variances are involved. Biometrika 1947. (The degrees of freedom (15) and (20).)
- Graybill FA, Deal RB. Combining unbiased estimators. Biometrics 1959. Meier P. Variance of a weighted mean. Biometrics 1953; 9(1):59-73. (The estimated-weights effect on (15) and its first-order correction (16), Section 4.5.)
- MacKinnon JG, White H. Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. Journal of Econometrics 1985. (The HC1 sandwich of Section 4.5.)
- Castel SE, Mohammadi P, Chung WK, Shen Y, Lappalainen T. Rare variant phasing and haplotypic expression from RNA sequencing with phASER. Nature Communications 2016. (The per-site counts used by the reference-bias gate of Section 6.3.)
