# Statistical methods of hapmixQTL

## cis-eQTL detection and effect-size estimation from haplotype-resolved expression with inferential uncertainty

This document specifies the statistics implemented in `tensorqtl/hapmixqtl.py` at the level needed to reproduce them: every quantity, its estimator, the null distribution used for inference, and the defaults. Section 7 lists the measured properties on which the choices rest, Section 8 the assumptions, Section 9 a symbol-to-code map. Equation numbers are referenced throughout.

### Summary

For each gene, hapmixQTL takes two measurements per sample from the posterior draws of a diploid (personalized) quantification: the log ratio of the two haplotypes' expression, and the log total. Both are regressed on the phased genotype of each cis variant in a known-variance generalized least squares (GLS) whose per-sample error variance is the sum of the quantifier's inferential variance, a Poisson counting term, and a per-gene between-sample variance $\tau$ estimated across samples. The two regressions estimate the same parameter, the log allelic fold change of the ALT haplotype relative to the REF haplotype, and are combined by inverse-variance weighting. The lead variant is the maximum of the combined statistic over the window; its gene-level significance is empirical, from a Freedman-Lane permutation of leverage-standardized whitened residuals, with a Beta approximation of the permutation distribution. Effect sizes are reported at the lead with $\tau$ re-estimated under the alternative so that a gene's own signal does not shrink its reported scale.

## 1. Notation and inputs

One gene is analysed at a time. Samples are indexed $i = 1,\dots,N$ and cis variants $v = 1,\dots,V$.

**Quantification draws.** A diploid quantification assigns each sample's reads to its two personalized haplotype transcriptomes and returns $K$ posterior draws (Salmon Gibbs draws; $K = 200$ in the BrainVar deployment). For sample $i$ and draw $k$ the gene has haplotype-resolved read counts $y^{(k)}_{L,i}$ and $y^{(k)}_{R,i}$ (haplotypes $L$ and $R$, summed over the gene's transcripts that were quantified per haplotype) and a total $y^{(k)}_{T,i}$ summed over all of the gene's transcripts.

**Phased genotypes.** For variant $v$, $x_{L,v,i}, x_{R,v,i} \in \{0,1\}$ indicate the ALT allele on haplotypes $L$ and $R$, the same haplotype labels the quantification used. The dosage is $g_{v,i} = x_{L,v,i} + x_{R,v,i} \in \{0,1,2\}$ and the signed heterozygosity is
$$ s_{v,i} = x_{L,v,i} - x_{R,v,i} \in \{-1, 0, +1\}, \tag{1} $$
so $s = +1$ when the ALT allele sits on haplotype $L$, $-1$ when on $R$, and $0$ for homozygotes.

**Covariates.** $C_t$ ($N \times p_t$) is the covariate matrix of the total channel and $C_a$ ($N \times p_a$) that of the allelic channel. The deployment driver uses $p_a = 0$ (intercept only; Section 3.3); the library default is $C_a = C_t$.

**Parameter of interest.** $\beta$ is the log allelic fold change (natural logarithm): if the REF haplotype of a heterozygote is expressed at level $e$ and the ALT haplotype at $e\,\mathrm{e}^{\beta}$, then $\beta = \log(e_{\mathrm{ALT}} / e_{\mathrm{REF}})$.

## 2. Per-sample summaries from the posterior draws

With pseudocount $\kappa = 0.5$, for every sample and draw
$$ a^{(k)}_i = \log\!\big(y^{(k)}_{L,i} + \kappa\big) - \log\!\big(y^{(k)}_{R,i} + \kappa\big), \qquad t^{(k)}_i = \log\!\big(y^{(k)}_{T,i}/2 + \kappa\big). \tag{2} $$
The summaries are the draw means and population variances (divisor $K$):
$$ a_i = \frac{1}{K}\sum_k a^{(k)}_i, \quad t_i = \frac{1}{K}\sum_k t^{(k)}_i, \quad v^{\mathrm{inf}}_{a,i} = \frac{1}{K}\sum_k \big(a^{(k)}_i - a_i\big)^2, \quad v^{\mathrm{inf}}_{t,i} = \frac{1}{K}\sum_k \big(t^{(k)}_i - t_i\big)^2. \tag{3} $$
$v^{\mathrm{inf}}$ is the inferential variance: the uncertainty of assigning the sample's reads to haplotypes and transcripts. The inferential covariance $\mathrm{Cov}_k(a^{(k)}_i, t^{(k)}_i)$ is computed for inspection and deliberately not used (Section 4.4).

**Counting noise.** Gibbs draws reassign one fixed set of reads, so their across-draw variance excludes the sampling variance of the counts themselves; a sample whose count is identical in every draw (zero reads, or reads compatible with nothing else) has $v^{\mathrm{inf}} = 0$ and would otherwise receive the largest weight in the gene. With $\bar y$ denoting the draw mean, the plug-in Poisson variance of a log count ($\mathrm{Var}\log(y+\kappa) \approx 1/(y+\kappa)$) is added:
$$ v_{a,i} = \begin{cases} v^{\mathrm{inf}}_{a,i} + \dfrac{1}{\bar y_{L,i} + \kappa} + \dfrac{1}{\bar y_{R,i} + \kappa} & \bar y_{L,i} + \bar y_{R,i} > 0 \\[2ex] 0 & \text{otherwise,} \end{cases} \qquad v_{t,i} = v^{\mathrm{inf}}_{t,i} + \frac{1}{\bar y_{T,i} + 2\kappa}. \tag{4} $$
Bootstrap draws resample reads and already carry this term, so it is added for Gibbs draws only (`count_noise=True` in the deployment; the library default is off).

**Normalization of the total.** $t_i$ carries no library-size offset. mixQTL's total-count model has one explicitly; here between-sample depth variation is left to the covariates of (6). On BrainVar the log library size has a standard deviation of 0.305 (a 4.9-fold range) and correlates at $-0.93$ with the first expression principal component, so the deployed covariate set absorbs it: the median $\hat\tau_t$ is 0.151 under an intercept alone, 0.043 with log library size alone (71% of it is depth), and 0.0066 with the 17 covariates. A run with no covariates therefore inflates the total channel's standard error by about $\sqrt{0.151/0.0066} \approx 4.8$ (Section 8, item 10).

**Why the total is not $y_L + y_R$.** Against a personalized diploid transcriptome the second haplotype copy of a transcript exists only where the sample is heterozygous, so $y_L + y_R$ is a heterozygous-transcript subtotal whose level is set by local heterozygosity, which is in linkage disequilibrium with the variants being tested. The total must be summed over all transcripts of the gene.

**Informative samples.** A sample is informative for channel $c \in \{a, t\}$ if $v_{c,i} > \varepsilon$ with $\varepsilon = 10^{-12}$; the informative set is $I_c$. A sample with no allele-specific reads has $a_i = 0$ and $v_{a,i} = 0$ and is never informative for the allelic channel.

## 3. Model

### 3.1 Two channels for one parameter

For a causal variant $v$ with log aFC $\beta$,
$$ a_i = \alpha_a + \beta\, s_{v,i} + C_{a,i}\, b_a + e_{a,i}, \tag{5} $$
$$ t_i = \alpha_t + \beta\, \tfrac{1}{2} g_{v,i} + C_{t,i}\, b_t + e_{t,i}. \tag{6} $$

Equation (5) is exact under the definition of $\beta$: a heterozygote with the ALT allele on $L$ has $\mathbb{E}[\log(y_L/y_R)] = \beta$, one with the ALT allele on $R$ has $-\beta$, and a homozygote has $0$, which is $\beta s_{v,i}$. Equation (6) is first order: relative to a diploid REF baseline the expected total is $(2 - g + g\,\mathrm{e}^{\beta})/2$, so $\mathbb{E}[t \mid g] - \text{const} = \log\!\big(1 + g(\mathrm{e}^{\beta} - 1)/2\big)$, which equals $\beta\,g/2$ exactly at $g \in \{0, 2\}$ and equals $\log\!\big((1 + \mathrm{e}^{\beta})/2\big) = \beta/2 + \beta^2/8 + O(\beta^3)$ at $g = 1$. The predictor $g/2$ therefore estimates the same parameter as the allelic channel, with a heterozygote excess of $\beta^2/8$ (12% of $\beta/2$ at $\beta = 0.5$).

### 3.2 Error variance

$$ \mathrm{Var}(e_{c,i}) = v_{c,i} + \tau_c, \qquad c \in \{a, t\}, \tag{7} $$
with errors independent across samples. $\tau_c \ge 0$ is a per-gene, per-channel between-sample variance: biological variation in expression or in the allelic ratio, plus unmodelled technical variation. It is the biological-variance term of sleuth's decomposition and the residual heterogeneity of a random-effects meta-regression, and no within-sample sampler can measure it; it is estimated across samples (Section 4.3). Once $\tau_c$ is set the variance is treated as known: standard errors are not rescaled by an estimated dispersion, which is what lets the inferential variance propagate into the standard error in absolute terms (inflating every $v$ inflates every SE).

**Relation to mixQTL's variance model.** This is the one place where hapmixQTL departs structurally from its parent. mixQTL writes the allele-specific error as $\mathcal{N}\big(0, \sigma^2 (1/Y_1 + 1/Y_2)\big)$: the counts set only the *shape* of the weights and $\sigma^2$ is a free *multiplicative* scale inferred from the data. Equation (7) instead adds a free *additive* term. The two coincide only when $v$ is constant across samples. A multiplicative scale is the right correction when the quantifier's variance has the right shape and the wrong size; an additive term is the right one for biological variance, which does not shrink with read depth, and it is the more conservative choice when $v$ spans orders of magnitude, since the weights then approach uniformity rather than tracking $v$. The additive form, the multiplicative form and the nested $\sigma^2 v_i + \tau_c$ were compared on mixQTL's own simulation design (allelic fold changes 1 to 3, $N$ = 25 to 100, power at empirical false-positive rate 10%): the additive form is the best of the three and the nested model adds nothing to it.

### 3.3 Covariates per channel

$a_i$ is a within-sample contrast. Any covariate that acts on both haplotypes alike (library size, expression principal components, sex, age, RNA integrity) cancels in it, so the total channel's covariates have nothing to remove from the allelic channel, and each column projected out of it costs one informative sample. $C_a$ should therefore contain only allelic nuisance terms, if any; the deployment uses an intercept alone. Section 7 gives the measurement behind this.

## 4. Estimation at one variant

### 4.1 Whitening and residualization

Per channel, with $\tau_c$ from Section 4.3, the weights and square-root weights are
$$ w_{c,i} = \frac{1}{\max(v_{c,i}, 10^{-8}) + \tau_c}, \qquad r_{c,i} = \sqrt{w_{c,i}}, \tag{8} $$
and for the allelic channel $r_{a,i} = 0$ for every non-informative sample. Whitened responses and predictors are $y^{*}_c = r_c \circ y_c$ and $x^{*}_{c,v} = r_c \circ x_{c,v}$ (elementwise), with $y_a = a$, $y_t = t$, $x_{a,v} = s_v$ and $x_{t,v} = g_v/2$. The whitened null design is
$$ D_c = \big[\, r_c \;\; r_c \circ C_c \,\big] \in \mathbb{R}^{N \times (1 + p_c)}, \tag{9} $$
whose first column is the whitened intercept (a constant $\alpha$ becomes $\alpha r_i$ after whitening). With the thin QR factorization $D_c = Q_c R_c$, residualization is the projection
$$ P_c z = z - Q_c\,(Q_c^{\top} z). \tag{10} $$
If every $r_{c,i}$ is zero (a switched-off channel, Section 4.3) $Q_c$ has no columns and $P_c$ is the identity.

### 4.2 Known-variance GLS

$$ xy_{c,v} = \langle P_c x^{*}_{c,v},\, P_c y^{*}_c \rangle, \qquad xx_{c,v} = \lVert P_c x^{*}_{c,v} \rVert^2, \tag{11} $$
$$ \hat\beta_{c,v} = \frac{xy_{c,v}}{xx_{c,v}}, \qquad \mathrm{SE}_{c,v} = \frac{1}{\sqrt{xx_{c,v}}}. \tag{12} $$
Equation (12) is the GLS estimator and its exact variance when (7) holds with known $\tau_c$; no residual sum of squares enters the SE. A predictor is valid only if $xx_{c,v} > 10^{-12} \max(\lVert x^{*}_{c,v}\rVert^2, 10^{-30})$; otherwise $\hat\beta_{c,v} = 0$ and $\mathrm{SE}_{c,v} = \infty$, and the channel contributes nothing at that variant (every $s = 0$, or a predictor in the span of the design).

### 4.3 Estimating $\tau$

$\tau_c$ is a moment estimator on the informative samples. With inferential-only weights $w^{0}_i = 1/\max(v_{c,i}, 10^{-8})$, whiten $y_c$ and the null design by $\sqrt{w^{0}}$, residualize as in (10), and let $\mathrm{RSS}_c$ be the residual sum of squares, $\nu_0 = |I_c| - (1 + p_c)$ and $\hat\sigma^2_c = \mathrm{RSS}_c / \max(\nu_0, 1)$. Then
$$ \hat\tau_c = \max\Big\{0,\; \frac{\hat\sigma^2_c - 1}{\overline{w^{0}}}\Big\}, \qquad \overline{w^{0}} = \frac{1}{|I_c|}\sum_{i \in I_c} w^{0}_i. \tag{13} $$
Under (7), $\mathbb{E}[e_i^2 / v_i] = 1 + \tau_c / v_i$, so $\mathbb{E}[\hat\sigma^2_c] \approx 1 + \tau_c\, \overline{w^{0}}$ (leverage ignored); (13) is the DerSimonian-Laird estimator of a random-effects meta-regression. Zero-variance samples never enter it: a sample with $v = 0$ would enter at weight $10^{8}$, dominate $\overline{w^{0}}$ and collapse $\hat\tau$.

**Sparse-channel rule.** If $|I_c| < (1 + p_c) + 2$, neither the regression nor $\tau_c$ is identifiable from the channel; every $r_{c,i}$ is set to zero and the channel contributes nothing to the gene (the other channel is used alone).

**Lead refit.** After the scan (Section 5), $\hat\tau_c$ is re-estimated with the lead's predictor $x_{c,\mathrm{lead}}$ appended to the design of (13) (so $\nu_0 = |I_c| - (2 + p_c)$), while the residualizer of (10) still projects out the null design only. If $|I_c| < (2 + p_c) + 2$ the channel keeps its null $\hat\tau_c$. Under (5)-(6) the null-design residuals contain the effect, so the null estimate exceeds $\tau_c$ by about $\beta^2 f_{\mathrm{het}}$ ($f_{\mathrm{het}}$ the heterozygote fraction among informative samples) and every statistic in the window is shrunk by about $1/(1 + \beta^2 f_{\mathrm{het}}/\sigma^2)$. That factor cancels in the empirical p (Section 5.3) but not in the nominal scale on which genes and methods are compared, which is what the refit restores.

### 4.4 Combining the channels

Since $1/\mathrm{SE}^2_{c,v} = xx_{c,v}$, inverse-variance weighting of the two channel estimates reduces to
$$ \hat\beta_v = \frac{xx_{a,v}\hat\beta_{a,v} + xx_{t,v}\hat\beta_{t,v}}{xx_{a,v} + xx_{t,v}} = \frac{xy_{a,v} + xy_{t,v}}{xx_{a,v} + xx_{t,v}}, \qquad \mathrm{SE}_v = \big(xx_{a,v} + xx_{t,v}\big)^{-1/2}, \tag{14} $$
$$ T_v = \frac{\hat\beta_v}{\mathrm{SE}_v}, \qquad T^2_v = \frac{(xy_{a,v} + xy_{t,v})^2}{xx_{a,v} + xx_{t,v}}, \tag{15} $$
where an invalid channel enters with $xx = xy = 0$. The two channel estimators are treated as independent. This is not an approximation of convenience: $s_v$ and $g_v/2$ are orthogonal predictors under random phase ($\mathbb{E}[s_v \mid g_v = 1] = 0$), so any noise shared by $a_i$ and $t_i$ within a sample, including the inferential covariance, is projected onto orthogonal directions by the two regressions, and the correlation of $\hat\beta_a$ and $\hat\beta_t$ is zero even when $a$ and $t$ are strongly correlated (measured at an inferential correlation of 0.9).

### 4.5 Nominal p-value

$$ p_{\mathrm{nom},v} = 2\, \Pr\big(t_{\nu} > |T_v|\big), \qquad \nu = N - 2 - \max(p_t, p_a). \tag{16} $$
The known-variance statistic is asymptotically standard normal; the $t$ reference is a finite-sample convention, shared by both channels and taken from the larger design. The implementation computes it through $r^2 = T^2/(T^2 + \nu)$ and the correlation-to-p mapping, which is the same quantity.

## 5. Cis scan and gene-level inference

### 5.1 Window and variant filters

The window is the transcription start site $\pm W$ (or the gene span extended by $W$ when a start and end are given), $W = 10^6$ by default. Missing genotypes are imputed to the variant's mean dosage. Optionally, variants with in-sample minor allele frequency $\min(\mathrm{AF}_v, 1 - \mathrm{AF}_v) < m$ are removed; monomorphic variants are always removed. The phase matrices $X_L$, $X_R$ are subset identically to the dosage matrix.

### 5.2 Scan

$\hat\tau_a$, $\hat\tau_t$, the weights (8) and the projections (10) are computed once per gene. With $S$ the $V \times N$ matrix of $s_{v,i}$ and $G$ that of $g_{v,i}$,
$$ \tilde S = P_a\big(S \circ r_a^{\top}\big), \quad \tilde G = P_t\big(\tfrac{1}{2} G \circ r_t^{\top}\big), \quad \tilde y_a = P_a y^{*}_a, \quad \tilde y_t = P_t y^{*}_t, $$
$$ xy_a = \tilde S\, \tilde y_a, \quad xx_a = \mathrm{rowsum}(\tilde S \circ \tilde S), \quad xy_t = \tilde G\, \tilde y_t, \quad xx_t = \mathrm{rowsum}(\tilde G \circ \tilde G), $$
$T^2_v$ follows from (15) and the lead is $\arg\max_v T^2_v$ (undefined statistics rank last).

### 5.3 Permutation null

The null is Freedman-Lane in whitened space. Under $H_0$ the whitened null residuals $\tilde y_c$ have mean zero and covariance $I - Q_c Q_c^{\top}$, so entry $i$ has variance $1 - h_{c,i}$ with leverage $h_{c,i} = \sum_k Q^2_{c,ik}$. They are standardized,
$$ \tilde e_{c,i} = \frac{\tilde y_{c,i}}{\sqrt{\max(1 - h_{c,i},\, 10^{-3})}}, \tag{17} $$
because the statistic (15) is not pivotal: it is built from $xy$ and $xx$ rather than refitted per permutation, so a deficit of $(N - 1 - p_c)/N$ in the residual variance would carry straight into the null (Section 7).

Draw $P$ permutations $\pi_1, \dots, \pi_P$ of $\{1, \dots, N\}$ from a seeded generator, shared across genes and across the two channels. For channel $c$, the restriction of $\pi$ to $I_c$, the informative samples in the order $\pi$ visits them, is a uniformly distributed permutation of $I_c$; $\tilde e^{\pi}_c$ carries the entries of $I_c$ so permuted and leaves non-informative entries unchanged (they are zero). No information therefore lands on a zero weight, no zero lands on an informative sample, and the two channels share one draw. Then, as one matrix product per channel,
$$ xy^{\pi}_c = \tilde X_c\, \tilde e^{\pi}_c \;\; (\tilde X_a = \tilde S,\; \tilde X_t = \tilde G), \qquad T^{2,\pi}_v = \frac{(xy^{\pi}_{a,v} + xy^{\pi}_{t,v})^2}{xx_{a,v} + xx_{t,v}}, \qquad M_\pi = \max_v T^{2,\pi}_v. \tag{18} $$
Because $\tilde X_c$ is orthogonal to the null design, re-residualizing the permuted residuals would leave $xy^{\pi}$ unchanged, and $yy$ does not enter the statistic, so that step is omitted. Permuting the raw $a$ or $t$ at fixed weights, the alternative, would hand sample $i$ another sample's value at $i$'s own precision and mis-scale the null under heteroskedastic $v$ (Section 7).

**Empirical p.** With $r^2_{\mathrm{nom}} = T^2_{\mathrm{lead}}/(T^2_{\mathrm{lead}} + \nu)$ and $r^2_{\pi} = M_\pi/(M_\pi + \nu)$,
$$ p_{\mathrm{perm}} = \frac{\#\{\pi : r^2_{\pi} \ge r^2_{\mathrm{nom}}\} + 1}{P + 1}. \tag{19} $$

**Beta approximation.** As in FastQTL and tensorQTL: an effective degrees of freedom $\nu^{*}$ is found (Newton iteration on $\log \nu$, starting at $\nu$) such that the method-of-moments first shape parameter of a Beta distribution fitted to $\{p_t(r^2_\pi; \nu^{*})\}$ equals 1, where $p_t(r^2; \nu) = 2\Pr(t_\nu > \sqrt{\nu r^2/(1-r^2)})$; a $\mathrm{Beta}(k, n)$ is then fitted to those values by maximum likelihood from the moment estimates, and
$$ p_{\mathrm{beta}} = F_{\mathrm{Beta}(k,n)}\big(p_t(r^2_{\mathrm{nom}}; \nu^{*})\big). \tag{20} $$
$p_{\mathrm{perm}}$ and $p_{\mathrm{beta}}$ are the gene-level p-values. Both are invariant to a common rescaling of the weights, hence to the value of $\hat\tau$ used in the scan.

### 5.4 Reported lead quantities

On the scan scale the lead's $\hat\beta$ and SE are (14) and its nominal p is (16). With the lead refit (Section 4.3) the weights and projections are rebuilt with the refit $\hat\tau_a$, $\hat\tau_t$ and the lead's $\hat\beta$, SE, $p_{\mathrm{nom}}$, $\hat\beta_a$, $\mathrm{SE}_a$, $\hat\beta_t$, $\mathrm{SE}_t$ are recomputed from (11)-(16); $p_{\mathrm{perm}}$ and $p_{\mathrm{beta}}$ are left on the scan scale. Both $\hat\tau$ values are reported. A lead's nominal p is never a gene-level p, being the best of the window, and the refit makes it more selective still; the detection call is $p_{\mathrm{beta}}$ (or $p_{\mathrm{perm}}$).

### 5.5 Algorithm

Algorithm 1 (one gene; `map_cis`).

1. Summaries: $a_i$, $t_i$, $v_{a,i}$, $v_{t,i}$ from the draws by (2)-(4); informative sets $I_a$, $I_t$.
2. Genotypes: $G$, $S$ for the window (Section 5.1), imputation, MAF and monomorphic filters applied to $G$ and $S$ together.
3. Per channel: $\hat\tau_c$ by (13) on $I_c$ under the null design, or the channel switched off by the sparse rule; weights (8); allelic weights zeroed off $I_a$; $Q_c$ from (9).
4. Scan: $\tilde S$, $\tilde G$, $\tilde y_a$, $\tilde y_t$, $xy$, $xx$ (Section 5.2); $T^2_v$ by (15); lead $= \arg\max_v T^2_v$.
5. Null: $\tilde e_c$ by (17); $P$ seeded permutations restricted to $I_c$; $xy^{\pi}_c$, $M_\pi$ by (18); $r^2_{\mathrm{nom}}$, $r^2_\pi$; $p_{\mathrm{perm}}$ by (19); $p_{\mathrm{beta}}$ by (20).
6. Lead: per-channel $\hat\beta_c$, $\mathrm{SE}_c$ at the lead; with the refit, $\hat\tau_c$ re-estimated with $x_{c,\mathrm{lead}}$ in the $\tau$ design and steps 3-4 repeated for the lead only; $\hat\beta$, SE, $p_{\mathrm{nom}}$ by (14)-(16); $\alpha_{\mathrm{cis}}$ and the cis/trans p (Section 6.2).
7. Output one row: lead id and distances, allele statistics, $p_{\mathrm{nom}}$, $\hat\beta$, SE, $\hat\beta_a$, $\mathrm{SE}_a$, $\hat\beta_t$, $\mathrm{SE}_t$, $\alpha_{\mathrm{cis}}$, $p_{\mathrm{cis/trans}}$, $p_{\mathrm{perm}}$, $p_{\mathrm{beta}}$, Beta shape parameters and $\nu^{*}$, $\hat\tau_a$, $\hat\tau_t$, their null-design values, and whether the refit was applied.

The scan costs $O(VN)$ and the permutations $O(VNP)$, as dense matrix products; arithmetic is single precision by default and runs on a GPU when one is available.

## 6. Effect sizes and diagnostics

### 6.1 The allelic fold change

$\hat\beta$ estimates $\log(e_{\mathrm{ALT}}/e_{\mathrm{REF}})$, so the allelic fold change is $\mathrm{aFC} = \mathrm{e}^{\hat\beta}$ with the 95% interval $\exp(\hat\beta \pm 1.96\,\mathrm{SE})$. $\hat\beta_a$ is estimated from the heterozygotes alone (homozygotes have $s = 0$ and contribute to neither $xy_a$ nor $xx_a$); $\hat\beta_t$ uses every sample. The effect allele is the VCF's ALT allele, the same orientation as RASQUAL's $\log(\pi/(1-\pi))$, so the two are directly comparable at the same variant.

### 6.2 The cis/trans diagnostic

The combination (14) assumes both channels estimate the same $\beta$, that is, a purely cis effect. A trans component acting on total expression only, reference mapping bias attenuating the allelic channel, phasing error, or feature-level misquantification violate it, and the combined slope then averages two different quantities. Because the channel estimators are uncorrelated, the difference has variance $\mathrm{SE}_a^2 + \mathrm{SE}_t^2$ with no covariance term and
$$ z = \frac{\hat\beta_a - \hat\beta_t}{\sqrt{\mathrm{SE}_a^2 + \mathrm{SE}_t^2}}, \qquad p_{\mathrm{cis/trans}} = 2\Pr(t_\nu > |z|), \qquad \alpha_{\mathrm{cis}} = \frac{\hat\beta_a}{\hat\beta_t}. \tag{21} $$
A small $p_{\mathrm{cis/trans}}$ flags a lead whose combined slope should not be read as a cis log aFC; it is a diagnostic column, not a filter, and is undefined when a channel has no finite SE.

### 6.3 The reference-bias gate

hapmixQTL does not model reference mapping bias and is anticonservative in its presence, so the precondition is tested rather than assumed. For gene $g$ and sample $i$ the haplotype orientation is
$$ o_{g,i} = \operatorname{sign}\Big(\sum_{v \in F_g} d_{v,i}\, s_{v,i}\Big), \tag{22} $$
over the gene's feature sites $F_g$ (the heterozygous sites inside the exon union, else the gene body, else the het site nearest the TSS), with $d_{v,i}$ the allele-specific read depth at the site (uniform when unavailable): mapping bias favours REF at every site carrying reads, so the bias it adds to the haplotype totals is proportional to this depth-weighted sum. The reference count of the gene-sample is $\bar y_R$ when $o = +1$ (ALT on $L$) and $\bar y_L$ when $o = -1$, the alternate count the other; a gene-sample is usable if $o \ne 0$ and its allele-specific total is at least 10. For each gene with at least 3 usable samples, $f_g$ is the pooled reference fraction; over $G \ge 5$ such genes,
$$ m = \frac{1}{G}\sum_g f_g, \qquad z = \frac{m - 0.5}{\mathrm{sd}(f_g)/\sqrt{G}}, \qquad p = 2\Phi(-|z|), \tag{23} $$
and bias is declared at $p < 10^{-3}$. Genuine allele-specific expression raises REF or ALT with a sign that is arbitrary per gene and cancels in $m$; mapping bias accumulates. The test is clustered at the gene level because that is the unit of randomization of the sign.

## 7. Measured properties

The choices above rest on measurements from the calibration suite (`tests/test_hapmixqtl_calibration.py`, `docs/ase_validation.md`) and from the BrainVar deployment (92 prenatal brain samples; `docs/brainvar_deploy_runbook.md`).

| Property | Measurement |
|---|---|
| Known variance without $\tau$ (`tau_mode='zero'`) | Type-I error up to 107x nominal at $\alpha = 10^{-3}$ on simulations with unmodelled variance; 64.5% coverage of nominal 95% intervals |
| With $\tau$ from (13) | Type-I within 0.85-1.02x nominal across the same conditions; interval coverage 0.96-0.99 |
| Channel estimators | Correlation of $\hat\beta_a$, $\hat\beta_t$ indistinguishable from 0 at inferential correlation 0.9 |
| What the draws measure (30 BrainVar genes, medians, per sample) | Allelic channel: Gibbs variance 0.013, counting term 0.001, $\hat\tau_a$ 0.029 (intercept-only design, which is the deployed allelic design). Total channel: Gibbs 0.0002, counting 0.0002, $\hat\tau_t$ 0.151 under an intercept alone but **0.0066 under the deployed 17-covariate design**; the deployed ratio of $\hat\tau_t$ to the per-sample quantification variance is 22, not the 373 an intercept-only design implies |
| Covariates per channel (30 genes, 17 covariates) | Total channel: 96% of whitened residual variance explained, $F$ test $p < 10^{-3}$ on 30/30 genes ($\hat\tau_t$ 0.151 to 0.007). Allelic channel: 24% explained against 22% expected by chance, $\hat\tau_a$ 0.0294 to 0.0292, 3/30 genes at $p < 0.05$ against 1.5 expected |
| Zero-variance samples in the $\tau$ estimator | Three zero-count samples carried 99.8% of a gene's total-channel weight before the counting term and the informative-sample rule; the allelic channel's permutation null reached $\chi^2$ 40-120 where 12-16 is calibrated |
| Permuting raw values at fixed weights | Null genes with $v \in [0.01, 2]$: mean empirical p 0.94, no rejection at 0.05 in 100 genes |
| Permutation under a misspecified variance *shape* (60 null genes, $v$ spanning 200x, Monte Carlo SE 0.037 on the mean) | Truth additive, as (7) assumes: mean empirical p 0.499, 5.0% below 0.05. Truth multiplicative ($\mathrm{Var} = c\,v_i$, mixQTL's structure): 0.597, 1.7% (conservative). Truth constant ($v$ carrying no information): 0.441, 8.3% |
| Whitened residuals, unstandardized | Mean empirical p 0.44 on the same design (deficit $(N-1-p)/N$) |
| Whitened residuals, leverage-standardized (17) | 400 null genes: mean empirical p 0.503, 5.2% below 0.05, 49.7% below 0.5; the permuted statistic's variance equals $xx$ (0.70 of it before standardization with 18 columns on 60 samples) |
| $\tau$ under the null versus with the lead in the model | Nominal statistic of strong genes shrunk 30-110% (CCNI allelic 19.7 vs 40.2); the lead's empirical p identical at 20,000 draws; simulated power at matched false-positive rate identical (0.49 vs 0.45, 0.92 vs 0.91, 1.00 vs 1.00) |
| Matched-variant effects against RASQUAL (30 genes) | Sign agreement 52 of 55 pairs, $r$ 0.78-0.86 at the same variant; 0.24 gene-wise, where the leads differ on 29 of 30 genes |

## 8. Assumptions and limitations

1. **Phase.** $s$ is taken from the phased VCF; a phase switch between the tested variant and the gene flips the sign of that sample's contribution and attenuates $\hat\beta_a$. Read-backed phasing of the gene's own heterozygous sites limits this.
2. **Reference mapping bias** is not modelled; it is tested for (Section 6.3) and the method is not valid when the gate fails.
3. **Log-normal errors.** Counts enter through $\log(y + \kappa)$ with the delta-method variance $1/(y + \kappa)$, a first-order approximation to a negative binomial that is accurate at moderate counts and coarse below about 10 reads; genes are given an expression floor in the deployment.
4. **A single between-sample variance per channel.** $\tau_c$ is shared by all samples; sample-specific biological variance beyond the counting term is not modelled.
5. **Null-model $\tau$ in the scan.** The scan's ranking and the gene-level p are unaffected (Section 7); the reported scale relies on the lead refit.
6. **First-order total channel** (Section 3.1) and a single causal variant per window (the combined statistic is a marginal test; the stacked-design SuSiE extension in `map_susie` addresses multiple effects).
7. **The $t$ reference** of (16) is a convention; the empirical p (19)-(20) is the calibrated quantity.
8. **Independence across samples**: no relatedness or repeated measures.
9. **Sparse channels** are admitted down to $(1 + p_c) + 2$ informative samples; calibration was measured with about 70 and the deployment's genes had 46 to 85.
10. **No library-size offset** (Section 2). Depth is absorbed by the covariates of (6) or not at all; a run without covariates loses power in the total channel rather than becoming miscalibrated.
11. **Exchangeability of the whitened residuals** requires the variance model (7) to have the right *shape* up to a constant, not merely the right average. $\hat\tau_c$ by (13) fixes the average, so a misspecified shape leaves the permutation null approximately, not exactly, calibrated; the measured degradation over a 200-fold spread in $v$ is in Section 7, and is conservative when the truth is multiplicative.

## 9. Symbols, defaults and implementation

| Symbol | Meaning | Default | Code |
|---|---|---|---|
| $\kappa$ | pseudocount in (2) | 0.5 | `compute_summaries_from_gibbs(kappa)` |
| counting term (4) | Poisson plug-in variance | off (library); on (driver, runner) | `count_noise` |
| $\varepsilon$ | informative threshold | $10^{-12}$ | `_channel_weights(eps)` |
| variance floor in (8) | | $10^{-8}$ | `_channel_weights` |
| sparse-channel rule | minimum informative samples | $(1 + p_c) + 2$ | `_min_informative` |
| $C_a$ | allelic covariate design | same as $C_t$ (library); intercept only (driver) | `ase_covariates_df` |
| $\hat\tau$ | moment estimator (13) | `tau_mode='estimate'` | `_estimate_tau`, `_estimate_tau_informative` |
| lead refit | $\tau$ with the lead in the design | off (library, CLI); on (driver, runner) | `map_cis(tau_refit)` |
| $\nu$ | t reference in (16) | $N - 2 - \max(p_t, p_a)$ | `map_cis`, `map_nominal` |
| $W$ | cis window | $10^6$ | `window` |
| $m$ | in-sample MAF filter | 0 (off) | `maf_threshold` |
| $P$ | permutations | 10,000 | `nperm` |
| leverage floor in (17) | | $10^{-3}$ | `_leverage_standardized` |
| seed | permutation generator | none | `seed` |
| predictor validity | $xx > 10^{-12}\max(\lVert x^{*}\rVert^2, 10^{-30})$ | | `_wls_regression` |
| gate thresholds | min depth 10, min usable gene-samples 20, $p < 10^{-3}$ | | `reference_bias_diagnostic` |

Functions: `compute_summaries_from_gibbs` (Section 2), `_prepare_channels` and `_channel_weights` (4.1, 4.3), `WeightedResidualizer` (4.1), `_wls_regression` (4.2), `_estimate_tau` (4.3), `calculate_hapmixqtl_nominal` and `_combined_tstat2` (4.4), `calculate_hapmixqtl_permutations`, `_leverage_standardized`, `_permute_within_informative` (5.3), `map_cis` (5), `map_nominal` (per-pair statistics on the null-model $\tau$ scale), `cis_trans_diagnostic` (6.2), `orient_haplotypes` and `reference_bias_diagnostic` (6.3), `calculate_beta_approx_pval` in `core.py` (5.3).

## References

- Kumasaka N, Knights AJ, Gaffney DJ. Fine-mapping cellular QTLs with RASQUAL and ATAC-seq. Nature Genetics 2016.
- Liang Y, Aguet F, Barbeira AN, Ardlie K, Im HK. A scalable unified framework of total and allele-specific counts for cis-QTL mapping (mixQTL). Nature Communications 2021.
- Mohammadi P, Castel SE, Brown AA, Lappalainen T. Quantifying the regulatory effect size of cis-acting genetic variation using allelic fold change. Genome Research 2017.
- Patro R, Duggal G, Love MI, Irizarry RA, Kingsford C. Salmon provides fast and bias-aware quantification of transcript expression. Nature Methods 2017.
- Pimentel H, Bray NL, Puente S, Melsted P, Pachter L. Differential analysis of RNA-seq incorporating quantification uncertainty (sleuth). Nature Methods 2017.
- Ongen H, Buil A, Brown AA, Dermitzakis ET, Delaneau O. Fast and efficient QTL mapper for thousands of molecular phenotypes (FastQTL). Bioinformatics 2016.
- Taylor-Weiner A, Aguet F, et al. Scaling computational genomics to millions of individuals with GPUs (tensorQTL). Genome Biology 2019.
- Freedman D, Lane D. A nonstochastic interpretation of reported significance levels. Journal of Business and Economic Statistics 1983.
- DerSimonian R, Laird N. Meta-analysis in clinical trials. Controlled Clinical Trials 1986.
- Winkler AM, Ridgway GR, Webster MA, Smith SM, Nichols TE. Permutation inference for the general linear model. NeuroImage 2014.
- Castel SE, Mohammadi P, Chung WK, Shen Y, Lappalainen T. Rare variant phasing and haplotypic expression from RNA sequencing with phASER. Nature Communications 2016.
