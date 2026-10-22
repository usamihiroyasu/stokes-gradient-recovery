# Primary-source locators

The manuscript compares its recovery statements with the following results.

- Levin, Kileel, Boumal: [arXiv:2207.03512v5](https://arxiv.org/abs/2207.03512v5). Theorem 2.4; Lemma 3.24; Proposition 3.26; Theorem 3.23. Checked PDF SHA-256: `6fef82bd5501dd2c7dc87a19dc20565945abe679faad67412e58a8fcb2278d30`. The journal DOI is given in the bibliography; theorem locators are explicitly pinned to this arXiv version.
- Wang, Zhou, Xu, Gao: [author PDF](https://zhepeiwang.github.io/pubs/tro_2022_gcopter.pdf). Section V-B, equations (78)-(79). Checked local source SHA-256: `dfab71a469ddd9182fa024bebf336a07911afe16eef2af5a0d107e1bec55db56`.

Wang et al. supply the rational map and its first derivative. Levin et al. supply the smooth-lift differential framework and stationarity-transfer results. The manuscript derives explicit Stokes inverse formulas and input-sensitivity coefficients within that framework.

## AD implementation and software citation

Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, NeurIPS 2019, volume 32. Primary proceedings page: https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html . The recorded experimental version is PyTorch 2.14.1. Official autograd.grad API: https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html . Acquisition measurements are in `repro/results/ad_route_benchmark.json`.

## Hadamard parametrization

Li, McKenzie, and Yin, [arXiv:2112.05273v2](https://arxiv.org/html/2112.05273v2), Sections 1 and 4: correspondence of KKT and strict-saddle points under the simplex-to-sphere squaring map. Journal metadata: Information and Inference 12(3), 1898–1937, 2023, [doi:10.1093/imaiai/iaad017](https://doi.org/10.1093/imaiai/iaad017). This source supports the related landscape comparison; it is not evidence for the Stokes recovery coefficients.


## Sequential initialization and stopping

Lee, Simchowitz, Jordan, and Recht, *Gradient Descent Only Converges to
Minimizers*, PMLR 49, 2016: https://proceedings.mlr.press/v49/lee16.html.
The theorem assumes absolutely continuous initialization, Lipschitz gradient,
and a sufficiently small fixed step. Boundary-derived warm starts and finite
quasi-Newton termination require separate analysis.

The boundary-restricted Stokes-flow proposition applies the rational ball map
and smooth-lift differential geometry to initialization on a nonzero Stokes
boundary state. The finite-solver experiments evaluate discrete stopping
separately from this continuous-flow statement.
