# References

> **Living bibliography for the Koopman-EML project.**
> Add new references as the project evolves. Organized by topic.

---

## 1. The EML Operator

The foundational paper for this project. Odrzywołek shows that a single binary operator
`eml(x, y) = exp(x) - ln(y)`, paired with the constant `1`, generates all standard
elementary functions — the continuous-math analog of the NAND gate.

- **Odrzywołek, A.** (2026).
  *All elementary functions from a single operator.*
  arXiv preprint.
  arXiv:[2603.21852v2](https://arxiv.org/abs/2603.21852)

- **Odrzywołek, A.** (2026).
  *SymbolicRegressionPackage: basic building blocks for symbolic regression.*
  GitHub: [VA00/SymbolicRegressionPackage](https://github.com/VA00/SymbolicRegressionPackage).
  Zenodo DOI: [10.5281/zenodo.19183008](https://doi.org/10.5281/zenodo.19183008)

- **Odrzywołek, A.** (2026).
  *A ternary Sheffer operator for elementary functions?*
  Acta Physica Polonica B (in preparation).

---

## 2. Koopman Operator Theory

### 2.1 Foundations

- **Koopman, B. O.** (1931).
  *Hamiltonian systems and transformations in Hilbert space.*
  Proceedings of the National Academy of Sciences, 17(5), 315–318.
  DOI:[10.1073/pnas.17.5.315](https://doi.org/10.1073/pnas.17.5.315)

- **Mezić, I.** (2005).
  *Spectral properties of dynamical systems, model reduction and decompositions.*
  Nonlinear Dynamics, 41(1–3), 309–325.
  DOI:[10.1007/s11071-005-2824-x](https://doi.org/10.1007/s11071-005-2824-x)

- **Banaszuk, A. & Mezić, I.** (2004).
  *Comparison of systems with complex behavior.*
  Physica D: Nonlinear Phenomena, 197(1–2), 101–133.
  DOI:[10.1016/j.physd.2004.06.015](https://doi.org/10.1016/j.physd.2004.06.015)

- **Budišić, M., Mohr, R. & Mezić, I.** (2012).
  *Applied Koopmanism.*
  Chaos, 22(4), 047510.
  DOI:[10.1063/1.4772195](https://doi.org/10.1063/1.4772195);
  arXiv:[1206.3164](https://arxiv.org/abs/1206.3164)

- **Mezić, I.** (2013).
  *Analysis of fluid flows via spectral properties of the Koopman operator.*
  Annual Review of Fluid Mechanics, 45, 357–378.
  DOI:[10.1146/annurev-fluid-011212-140652](https://doi.org/10.1146/annurev-fluid-011212-140652)

### 2.2 Dynamic Mode Decomposition (DMD)

- **Schmid, P. J.** (2010).
  *Dynamic mode decomposition of numerical and experimental data.*
  Journal of Fluid Mechanics, 656, 5–28.
  DOI:[10.1017/S0022112010001217](https://doi.org/10.1017/S0022112010001217)

- **Tu, J. H., Rowley, C. W., Luchtenburg, D. M., Brunton, S. L. & Kutz, J. N.** (2014).
  *On dynamic mode decomposition: theory and applications.*
  Journal of Computational Dynamics, 1(2), 391–421.
  DOI:[10.3934/jcd.2014.1.391](https://doi.org/10.3934/jcd.2014.1.391);
  arXiv:[1312.0041](https://arxiv.org/abs/1312.0041)

- **Kutz, J. N., Brunton, S. L., Brunton, B. W. & Proctor, J. L.** (2016).
  *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems.*
  SIAM.
  DOI:[10.1137/1.9781611974508](https://doi.org/10.1137/1.9781611974508)

- **Brunton, S. L., Proctor, J. L. & Kutz, J. N.** (2015).
  *Compressive sampling and dynamic mode decomposition.*
  Journal of Computational Dynamics, 2(2).
  DOI:[10.3934/jcd.2015002](https://doi.org/10.3934/jcd.2015002);
  arXiv:[1312.5186](https://arxiv.org/abs/1312.5186)

- **Proctor, J. L., Brunton, S. L. & Kutz, J. N.** (2016).
  *Dynamic mode decomposition with control.*
  SIAM Journal on Applied Dynamical Systems, 15(1), 142–161.
  DOI:[10.1137/15M1013857](https://doi.org/10.1137/15M1013857);
  arXiv:[1409.6358](https://arxiv.org/abs/1409.6358)

- **Kutz, J. N., Fu, X. & Brunton, S. L.** (2016).
  *Multi-resolution dynamic mode decomposition.*
  SIAM Journal on Applied Dynamical Systems, 15(2), 713–735.
  DOI:[10.1137/15M1023543](https://doi.org/10.1137/15M1023543);
  arXiv:[1506.00564](https://arxiv.org/abs/1506.00564)

### 2.3 Extended DMD (EDMD) and Kernel Methods

- **Williams, M. O., Kevrekidis, I. G. & Rowley, C. W.** (2015).
  *A data-driven approximation of the Koopman operator: extending dynamic mode decomposition.*
  Journal of Nonlinear Science, 25(6), 1307–1346.
  DOI:[10.1007/s00332-015-9258-5](https://doi.org/10.1007/s00332-015-9258-5);
  arXiv:[1408.4408](https://arxiv.org/abs/1408.4408)

- **Williams, M. O., Rowley, C. W. & Kevrekidis, I. G.** (2015).
  *A kernel-based method for data-driven Koopman spectral analysis.*
  Journal of Computational Dynamics, 2(2), 247–265.
  DOI:[10.3934/jcd.2015005](https://doi.org/10.3934/jcd.2015005)

- **Li, Q., Dietrich, F., Bollt, E. M. & Kevrekidis, I. G.** (2017).
  *Extended dynamic mode decomposition with dictionary learning.*
  Chaos, 27(10), 103111.
  DOI:[10.1063/1.4993854](https://doi.org/10.1063/1.4993854);
  arXiv:[1707.00225](https://arxiv.org/abs/1707.00225)

### 2.4 Deep Learning for Koopman

- **Takeishi, N., Kawahara, Y. & Yairi, T.** (2017).
  *Learning Koopman invariant subspaces for dynamic mode decomposition.*
  Advances in Neural Information Processing Systems (NeurIPS), 30.
  arXiv:[1710.04340](https://arxiv.org/abs/1710.04340)

- **Lusch, B., Kutz, J. N. & Brunton, S. L.** (2018).
  *Deep learning for universal linear embeddings of nonlinear dynamics.*
  Nature Communications, 9, 4950.
  DOI:[10.1038/s41467-018-07210-0](https://doi.org/10.1038/s41467-018-07210-0);
  arXiv:[1712.09707](https://arxiv.org/abs/1712.09707)

- **Otto, S. E. & Rowley, C. W.** (2019).
  *Linearly recurrent autoencoder networks for learning dynamics.*
  SIAM Journal on Applied Dynamical Systems, 18(1), 558–593.
  DOI:[10.1137/18M1177846](https://doi.org/10.1137/18M1177846);
  arXiv:[1712.01378](https://arxiv.org/abs/1712.01378)

### 2.5 Koopman Reviews and Books

- **Brunton, S. L., Budišić, M., Kaiser, E. & Kutz, J. N.** (2022).
  *Modern Koopman theory for dynamical systems.*
  SIAM Review, 64(2), 229–340.
  DOI:[10.1137/21M1401243](https://doi.org/10.1137/21M1401243);
  arXiv:[2102.12086](https://arxiv.org/abs/2102.12086)

- **Mauroy, A., Mezić, I. & Susuki, Y.** (Eds.) (2020).
  *The Koopman Operator in Systems and Control: Concepts, Methodologies, and Applications.*
  Lecture Notes in Control and Information Sciences, vol. 484. Springer.
  DOI:[10.1007/978-3-030-35713-9](https://doi.org/10.1007/978-3-030-35713-9)

- **Otto, S. E. & Rowley, C. W.** (2021).
  *Koopman operators for estimation and control of dynamical systems.*
  Annual Review of Control, Robotics, and Autonomous Systems, 4, 425–449.
  DOI:[10.1146/annurev-control-071020-010108](https://doi.org/10.1146/annurev-control-071020-010108)

- **Bevanda, P., Sosnowski, S. & Hirche, S.** (2021).
  *Koopman operator dynamical models: learning, analysis and control.*
  Annual Reviews in Control, 52, 197–212.
  DOI:[10.1016/j.arcontrol.2021.09.002](https://doi.org/10.1016/j.arcontrol.2021.09.002)

---

## 3. SINDy and PySINDy

### 3.1 SINDy (Sparse Identification of Nonlinear Dynamics)

- **Brunton, S. L., Proctor, J. L. & Kutz, J. N.** (2016).
  *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.*
  Proceedings of the National Academy of Sciences, 113(15), 3932–3937.
  DOI:[10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113);
  arXiv:[1509.03580](https://arxiv.org/abs/1509.03580)

- **Brunton, S. L., Proctor, J. L. & Kutz, J. N.** (2016).
  *Sparse identification of nonlinear dynamics with control (SINDYc).*
  arXiv:[1605.06682](https://arxiv.org/abs/1605.06682)

- **Kaiser, E., Kutz, J. N. & Brunton, S. L.** (2018).
  *Sparse identification of nonlinear dynamics for model predictive control in the low-data limit.*
  Proceedings of the Royal Society A, 474(2219).
  DOI:[10.1098/rspa.2018.0335](https://doi.org/10.1098/rspa.2018.0335);
  arXiv:[1711.05501](https://arxiv.org/abs/1711.05501)

- **Mangan, N. M., Brunton, S. L., Proctor, J. L. & Kutz, J. N.** (2016).
  *Inferring biological networks by sparse identification of nonlinear dynamics.*
  IEEE Transactions on Molecular, Biological, and Multi-Scale Communications, 2(1), 52–63.
  DOI:[10.1109/TMBMC.2016.2633265](https://doi.org/10.1109/TMBMC.2016.2633265);
  arXiv:[1605.08368](https://arxiv.org/abs/1605.08368)

### 3.2 PySINDy

- **de Silva, B. M., Champion, K., Quade, M., Loiseau, J.-C., Kutz, J. N. & Brunton, S. L.** (2020).
  *PySINDy: a Python package for the sparse identification of nonlinear dynamical systems from data.*
  Journal of Open Source Software, 5(49), 2104.
  DOI:[10.21105/joss.02104](https://doi.org/10.21105/joss.02104);
  arXiv:[2004.08424](https://arxiv.org/abs/2004.08424)

- **PySINDy Documentation.**
  [https://pysindy.readthedocs.io](https://pysindy.readthedocs.io);
  GitHub: [dynamicslab/pysindy](https://github.com/dynamicslab/pysindy)

---

## 4. Brunton & Kutz — Data-Driven Dynamics (Additional Works)

### 4.1 Textbooks

- **Brunton, S. L. & Kutz, J. N.** (2019; 2nd ed. 2022).
  *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control.*
  Cambridge University Press.
  DOI:[10.1017/9781108380690](https://doi.org/10.1017/9781108380690)

### 4.2 Koopman for Control

- **Brunton, S. L., Brunton, B. W., Proctor, J. L. & Kutz, J. N.** (2016).
  *Koopman invariant subspaces and finite linear representations of nonlinear dynamical systems for control.*
  PLOS ONE, 11(2), e0150171.
  DOI:[10.1371/journal.pone.0150171](https://doi.org/10.1371/journal.pone.0150171);
  arXiv:[1510.03007](https://arxiv.org/abs/1510.03007)

- **Kaiser, E., Kutz, J. N. & Brunton, S. L.** (2017).
  *Data-driven discovery of Koopman eigenfunctions for control.*
  arXiv:[1707.01146](https://arxiv.org/abs/1707.01146)

- **Kaiser, E., Kutz, J. N. & Brunton, S. L.** (2020).
  *Data-driven approximations of dynamical systems operators for control.*
  In: Mauroy et al. (Eds.), The Koopman Operator in Systems and Control, Springer.
  DOI:[10.1007/978-3-030-35713-9_8](https://doi.org/10.1007/978-3-030-35713-9_8);
  arXiv:[1902.10239](https://arxiv.org/abs/1902.10239)

### 4.3 Data-Driven Discovery and Coordinates

- **Champion, K., Lusch, B., Kutz, J. N. & Brunton, S. L.** (2019).
  *Data-driven discovery of coordinates and governing equations.*
  Proceedings of the National Academy of Sciences, 116(45), 22445–22451.
  DOI:[10.1073/pnas.1906995116](https://doi.org/10.1073/pnas.1906995116);
  arXiv:[1904.02107](https://arxiv.org/abs/1904.02107)

- **Brunton, S. L., Brunton, B. W., Proctor, J. L., Kaiser, E. & Kutz, J. N.** (2017).
  *Chaos as an intermittently forced linear system.*
  Nature Communications, 8, 19.
  DOI:[10.1038/s41467-017-00030-8](https://doi.org/10.1038/s41467-017-00030-8)

### 4.4 Reviews and Tutorials

- **Brunton, S. L., Noack, B. R. & Koumoutsakos, P.** (2020).
  *Machine learning for fluid mechanics.*
  Annual Review of Fluid Mechanics, 52, 477–508.
  DOI:[10.1146/annurev-fluid-010719-060214](https://doi.org/10.1146/annurev-fluid-010719-060214)

- **Manohar, K., Brunton, S. L., Brunton, B. W. & Kutz, J. N.** (2018).
  *Data-driven sparse sensor placement for reconstruction.*
  IEEE Control Systems Magazine, 38(3), 63–86.
  DOI:[10.1109/MCS.2018.2810460](https://doi.org/10.1109/MCS.2018.2810460);
  arXiv:[1701.07569](https://arxiv.org/abs/1701.07569)

---

## 5. Common Task Framework (CTF) — dynamicsai.org

- **Wyder, P. M., Goldfeder, J., Yermakov, A., Zhao, Y., Riva, S., Williams, J., Zoro, D.,
  Rude, A. S., Tomasetto, M., Germany, J., Bakarji, J., Maierhofer, G., Cranmer, M.
  & Kutz, J. N.** (2025).
  *Common Task Framework for a Critical Evaluation of Scientific Machine Learning Algorithms.*
  NeurIPS 2025 Datasets and Benchmarks Track.
  arXiv:[2510.23166](https://arxiv.org/abs/2510.23166)

- **CTF-for-Science/ctf4science.**
  GitHub: [CTF-for-Science/ctf4science](https://github.com/CTF-for-Science/ctf4science);
  Datasets on Kaggle: [Dynamics AI organization](https://www.kaggle.com/organizations/dynamicsai)

- **AI Institute in Dynamic Systems.** CTF overview page:
  [https://dynamicsai.org/CTF/](https://dynamicsai.org/CTF/)

---

## 6. Symbolic Regression

### 6.1 PySR

- **Cranmer, M.** (2023).
  *Interpretable machine learning for science with PySR and SymbolicRegression.jl.*
  arXiv:[2305.01582](https://arxiv.org/abs/2305.01582)

- **Cranmer, M., Sanchez-Gonzalez, A., Battaglia, P., Xu, R., Cranmer, K., Spergel, D. & Ho, S.** (2020).
  *Discovering symbolic models from deep learning with inductive biases.*
  Advances in Neural Information Processing Systems (NeurIPS), 33, 17429–17442.

### 6.2 AI Feynman

- **Udrescu, S.-M. & Tegmark, M.** (2020).
  *AI Feynman: a physics-inspired method for symbolic regression.*
  Science Advances, 6(16), eaay2631.
  DOI:[10.1126/sciadv.aay2631](https://doi.org/10.1126/sciadv.aay2631)

- **Udrescu, S.-M., Tan, A., Feng, J., Neto, O., Wu, T. & Tegmark, M.** (2020).
  *AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity.*
  NeurIPS 2020.
  arXiv:[2006.10782](https://arxiv.org/abs/2006.10782)

### 6.3 Kolmogorov-Arnold Networks (KAN)

- **Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y.
  & Tegmark, M.** (2025).
  *KAN: Kolmogorov-Arnold Networks.*
  ICLR 2025.
  arXiv:[2404.19756](https://arxiv.org/abs/2404.19756)

### 6.4 Surveys and Benchmarks

- **La Cava, W., Orzechowski, P., Burlacu, B., de França, F. O., Virgolin, M., Jin, Y.,
  Kommenda, M. & Moore, J. H.** (2021).
  *Contemporary symbolic regression methods and their relative performance.*
  NeurIPS Datasets and Benchmarks Track.
  arXiv:[2107.14351](https://arxiv.org/abs/2107.14351)

- **Makke, N. & Chawla, S.** (2024).
  *Interpretable scientific discovery with symbolic regression: a review.*
  Artificial Intelligence Review, 57, 2.
  DOI:[10.1007/s10462-023-10622-0](https://doi.org/10.1007/s10462-023-10622-0)

### 6.5 Early Data-Driven Law Discovery

- **Schmidt, M. & Lipson, H.** (2009).
  *Distilling free-form natural laws from experimental data.*
  Science, 324(5923), 81–85.
  DOI:[10.1126/science.1165893](https://doi.org/10.1126/science.1165893)

### 6.6 Deep Symbolic Regression

- **Petersen, B. K., Landajuela, M., Mundhenk, T. N., Santiago, C. P., Kim, S. & Kim, J. T.** (2021).
  *Deep symbolic regression: recovering mathematical expressions from data via risk-seeking policy gradients.*
  ICLR 2021.

---

## 7. Numerical and Computational Background

- **Sheffer, H. M.** (1913).
  *A set of five independent postulates for Boolean algebras, with application to logical constants.*
  Transactions of the American Mathematical Society, 14(4), 481–488.
  DOI:[10.1090/S0002-9947-1913-1500960-1](https://doi.org/10.1090/S0002-9947-1913-1500960-1)

- **Jang, E., Gu, S. & Poole, B.** (2017).
  *Categorical reparameterization with Gumbel-Softmax.*
  ICLR 2017.

- **Paszke, A.** et al. (2019).
  *PyTorch: an imperative style, high-performance deep learning library.*
  Advances in Neural Information Processing Systems (NeurIPS), 32, 8024–8035.

- **Harris, C. R.** et al. (2020).
  *Array programming with NumPy.*
  Nature, 585(7825), 357–362.
  DOI:[10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)

---

## 8. Additional References

*(Add new references below as the project develops.)*

<!-- Template for new entries:

- **Author, A. B., Author, C. D. & Author, E. F.** (Year).
  *Title of the paper.*
  Journal Name, Volume(Issue), Pages.
  DOI:[10.xxxx/xxxxxxx](https://doi.org/10.xxxx/xxxxxxx);
  arXiv:[XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

-->
