# Devito Self Adjoint modeling operators

## These operators are contributed by Chevron Energy Technology Company (2020)

These operators are based on simplfications of the systems presented in:
<br>**Self-adjoint, energy-conserving second-order pseudoacoustic systems for VTI and TTI media for reverse migration and full-waveform inversion** (2016)
<br>Kenneth Bube, John Washbourne, Raymond Ergas, and Tamas Nemeth
<br>SEG Technical Program Expanded Abstracts
<br>https://library.seg.org/doi/10.1190/segam2016-13878451.1

## Tutorial goal

The goal of this series of tutorials is to generate -- and then test for correctness -- the modeling and inversion capability in Devito for variable density visco- acoustics. We use an energy conserving form of the wave equation that is *self adjoint*, which allows the same modeling system to be used for all for all phases of finite difference evolution required for quasi-Newton optimization:
- **nonlinear forward**, nonlinear with respect to the model parameters
- **Jacobian forward**, linearized with respect to the model parameters 
- **Jacobian adjoint**, linearized with respect to the model parameters

These notebooks first implement and then test for correctness for three types of modeling physics.

| Physics         | Implementation            | Notebook                          |
|:----------------|:--------------------------|:----------------------------------|
| Isotropic       | Nonlinear ops             | [sa_01_iso_implementation1.ipynb] |
| Isotropic       | Linearized ops            | [sa_02_iso_implementation2.ipynb] |
| Isotropic       | Correctness tests         | [sa_03_iso_correctness.ipynb]     |
|-----------------|---------------------------|-----------------------------------|
| VTI Anisotropic | Nonlinear/linearized ops  | [sa_11_vti_implementation.ipynb]  |
| VTI Anisotropic | Correctness tests         | [sa_12_vti_correctness.ipynb]     |
|-----------------|---------------------------|-----------------------------------|
| TTI Anisotropic | Nonlinear/linearized ops  | [sa_21_tti_implementation.ipynb]  |
| TTI Anisotropic | Correctness tests         | [sa_22_tti_correctness.ipynb]     |
|:----------------|:--------------------------|:----------------------------------|

[sa_01_iso_implementation1.ipynb]: sa_01_iso_implementation1.ipynb
[sa_02_iso_implementation2.ipynb]: sa_02_iso_implementation2.ipynb
[sa_03_iso_correctness.ipynb]:     sa_03_iso_correctness.ipynb
[sa_11_vti_implementation.ipynb]: sa_11_vti_implementation.ipynb
[sa_12_vti_correctness.ipynb]:     sa_12_vti_correctness.ipynb
[sa_21_tti_implementation.ipynb]: sa_21_tti_implementation.ipynb
[sa_22_tti_correctness.ipynb]:     sa_22_tti_correctness.ipynb

## Running unit tests
- if you would like to see stdout when running the tests, use
```py.test -c testUtils.py```

## TODO
- [X] Devito-esque equation version of setup_w_over_q
- [ ] figure out weird test failure depending on the order of equations in operator

**Equation order 1**
```
    return Operator([dm_update] + eqn + rec_term, subs=spacing_map,
                    name='IsoJacobianAdjOperator', **kwargs)
```
**Equation order 2**
```
    return Operator(eqn + rec_term + [dm_update], subs=spacing_map,
                    name='IsoJacobianAdjOperator', **kwargs)
```
    - With Equation order 1, all tests pass
    - With Equation order 2, there are different outcomes for tests 
    - Possibly there is a different path chosen through the AST, and different c code is generated?

- [ ] replace the conditional logic in the stencil with comprehension
```
    space_fd = sum([getattr(b * getattr(field, 'd%s'%d.name)(x0=d+d.spacing/2)),
        'd%s'%d.name)(x0=d-d.spacing/2)) for d in field.dimensions[1:]])
```
- [ ] Add memoized methods back to wavesolver.py
- [ ] Add ensureSanityOfFields methods for iso, vti, tti
- [ ] Add timing info via logging for the w_over_q setup, as in initialize_damp
- [ ] Add smoother back to setup_w_over_q method
```
     eqn1 = Eq(wOverQ, val)
     Operator([eqn1], name='WOverQ_Operator_init')()
     # If we apply the smoother, we must renormalize output to [qmin,qmax]
     if sigma > 0:
         smooth = gaussian_smooth(wOverQ.data, sigma=sigma)
         smin, smax = np.min(smooth), np.max(smooth)
         smooth[:] = qmin + (qmax - qmin) * (smooth - smin) / (smax - smin)
         wOverQ.data[:] = smooth
     eqn2 = Eq(wOverQ, w / wOverQ)
     Operator([eqn2], name='WOverQ_Operator_recip')()
```
- [X] Correctness tests
  - [X] Analytic response in the far field
  - [X] Modeling operator linearity test, with respect to source
  - [X] Modeling operator adjoint test, with respect to source
  - [X] Nonlinear operator linearization test, with respect to model/data
  - [X] Jacobian operator linearity test, with respect to model/data
  - [X] Jacobian operator adjoint test, with respect to model/data
  - [X] Skew symmetry test for shifted derivatives

## To save generated code 

```
f = open("operator.c", "w")
print(op, file=f)
f.close()
```