# K3 Mass Share Sweep Review: 2026-09-03

## Evidence

Result commit `a124704` contains five A100-share runs for each compiler. The
tested code is commit `efb41af`, with the K3 heterogeneous mass policy from
`733e7ea`.

| Compiler | Shares tested | Placement/VLEN | Mass max error |
|---|---|---|---:|
| GCC 15.2 | 0, 25, 50, 75, 100 | pass for all 5 | 0 for all 5 |
| Clang 24 | 0, 25, 50, 75, 100 | pass for all 5 | 0 for all 5 |

Every run used 16 OpenMP workers and 64 placement/VLEN stability rounds. X100
workers reported singleton CPU masks for CPUs 0-7. A100 workers remained in
the `0xff00` A100-class mask and reported `vlenb=128`, e64/m1 VLMAX=16.

Shares 0 and 100 validate the edge cases where one worker group receives no
cell indices. Shares 25, 50, and 75 validate mixed range ownership. All runs
reported `K3 heterogeneous mass max error: 0` against the existing host mass
operator.

## Assessment

The cell-range partition and barrier/placement policy are correct for the
current order-1, 4x4x4 mass case under both compilers. No runner or policy bug
was exposed by the share sweep.

This is not yet performance evidence and does not validate high-order tensor
stack pressure, large meshes, advection, shared H1 DoFs, or face operators.
The next test must increase polynomial order and mesh size before introducing
separate tuned X100/A100 kernel objects.

## Next Commands

The current mass target uses a small compile-time order-1 case. Extend that
target with explicit order/mesh variants, then repeat the same share matrix:

```sh
scripts/machines/spacemit-k3/run-k3-mass-share-sweep.sh
```

Do not compare throughput until the larger-order cases pass for both compilers
and every share.
