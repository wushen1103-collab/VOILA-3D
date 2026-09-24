# Scaffold-partition robustness summary

Each partition estimate averages five model seeds; summary SDs are across five additional scaffold partitions (seeds 1--5).

- R9 macro BIG: -0.000168 +/- 0.000535 across partitions.
- Random macro BIG: 0.000980 +/- 0.010448 across partitions.
- R9 exceeded random BIG in 19/30 task-partition comparisons.
- R9 mean macro call rate: 4.34% +/- 0.77% across partitions.

## Split-level macro values

|   split_seed | policy   |   macro_BIG |   macro_normalized_BIG |   macro_call_rate |
|-------------:|:---------|------------:|-----------------------:|------------------:|
|            1 | R7       |   -0.003991 |              -0.049268 |         20.009412 |
|            1 | R8       |   -0.002124 |              -0.023834 |         13.114233 |
|            1 | R9       |   -0.000750 |              -0.017062 |          3.863567 |
|            1 | Random   |   -0.007466 |              -0.088814 |         20.009412 |
|            2 | R7       |   -0.007446 |              -0.188660 |         20.022788 |
|            2 | R8       |   -0.001356 |              -0.086724 |         12.700475 |
|            2 | R9       |    0.000819 |               0.020858 |          3.337559 |
|            2 | Random   |   -0.006702 |              -0.137909 |         20.022788 |
|            3 | R7       |    0.007323 |              -0.005626 |         19.980716 |
|            3 | R8       |    0.003427 |               0.004178 |         13.283296 |
|            3 | R9       |   -0.000454 |              -0.010040 |          4.545868 |
|            3 | Random   |    0.004363 |              -0.055096 |         19.980716 |
|            4 | R7       |    0.025411 |               0.083158 |         19.991610 |
|            4 | R8       |    0.009106 |               0.045876 |         13.029040 |
|            4 | R9       |   -0.000112 |              -0.011255 |          4.302465 |
|            4 | Random   |    0.020057 |               0.018307 |         19.991610 |
|            5 | R7       |    0.001163 |               0.019335 |         20.028097 |
|            5 | R8       |    0.002328 |               0.025858 |         12.860162 |
|            5 | R9       |   -0.000344 |              -0.009377 |          5.631625 |
|            5 | Random   |   -0.005353 |              -0.073142 |         20.028097 |
