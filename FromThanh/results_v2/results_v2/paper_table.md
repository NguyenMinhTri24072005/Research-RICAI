| Group | Method | K=0 MAE / RMSE / MAPE% | K=1 MAE / RMSE / MAPE% | K=5 MAE / RMSE / MAPE% | K=10 MAE / RMSE / MAPE% |
|---|---|---|---|---|---|
| Gaussian process | GP on V_bulk, support-only | – | – | 14.6 / 20.1 / 11.11 | 7.2 / 9.6 / 5.50 |
| Gaussian process | GP residual on source log-log, pooled+sup. | 10.1 / 13.3 / 5.63 | 10.0 / 13.3 / 5.59 | 10.0 / 13.1 / 5.56 | 9.6 / 12.6 / 5.36 |
| Physics-based | Physics V_bulk/V_g, source offset | 60.1 / 82.0 / 28.78 | 60.2 / 82.0 / 28.78 | 60.5 / 82.3 / 29.02 | 60.0 / 81.1 / 28.78 |
| Physics-based | Repo physical estimate | 53.2 / 76.4 / 26.11 | 53.2 / 76.5 / 26.11 | 53.6 / 76.7 / 26.36 | 53.1 / 75.5 / 26.12 |
| Physics-based | Repo physical estimate + MLE scale | – | 50.0 / 71.3 / 26.95 | 39.8 / 59.2 / 20.69 | 37.9 / 56.3 / 19.56 |
| Tabular ML | Gradient Boosting, pooled + support | 14.4 / 17.4 / 11.57 | 13.6 / 16.7 / 9.95 | 11.6 / 14.3 / 7.92 | 10.3 / 12.8 / 6.85 |
| Tabular ML | Random Forest, pooled + support | 17.0 / 21.1 / 13.24 | 16.2 / 20.0 / 12.22 | 14.7 / 18.0 / 10.14 | 12.8 / 15.8 / 8.44 |
| Tabular ML | Ridge, all features, pooled + support | 14.3 / 17.9 / 7.47 | 12.4 / 16.1 / 6.94 | 10.1 / 13.5 / 5.81 | 8.9 / 11.7 / 5.10 |
| Vision-only | Ridge, image features only | 89.9 / 106.6 / 77.68 | 89.9 / 106.6 / 77.52 | 89.7 / 106.4 / 76.85 | 89.3 / 105.7 / 76.48 |
| Volume log-log | V_bulk log-log, pooled source | 10.1 / 13.3 / 5.63 | 10.1 / 13.3 / 5.62 | 10.2 / 13.4 / 5.66 | 10.0 / 13.2 / 5.56 |
| Volume log-log | V_bulk log-log, source slope + MLE shift | – | 10.6 / 13.1 / 6.73 | 8.5 / 11.0 / 5.15 | 7.8 / 9.9 / 4.81 |
| Volume log-log | V_bulk log-log, support-only | – | – | 7.5 / 9.4 / 4.97 | 6.7 / 8.4 / 4.32 |
| Weight-based | Weight + V_bulk log-log | 6.7 / 8.0 / 4.03 | 6.7 / 8.0 / 4.04 | 6.8 / 8.0 / 4.07 | 6.7 / 7.9 / 3.98 |
| Weight-based | Weight + V_bulk log-log + MLE shift | – | 3.0 / 3.8 / 2.44 | 2.3 / 3.1 / 1.91 | 2.1 / 2.8 / 1.81 |
| Weight-based | Weight only + MLE scale | – | 3.2 / 4.1 / 2.80 | 2.5 / 3.3 / 2.14 | 2.3 / 3.0 / 2.01 |
| Weight-based | Weight only, source scale | 7.0 / 8.2 / 4.08 | 7.0 / 8.2 / 4.07 | 7.0 / 8.3 / 4.11 | 6.9 / 8.2 / 4.02 |
| Ours | Ours+W (main): log W, g(z)=W+V_bulk | 6.0 / 7.3 / 3.76 | 2.8 / 3.6 / 2.33 | 2.2 / 3.0 / 1.88 | 2.0 / 2.7 / 1.78 |
| Ours | Ours+W ablation: + V_g in g(z) | 7.9 / 9.4 / 4.50 | 3.5 / 4.5 / 2.64 | 2.8 / 3.7 / 2.12 | 2.6 / 3.3 / 2.03 |
| Ours | Ours+W ablation: no NIG (fixed noise) | 6.0 / 7.3 / 3.76 | 2.8 / 3.6 / 2.33 | 2.2 / 3.0 / 1.88 | 2.0 / 2.7 / 1.78 |
| Ours | Ours+W ablation: scale only (g=0) | 6.6 / 7.9 / 3.93 | 3.3 / 4.2 / 2.80 | 2.5 / 3.3 / 2.15 | 2.3 / 3.0 / 2.02 |
| Ours | Ours+W ablation: uncalibrated (lam=1) | 6.0 / 7.3 / 3.76 | 2.8 / 3.6 / 2.33 | 2.2 / 3.0 / 1.88 | 2.0 / 2.7 / 1.78 |
| Ours | Ours+W ablation: within-lot g(z) | 6.1 / 7.7 / 3.53 | 3.9 / 4.9 / 3.11 | 3.3 / 4.3 / 2.53 | 2.9 / 3.8 / 2.32 |
| Ours | Ours-V (no scale): g(z)=V_g | 10.4 / 13.9 / 5.73 | 10.2 / 13.0 / 6.09 | 9.4 / 12.1 / 5.37 | 8.6 / 10.9 / 5.02 |
| Ours | Ours-V ablation: g=0 | 56.8 / 76.8 / 27.28 | 51.2 / 72.3 / 25.53 | 41.0 / 61.2 / 20.92 | 37.5 / 56.6 / 19.35 |
| Ours | Ours-V ablation: no V_g term | 12.7 / 16.6 / 7.71 | 10.8 / 13.6 / 6.91 | 9.3 / 11.9 / 5.71 | 8.5 / 10.7 / 5.36 |
| Ours | Ours-V ablation: uncalibrated | 10.4 / 13.9 / 5.73 | 10.2 / 13.0 / 6.09 | 9.4 / 12.1 / 5.37 | 8.6 / 10.9 / 5.02 |

90% interval coverage (%)

| model                                 |     0 |    1 |    3 |    5 |   10 |   20 |
|:--------------------------------------|------:|-----:|-----:|-----:|-----:|-----:|
| Ours+W (main): log W, g(z)=W+V_bulk   |  99   | 85.4 | 92.8 | 93   | 96   | 95.2 |
| Ours+W ablation: + V_g in g(z)        |  99   | 85.1 | 92.6 | 92.7 | 95.3 | 94.4 |
| Ours+W ablation: no NIG (fixed noise) |  98   | 84.8 | 83.5 | 84.8 | 83.1 | 83.5 |
| Ours+W ablation: scale only (g=0)     | 100   | 81.1 | 90.8 | 92.4 | 95.9 | 94.6 |
| Ours+W ablation: uncalibrated (lam=1) |  99   | 85.4 | 92.5 | 92.6 | 95.5 | 94.9 |
| Ours+W ablation: within-lot g(z)      | 100   | 81.4 | 89.9 | 91.3 | 94.7 | 93.1 |
| Ours-V (no scale): g(z)=V_g           |  99   | 93   | 91.7 | 90.9 | 93.2 | 92.6 |
| Ours-V ablation: g=0                  |  96.4 | 92.6 | 92.4 | 91.2 | 92.6 | 92.7 |
| Ours-V ablation: no V_g term          |  94   | 87.5 | 91.3 | 91.4 | 93.6 | 92.4 |
| Ours-V ablation: uncalibrated         |  99   | 91.7 | 90   | 88.9 | 91.6 | 90.9 |