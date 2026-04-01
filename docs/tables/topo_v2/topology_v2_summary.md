# Topology V2 Metrics

Computed with per-sample-first averaging on the existing Table2 predictions.

- T_sens is reported as requested.
- Betti Error uses: |CC_pred-CC_gt| + |Holes_pred-Holes_gt| (per sample, then averaged).

| Model | Samples | T_prec | T_sens | clDice | Betti Error | |dBeta0| | |dBeta1| |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B2 (baseline) | 281 | 0.6946 | 0.8637 | 0.7103 | 148.6619 | 79.1815 | 69.4804 |
| SegFormer-B2 + Skeleton | 281 | 0.7153 | 0.8741 | 0.7489 | 66.9004 | 38.6335 | 28.2669 |
| SegFormer-B2 + Skeleton + clDice-v2 | 281 | 0.8560 | 0.9009 | 0.8545 | 185.1601 | 41.9573 | 143.2028 |
| DeepLabV3+ (R50) | 281 | 0.8327 | 0.8086 | 0.7905 | 76.9822 | 44.8078 | 32.1744 |
| PSPNet (R50) | 281 | 0.8497 | 0.8077 | 0.7986 | 24.1993 | 16.7438 | 7.4555 |
| UPerNet (R50) | 281 | 0.7308 | 0.7742 | 0.7038 | 38.6584 | 30.2562 | 8.4021 |
