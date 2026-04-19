| model | samples_before | samples_after | removed | mean_T_prec | mean_T_sens | mean_clDice | mean_Betti_Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SegFormer-B2 (baseline) | 281 | 249 | 32 | 0.743617 | 0.860458 | 0.748242 | 75.481928 |
| SegFormer-B2 + Skeleton | 281 | 254 | 27 | 0.749170 | 0.876725 | 0.775948 | 35.326772 |
| SegFormer-B2 + Skeleton + clDice-v2 | 281 | 239 | 42 | 0.888866 | 0.910910 | 0.882459 | 47.891213 |
| DeepLabV3+ (R50) | 281 | 257 | 24 | 0.850902 | 0.826070 | 0.810783 | 42.964981 |
| PSPNet (R50) | 281 | 254 | 27 | 0.867941 | 0.819959 | 0.815492 | 13.909449 |
| UPerNet (R50) | 281 | 276 | 5 | 0.737446 | 0.775262 | 0.709225 | 35.873188 |
