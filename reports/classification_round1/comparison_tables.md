# Classification Round 1 — comparison tables

Protocol `samuel-classification-round1-v1`; run `classification_round1_v1`; 56 evaluated recipe-window jobs.
Validation only. Macro F1 is computed with fixed labels `[0,1]`; AP is sklearn average precision. Selected thresholds are fit-history inner-only; no outer validation label chooses its threshold.

## Primary pooled metrics

| Model | Method | Inner threshold median [range] | Pooled F1 @0.50 | Pooled F1 inner | Δ F1 | Actual class 1 rate | Predicted class 1 @0.50 | Predicted class 1 inner | Balanced accuracy inner | AP | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_A | Majority class | 0.50 [0.50, 0.50] | 0.3894 | 0.3894 | 0.0000 | 0.3623 | 0.0000 | 0.0000 | 0.5000 | 0.3420 | 0.2334 |
| C0_B | EF positive-rate lookup | 0.39 [0.30, 0.42] | 0.3894 | 0.5469 | 0.1575 | 0.3623 | 0.0000 | 0.3577 | 0.5468 | 0.3800 | 0.2313 |
| C0_C | Logistic regression, impact + EF | 0.36 [0.32, 0.39] | 0.3894 | 0.5549 | 0.1656 | 0.3623 | 0.0000 | 0.4170 | 0.5582 | 0.4107 | 0.2297 |
| C1_A | Logistic regression, structured | 0.45 [0.32, 0.51] | 0.7145 | 0.7112 | -0.0033 | 0.3623 | 0.4169 | 0.4929 | 0.7330 | 0.6734 | 0.1690 |
| C1_B | Histogram gradient boosting, structured | 0.37 [0.29, 0.44] | 0.7310 | 0.7333 | 0.0023 | 0.3623 | 0.3264 | 0.4551 | 0.7487 | 0.7086 | 0.1593 |
| C2_A | TF-IDF + logistic regression | 0.42 [0.39, 0.48] | 0.8014 | 0.8057 | 0.0043 | 0.3623 | 0.2969 | 0.3338 | 0.8006 | 0.8181 | 0.1259 |
| C2_B | Structured + TF-IDF logistic regression | 0.47 [0.34, 0.53] | 0.8206 | 0.8188 | -0.0019 | 0.3623 | 0.3509 | 0.3752 | 0.8212 | 0.8276 | 0.1187 |

Pooled F1 uses all primary validation rows once. Delta is selected minus fixed. AP uses the probability vector and is independent of threshold.

## Per-class precision, recall and F1

| Model | Policy | Actual class | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| C0_A | 0.50 | 0 | 24607 | 0.6377 | 1.0000 | 0.7788 |
| C0_A | 0.50 | 1 | 13981 | 0.0000 | 0.0000 | 0.0000 |
| C0_A | inner-selected | 0 | 24607 | 0.6377 | 1.0000 | 0.7788 |
| C0_A | inner-selected | 1 | 13981 | 0.0000 | 0.0000 | 0.0000 |
| C0_B | 0.50 | 0 | 24607 | 0.6377 | 1.0000 | 0.7788 |
| C0_B | 0.50 | 1 | 13981 | 0.0000 | 0.0000 | 0.0000 |
| C0_B | inner-selected | 0 | 24607 | 0.6713 | 0.6762 | 0.6738 |
| C0_B | inner-selected | 1 | 13981 | 0.4227 | 0.4174 | 0.4200 |
| C0_C | 0.50 | 0 | 24607 | 0.6377 | 1.0000 | 0.7788 |
| C0_C | 0.50 | 1 | 13981 | 0.0000 | 0.0000 | 0.0000 |
| C0_C | inner-selected | 0 | 24607 | 0.6838 | 0.6252 | 0.6532 |
| C0_C | inner-selected | 1 | 13981 | 0.4268 | 0.4912 | 0.4567 |
| C1_A | 0.50 | 0 | 24607 | 0.8140 | 0.7443 | 0.7775 |
| C1_A | 0.50 | 1 | 13981 | 0.6088 | 0.7006 | 0.6515 |
| C1_A | inner-selected | 0 | 24607 | 0.8500 | 0.6759 | 0.7530 |
| C1_A | inner-selected | 1 | 13981 | 0.5807 | 0.7900 | 0.6694 |
| C1_B | 0.50 | 0 | 24607 | 0.7930 | 0.8376 | 0.8147 |
| C1_B | 0.50 | 1 | 13981 | 0.6829 | 0.6153 | 0.6473 |
| C1_B | inner-selected | 0 | 24607 | 0.8486 | 0.7251 | 0.7820 |
| C1_B | inner-selected | 1 | 13981 | 0.6149 | 0.7723 | 0.6847 |
| C2_A | 0.50 | 0 | 24607 | 0.8286 | 0.9136 | 0.8691 |
| C2_A | 0.50 | 1 | 13981 | 0.8145 | 0.6675 | 0.7337 |
| C2_A | inner-selected | 0 | 24607 | 0.8462 | 0.8840 | 0.8647 |
| C2_A | inner-selected | 1 | 13981 | 0.7785 | 0.7173 | 0.7466 |
| C2_B | 0.50 | 0 | 24607 | 0.8644 | 0.8799 | 0.8721 |
| C2_B | 0.50 | 1 | 13981 | 0.7817 | 0.7570 | 0.7692 |
| C2_B | inner-selected | 0 | 24607 | 0.8753 | 0.8576 | 0.8663 |
| C2_B | inner-selected | 1 | 13981 | 0.7579 | 0.7849 | 0.7712 |

Class-specific metrics use fixed labels 0 and 1. The recorded zero-division convention sets undefined precision/recall/F1 to zero.

## Primary pooled confusion counts

| Model | Policy | TN | FP | FN | TP | False positives | False negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0_A | 0.50 | 24,607 | 0 | 13,981 | 0 | 0 | 13,981 |
| C0_A | inner-selected | 24,607 | 0 | 13,981 | 0 | 0 | 13,981 |
| C0_B | 0.50 | 24,607 | 0 | 13,981 | 0 | 0 | 13,981 |
| C0_B | inner-selected | 16,639 | 7,968 | 8,146 | 5,835 | 7,968 | 8,146 |
| C0_C | 0.50 | 24,607 | 0 | 13,981 | 0 | 0 | 13,981 |
| C0_C | inner-selected | 15,384 | 9,223 | 7,114 | 6,867 | 9,223 | 7,114 |
| C1_A | 0.50 | 18,314 | 6,293 | 4,186 | 9,795 | 6,293 | 4,186 |
| C1_A | inner-selected | 16,633 | 7,974 | 2,936 | 11,045 | 7,974 | 2,936 |
| C1_B | 0.50 | 20,612 | 3,995 | 5,379 | 8,602 | 3,995 | 5,379 |
| C1_B | inner-selected | 17,843 | 6,764 | 3,183 | 10,798 | 6,764 | 3,183 |
| C2_A | 0.50 | 22,482 | 2,125 | 4,649 | 9,332 | 2,125 | 4,649 |
| C2_A | inner-selected | 21,753 | 2,854 | 3,953 | 10,028 | 2,854 | 3,953 |
| C2_B | 0.50 | 21,652 | 2,955 | 3,397 | 10,584 | 2,955 | 3,397 |
| C2_B | inner-selected | 21,102 | 3,505 | 3,007 | 10,974 | 3,505 | 3,007 |

Rows are actual classes 0/1; columns are predicted classes 0/1. FP means actual 0/predicted 1; FN means actual 1/predicted 0.

## Equal-cycle F1

| Model | Equal-cycle F1 @0.50 | Equal-cycle F1 inner |
| --- | --- | --- |
| C0_A | 0.3868 | 0.3868 |
| C0_B | 0.3868 | 0.4533 |
| C0_C | 0.3868 | 0.5393 |
| C1_A | 0.7121 | 0.7101 |
| C1_B | 0.7262 | 0.7295 |
| C2_A | 0.7916 | 0.7956 |
| C2_B | 0.8124 | 0.8110 |

This is the arithmetic mean over 24 per-cycle Macro F1 scores; it is distinct from pooled F1.

## Primary-window F1 — fixed threshold 0.50

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
| --- | --- | --- | --- | --- | --- | --- |
| C0_A | 0.3662 | 0.3531 | 0.3892 | 0.3849 | 0.4054 | 0.4137 |
| C0_B | 0.3662 | 0.3531 | 0.3892 | 0.3849 | 0.4054 | 0.4137 |
| C0_C | 0.3662 | 0.3531 | 0.3892 | 0.3849 | 0.4054 | 0.4137 |
| C1_A | 0.7200 | 0.7764 | 0.7285 | 0.7176 | 0.6605 | 0.6756 |
| C1_B | 0.7244 | 0.7662 | 0.7228 | 0.7241 | 0.7212 | 0.7037 |
| C2_A | 0.7493 | 0.8193 | 0.8037 | 0.8178 | 0.7937 | 0.7974 |
| C2_B | 0.7767 | 0.8471 | 0.8257 | 0.8409 | 0.8040 | 0.8056 |

## Primary-window F1 — inner-selected threshold

| Model | W1 | W2 | W3 | W4 | W5 | W6 |
| --- | --- | --- | --- | --- | --- | --- |
| C0_A | 0.3662 | 0.3531 | 0.3892 | 0.3849 | 0.4054 | 0.4137 |
| C0_B | 0.3662 | 0.3123 | 0.5304 | 0.3849 | 0.5488 | 0.5732 |
| C0_C | 0.5454 | 0.5492 | 0.5032 | 0.5641 | 0.5433 | 0.5542 |
| C1_A | 0.7445 | 0.7452 | 0.7232 | 0.7153 | 0.6663 | 0.6733 |
| C1_B | 0.7451 | 0.7560 | 0.7210 | 0.7410 | 0.7053 | 0.7142 |
| C2_A | 0.7528 | 0.8253 | 0.8045 | 0.8202 | 0.7959 | 0.8046 |
| C2_B | 0.7857 | 0.8233 | 0.8250 | 0.8408 | 0.8029 | 0.8089 |

## Per-cycle prevalence and positive-prediction rates

| Model | Cycle | n | Actual class 1 rate | Predicted class 1 rate @0.50 | Predicted class 1 rate inner | Macro F1 inner |
| --- | --- | --- | --- | --- | --- | --- |
| C0_C | 56 | 564 | 0.2837 | 0.0000 | 0.1082 | 0.5357 |
| C0_C | 57 | 962 | 0.4179 | 0.0000 | 0.1538 | 0.4984 |
| C0_C | 58 | 1205 | 0.4954 | 0.0000 | 0.2232 | 0.5494 |
| C0_C | 59 | 1383 | 0.4179 | 0.0000 | 0.1656 | 0.5600 |
| C0_C | 60 | 1478 | 0.4790 | 0.0000 | 0.6042 | 0.5401 |
| C0_C | 61 | 1632 | 0.4822 | 0.0000 | 0.5607 | 0.5531 |
| C0_C | 62 | 1500 | 0.3740 | 0.0000 | 0.5807 | 0.5444 |
| C0_C | 63 | 1883 | 0.4742 | 0.0000 | 0.6054 | 0.5527 |
| C0_C | 64 | 1807 | 0.3857 | 0.0000 | 0.6082 | 0.4709 |
| C0_C | 65 | 1567 | 0.3638 | 0.0000 | 0.5858 | 0.4958 |
| C0_C | 66 | 1751 | 0.3444 | 0.0000 | 0.5460 | 0.5162 |
| C0_C | 67 | 1953 | 0.3574 | 0.0000 | 0.5346 | 0.5244 |
| C0_C | 68 | 577 | 0.2964 | 0.0000 | 0.3674 | 0.5095 |
| C0_C | 69 | 1079 | 0.4087 | 0.0000 | 0.4133 | 0.5588 |
| C0_C | 70 | 1605 | 0.3807 | 0.0000 | 0.4025 | 0.5404 |
| C0_C | 71 | 1689 | 0.3730 | 0.0000 | 0.3564 | 0.6052 |
| C0_C | 72 | 1811 | 0.3512 | 0.0000 | 0.2766 | 0.5391 |
| C0_C | 73 | 1880 | 0.3309 | 0.0000 | 0.2362 | 0.5575 |
| C0_C | 74 | 1814 | 0.3060 | 0.0000 | 0.2734 | 0.5238 |
| C0_C | 75 | 1957 | 0.2867 | 0.0000 | 0.2892 | 0.5508 |
| C0_C | 76 | 1822 | 0.3425 | 0.0000 | 0.3924 | 0.5756 |
| C0_C | 77 | 2335 | 0.2852 | 0.0000 | 0.4338 | 0.5494 |
| C0_C | 78 | 2146 | 0.2502 | 0.0000 | 0.4226 | 0.5168 |
| C0_C | 79 | 2188 | 0.3076 | 0.0000 | 0.4566 | 0.5741 |
| C1_A | 56 | 564 | 0.2837 | 0.2057 | 0.2394 | 0.7223 |
| C1_A | 57 | 962 | 0.4179 | 0.3233 | 0.3909 | 0.7432 |
| C1_A | 58 | 1205 | 0.4954 | 0.3477 | 0.4257 | 0.7294 |
| C1_A | 59 | 1383 | 0.4179 | 0.3333 | 0.3897 | 0.7545 |
| C1_A | 60 | 1478 | 0.4790 | 0.4878 | 0.6292 | 0.7960 |
| C1_A | 61 | 1632 | 0.4822 | 0.5135 | 0.6679 | 0.7524 |
| C1_A | 62 | 1500 | 0.3740 | 0.4260 | 0.5887 | 0.7089 |
| C1_A | 63 | 1883 | 0.4742 | 0.5619 | 0.6952 | 0.7222 |
| C1_A | 64 | 1807 | 0.3857 | 0.5180 | 0.5567 | 0.6941 |
| C1_A | 65 | 1567 | 0.3638 | 0.4065 | 0.4474 | 0.7611 |
| C1_A | 66 | 1751 | 0.3444 | 0.4654 | 0.5174 | 0.7176 |
| C1_A | 67 | 1953 | 0.3574 | 0.4270 | 0.4849 | 0.7233 |
| C1_A | 68 | 577 | 0.2964 | 0.3293 | 0.3241 | 0.7327 |
| C1_A | 69 | 1079 | 0.4087 | 0.4551 | 0.4402 | 0.7041 |
| C1_A | 70 | 1605 | 0.3807 | 0.4866 | 0.4723 | 0.7192 |
| C1_A | 71 | 1689 | 0.3730 | 0.4784 | 0.4671 | 0.7090 |
| C1_A | 72 | 1811 | 0.3512 | 0.4909 | 0.5726 | 0.6818 |
| C1_A | 73 | 1880 | 0.3309 | 0.4324 | 0.5362 | 0.6773 |
| C1_A | 74 | 1814 | 0.3060 | 0.4019 | 0.4768 | 0.6529 |
| C1_A | 75 | 1957 | 0.2867 | 0.4343 | 0.5156 | 0.6490 |
| C1_A | 76 | 1822 | 0.3425 | 0.3650 | 0.4407 | 0.7057 |
| C1_A | 77 | 2335 | 0.2852 | 0.3169 | 0.4236 | 0.6635 |
| C1_A | 78 | 2146 | 0.2502 | 0.2870 | 0.3779 | 0.6572 |
| C1_A | 79 | 2188 | 0.3076 | 0.3341 | 0.4356 | 0.6663 |
| C1_B | 56 | 564 | 0.2837 | 0.1915 | 0.2766 | 0.7230 |
| C1_B | 57 | 962 | 0.4179 | 0.3119 | 0.4137 | 0.7796 |
| C1_B | 58 | 1205 | 0.4954 | 0.3361 | 0.4440 | 0.7101 |
| C1_B | 59 | 1383 | 0.4179 | 0.3261 | 0.4158 | 0.7494 |
| C1_B | 60 | 1478 | 0.4790 | 0.4486 | 0.6042 | 0.7868 |
| C1_B | 61 | 1632 | 0.4822 | 0.4522 | 0.6250 | 0.7688 |
| C1_B | 62 | 1500 | 0.3740 | 0.3660 | 0.5347 | 0.7277 |
| C1_B | 63 | 1883 | 0.4742 | 0.4679 | 0.6431 | 0.7372 |
| C1_B | 64 | 1807 | 0.3857 | 0.3935 | 0.5285 | 0.7045 |
| C1_B | 65 | 1567 | 0.3638 | 0.3357 | 0.4263 | 0.7396 |
| C1_B | 66 | 1751 | 0.3444 | 0.3752 | 0.4991 | 0.7055 |
| C1_B | 67 | 1953 | 0.3574 | 0.3246 | 0.4491 | 0.7335 |
| C1_B | 68 | 577 | 0.2964 | 0.2218 | 0.3397 | 0.7423 |
| C1_B | 69 | 1079 | 0.4087 | 0.3448 | 0.4838 | 0.7665 |
| C1_B | 70 | 1605 | 0.3807 | 0.3502 | 0.4916 | 0.7327 |
| C1_B | 71 | 1689 | 0.3730 | 0.3410 | 0.4926 | 0.7275 |
| C1_B | 72 | 1811 | 0.3512 | 0.3280 | 0.5025 | 0.7325 |
| C1_B | 73 | 1880 | 0.3309 | 0.3005 | 0.4590 | 0.7145 |
| C1_B | 74 | 1814 | 0.3060 | 0.2426 | 0.4520 | 0.7172 |
| C1_B | 75 | 1957 | 0.2867 | 0.2867 | 0.4609 | 0.6567 |
| C1_B | 76 | 1822 | 0.3425 | 0.2827 | 0.3392 | 0.7447 |
| C1_B | 77 | 2335 | 0.2852 | 0.2283 | 0.2981 | 0.7543 |
| C1_B | 78 | 2146 | 0.2502 | 0.2036 | 0.2647 | 0.6947 |
| C1_B | 79 | 2188 | 0.3076 | 0.3158 | 0.4031 | 0.6593 |
| C2_A | 56 | 564 | 0.2837 | 0.1277 | 0.1348 | 0.6785 |
| C2_A | 57 | 962 | 0.4179 | 0.3046 | 0.3077 | 0.7594 |
| C2_A | 58 | 1205 | 0.4954 | 0.3261 | 0.3353 | 0.7599 |
| C2_A | 59 | 1383 | 0.4179 | 0.2957 | 0.3022 | 0.7505 |
| C2_A | 60 | 1478 | 0.4790 | 0.3904 | 0.4438 | 0.8462 |
| C2_A | 61 | 1632 | 0.4822 | 0.3940 | 0.4473 | 0.8479 |
| C2_A | 62 | 1500 | 0.3740 | 0.3020 | 0.3540 | 0.7955 |
| C2_A | 63 | 1883 | 0.4742 | 0.3861 | 0.4472 | 0.8071 |
| C2_A | 64 | 1807 | 0.3857 | 0.3414 | 0.4051 | 0.7529 |
| C2_A | 65 | 1567 | 0.3638 | 0.3146 | 0.3772 | 0.8406 |
| C2_A | 66 | 1751 | 0.3444 | 0.2895 | 0.3655 | 0.8111 |
| C2_A | 67 | 1953 | 0.3574 | 0.2934 | 0.3472 | 0.8182 |
| C2_A | 68 | 577 | 0.2964 | 0.1976 | 0.2392 | 0.7503 |
| C2_A | 69 | 1079 | 0.4087 | 0.3587 | 0.3976 | 0.8325 |
| C2_A | 70 | 1605 | 0.3807 | 0.3427 | 0.3819 | 0.8297 |
| C2_A | 71 | 1689 | 0.3730 | 0.3173 | 0.3606 | 0.8196 |
| C2_A | 72 | 1811 | 0.3512 | 0.3092 | 0.3236 | 0.7999 |
| C2_A | 73 | 1880 | 0.3309 | 0.2319 | 0.2468 | 0.7915 |
| C2_A | 74 | 1814 | 0.3060 | 0.2883 | 0.3015 | 0.8254 |
| C2_A | 75 | 1957 | 0.2867 | 0.2529 | 0.2611 | 0.7649 |
| C2_A | 76 | 1822 | 0.3425 | 0.2750 | 0.3095 | 0.7739 |
| C2_A | 77 | 2335 | 0.2852 | 0.2325 | 0.2668 | 0.8012 |
| C2_A | 78 | 2146 | 0.2502 | 0.1985 | 0.2335 | 0.8208 |
| C2_A | 79 | 2188 | 0.3076 | 0.2875 | 0.3213 | 0.8166 |
| C2_B | 56 | 564 | 0.2837 | 0.1613 | 0.1862 | 0.7361 |
| C2_B | 57 | 962 | 0.4179 | 0.3368 | 0.3607 | 0.8022 |
| C2_B | 58 | 1205 | 0.4954 | 0.3602 | 0.3718 | 0.7812 |
| C2_B | 59 | 1383 | 0.4179 | 0.3218 | 0.3348 | 0.7836 |
| C2_B | 60 | 1478 | 0.4790 | 0.4614 | 0.5501 | 0.8517 |
| C2_B | 61 | 1632 | 0.4822 | 0.4841 | 0.5607 | 0.8527 |
| C2_B | 62 | 1500 | 0.3740 | 0.3620 | 0.4707 | 0.7821 |
| C2_B | 63 | 1883 | 0.4742 | 0.4721 | 0.5884 | 0.8033 |
| C2_B | 64 | 1807 | 0.3857 | 0.4073 | 0.4294 | 0.7954 |
| C2_B | 65 | 1567 | 0.3638 | 0.3657 | 0.3810 | 0.8669 |
| C2_B | 66 | 1751 | 0.3444 | 0.3821 | 0.4021 | 0.8151 |
| C2_B | 67 | 1953 | 0.3574 | 0.3692 | 0.3871 | 0.8280 |
| C2_B | 68 | 577 | 0.2964 | 0.2617 | 0.2548 | 0.7786 |
| C2_B | 69 | 1079 | 0.4087 | 0.4050 | 0.3837 | 0.8596 |
| C2_B | 70 | 1605 | 0.3807 | 0.4019 | 0.3900 | 0.8507 |
| C2_B | 71 | 1689 | 0.3730 | 0.3925 | 0.3712 | 0.8347 |
| C2_B | 72 | 1811 | 0.3512 | 0.3556 | 0.3490 | 0.8131 |
| C2_B | 73 | 1880 | 0.3309 | 0.3005 | 0.2963 | 0.7930 |
| C2_B | 74 | 1814 | 0.3060 | 0.3214 | 0.3181 | 0.8292 |
| C2_B | 75 | 1957 | 0.2867 | 0.3005 | 0.2954 | 0.7752 |
| C2_B | 76 | 1822 | 0.3425 | 0.3085 | 0.3337 | 0.8001 |
| C2_B | 77 | 2335 | 0.2852 | 0.2784 | 0.3054 | 0.7916 |
| C2_B | 78 | 2146 | 0.2502 | 0.2269 | 0.2540 | 0.8134 |
| C2_B | 79 | 2188 | 0.3076 | 0.3048 | 0.3277 | 0.8271 |

Cycle-level output rates expose changes due to both observed prevalence and policy threshold; they are descriptive validation behavior, not prevalence estimates for test.

## H12 window Macro F1

| H12 window | Model | Window Macro F1 @0.50 | Window Macro F1 inner | Δ F1 |
| --- | --- | --- | --- | --- |
| H12_E | C0_A | 0.3710 | 0.3710 | 0.0000 |
| H12_E | C0_B | 0.3710 | 0.3710 | 0.0000 |
| H12_E | C0_C | 0.3710 | 0.5390 | 0.1680 |
| H12_E | C1_A | 0.6973 | 0.7259 | 0.0286 |
| H12_E | C1_B | 0.6934 | 0.7252 | 0.0318 |
| H12_E | C2_A | 0.7669 | 0.7693 | 0.0024 |
| H12_E | C2_B | 0.7759 | 0.7827 | 0.0068 |
| H12_L | C0_A | 0.4041 | 0.4041 | 0.0000 |
| H12_L | C0_B | 0.4041 | 0.4041 | 0.0000 |
| H12_L | C0_C | 0.4041 | 0.5557 | 0.1516 |
| H12_L | C1_A | 0.6592 | 0.6583 | -0.0009 |
| H12_L | C1_B | 0.6849 | 0.6719 | -0.0131 |
| H12_L | C2_A | 0.7940 | 0.7933 | -0.0007 |
| H12_L | C2_B | 0.7994 | 0.8001 | 0.0007 |

H12 views overlap primary validation rows; they are longer-horizon sensitivity checks, not independent replications.

## Threshold decisions by fit history

| Outer fit history | Model | Threshold | Inner F1 | Inner n | Rule | Selection record |
| --- | --- | --- | --- | --- | --- | --- |
| 47–55 | C0_A | 0.50 | — | 0 | fixed control | Majority control; no threshold selection |
| 47–55 | C0_B | 0.31 | 0.5032 | 5800 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–55 | C0_C | 0.33 | 0.5350 | 5800 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–55 | C1_A | 0.45 | 0.7361 | 5800 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–55 | C1_B | 0.38 | 0.7242 | 5800 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–55 | C2_A | 0.48 | 0.7867 | 5800 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–55 | C2_B | 0.47 | 0.8072 | 5800 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–59 | C0_A | 0.50 | — | 0 | fixed control | Majority control; no threshold selection |
| 47–59 | C0_B | 0.30 | 0.5611 | 4114 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–59 | C0_C | 0.32 | 0.5620 | 4114 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–59 | C1_A | 0.32 | 0.7640 | 4114 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–59 | C1_B | 0.29 | 0.7612 | 4114 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–59 | C2_A | 0.40 | 0.7587 | 4114 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–59 | C2_B | 0.34 | 0.8096 | 4114 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–63 | C0_A | 0.50 | — | 0 | fixed control | Majority control; no threshold selection |
| 47–63 | C0_B | 0.36 | 0.5542 | 6493 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–63 | C0_C | 0.35 | 0.5757 | 6493 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–63 | C1_A | 0.46 | 0.7776 | 6493 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–63 | C1_B | 0.36 | 0.7708 | 6493 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–63 | C2_A | 0.39 | 0.8255 | 6493 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–63 | C2_B | 0.47 | 0.8477 | 6493 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–67 | C0_A | 0.50 | — | 0 | fixed control | Majority control; no threshold selection |
| 47–67 | C0_B | 0.42 | 0.5304 | 7078 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–67 | C0_C | 0.39 | 0.5359 | 7078 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–67 | C1_A | 0.51 | 0.7296 | 7078 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–67 | C1_B | 0.39 | 0.7291 | 7078 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–67 | C2_A | 0.42 | 0.8070 | 7078 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–67 | C2_B | 0.53 | 0.8272 | 7078 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–71 | C0_A | 0.50 | — | 0 | fixed control | Majority control; no threshold selection |
| 47–71 | C0_B | 0.41 | 0.5302 | 4950 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–71 | C0_C | 0.39 | 0.5641 | 4950 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–71 | C1_A | 0.42 | 0.7321 | 4950 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–71 | C1_B | 0.36 | 0.7461 | 4950 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–71 | C2_A | 0.47 | 0.8221 | 4950 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–71 | C2_B | 0.51 | 0.8436 | 4950 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–75 | C0_A | 0.50 | — | 0 | fixed control | Majority control; no threshold selection |
| 47–75 | C0_B | 0.41 | 0.5488 | 7462 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–75 | C0_C | 0.38 | 0.5457 | 7462 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–75 | C1_A | 0.45 | 0.6704 | 7462 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–75 | C1_B | 0.44 | 0.7338 | 7462 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–75 | C2_A | 0.43 | 0.7975 | 7462 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |
| 47–75 | C2_B | 0.46 | 0.8050 | 7462 | selected | Max inner pooled Macro F1; ties closest to 0.5 then lower |

Majority class has no selection and uses 0.50 as a fixed control. Other recipes use inner histories described in the frozen protocol; `fallback` rows explicitly retain 0.50.

## Selected-threshold subgroups — primary pooled

| Model | Group | n | Macro F1 | F1 class 0 | F1 class 1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1_A | EF0 | 26048 | 0.7071 | 0.7672 | 0.6471 | 5502 | 1805 |
| C1_A | EF1 | 12540 | 0.7126 | 0.7181 | 0.7070 | 2472 | 1131 |
| C1_A | stream_missing | 2578 | 0.7525 | 0.7539 | 0.7512 | 503 | 135 |
| C1_A | stream_present | 36010 | 0.7077 | 0.7530 | 0.6625 | 7471 | 2801 |
| C1_A | all_codes_known_nonmissing | 14452 | 0.7519 | 0.8326 | 0.6712 | 2246 | 960 |
| C1_A | any_unseen_code | 18645 | 0.6446 | 0.6150 | 0.6743 | 5033 | 1547 |
| C1_A | any_missing_code | 9348 | 0.7275 | 0.7850 | 0.6700 | 1554 | 880 |
| C1_B | EF0 | 26048 | 0.7378 | 0.8051 | 0.6705 | 4369 | 2012 |
| C1_B | EF1 | 12540 | 0.7154 | 0.7236 | 0.7072 | 2395 | 1171 |
| C1_B | stream_missing | 2578 | 0.7907 | 0.8019 | 0.7795 | 391 | 147 |
| C1_B | stream_present | 36010 | 0.7287 | 0.7808 | 0.6767 | 6373 | 3036 |
| C1_B | all_codes_known_nonmissing | 14452 | 0.7560 | 0.8384 | 0.6737 | 2117 | 1007 |
| C1_B | any_unseen_code | 18645 | 0.6909 | 0.6888 | 0.6931 | 3911 | 1851 |
| C1_B | any_missing_code | 9348 | 0.7468 | 0.7970 | 0.6965 | 1532 | 742 |
| C2_A | EF0 | 26048 | 0.8119 | 0.8845 | 0.7394 | 1579 | 2590 |
| C2_A | EF1 | 12540 | 0.7858 | 0.8144 | 0.7573 | 1275 | 1363 |
| C2_A | stream_missing | 2578 | 0.3647 | 0.7294 | 0.0000 | 0 | 1098 |
| C2_A | stream_present | 36010 | 0.8275 | 0.8766 | 0.7784 | 2854 | 2855 |
| C2_A | all_codes_known_nonmissing | 14452 | 0.7892 | 0.8853 | 0.6930 | 905 | 1508 |
| C2_A | any_unseen_code | 18645 | 0.8008 | 0.8254 | 0.7762 | 1643 | 2014 |
| C2_A | any_missing_code | 9348 | 0.8267 | 0.8798 | 0.7737 | 627 | 841 |
| C2_B | EF0 | 26048 | 0.8277 | 0.8867 | 0.7688 | 2042 | 1919 |
| C2_B | EF1 | 12540 | 0.7947 | 0.8145 | 0.7749 | 1463 | 1088 |
| C2_B | stream_missing | 2578 | 0.7186 | 0.7255 | 0.7117 | 522 | 203 |
| C2_B | stream_present | 36010 | 0.8257 | 0.8744 | 0.7770 | 2983 | 2804 |
| C2_B | all_codes_known_nonmissing | 14452 | 0.8150 | 0.8923 | 0.7378 | 1080 | 1127 |
| C2_B | any_unseen_code | 18645 | 0.8049 | 0.8190 | 0.7908 | 2101 | 1518 |
| C2_B | any_missing_code | 9348 | 0.8345 | 0.8821 | 0.7868 | 687 | 732 |

Subgroup metrics use fit-derived category vocabulary and impact cutpoints recorded per job. Missing/unseen groups may overlap. Single-class subsets retain fixed labels; AP/balanced accuracy are null where only one actual class occurs.

## Calibration-bin coverage

Ten equal-width probability bins; probability vectors are identical for both threshold policies.

| Model | Fixed probability bin | n | Mean probability | Actual class 1 share |
| --- | --- | --- | --- | --- |
| C1_A | [0.0, 0.1) | 9959 | 0.0297 | 0.0246 |
| C1_A | [0.1, 0.2) | 2397 | 0.1490 | 0.1331 |
| C1_A | [0.2, 0.3) | 2378 | 0.2517 | 0.2431 |
| C1_A | [0.3, 0.4) | 3131 | 0.3520 | 0.3408 |
| C1_A | [0.4, 0.5) | 4635 | 0.4525 | 0.4265 |
| C1_A | [0.5, 0.6) | 4892 | 0.5476 | 0.4978 |
| C1_A | [0.6, 0.7) | 4493 | 0.6521 | 0.5700 |
| C1_A | [0.7, 0.8) | 3585 | 0.7457 | 0.6485 |
| C1_A | [0.8, 0.9) | 2448 | 0.8456 | 0.7917 |
| C1_A | [0.9, 1.0] | 670 | 0.9477 | 0.8000 |
| C1_B | [0.0, 0.1) | 10812 | 0.0240 | 0.0281 |
| C1_B | [0.1, 0.2) | 3502 | 0.1528 | 0.1922 |
| C1_B | [0.2, 0.3) | 3910 | 0.2518 | 0.2905 |
| C1_B | [0.3, 0.4) | 3871 | 0.3505 | 0.3712 |
| C1_B | [0.4, 0.5) | 3896 | 0.4499 | 0.4695 |
| C1_B | [0.5, 0.6) | 3867 | 0.5492 | 0.5839 |
| C1_B | [0.6, 0.7) | 3702 | 0.6499 | 0.6442 |
| C1_B | [0.7, 0.8) | 2546 | 0.7444 | 0.7372 |
| C1_B | [0.8, 0.9) | 1516 | 0.8416 | 0.8292 |
| C1_B | [0.9, 1.0] | 966 | 0.9505 | 0.8540 |
| C2_A | [0.0, 0.1) | 10040 | 0.0438 | 0.0229 |
| C2_A | [0.1, 0.2) | 5738 | 0.1468 | 0.1128 |
| C2_A | [0.2, 0.3) | 4366 | 0.2468 | 0.2350 |
| C2_A | [0.3, 0.4) | 4489 | 0.3578 | 0.3633 |
| C2_A | [0.4, 0.5) | 2498 | 0.4379 | 0.4464 |
| C2_A | [0.5, 0.6) | 1587 | 0.5485 | 0.5822 |
| C2_A | [0.6, 0.7) | 1622 | 0.6512 | 0.7324 |
| C2_A | [0.7, 0.8) | 2119 | 0.7526 | 0.8259 |
| C2_A | [0.8, 0.9) | 2964 | 0.8517 | 0.8809 |
| C2_A | [0.9, 1.0] | 3165 | 0.9462 | 0.9033 |
| C2_B | [0.0, 0.1) | 12469 | 0.0301 | 0.0240 |
| C2_B | [0.1, 0.2) | 4354 | 0.1476 | 0.1332 |
| C2_B | [0.2, 0.3) | 3288 | 0.2476 | 0.2175 |
| C2_B | [0.3, 0.4) | 2634 | 0.3488 | 0.3106 |
| C2_B | [0.4, 0.5) | 2304 | 0.4505 | 0.4275 |
| C2_B | [0.5, 0.6) | 2263 | 0.5483 | 0.5621 |
| C2_B | [0.6, 0.7) | 2206 | 0.6490 | 0.6754 |
| C2_B | [0.7, 0.8) | 2210 | 0.7509 | 0.7855 |
| C2_B | [0.8, 0.9) | 3248 | 0.8540 | 0.8716 |
| C2_B | [0.9, 1.0] | 3612 | 0.9443 | 0.9012 |

No model is selected for final use in this artifact.
