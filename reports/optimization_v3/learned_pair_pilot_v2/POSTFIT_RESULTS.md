# Learned-pair FM V2 — hasil diagnostic

**Status:** `COMPLETED_DIAGNOSTIC_ONLY`  
**MECHANISM:** `MECHANISM_FAIL`  
**COMPETITION:** `COMPETITION_FAIL`

Kedua verdict dilaporkan terpisah. Hasil ini diagnostic-only dan tidak mengubah champion, tidak mengizinkan continuation/refit/Test, dan tidak mengizinkan submission.

## Rancangan dan batas inferensi

FM rank-16 memakai binary logistic SUM + penalti L2 pada main weights, factors, dan intercept. Kontrol no-pair memakai baris, matriks, objective, inisialisasi w/b, dan solver yang sama. Present stream memakai expert langsung; missing/blank memakai fallback C1_B HGB tersimpan secara bit-exact. A02 observed adalah comparator terpisah dan tidak menjadi input training maupun prediksi.
V2 hanya mengubah `gtol` dari V1 menjadi `n_fit × 1e-4` pada objective SUM. Fit tetap harus memenuhi SciPy success, stationarity mean-gradient <=1e-4, finite state, dan objective nonincrease. V1 tetap `FAIL_OPTIMIZER_INCONCLUSIVE`; perubahan ini tidak memperbaiki atau menulis ulang verdict V1.
E67 H12 adalah view agregat utama. E67 W4/W5/W6 dan E75 W6 adalah pemeriksaan temporal/history sensitivity; W6 E75 tumpang tindih dengan W6 E67, jadi bukan replikasi independen.

## Metrik dan denominator

Macro F1/precision/recall memakai threshold 0,50. Denominator, jumlah kelas, confusion matrix, dan probability metrics dihitung untuk setiap arm di setiap view.

| Origin | View | Arm | n | Pos/Neg | Macro F1 | Pos recall | Precision | ROC AUC | AP | Log loss | Brier | Confusion [[TN,FP],[FN,TP]] |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E67 | W4 | bag | 4950 | 1853/3097 | 0.842554 | 0.805720 | 0.800966 | 0.909754 | 0.815311 | 0.372761 | 0.114696 | [[2726, 371], [360, 1493]] |
| E67 | W4 | nopair | 4950 | 1853/3097 | 0.842056 | 0.804101 | 0.801075 | 0.909743 | 0.815244 | 0.372907 | 0.114739 | [[2727, 370], [363, 1490]] |
| E67 | W4 | a02_observed | 4950 | 1853/3097 | 0.844405 | 0.803562 | 0.806609 | 0.915358 | 0.830957 | 0.421903 | 0.111075 | [[2740, 357], [364, 1489]] |
| E67 | W4 | fm | 4950 | 1853/3097 | 0.814982 | 0.726929 | 0.800832 | 0.896051 | 0.816422 | 0.413812 | 0.126653 | [[2762, 335], [506, 1347]] |
| E67 | W5 | bag | 7462 | 2374/5088 | 0.799093 | 0.726201 | 0.725895 | 0.886525 | 0.769208 | 0.396633 | 0.126506 | [[4437, 651], [650, 1724]] |
| E67 | W5 | nopair | 7462 | 2374/5088 | 0.799270 | 0.726622 | 0.726010 | 0.886417 | 0.769461 | 0.396751 | 0.126542 | [[4437, 651], [649, 1725]] |
| E67 | W5 | a02_observed | 7462 | 2374/5088 | 0.805176 | 0.722409 | 0.743068 | 0.893292 | 0.781576 | 0.415382 | 0.122400 | [[4495, 593], [659, 1715]] |
| E67 | W5 | fm | 7462 | 2374/5088 | 0.758471 | 0.573715 | 0.755408 | 0.867457 | 0.738294 | 0.453407 | 0.141730 | [[4647, 441], [1012, 1362]] |
| E67 | W6 | bag | 8491 | 2500/5991 | 0.784403 | 0.718400 | 0.681077 | 0.882620 | 0.758180 | 0.396567 | 0.127407 | [[5150, 841], [704, 1796]] |
| E67 | W6 | nopair | 8491 | 2500/5991 | 0.784076 | 0.717600 | 0.680835 | 0.882630 | 0.758517 | 0.396416 | 0.127346 | [[5150, 841], [706, 1794]] |
| E67 | W6 | a02_observed | 8491 | 2500/5991 | 0.792999 | 0.721200 | 0.698837 | 0.892884 | 0.770734 | 0.420955 | 0.121561 | [[5214, 777], [697, 1803]] |
| E67 | W6 | fm | 8491 | 2500/5991 | 0.724294 | 0.467600 | 0.762060 | 0.856495 | 0.708110 | 0.464627 | 0.144775 | [[5626, 365], [1331, 1169]] |
| E67 | H12_E67 | bag | 20903 | 6727/14176 | 0.805106 | 0.745206 | 0.729058 | 0.892228 | 0.780164 | 0.390953 | 0.124075 | [[12313, 1863], [1714, 5013]] |
| E67 | H12_E67 | nopair | 20903 | 6727/14176 | 0.804906 | 0.744611 | 0.729006 | 0.892198 | 0.780368 | 0.390968 | 0.124073 | [[12314, 1862], [1718, 5009]] |
| E67 | H12_E67 | a02_observed | 20903 | 6727/14176 | 0.811159 | 0.744314 | 0.743540 | 0.899755 | 0.792814 | 0.419190 | 0.119377 | [[12449, 1727], [1720, 5007]] |
| E67 | H12_E67 | fm | 20903 | 6727/14176 | 0.763788 | 0.576483 | 0.772664 | 0.871872 | 0.752178 | 0.448589 | 0.139397 | [[13035, 1141], [2849, 3878]] |
| E75 | W6 | bag | 8491 | 2500/5991 | 0.810384 | 0.690000 | 0.767690 | 0.904475 | 0.780881 | 0.355016 | 0.111401 | [[5469, 522], [775, 1725]] |
| E75 | W6 | nopair | 8491 | 2500/5991 | 0.809872 | 0.688800 | 0.767380 | 0.904778 | 0.781786 | 0.354391 | 0.111262 | [[5469, 522], [778, 1722]] |
| E75 | W6 | a02_observed | 8491 | 2500/5991 | 0.819604 | 0.708000 | 0.776656 | 0.911107 | 0.792959 | 0.379397 | 0.107386 | [[5482, 509], [730, 1770]] |
| E75 | W6 | fm | 8491 | 2500/5991 | 0.785334 | 0.630400 | 0.754067 | 0.886085 | 0.742772 | 0.412419 | 0.124001 | [[5477, 514], [924, 1576]] |

## Gate checks

| Track | Origin | View | Comparator | Metric | Delta | Threshold | Result |
|---|---:|---|---|---|---:|---:|---|
| mechanism | 67 | H12_E67 | bag | macro_f1 | -0.04131872922692459 | 0.0005 | FAIL |
| mechanism | 67 | H12_E67 | nopair | macro_f1 | -0.041118138392937986 | 0.0002 | FAIL |
| competition | 67 | H12_E67 | a02_observed | macro_f1 | -0.04737139440165827 | 0.0005 | FAIL |
| mechanism | 67 | W4 | bag | macro_f1 | -0.027572051657896224 | -0.005 | FAIL |
| mechanism | 67 | W4 | nopair | macro_f1 | -0.02707338378706925 | -0.005 | FAIL |
| competition | 67 | W4 | a02_observed | macro_f1 | -0.0294231814799496 | -0.005 | FAIL |
| mechanism | 67 | W5 | bag | macro_f1 | -0.040621508020481256 | -0.005 | FAIL |
| mechanism | 67 | W5 | nopair | macro_f1 | -0.04079846937080167 | -0.005 | FAIL |
| competition | 67 | W5 | a02_observed | macro_f1 | -0.04670449714612235 | -0.005 | FAIL |
| mechanism | 67 | W6 | bag | macro_f1 | -0.06010893780227766 | -0.005 | FAIL |
| mechanism | 67 | W6 | nopair | macro_f1 | -0.05978222578442449 | -0.005 | FAIL |
| competition | 67 | W6 | a02_observed | macro_f1 | -0.06870482194797178 | -0.005 | FAIL |
| both | 67 | W4 |  | support | {"n": 4608, "negative": 2895, "positive": 1713} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W4 | bag | macro_f1 | -0.029833055406669695 | -0.01 | FAIL |
| mechanism | 67 | W4 | bag | positive_recall | -0.08523058960887331 | -0.02 | FAIL |
| mechanism | 67 | W4 | nopair | macro_f1 | -0.02929496810768395 | -0.01 | FAIL |
| mechanism | 67 | W4 | nopair | positive_recall | -0.0834792761237595 | -0.02 | FAIL |
| competition | 67 | W4 | a02_observed | macro_f1 | -0.03426568453583967 | -0.01 | FAIL |
| competition | 67 | W4 | a02_observed | positive_recall | -0.08756567425569184 | -0.02 | FAIL |
| both | 67 | W4 |  | support | {"n": 342, "negative": 202, "positive": 140} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W4 | bag | macro_f1 | 0.0 | -0.01 | PASS |
| mechanism | 67 | W4 | bag | positive_recall | 0.0 | -0.02 | PASS |
| mechanism | 67 | W4 | nopair | macro_f1 | 0.0 | -0.01 | PASS |
| mechanism | 67 | W4 | nopair | positive_recall | 0.0 | -0.02 | PASS |
| competition | 67 | W4 | a02_observed | macro_f1 | 0.031247312806773397 | -0.01 | PASS |
| competition | 67 | W4 | a02_observed | positive_recall | 0.05714285714285716 | -0.02 | PASS |
| both | 67 | W4 |  | support | {"n": 1765, "negative": 1085, "positive": 680} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W4 | bag | macro_f1 | -0.025732480765241372 | -0.01 | FAIL |
| mechanism | 67 | W4 | bag | positive_recall | -0.0794117647058824 | -0.02 | FAIL |
| mechanism | 67 | W4 | nopair | macro_f1 | -0.02506995025920311 | -0.01 | FAIL |
| mechanism | 67 | W4 | nopair | positive_recall | -0.07794117647058829 | -0.02 | FAIL |
| competition | 67 | W4 | a02_observed | macro_f1 | -0.030358624358871156 | -0.01 | FAIL |
| competition | 67 | W4 | a02_observed | positive_recall | -0.07500000000000007 | -0.02 | FAIL |
| both | 67 | W4 |  | support | {"n": 3185, "negative": 2012, "positive": 1173} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W4 | bag | macro_f1 | -0.02867346377203661 | -0.01 | FAIL |
| mechanism | 67 | W4 | bag | positive_recall | -0.07843137254901955 | -0.02 | FAIL |
| mechanism | 67 | W4 | nopair | macro_f1 | -0.028267811931036557 | -0.01 | FAIL |
| mechanism | 67 | W4 | nopair | positive_recall | -0.07672634271099743 | -0.02 | FAIL |
| competition | 67 | W4 | a02_observed | macro_f1 | -0.028936257691804967 | -0.01 | FAIL |
| competition | 67 | W4 | a02_observed | positive_recall | -0.07757885763000849 | -0.02 | FAIL |
| both | 67 | W4 |  | support | {"n": 0, "negative": 0, "positive": 0} | {'n': 200, 'each_class': 30} | UNKNOWN_UNSUPPORTED |
| both | 67 | W4 |  | support | {"n": 3418, "negative": 2211, "positive": 1207} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W4 | bag | macro_f1 | -0.03229348600334192 | -0.01 | FAIL |
| mechanism | 67 | W4 | bag | positive_recall | -0.07953603976801993 | -0.02 | FAIL |
| mechanism | 67 | W4 | nopair | macro_f1 | -0.03194405756375418 | -0.01 | FAIL |
| mechanism | 67 | W4 | nopair | positive_recall | -0.07870753935376973 | -0.02 | FAIL |
| competition | 67 | W4 | a02_observed | macro_f1 | -0.034197948661899846 | -0.01 | FAIL |
| competition | 67 | W4 | a02_observed | positive_recall | -0.07705053852526933 | -0.02 | FAIL |
| both | 67 | W4 |  | support | {"n": 1532, "negative": 886, "positive": 646} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W4 | bag | macro_f1 | -0.017432926112960545 | -0.01 | FAIL |
| mechanism | 67 | W4 | bag | positive_recall | -0.07739938080495357 | -0.02 | FAIL |
| mechanism | 67 | W4 | nopair | macro_f1 | -0.016666416809064777 | -0.01 | FAIL |
| mechanism | 67 | W4 | nopair | positive_recall | -0.07430340557275539 | -0.02 | FAIL |
| competition | 67 | W4 | a02_observed | macro_f1 | -0.019254126990431275 | -0.01 | FAIL |
| competition | 67 | W4 | a02_observed | positive_recall | -0.07585139318885448 | -0.02 | FAIL |
| both | 67 | W5 |  | support | {"n": 6955, "negative": 4757, "positive": 2198} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W5 | bag | macro_f1 | -0.04491173597635256 | -0.01 | FAIL |
| mechanism | 67 | W5 | bag | positive_recall | -0.16469517743403095 | -0.02 | FAIL |
| mechanism | 67 | W5 | nopair | macro_f1 | -0.045104184863234975 | -0.01 | FAIL |
| mechanism | 67 | W5 | nopair | positive_recall | -0.16515013648771615 | -0.02 | FAIL |
| competition | 67 | W5 | a02_observed | macro_f1 | -0.05298172016530911 | -0.01 | FAIL |
| competition | 67 | W5 | a02_observed | positive_recall | -0.16515013648771615 | -0.02 | FAIL |
| both | 67 | W5 |  | support | {"n": 507, "negative": 331, "positive": 176} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W5 | bag | macro_f1 | 0.0 | -0.01 | PASS |
| mechanism | 67 | W5 | bag | positive_recall | 0.0 | -0.02 | PASS |
| mechanism | 67 | W5 | nopair | macro_f1 | 0.0 | -0.01 | PASS |
| mechanism | 67 | W5 | nopair | positive_recall | 0.0 | -0.02 | PASS |
| competition | 67 | W5 | a02_observed | macro_f1 | 0.017097784981970454 | -0.01 | PASS |
| competition | 67 | W5 | a02_observed | positive_recall | 0.05681818181818177 | -0.02 | PASS |
| both | 67 | W5 |  | support | {"n": 2018, "negative": 1353, "positive": 665} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W5 | bag | macro_f1 | -0.03720610364202548 | -0.01 | FAIL |
| mechanism | 67 | W5 | bag | positive_recall | -0.14586466165413542 | -0.02 | FAIL |
| mechanism | 67 | W5 | nopair | macro_f1 | -0.03837389426376403 | -0.01 | FAIL |
| mechanism | 67 | W5 | nopair | positive_recall | -0.14736842105263165 | -0.02 | FAIL |
| competition | 67 | W5 | a02_observed | macro_f1 | -0.044601644959806896 | -0.01 | FAIL |
| competition | 67 | W5 | a02_observed | positive_recall | -0.12781954887218044 | -0.02 | FAIL |
| both | 67 | W5 |  | support | {"n": 5444, "negative": 3735, "positive": 1709} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W5 | bag | macro_f1 | -0.04221261052185943 | -0.01 | FAIL |
| mechanism | 67 | W5 | bag | positive_recall | -0.155061439438268 | -0.02 | FAIL |
| mechanism | 67 | W5 | nopair | macro_f1 | -0.042030470069607806 | -0.01 | FAIL |
| mechanism | 67 | W5 | nopair | positive_recall | -0.155061439438268 | -0.02 | FAIL |
| competition | 67 | W5 | a02_observed | macro_f1 | -0.04749421110091201 | -0.01 | FAIL |
| competition | 67 | W5 | a02_observed | positive_recall | -0.15681685196021067 | -0.02 | FAIL |
| both | 67 | W5 |  | support | {"n": 0, "negative": 0, "positive": 0} | {'n': 200, 'each_class': 30} | UNKNOWN_UNSUPPORTED |
| both | 67 | W5 |  | support | {"n": 5188, "negative": 3694, "positive": 1494} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W5 | bag | macro_f1 | -0.05265771360085936 | -0.01 | FAIL |
| mechanism | 67 | W5 | bag | positive_recall | -0.18273092369477906 | -0.02 | FAIL |
| mechanism | 67 | W5 | nopair | macro_f1 | -0.05292947969211259 | -0.01 | FAIL |
| mechanism | 67 | W5 | nopair | positive_recall | -0.18340026773761708 | -0.02 | FAIL |
| competition | 67 | W5 | a02_observed | macro_f1 | -0.05752536788266549 | -0.01 | FAIL |
| competition | 67 | W5 | a02_observed | positive_recall | -0.17402945113788482 | -0.02 | FAIL |
| both | 67 | W5 |  | support | {"n": 2274, "negative": 1394, "positive": 880} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W5 | bag | macro_f1 | -0.018356930398988003 | -0.01 | FAIL |
| mechanism | 67 | W5 | bag | positive_recall | -0.10113636363636358 | -0.02 | FAIL |
| mechanism | 67 | W5 | nopair | macro_f1 | -0.018356930398988003 | -0.01 | FAIL |
| mechanism | 67 | W5 | nopair | positive_recall | -0.10113636363636358 | -0.02 | FAIL |
| competition | 67 | W5 | a02_observed | macro_f1 | -0.02701056036491789 | -0.01 | FAIL |
| competition | 67 | W5 | a02_observed | positive_recall | -0.10568181818181821 | -0.02 | FAIL |
| both | 67 | W6 |  | support | {"n": 7965, "negative": 5636, "positive": 2329} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W6 | bag | macro_f1 | -0.0665775772270163 | -0.01 | FAIL |
| mechanism | 67 | W6 | bag | positive_recall | -0.2692142550450838 | -0.02 | FAIL |
| mechanism | 67 | W6 | nopair | macro_f1 | -0.06622543691432936 | -0.01 | FAIL |
| mechanism | 67 | W6 | nopair | positive_recall | -0.2683555173894375 | -0.02 | FAIL |
| competition | 67 | W6 | a02_observed | macro_f1 | -0.07664058313677069 | -0.01 | FAIL |
| competition | 67 | W6 | a02_observed | positive_recall | -0.27179046801202233 | -0.02 | FAIL |
| both | 67 | W6 |  | support | {"n": 526, "negative": 355, "positive": 171} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W6 | bag | macro_f1 | 0.0 | -0.01 | PASS |
| mechanism | 67 | W6 | bag | positive_recall | 0.0 | -0.02 | PASS |
| mechanism | 67 | W6 | nopair | macro_f1 | 0.0 | -0.01 | PASS |
| mechanism | 67 | W6 | nopair | positive_recall | 0.0 | -0.02 | PASS |
| competition | 67 | W6 | a02_observed | macro_f1 | 0.011870186495782309 | -0.01 | PASS |
| competition | 67 | W6 | a02_observed | positive_recall | -0.005847953216374213 | -0.02 | PASS |
| both | 67 | W6 |  | support | {"n": 2080, "negative": 1387, "positive": 693} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W6 | bag | macro_f1 | -0.06254160135058517 | -0.01 | FAIL |
| mechanism | 67 | W6 | bag | positive_recall | -0.20490620490620493 | -0.02 | FAIL |
| mechanism | 67 | W6 | nopair | macro_f1 | -0.062106610831588305 | -0.01 | FAIL |
| mechanism | 67 | W6 | nopair | positive_recall | -0.2020202020202021 | -0.02 | FAIL |
| competition | 67 | W6 | a02_observed | macro_f1 | -0.056655211362080515 | -0.01 | FAIL |
| competition | 67 | W6 | a02_observed | positive_recall | -0.17604617604617612 | -0.02 | FAIL |
| both | 67 | W6 |  | support | {"n": 6411, "negative": 4604, "positive": 1807} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W6 | bag | macro_f1 | -0.05896454639978943 | -0.01 | FAIL |
| mechanism | 67 | W6 | bag | positive_recall | -0.2684006640841173 | -0.02 | FAIL |
| mechanism | 67 | W6 | nopair | macro_f1 | -0.058654016583737256 | -0.01 | FAIL |
| mechanism | 67 | W6 | nopair | positive_recall | -0.2684006640841173 | -0.02 | FAIL |
| competition | 67 | W6 | a02_observed | macro_f1 | -0.07212367231059191 | -0.01 | FAIL |
| competition | 67 | W6 | a02_observed | positive_recall | -0.2833425567238517 | -0.02 | FAIL |
| both | 67 | W6 |  | support | {"n": 0, "negative": 0, "positive": 0} | {'n': 200, 'each_class': 30} | UNKNOWN_UNSUPPORTED |
| both | 67 | W6 |  | support | {"n": 5738, "negative": 4316, "positive": 1422} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W6 | bag | macro_f1 | -0.06260238519920946 | -0.01 | FAIL |
| mechanism | 67 | W6 | bag | positive_recall | -0.2545710267229255 | -0.02 | FAIL |
| mechanism | 67 | W6 | nopair | macro_f1 | -0.06212808119041968 | -0.01 | FAIL |
| mechanism | 67 | W6 | nopair | positive_recall | -0.2524613220815753 | -0.02 | FAIL |
| competition | 67 | W6 | a02_observed | macro_f1 | -0.07046156699573691 | -0.01 | FAIL |
| competition | 67 | W6 | a02_observed | positive_recall | -0.2517580872011252 | -0.02 | FAIL |
| both | 67 | W6 |  | support | {"n": 2753, "negative": 1675, "positive": 1078} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | W6 | bag | macro_f1 | -0.05577822684393374 | -0.01 | FAIL |
| mechanism | 67 | W6 | bag | positive_recall | -0.24582560296846012 | -0.02 | FAIL |
| mechanism | 67 | W6 | nopair | macro_f1 | -0.055500930070132126 | -0.01 | FAIL |
| mechanism | 67 | W6 | nopair | positive_recall | -0.24675324675324672 | -0.02 | FAIL |
| competition | 67 | W6 | a02_observed | macro_f1 | -0.06544818040262879 | -0.01 | FAIL |
| competition | 67 | W6 | a02_observed | positive_recall | -0.2560296846011132 | -0.02 | FAIL |
| both | 67 | H12_E67 |  | support | {"n": 19528, "negative": 13288, "positive": 6240} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | H12_E67 | bag | macro_f1 | -0.045398920144042676 | -0.01 | FAIL |
| mechanism | 67 | H12_E67 | bag | positive_recall | -0.18189102564102566 | -0.02 | FAIL |
| mechanism | 67 | H12_E67 | nopair | macro_f1 | -0.04518182657417835 | -0.01 | FAIL |
| mechanism | 67 | H12_E67 | nopair | positive_recall | -0.18125000000000002 | -0.02 | FAIL |
| competition | 67 | H12_E67 | a02_observed | macro_f1 | -0.053419016539718145 | -0.01 | FAIL |
| competition | 67 | H12_E67 | a02_observed | positive_recall | -0.18365384615384617 | -0.02 | FAIL |
| both | 67 | H12_E67 |  | support | {"n": 1375, "negative": 888, "positive": 487} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | H12_E67 | bag | macro_f1 | 0.0 | -0.01 | PASS |
| mechanism | 67 | H12_E67 | bag | positive_recall | 0.0 | -0.02 | PASS |
| mechanism | 67 | H12_E67 | nopair | macro_f1 | 0.0 | -0.01 | PASS |
| mechanism | 67 | H12_E67 | nopair | positive_recall | 0.0 | -0.02 | PASS |
| competition | 67 | H12_E67 | a02_observed | macro_f1 | 0.019005096896050788 | -0.01 | PASS |
| competition | 67 | H12_E67 | a02_observed | positive_recall | 0.03490759753593431 | -0.02 | PASS |
| both | 67 | H12_E67 |  | support | {"n": 5863, "negative": 3825, "positive": 2038} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | H12_E67 | bag | macro_f1 | -0.03887077041147824 | -0.01 | FAIL |
| mechanism | 67 | H12_E67 | bag | positive_recall | -0.14376840039254168 | -0.02 | FAIL |
| mechanism | 67 | H12_E67 | nopair | macro_f1 | -0.038914388493954566 | -0.01 | FAIL |
| mechanism | 67 | H12_E67 | nopair | positive_recall | -0.14278704612365067 | -0.02 | FAIL |
| competition | 67 | H12_E67 | a02_observed | macro_f1 | -0.04115466224275899 | -0.01 | FAIL |
| competition | 67 | H12_E67 | a02_observed | positive_recall | -0.126594700686948 | -0.02 | FAIL |
| both | 67 | H12_E67 |  | support | {"n": 15040, "negative": 10351, "positive": 4689} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | H12_E67 | bag | macro_f1 | -0.04218207319818146 | -0.01 | FAIL |
| mechanism | 67 | H12_E67 | bag | positive_recall | -0.17956920452121983 | -0.02 | FAIL |
| mechanism | 67 | H12_E67 | nopair | macro_f1 | -0.041878955627925096 | -0.01 | FAIL |
| mechanism | 67 | H12_E67 | nopair | positive_recall | -0.17914267434420983 | -0.02 | FAIL |
| competition | 67 | H12_E67 | a02_observed | macro_f1 | -0.04956417428070814 | -0.01 | FAIL |
| competition | 67 | H12_E67 | a02_observed | positive_recall | -0.1857538920878652 | -0.02 | FAIL |
| both | 67 | H12_E67 |  | support | {"n": 0, "negative": 0, "positive": 0} | {'n': 200, 'each_class': 30} | UNKNOWN_UNSUPPORTED |
| both | 67 | H12_E67 |  | support | {"n": 14344, "negative": 10221, "positive": 4123} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | H12_E67 | bag | macro_f1 | -0.046097748052518206 | -0.01 | FAIL |
| mechanism | 67 | H12_E67 | bag | positive_recall | -0.1772980839194761 | -0.02 | FAIL |
| mechanism | 67 | H12_E67 | nopair | macro_f1 | -0.045947594291139815 | -0.01 | FAIL |
| mechanism | 67 | H12_E67 | nopair | positive_recall | -0.1765704584040747 | -0.02 | FAIL |
| competition | 67 | H12_E67 | a02_observed | macro_f1 | -0.05143701696348735 | -0.01 | FAIL |
| competition | 67 | H12_E67 | a02_observed | positive_recall | -0.17244724715013338 | -0.02 | FAIL |
| both | 67 | H12_E67 |  | support | {"n": 6559, "negative": 3955, "positive": 2604} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 67 | H12_E67 | bag | macro_f1 | -0.03194784757360347 | -0.01 | FAIL |
| mechanism | 67 | H12_E67 | bag | positive_recall | -0.15514592933947768 | -0.02 | FAIL |
| mechanism | 67 | H12_E67 | nopair | macro_f1 | -0.031631976788304206 | -0.01 | FAIL |
| mechanism | 67 | H12_E67 | nopair | positive_recall | -0.15476190476190477 | -0.02 | FAIL |
| competition | 67 | H12_E67 | a02_observed | macro_f1 | -0.039410761059463395 | -0.01 | FAIL |
| competition | 67 | H12_E67 | a02_observed | positive_recall | -0.16052227342549918 | -0.02 | FAIL |
| mechanism | 75 | W6 | bag | macro_f1 | -0.025049908838448043 | 0.0 | FAIL |
| mechanism | 75 | W6 | nopair | macro_f1 | -0.024537753376024263 | 0.0 | FAIL |
| competition | 75 | W6 | a02_observed | macro_f1 | -0.03427034958025277 | 0.0005 | FAIL |
| both | 75 | W6 |  | support | {"n": 7965, "negative": 5636, "positive": 2329} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 75 | W6 | bag | macro_f1 | -0.026736369722495823 | -0.01 | FAIL |
| mechanism | 75 | W6 | bag | positive_recall | -0.06397595534564193 | -0.02 | FAIL |
| mechanism | 75 | W6 | nopair | macro_f1 | -0.02619052565922786 | -0.01 | FAIL |
| mechanism | 75 | W6 | nopair | positive_recall | -0.0626878488621726 | -0.02 | FAIL |
| competition | 75 | W6 | a02_observed | macro_f1 | -0.03334657304335953 | -0.01 | FAIL |
| competition | 75 | W6 | a02_observed | positive_recall | -0.07685702018033491 | -0.02 | FAIL |
| both | 75 | W6 |  | support | {"n": 526, "negative": 355, "positive": 171} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 75 | W6 | bag | macro_f1 | 0.0 | -0.01 | PASS |
| mechanism | 75 | W6 | bag | positive_recall | 0.0 | -0.02 | PASS |
| mechanism | 75 | W6 | nopair | macro_f1 | 0.0 | -0.01 | PASS |
| mechanism | 75 | W6 | nopair | positive_recall | 0.0 | -0.02 | PASS |
| competition | 75 | W6 | a02_observed | macro_f1 | -0.04748453557769272 | -0.01 | FAIL |
| competition | 75 | W6 | a02_observed | positive_recall | -0.08771929824561409 | -0.02 | FAIL |
| both | 75 | W6 |  | support | {"n": 1993, "negative": 1328, "positive": 665} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 75 | W6 | bag | macro_f1 | -0.0052574154554844155 | -0.01 | PASS |
| mechanism | 75 | W6 | bag | positive_recall | -0.015037593984962405 | -0.02 | PASS |
| mechanism | 75 | W6 | nopair | macro_f1 | -0.005742152860379934 | -0.01 | PASS |
| mechanism | 75 | W6 | nopair | positive_recall | -0.015037593984962405 | -0.02 | PASS |
| competition | 75 | W6 | a02_observed | macro_f1 | -0.005280256774238712 | -0.01 | PASS |
| competition | 75 | W6 | a02_observed | positive_recall | -0.012030075187969946 | -0.02 | PASS |
| both | 75 | W6 |  | support | {"n": 6498, "negative": 4663, "positive": 1835} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 75 | W6 | bag | macro_f1 | -0.03139219684903394 | -0.01 | FAIL |
| mechanism | 75 | W6 | bag | positive_recall | -0.07574931880108993 | -0.02 | FAIL |
| mechanism | 75 | W6 | nopair | macro_f1 | -0.030556980777829534 | -0.01 | FAIL |
| mechanism | 75 | W6 | nopair | positive_recall | -0.07411444141689372 | -0.02 | FAIL |
| competition | 75 | W6 | a02_observed | macro_f1 | -0.043464094436301615 | -0.01 | FAIL |
| competition | 75 | W6 | a02_observed | positive_recall | -0.10136239782016343 | -0.02 | FAIL |
| both | 75 | W6 |  | support | {"n": 0, "negative": 0, "positive": 0} | {'n': 200, 'each_class': 30} | UNKNOWN_UNSUPPORTED |
| both | 75 | W6 |  | support | {"n": 5738, "negative": 4316, "positive": 1422} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 75 | W6 | bag | macro_f1 | -0.023601124364994286 | -0.01 | FAIL |
| mechanism | 75 | W6 | bag | positive_recall | -0.053445850914205395 | -0.02 | FAIL |
| mechanism | 75 | W6 | nopair | macro_f1 | -0.023003640075504928 | -0.01 | FAIL |
| mechanism | 75 | W6 | nopair | positive_recall | -0.0520393811533052 | -0.02 | FAIL |
| competition | 75 | W6 | a02_observed | macro_f1 | -0.034147204482198656 | -0.01 | FAIL |
| competition | 75 | W6 | a02_observed | positive_recall | -0.07665260196905765 | -0.02 | FAIL |
| both | 75 | W6 |  | support | {"n": 2753, "negative": 1675, "positive": 1078} | {'n': 200, 'each_class': 30} | PASS |
| mechanism | 75 | W6 | bag | macro_f1 | -0.027665758046644284 | -0.01 | FAIL |
| mechanism | 75 | W6 | bag | positive_recall | -0.06771799628942488 | -0.02 | FAIL |
| mechanism | 75 | W6 | nopair | macro_f1 | -0.027247442127660282 | -0.01 | FAIL |
| mechanism | 75 | W6 | nopair | positive_recall | -0.06679035250463827 | -0.02 | FAIL |
| competition | 75 | W6 | a02_observed | macro_f1 | -0.03581551866240218 | -0.01 | FAIL |
| competition | 75 | W6 | a02_observed | positive_recall | -0.07884972170686455 | -0.02 | FAIL |

### Per-cycle direction

| Origin | Cycle | n | FM Δ vs C2_B | FM Δ vs no-pair | FM Δ vs A02 observed |
|---|---:|---:|---:|---:|---:|
| E67 | 68 | 577 | -0.018856 | -0.018856 | -0.018856 |
| E67 | 69 | 1079 | -0.034044 | -0.033030 | -0.036070 |
| E67 | 70 | 1605 | -0.016544 | -0.017156 | -0.020222 |
| E67 | 71 | 1689 | -0.036766 | -0.035412 | -0.037334 |
| E67 | 72 | 1811 | -0.025527 | -0.025527 | -0.030012 |
| E67 | 73 | 1880 | -0.030950 | -0.031835 | -0.041939 |
| E67 | 74 | 1814 | -0.060507 | -0.060507 | -0.062589 |
| E67 | 75 | 1957 | -0.051975 | -0.051736 | -0.058457 |
| E67 | 76 | 1822 | -0.059452 | -0.058397 | -0.058196 |
| E67 | 77 | 2335 | -0.053927 | -0.053554 | -0.058756 |
| E67 | 78 | 2146 | -0.093385 | -0.092645 | -0.100528 |
| E67 | 79 | 2188 | -0.040507 | -0.041094 | -0.062622 |

E67 cycle signs (positive / negative / tie): C2_B 0/12/0; no-pair 0/12/0; A02 0/12/0.
| E75 | 76 | 1822 | -0.024082 | -0.022282 | -0.027871 |
| E75 | 77 | 2335 | -0.012472 | -0.012472 | -0.026538 |
| E75 | 78 | 2146 | -0.033984 | -0.033465 | -0.055041 |
| E75 | 79 | 2188 | -0.031654 | -0.031654 | -0.031019 |

E75 cycle signs (positive / negative / tie): C2_B 0/4/0; no-pair 0/4/0; A02 0/4/0.

### Subgroup support

Full subgroup metrics and all comparator deltas remain in `RESULTS.json`. A group is supported only for n>=200 and each class>=30; unsupported groups are `UNKNOWN_UNSUPPORTED`, with no numeric harm estimate.

#### E67
- W4 / availability=present: n=4608 (positive=1713, negative=2895), FM Macro F1=0.817063929217092, recall=0.7221249270286048; supported.
- W4 / availability=missing_or_blank: n=342 (positive=140, negative=202), FM Macro F1=0.7848755940732457, recall=0.7857142857142857; supported.
- W4 / text_load=Q4: n=1765 (positive=680, negative=1085), FM Macro F1=0.7941853390587937, recall=0.6676470588235294; supported.
- W4 / text_load=below_Q4: n=3185 (positive=1173, negative=2012), FM Macro F1=0.8263452351018356, recall=0.7612958226768969; supported.
- W4 / text_load=missing: n=0 (positive=0, negative=0), FM Macro F1=None, recall=None; UNKNOWN_UNSUPPORTED.
- W4 / entry_friction=0: n=3418 (positive=1207, negative=2211), FM Macro F1=0.8285306790694593, recall=0.728251864125932; supported.
- W4 / entry_friction=1: n=1532 (positive=646, negative=886), FM Macro F1=0.7841158897469138, recall=0.7244582043343654; supported.
- W5 / availability=present: n=6955 (positive=2198, negative=4757), FM Macro F1=0.7566303060814896, recall=0.5527752502274795; supported.
- W5 / availability=missing_or_blank: n=507 (positive=176, negative=331), FM Macro F1=0.7625100952996284, recall=0.8352272727272727; supported.
- W5 / text_load=Q4: n=2018 (positive=665, negative=1353), FM Macro F1=0.7239807142124399, recall=0.47368421052631576; supported.
- W5 / text_load=below_Q4: n=5444 (positive=1709, negative=3735), FM Macro F1=0.7704395116734286, recall=0.6126389701579871; supported.
- W5 / text_load=missing: n=0 (positive=0, negative=0), FM Macro F1=None, recall=None; UNKNOWN_UNSUPPORTED.
- W5 / entry_friction=0: n=5188 (positive=1494, negative=3694), FM Macro F1=0.766889293083244, recall=0.5568942436412316; supported.
- W5 / entry_friction=1: n=2274 (positive=880, negative=1394), FM Macro F1=0.7343724487263302, recall=0.6022727272727273; supported.
- W6 / availability=present: n=7965 (positive=2329, negative=5636), FM Macro F1=0.7201114995663409, recall=0.4439673679690854; supported.
- W6 / availability=missing_or_blank: n=526 (positive=171, negative=355), FM Macro F1=0.7484146698964713, recall=0.7894736842105263; supported.
- W6 / text_load=Q4: n=2080 (positive=693, negative=1387), FM Macro F1=0.6866992263389059, recall=0.3997113997113997; supported.
- W6 / text_load=below_Q4: n=6411 (positive=1807, negative=4604), FM Macro F1=0.7369974093193749, recall=0.4936358605423354; supported.
- W6 / text_load=missing: n=0 (positive=0, negative=0), FM Macro F1=None, recall=None; UNKNOWN_UNSUPPORTED.
- W6 / entry_friction=0: n=5738 (positive=1422, negative=4316), FM Macro F1=0.7234038677707224, recall=0.4409282700421941; supported.
- W6 / entry_friction=1: n=2753 (positive=1078, negative=1675), FM Macro F1=0.7120880872107211, recall=0.5027829313543599; supported.
- H12_E67 / availability=present: n=19528 (positive=6240, negative=13288), FM Macro F1=0.7625840999669777, recall=0.5586538461538462; supported.
- H12_E67 / availability=missing_or_blank: n=1375 (positive=487, negative=888), FM Macro F1=0.7633474258378273, recall=0.8049281314168378; supported.
- H12_E67 / text_load=Q4: n=5863 (positive=2038, negative=3825), FM Macro F1=0.7379531144632968, recall=0.5132482826300294; supported.
- H12_E67 / text_load=below_Q4: n=15040 (positive=4689, negative=10351), FM Macro F1=0.7739771204413639, recall=0.6039667306461932; supported.
- H12_E67 / text_load=missing: n=0 (positive=0, negative=0), FM Macro F1=None, recall=None; UNKNOWN_UNSUPPORTED.
- H12_E67 / entry_friction=0: n=14344 (positive=4123, negative=10221), FM Macro F1=0.7731534733693992, recall=0.567062818336163; supported.
- H12_E67 / entry_friction=1: n=6559 (positive=2604, negative=3955), FM Macro F1=0.7390458330843555, recall=0.5913978494623656; supported.
#### E75
- W6 / availability=present: n=7965 (positive=2329, negative=5636), FM Macro F1=0.7879925051693795, recall=0.6337483898668956; supported.
- W6 / availability=missing_or_blank: n=526 (positive=171, negative=355), FM Macro F1=0.7466446229350274, recall=0.5847953216374269; supported.
- W6 / text_load=Q4: n=1993 (positive=665, negative=1328), FM Macro F1=0.7573110858549754, recall=0.5804511278195489; supported.
- W6 / text_load=below_Q4: n=6498 (positive=1835, negative=4663), FM Macro F1=0.794289674526849, recall=0.6485013623978202; supported.
- W6 / text_load=missing: n=0 (positive=0, negative=0), FM Macro F1=None, recall=None; UNKNOWN_UNSUPPORTED.
- W6 / entry_friction=0: n=5738 (positive=1422, negative=4316), FM Macro F1=0.784207053199949, recall=0.5956399437412095; supported.
- W6 / entry_friction=1: n=2753 (positive=1078, negative=1675), FM Macro F1=0.7733944876277112, recall=0.6762523191094619; supported.

## Fit diagnostics

| Fit ID | nit | nfev | SciPy success | gtol (SUM) | grad infinity / row | Initial loss | Final loss | Factor movement |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| LEARNED_PAIR_PILOT_V2_E67_FM | 585 | 607 | True | 2.9705000000000004 | 8.884112052093531e-05 | 20606.106148940537 | 6692.820350844362 | 73.7001206380774 |
| LEARNED_PAIR_PILOT_V2_E67_NOPAIR | 194 | 201 | True | 2.9705000000000004 | 6.90480078989099e-05 | 20589.936998533165 | 9071.845145901229 | 0.0 |
| LEARNED_PAIR_PILOT_V2_E75_FM | 683 | 710 | True | 4.2117 | 9.49095041000667e-05 | 29209.286489974325 | 8963.471467579078 | 87.69560598350672 |
| LEARNED_PAIR_PILOT_V2_E75_NOPAIR | 181 | 190 | True | 4.2117 | 7.880572183818845e-05 | 29193.279803643185 | 12879.374698006759 | 0.0 |

A02 reference bundles were required and hash/receipt/row-identity checked at both origins. No A02 label array was read. Resource maxima are sampled process/system values, not OS peak RSS. The independent metric/gate receipt is `SCORED_VERIFICATION.json`; it confirms recomputed metrics and verdicts without fitting.

## Interpretasi akhir
MECHANISM_PASS means the learned-pair model met its C2_B and matched no-pair gates. COMPETITION_PASS additionally means it met the exact-origin A02 gates. Either result remains diagnostic-only; A02 is an adaptive historical comparator, and E75 W6 overlaps E67 W6.

## Integritas amendment pasca-fit

Empat fit asli digunakan kembali tanpa refit. Hanya `score_record_ids` yang diekspor ulang sebagai Unicode tetap; payload object lama tidak dideserialisasi. Byte NPY semua member lain dipertahankan identik, dan identitas baris dicocokkan terhadap official Train, prediksi historis positional, serta referensi A02 per origin. Amendment ini tidak mengubah probabilitas, threshold, route, comparator, metric, atau gate.
