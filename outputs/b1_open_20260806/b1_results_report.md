# B1 registered opening results

- Registration SHA-256: `8ac9ab2e42ed9db80fa10db20b2c679f59b9052af94c1fc23385f17bd1c21dad`
- Raw integrity: **PASS** (13/13 SHA-256 and byte counts; manifest/config seal SHA values matched)
- Bootstrap: 4000 paired event replicates, seed 20260805, percentile linear quantiles
- Confirmatory orientation: `answer_abstain_seed_20260713 - comparator`; verdicts use unrounded atomic values below.
- Availability remains a pre-generation allocation rule, never a release gate.

## Confirmatory verdicts

### H1: **PASS**

등록된 세 IUT 조건을 모두 충족했다.

| Condition | Primary numerator/N (rate) | Comparator numerator/N (rate) | Delta and registered interval/bound | Pass |
|---|---:|---:|---:|---:|
| Correct-released yield (two-sided 95% CI lower > 0) | 247.0/260 (0.95) | 140.0/260 (0.5384615384615384) | 0.4115384615384615; 95% CI [0.35, 0.46923076923076923] | PASS |
| Selected-row wrong-release risk (one-sided 95% upper <= 0.05) | 0.0/260 (0.0) | 27.0/260 (0.10384615384615385) | -0.10384615384615385; one-sided 95% upper -0.07307692307692308 | PASS |
| Event any-negative-violation risk (one-sided 95% upper <= 0.05) | 1.0/260 (0.0038461538461538464) | 0.0/260 (0.0) | 0.0038461538461538464; one-sided 95% upper 0.011538461538461539 | PASS |

Descriptive paired companion: n11/n10/n01/n00 = 140/107/0/13; Newcombe method-10 95% CI [0.35020944746266486, 0.4709901871367287]; exact two-sided McNemar p=1.232595164407831e-32. This is not an additional condition.

### H2: **PASS**

등록된 세 IUT 조건을 모두 충족했다.

| Condition | Primary numerator/N (rate) | Comparator numerator/N (rate) | Delta and registered interval/bound | Pass |
|---|---:|---:|---:|---:|
| Correct-released yield (two-sided 95% CI lower > 0) | 247.0/260 (0.95) | 120.0/260 (0.46153846153846156) | 0.48846153846153845; 95% CI [0.4269230769230769, 0.5500961538461543] | PASS |
| Selected-row wrong-release risk (one-sided 95% upper <= 0.05) | 0.0/260 (0.0) | 70.0/260 (0.2692307692307692) | -0.2692307692307692; one-sided 95% upper -0.2230769230769231 | PASS |
| Event any-negative-violation risk (one-sided 95% upper <= 0.05) | 1.0/260 (0.0038461538461538464) | 2.0/260 (0.007692307692307693) | -0.0038461538461538464; one-sided 95% upper 0.0 | PASS |

Descriptive paired companion: n11/n10/n01/n00 = 120/127/0/13; Newcombe method-10 95% CI [0.42503194227172236, 0.547597488317664]; exact two-sided McNemar p=1.1754943508222875e-38. This is not an additional condition.

## Three-seed robustness (secondary; not a gate)

### H1 comparator

| Layer | N | Yield delta (95% CI) | Wrong-risk delta (one-sided 95% upper) | Any-negative delta (one-sided 95% upper) |
|---|---:|---:|---:|---:|
| 20260713 | 260 | 0.4115384615384615 [0.35, 0.46923076923076923] | -0.10384615384615385 (-0.07307692307692308) | 0.0038461538461538464 (0.011538461538461539) |
| 20260714 | 260 | 0.4346153846153846 [0.3730769230769231, 0.49615384615384617] | -0.10384615384615385 (-0.07307692307692308) | 0.011538461538461539 (0.023076923076923078) |
| 20260715 | 260 | 0.4230769230769231 [0.36153846153846153, 0.4846153846153846] | -0.10384615384615385 (-0.07307692307692308) | 0.007692307692307693 (0.019230769230769232) |
| event mean over 3 seeds | 260 | 0.4230769230769231 [0.36153846153846153, 0.4833333333333333] | -0.10384615384615385 (-0.07307692307692308) | 0.007692307692307693 (0.016666666666666666) |

Worst-seed upper bounds: wrong-release `-0.07307692307692308` (20260715), any-negative `0.023076923076923078` (20260714).

### H2 comparator

| Layer | N | Yield delta (95% CI) | Wrong-risk delta (one-sided 95% upper) | Any-negative delta (one-sided 95% upper) |
|---|---:|---:|---:|---:|
| 20260713 | 260 | 0.48846153846153845 [0.4269230769230769, 0.5500961538461543] | -0.2692307692307692 (-0.2230769230769231) | -0.0038461538461538464 (0.0) |
| 20260714 | 260 | 0.5115384615384615 [0.45, 0.573076923076923] | -0.2692307692307692 (-0.2230769230769231) | 0.0038461538461538464 (0.015384615384615385) |
| 20260715 | 260 | 0.5 [0.43846153846153846, 0.5615384615384615] | -0.2692307692307692 (-0.2230769230769231) | 0.0 (0.007692307692307693) |
| event mean over 3 seeds | 260 | 0.5 [0.43846153846153846, 0.5615384615384615] | -0.2692307692307692 (-0.2230769230769231) | -2.1350442781253012e-19 (0.007692307692307693) |

Worst-seed upper bounds: wrong-release `-0.2230769230769231` (20260715), any-negative `0.015384615384615385` (20260714).

## Auxiliary arms (secondary/descriptive; not a gate)

The Holm family contains the three answer-only release policies and the activity head, each compared with the primary exact policy on event correct-released yield using exact two-sided McNemar p-values.

| Auxiliary arm | Correct-released / 260 | Wrong-release / 260 | Any-negative / 260 | Raw p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| `answer_only_seed_20260713` | 0.0/260 | 0.0/260 | 0.0/260 | 8.843436600416711e-75 | 3.5373746401666845e-74 |
| `answer_only_seed_20260714` | 0.0/260 | 0.0/260 | 0.0/260 | 8.843436600416711e-75 | 3.5373746401666845e-74 |
| `answer_only_seed_20260715` | 0.0/260 | 0.0/260 | 0.0/260 | 8.843436600416711e-75 | 3.5373746401666845e-74 |
| `action_classification_head` | 97.0/260 | 0.0/260 | 12.0/260 | 1.401298464324817e-45 | 1.401298464324817e-45 |

Zero-shot is record-only with no tau or test: frozen-parser answered 260/260 selected rows, of which 124/260 were correct.

## Stage 2/3 decomposition (all registered arms)

All counts use 1,096 candidate-view rows; zero-shot has no registered release threshold, so Stage 3 is N/A.

| Arm | tau | Unparsable | Answered correct/wrong | Below-tau correct/wrong | Released correct/wrong |
|---|---:|---:|---:|---:|---:|
| `answer_abstain_seed_20260713` | 0.981128697326919 | 0/1096 | 254 (252/2) | 6 (5/1) | 248 (247/1) |
| `answer_abstain_seed_20260714` | 0.9722987314076553 | 0/1096 | 260 (256/4) | 4 (3/1) | 256 (253/3) |
| `answer_abstain_seed_20260715` | 0.9662443411375032 | 0/1096 | 256 (254/2) | 4 (4/0) | 252 (250/2) |
| `answer_only_seed_20260713` | 0.9999968725353879 | 0/1096 | 1096 (260/836) | 1096 (260/836) | 0 (0/0) |
| `answer_only_seed_20260714` | 0.9999948179767415 | 0/1096 | 1096 (260/836) | 1096 (260/836) | 0 (0/0) |
| `answer_only_seed_20260715` | 0.9999959819939936 | 0/1096 | 1096 (260/836) | 1096 (260/836) | 0 (0/0) |
| `explicit_abstention_prompt` | 0.9491225851331859 | 0/1096 | 170 (141/29) | 3 (1/2) | 167 (140/27) |
| `zero_shot` | N/A | 1096/1096 | 1096 (124/972) | N/A | N/A |
| `correctness_selector` | 0.9999998807907104 | 0/1096 | 237 (123/114) | 45 (3/42) | 192 (120/72) |
| `action_classification_head` | 0.9994823932647705 | 0/1096 | 135 (113/22) | 25 (16/9) | 110 (97/13) |

## Allocation and measured cost (record-only; no test)

| Policy | A(1) | U(1), primary exact policy | Router GPU s | Selected VLM GPU s | Total GPU s |
|---|---:|---:|---:|---:|---:|
| `availability` | 260/260 = 1.0 | 247/260 = 0.95 | 20.526596675872806 | 83.29480274963379 | 103.8213994255066 |
| `task_matched_selector` | 260/260 = 1.0 | 247/260 = 0.95 | 163.17363381385786 | 83.29480274963379 | 246.46843656349165 |

Measured feature CUDA timers (1,096 rows): Grounding-DINO `149.08231884765607` s, SigLIP2 image `14.066607894897464` s, SigLIP2 text `0.024707071304321288` s, DINOv2 `6.459988780975341` s.
All eight full generation passes sum to `2367.9418798828124` CUDA-timer seconds over 8 x 1,096 rows; per-pass values and checkpoint equality are in `b1_results.json`.
Successful resume-2 head accounting is `27890` active-GPU wall-seconds (2 x (11209 + 2736)); this is not a CUDA timer. Earlier failed attempts lack an exact additive GPU-seconds log and are not fabricated.

Task-matched auxiliary feature integrity: **PASS**. SigLIP2/DINOv2 tree, DINO JSONL, text-feature, and frozen selector SHA values all match their checkpoints; the auxiliary embeddings also reproduce the sealed availability top-1 for 260/260 events. These feature files remain outside the 13-file raw manifest and are used only for the registered record-only allocation comparison.

## Prespecified descriptive layers

- Proxy-hard union: 126/260 events; argmax reason height=50, FOV=76. Exact ties were assigned to height.
- Source-clip connected-component bootstrap: 63 components; 4,000 component replicates, seed 20260805.
- Site bootstrap: 3 sites; 4,000 site-cluster replicates, seed 20260805; descriptive because there are only three sites.
- Activity strata: 12 registered activities; descriptive rates and denominators are in `b1_results.json`.

### Proxy-hard event bootstrap

| Comparison | N | Yield delta (95% CI) | Wrong-risk delta (one-sided upper) | Any-negative delta (one-sided upper) |
|---|---:|---:|---:|---:|
| H1 | 126 | 0.5 [0.4126984126984127, 0.5873015873015873] | -0.07936507936507936 (-0.03968253968253968) | 0.0 (0.0) |
| H2 | 126 | 0.5158730158730159 [0.42857142857142855, 0.6031746031746031] | -0.1746031746031746 (-0.11904761904761904) | -0.007936507936507936 (0.0) |

### Cluster bootstrap

| Resampling unit / comparison | Clusters | Yield delta (95% CI) | Wrong-risk delta (one-sided upper) | Any-negative delta (one-sided upper) |
|---|---:|---:|---:|---:|
| source_clip / H1 | 63 | 0.4115384615384615 [0.3333333333333333, 0.5087034860947904] | -0.10384615384615385 (-0.07169811320754717) | 0.0038461538461538464 (0.01020408163265306) |
| source_clip / H2 | 63 | 0.48846153846153845 [0.40509848334232723, 0.5842239072256914] | -0.2692307692307692 (-0.22083333333333333) | -0.0038461538461538464 (0.0) |
| site / H1 | 3 | 0.4115384615384615 [0.2777777777777778, 0.6172839506172839] | -0.10384615384615385 (-0.068) | 0.0038461538461538464 (0.006968641114982578) |
| site / H2 | 3 | 0.48846153846153845 [0.32407407407407407, 0.654320987654321] | -0.2692307692307692 (-0.25252525252525254) | -0.0038461538461538464 (0.0) |

### Activity-stratified descriptive counts

Cells are correct-released / wrong-released / any-negative counts; the row N is the denominator for every cell.

| Activity | N | Primary | Explicit prompt | Correctness selector |
|---|---:|---:|---:|---:|
| `hand_interacts_with_person` | 10 | 10/0/0 | 9/1/0 | 10/0/0 |
| `person_carries_heavy_object` | 9 | 9/0/0 | 3/4/0 | 4/4/0 |
| `person_closes_vehicle_door` | 29 | 27/0/0 | 18/4/0 | 18/4/0 |
| `person_enters_scene_through_structure` | 18 | 18/0/0 | 14/0/0 | 9/5/0 |
| `person_enters_vehicle` | 8 | 7/0/0 | 6/1/0 | 3/3/0 |
| `person_exits_scene_through_structure` | 37 | 35/0/0 | 18/11/0 | 13/14/0 |
| `person_exits_vehicle` | 8 | 7/0/0 | 7/0/0 | 7/0/0 |
| `person_opens_facility_door` | 29 | 28/0/0 | 21/4/0 | 17/7/0 |
| `person_opens_vehicle_door` | 19 | 18/0/0 | 13/2/0 | 10/2/0 |
| `person_talks_on_phone` | 8 | 8/0/0 | 3/0/0 | 3/2/0 |
| `person_talks_to_person` | 70 | 65/0/1 | 21/0/0 | 22/19/2 |
| `person_texts_on_phone` | 15 | 15/0/0 | 7/0/0 | 4/10/0 |

## Transport sensitivity (nonconfirmatory)

The primary tau `0.981128697326919` is copied without reselection to the registered stored DEV responses:

| Target seed | Events/rows | P+ correct-released yield | E- release | P- release | Released/errors |
|---:|---:|---:|---:|---:|---:|
| 20260714 | 82/246 | 0.9390243902439024 | 0.036585365853658534 | 0.0 | 80/3 |
| 20260715 | 82/246 | 0.8780487804878049 | 0.024390243902439025 | 0.0 | 74/2 |

## Strict unparsable frequency

| Generation pass | Unparsable / 1,096 | Rate |
|---|---:|---:|
| `answer_abstain_seed_20260713` | 0/1096 | 0.0 |
| `answer_abstain_seed_20260714` | 0/1096 | 0.0 |
| `answer_abstain_seed_20260715` | 0/1096 | 0.0 |
| `answer_only_seed_20260713` | 0/1096 | 0.0 |
| `answer_only_seed_20260714` | 0/1096 | 0.0 |
| `answer_only_seed_20260715` | 0/1096 | 0.0 |
| `explicit_abstention_prompt` | 0/1096 | 0.0 |
| `zero_shot` | 1096/1096 | 1.0 |

The neutral zero-shot pass uses the frozen parser's stored non-abstention state only for the registered correctness-selector answer-source semantics. Its strict-wrapper unparsable count does not redefine H2.

## Retry and deviation disclosure

- Amendment 001 records the restored orientation-audit dependency and operator exit-code wrapper fix before opening.
- Amendments 003/004 record rejected host/container launch environments before valid generation started.
- The fixed head setting exhausted three attempts after 227 action-head and 221 correctness-head prefix rows.
- Amendment 002 records the B1-only no-truncation repair, 448-row prefix identity gate with 0 mismatches, completed heads, and final raw seal.
- No unresolved row is present in the final 13 raw files; the final files were sealed before this aggregation.

## Claim boundary

This is an offline sealed-cohort result. It does not claim human semantic sufficiency, new-site/camera transfer, formal risk guarantees, deployment safety, online latency/throughput, or availability-based release gating.

Analysis code SHA-256: `d7be279c159f2234fc864f3b38322be2a2623aad867061d0562292e20286786a`
