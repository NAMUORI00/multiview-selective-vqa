# B1 preregistration freeze summary v1.1

- Status: **FROZEN_NOT_EXECUTED**
- Registration SHA-256: `8ac9ab2e42ed9db80fa10db20b2c679f59b9052af94c1fc23385f17bd1c21dad`
- Threshold artifact SHA-256: `4a9f9c9eadaa1a98098f6c13563693a8e3b2c90101305ee1cfad1183b1b680e6`
- Scope: metadata/CPU freeze only; B1 media, pixels, model inference, GPU execution, services, and manuscript were not touched.
- Execution approval: **not granted**. Commit, push, server sync, and live resource checks remain pending.

## Cohort

- Prior exclusion: 641 clips, 160 site-by-5-minute slots; SHA `fee0e73393bf846a40b24d5af45dfcb2f2c7dd9fe45de89f59afd929a753dba8`.
- Eligible pool: 714 events / 3,354 candidate views.
- Selected B1: **260 events / 1,096 candidate views**, 254 unique clips, 63 unique site-slot keys; manifest SHA `1662e91c913511f0b5a3d31cccf56483ed65c589223a6ba8f5ca1e7f18f918eb`.
- Disjointness: clip intersection 0 and site-slot intersection 0. Same cameras and sites are intentionally reused.
- MILP: all 12 activities >=8; activity max 70/260 (26.92% displayed); camera max 104/260 (40%).
- Structure: `n_cand>=3` 249; `n_cand>=4` 164; 18 location-date blocks.
- Reserve: 454 constraint-preserving base candidates, ranks 261 onward; SHA `7b9a9a8aec8c4b4264639597760c2949d366d07be02180da6cdb621d3bfa37e8`.
- Double run: exclusion, manifest, and reserve SHA values were identical across two full same-seed executions.

### Activity distribution

| Activity | Events |
|---|---:|
| `hand_interacts_with_person` | 10 |
| `person_carries_heavy_object` | 9 |
| `person_closes_vehicle_door` | 29 |
| `person_enters_scene_through_structure` | 18 |
| `person_enters_vehicle` | 8 |
| `person_exits_scene_through_structure` | 37 |
| `person_exits_vehicle` | 8 |
| `person_opens_facility_door` | 29 |
| `person_opens_vehicle_door` | 19 |
| `person_talks_on_phone` | 8 |
| `person_talks_to_person` | 70 |
| `person_texts_on_phone` | 15 |

### Metadata proxy layers

| Proxy | Events |
|---|---:|
| minimum positive person height <=48 px | 38 |
| minimum positive person height <=64 px | 71 |
| minimum negative FOV clearance <=5 degrees | 46 |
| minimum negative FOV clearance <=10 degrees | 79 |

These are metadata proxies, not human semantic-sufficiency, occlusion, or ambiguous-timing labels.

## Frozen thresholds

The operative rule is W5a `dev_max_yield_role_guardrails`: over all 246 stored DEV rows, maximize P+ correct-released yield subject to E- release <=0.10 and P- release <=0.05; ties choose fewer released errors and then the higher tau. If no non-empty candidate satisfies both guardrails, tau is 1 and release is disabled. The M3 Clopper-Pearson track is non-operative in B1 v1.1.

| Seed / fixed pass | Arm | Tau | Fallback |
|---:|---|---:|---|
| 20260713 | `answer_abstain` exact policy | `0.981128697326919` | no |
| 20260713 | `answer_only` | `0.9999968725353879` | no |
| 20260714 | `answer_abstain` | `0.9722987314076553` | no |
| 20260714 | `answer_only` | `0.9999948179767415` | no |
| 20260715 | `answer_abstain` | `0.9662443411375032` | no |
| 20260715 | `answer_only` | `0.9999959819939936` | no |
| fixed | `explicit_abstention_prompt` | `0.9491225851331859` | no |
| fixed | `correctness_selector` | `0.9999998807907104` | no |
| fixed | `action_classification_head` | `0.9994823932647705` | no |

The executed W5a selection code reproduced the 20260713 answer/abstain B0 plan tau exactly: `0.981128697326919`. The reproduction uses W5a's 246-row selection source; the preserved M3 20260713 DEV response is recorded separately as an application diagnostic.

### Primary-tau transport sensitivity

The exact-policy tau `0.981128697326919` was transported without reselection to the new seeds' stored DEV responses. This is descriptive only.

| Target seed | P+ correct-released yield | E- release | P- release | Released / errors |
|---:|---:|---:|---:|---:|
| 20260714 | 0.939024 | 0.036585 | 0 | 80 / 3 |
| 20260715 | 0.878049 | 0.024390 | 0 | 74 / 2 |

## Power and confirmatory decision

- N=260, alpha=0.05, 10,000 Monte Carlo trials, 4,000 paired-bootstrap replicates per trial.
- Exact McNemar / paired-bootstrap power: 30 pp = 1.0000 / 1.0000; 15 pp = 0.9911 / 0.9931; 10 pp = 0.7999 / 0.8196.
- H1 compares the 20260713 exact-policy answer/abstain arm against `explicit_abstention_prompt`. H2, evaluated only if H1 passes, compares the same arm against `correctness_selector`.
- Each hypothesis uses the same three-condition IUT: the two-sided 95% event-bootstrap lower bound for paired correct-released-yield difference is >0; the one-sided 95% upper bound for selected-positive wrong-release risk difference is <=+0.05; and the one-sided 95% upper bound for event-level any-negative-violation risk difference is <=+0.05.
- Only the 20260713 exact-policy layer is confirmatory. The three-seed layer averages outcomes within event before event bootstrap, publishes every seed, forbids best-seed selection/reporting, and also reports worst-seed one-sided safety upper bounds. It is not another confirmatory gate.
- Availability-versus-task-matched allocation and preprocessing-inclusive GPU cost are record-only, with no test, pass condition, or gate.
- Full cohort is confirmatory. The proxy-hard layer is prespecified descriptive sensitivity only.

## Execution plan and updated resources

- Reused adapters: six; new training estimate **0 GPU-h**.
- Availability/generation/selector evaluation: 0.74--0.89 GPU-h.
- Separate frozen-prompt head passes: 8.22 GPU-h.
- Total planning estimate: **8.96--9.11 GPU-h**, about 5 wall-hours under the registered 3-GPU waves.
- Incremental disk reservation: **5--7 GiB**. These are linear offline planning estimates, not online latency or throughput claims.
- Two-stage seal: config before media access, raw artifacts before any aggregate. B1 opens only at a separately approved aggregate invocation.

## Pending pre-execution gates

1. Preserve or resolve the current user-managed manuscript change; obtain a clean execution worktree without sweeping it.
2. Create and push the exact execution commit, then verify local `HEAD == origin/main == server HEAD`.
3. Resolve the previously observed server dirty files through the owner-approved procedure.
4. Reverify all remote adapter/head hashes, >=80 GiB free disk, GPU ownership, and shard availability.
5. Obtain explicit user approval before any B1 materialization, GPU stage, service action, commit, push, or opening.

## v1.1 correction summary

- F1: Replaced the allocation-first gate with H1 exact-policy answer/abstain versus explicit-abstention prompt and gated H2 versus correctness-selector; moved allocation and GPU cost to record-only; made the three-seed analysis robustness-only.
- F2: Replaced operative M3 Clopper-Pearson thresholds with W5a `dev_max_yield_role_guardrails`, added the explicit-prompt tau, recomputed all seed/arm thresholds from stored DEV confidence, and restored the original primary-tau transport sensitivity.
- F3: Audited naming, claim scope, stop/retry rules, two-stage sealing, scorer/parser, deviation/disclosure, head thresholds, zero-shot pass, and all four GPU waves. No other design drift was found.
- Cohort, reserve, and power layers were not regenerated or edited. Their hashes remain unchanged.
