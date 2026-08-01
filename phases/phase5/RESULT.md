# Phase 5 result — model operations (rubric item 5, 10 marks)

Deliverables: `report/04_model_ops.md`, `outputs/figs/deploy_architecture.png`, and
`src/ops_arch_diagram.py` which regenerates the figure.

## DONE test coverage
| DONE-test line | Where |
|---|---|
| "Provide a mechanism (architecture) on how the model would be deployed" | `deploy_architecture.png` + section 1 of the report: camera, frame grabber, classifier triage with the defect-free bypass drawn as its own edge, segmentation on flagged frames only, QA database, operator UI, gated alarm / reject arm |
| ONNX export + TorchServe or Triton | section 2, Triton recommended with the tradeoff against TorchServe stated |
| latency budget from the line cadence | section 3, table summing to 100 ms against a 500 ms frame period |
| GPU vs edge box | section 4, comparison table plus a recommendation |
| shadow-mode rollout | section 5, five numeric exit criteria and a two-step ramp |
| "Provide a plan for model maintenance and a process for parameter update" | sections 6 to 9 |
| "Every retrain trigger has a number or a named statistical test attached" | section 6, triggers T1 to T8, each with its threshold on the label line |

## Assumptions I had to make, and why
The line cadence is not published for this dataset, so section 3 assumes a 1 to 2 m/s strip speed
with each 1600 x 256 frame covering about a metre of strip, giving a 500 to 1000 ms frame period.
Every latency number is derived from the shorter of those. The serving stack (Triton, ONNX opset
17), the drift thresholds, and the shadow-mode exit criteria are engineering choices, defended in
the text but not measured. The one place a measured metric belongs, the promotion rule in section
7, names `outputs/metrics/seg_unet.json` and `seg_deeplabv3p.json` as its source rather than
quoting a Dice value phase 4 has not produced yet.

## Scope
Phase 5 is documentation and one figure. It required no training, no GPU and no results from
phases 3 or 4, so it was built in parallel with them and touched no file belonging to another
phase.
