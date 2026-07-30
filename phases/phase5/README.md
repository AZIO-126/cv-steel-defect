# Phase 5 — Model Operations

**Owner** Segmentation+Ops · **Day 11–12** · **Worth 10 marks**

## Grading criteria (verbatim)
> Provide a mechanism (architecture) on how the model would be deployed
> Provide a plan for model maintenance and a process for parameter update.

These 10 marks are the cheapest in the whole rubric for this dataset, because the
production story is real: a steel line genuinely has a camera, a cadence, and a reject arm.

## Steps
1. **Deployment architecture** — one diagram (draw.io / excalidraw → PNG):

   line camera (grayscale, fixed lighting, constant speed) → frame grabber service →
   **classifier triages** (defect-free frames pass straight through, saving downstream
   compute) → only flagged frames enter the **segmentation model** for pixel contours →
   results to the QA database + operator UI highlight → alarm / reject-arm actuation

   Also specify: ONNX export + TorchServe or Triton, latency budget derived from the line
   cadence, GPU vs edge-box choice, and a **shadow-mode rollout** (log only, no
   intervention) before the model is allowed to actuate anything.

2. **Maintenance plan** — all four parts, and make every trigger **quantified** (writing
   "retrain periodically" reads as not having thought it through):
   - **Retrain triggers**: steel grade changeover · camera or lighting replacement ·
     input drift (KS test on the grayscale histogram exceeding a threshold) ·
     prediction drift (defect rate shifting more than X%) · N accumulated
     human-reviewed labels
   - **Retrain pipeline**: new data → human review/labelling → merged into the training
     set → retrain → compare against the incumbent on a **frozen golden test set** →
     promote only if it beats the incumbent
   - **Parameter update**: version names bound to metrics
     (`unet_r50_v20260814_dice0.8123.onnx`), rollback supported, shadow validation first
   - **Production monitoring**: inference latency P50/P99 · throughput · defect-rate time
     series · human-reviewed false-positive rate

## Artifacts
- `report/04_model_ops.md`, architecture PNG in `outputs/figs/`

## DONE test
- Every heading above has content
- Every retrain trigger has a number or a named statistical test attached
