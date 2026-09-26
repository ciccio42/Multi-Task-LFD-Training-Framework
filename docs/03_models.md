# 🧠 Models

[← Data](02_data.md) · [README](../README.md) · Next: [Training →](04_training.md)

Hydra instantiates every model from the `policy:` node. The model code is in [`training/multi_task_il/models/`](../training/multi_task_il/models/).

---

## The COD pipeline ⭐

The main line of work in this repo splits one-shot imitation into two steps: **what** to manipulate and **how** to manipulate it.

```
 demo video ─┐
             ├─► COD (Conditioned Target Object Detector) ─► target-object bbox(es)
 agent view ─┘                                                   │
                                                                 ▼
 demo video + agent view (+ wrist view) ─────────────► control policy ─► 7-D action
```

### 1. COD: `CondTargetObjectDetector`

Code: [`cond_target_obj_detector/cond_target_obj_detector.py`](../training/multi_task_il/models/cond_target_obj_detector/cond_target_obj_detector.py).

- A `CondModule` encodes the demonstration video into a task embedding.
- An `AgentModule` extracts features from the agent's frame, and **FiLM** blocks modulate those features with the task embedding.
- A `ProposalModule` and a `ClassificationModule` then predict an anchor-based bounding box, plus a class, for the target object (and the place location).
- Config: `config_cond_target_obj_detector.yaml`. Launchers: `train_cond_target_obj_detector*.sh`, `train_keypoint_detection_*.sh`.

### 2. Control policy on top of COD

| Variant | Selected with | Notes |
|---|---|---|
| **MOSAIC double policy** | `policy='${mosaic}'` + `mosaic._target_=multi_task_il.models.mt_rep_double_policy.VideoImitation` | Main model. Takes COD boxes (`mosaic.concat_bb=true`) and can optionally use the wrist image (`actions.use_wrist_img`). |
| COD policy | `policy='${cond_policy}'` (`policy.CondPolicy`) | Lightweight head on a pretrained COD |
| No-MOSAIC double policy | `mosaic._target_=…no_mosaic_double_policy.VideoImitation` | Ablation without the MOSAIC representation losses |

How COD is wired into the policy:

| Key | Meaning |
|---|---|
| `mosaic.load_target_obj_detector` | Load a pretrained COD |
| `mosaic.target_obj_detector_path` / `_step` | Which COD checkpoint to load |
| `mosaic.freeze_target_obj_detector` | Keep COD fixed, or fine-tune it jointly |
| `mosaic.concat_bb` | Feed the predicted boxes to the policy |
| `mosaic.concat_target_obj_embedding` | Feed COD's embedding to the policy |
| `augs.null_bb` | Ablation: zero out the boxes |

---

## MOSAIC (`${mosaic}`)

`mt_rep.VideoImitation` ([`mt_rep.py`](../training/multi_task_il/models/mt_rep.py)) is a re-implementation of MOSAIC. It combines a ResNet feature extractor, stacked non-local/temporal attention between demonstration and agent frames, and a discretized-logistic-mixture action head. Training adds several losses:

| Loss | Config weight |
|---|---|
| Behavior cloning | `bc_mul` |
| Inverse dynamics | `inv_mul` |
| Contrastive / BYOL representation losses ([`rep_modules.py`](../training/multi_task_il/models/rep_modules.py)) | `simclr.mul_pre`, `simclr.mul_pos`, `simclr.mul_intm` |

Main architecture knobs:

| Key | Meaning |
|---|---|
| `actions.n_mixtures`, `actions.out_dim`, `actions.adim` | Action head |
| `attn.attn_ff`, `attn.img_cfg.drop_dim`, `attn.img_cfg.out_feature` | Attention and image trunk |
| `simclr.compressor_dim`, `simclr.hidden_dim` | Projection heads |
| `mosaic.dim_H`, `mosaic.dim_W` | Feature-map size; depends on the input resolution (e.g. 13×23 for 100×180) |
| `actions.concat_img_emb`, `actions.concat_demo_emb`, `mosaic.concat_state`, `mosaic.concat_demo_act` | Which inputs the action head sees |

## Baselines

| Model | `policy=` | Code |
|---|---|---|
| TOSIL | `${tosil}` | `baselines.InverseImitation` |
| DAML | `${daml}` | `baselines.DAMLNetwork` |
| VIMA | `${vima}` (via `train_vima.py`, `config_vima.yaml`) | `models/vima/` |
| Target object detector (unconditioned) | `config_target_obj.yaml` / `target_object_detector_config.yaml` | `target_obj_detector*.py` |
| GNN scene-graph policy | `train_scripts/gnn/train.py`, `config_gnn.yaml` | `multi_task_il_gnn/` |

## Task encoders

| Model | `policy=` | Code |
|---|---|---|
| Video / command encoder | `${cond_module}` (via `train_video_encoder.py`, `experiments/video_encoder/`) | `models/command_encoder/cond_module.py` |
| MUSE text encoder | none | `models/command_encoder/muse/`. Encodes language commands; includes t-SNE and centroid tools. |
