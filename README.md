# SOR-BDNet Semantic Optical Representation for Boundary-Aware Video Anomaly Detection with GPT-4o
VAD is shifting to LLM-driven semantics. We present SOR-BDNet: annotation-free, multimodal. RGB + RAFT flow form spatiotemporal inputs; GPT-4o yields frame captions. Anomalies = semantic deviation from normal-caption memory. A Swin+contrastive boundary refiner sharpens timing. Ped2/Avenue/ShanghaiTech/Ucf-Crime: 97.96%, 82.86%, 87.36%, and 85.64%.
## 🧩 Model Overview

<p align="center">
  <img src="model/model.png" width="90%">
</p>

<p align="center">
  <i>RGB frames and RAFT optical flow are fused into a unified spatiotemporal representation, which is processed by GPT-4o and the boundary-aware network for anomaly detection.</i>
</p>

---

## 🎯 Boundary Refinement

<p align="center">
  <img src="model/bianjie.png" width="90%">
</p>

<p align="center">
  <i>The coarse anomaly interval from GPT-4o captions is refined by a Swin-based boundary module, sharpening the start and end frames of abnormal events.</i>
</p>
---
---

---

## 🎬 Demo 

<p align="center">
  <img src="assets/ped2.gif" width="80%">
</p>

<p align="center"><i>GIF demo generated from ped.mp4</i></p>



