# SOR-BDNet Semantic Optical Representation for Boundary-Aware Video Anomaly Detection with GPT-4o
VAD is shifting to LLM-driven semantics. We present SOR-BDNet: annotation-free, multimodal. RGB + RAFT flow form spatiotemporal inputs; GPT-4o yields frame captions. Anomalies = semantic deviation from normal-caption memory. A Swin+contrastive boundary refiner sharpens timing. Ped2/Avenue/ShanghaiTech/Ucf-Crime: 97.96%, 82.86%, 87.36%, and 85.64%.
## 🧩 Model Overview

<p align="center">
  <img src="model/model.png" width="60%">
</p>

<p align="center">
  <i>RGB frames and RAFT optical flow are fused into a unified spatiotemporal representation, which is processed by GPT-4o and the boundary-aware network for anomaly detection.</i>
</p>

---

## 🎯 Boundary Refinement

<p align="center">
  <img src="model/bianjie.png" width="60%">
</p>

<p align="center">
  <i>The coarse anomaly interval from GPT-4o captions is refined by a Swin-based boundary module, sharpening the start and end frames of abnormal events.</i>
</p>
---
---

---
---

## 📁 Directory Structure

The repository is organized into functional modules including caption generation, 
optical flow extraction, preprocessing, and the main SOR-BDNet architecture.

<pre>
SOR-BDNet/
├── caption/               # GPT-4o caption generation (semantic reasoning)
│
├── model/                 # Core SOR-BDNet components (fusion + Swin + boundary refinement)
│
├── processing/            # Preprocessing scripts & memory bank construction
│
├── prompt_gpt-4o/         # Prompt templates used for GPT-4o VQA/captioning
│
├── raft/                  # RAFT optical flow implementation
│
├── ped2.gif               # Demo GIF displayed in README
├── ped.mp4                # Subtitle + boundary refinement video demo
│
├── train_cli.py           # Main training entry point
├── raft.py                # RAFT optical flow extraction script
└── README.md              # Project documentation
</pre>



## 📥 RAFT Model Weight Download

To generate optical flow using RAFT, please download the pretrained model (.pth) and
place it inside the `raft/` folder:

https://drive.google.com/file/d/1p1CQUgYhZ1B6pR3pYrb_NpPGygERENgd/view?usp=drive_link

## 🎬 Demo (GIF)

<p align="center">
  <img src="ped2.gif" width="40%">
</p>

---

## 📝 Summary

SOR-BDNet provides a unified multimodal framework for annotation-free video anomaly detection by combining RGB appearance, RAFT-based motion dynamics, and GPT-4o semantic reasoning. The proposed boundary-aware refinement module further enhances temporal localization accuracy, enabling the model to better capture the start and end points of abnormal events. 

With strong performance across Ped2, Avenue, ShanghaiTech, and UCF-Crime, SOR-BDNet demonstrates the effectiveness of integrating vision, motion, and high-level semantics for robust real-world anomaly detection.




