# HoloLLM: Interactive AR Transformer Architecture Visualizer 🌌🦾

An Iron-Man / Jarvis-inspired interactive Augmented Reality (AR) visualizer of a modern **Transformer / Large Language Model (LLM)** architecture (LLaMA / GPT series) built with **OpenCV**, **MediaPipe CNN Hand Landmark Tracking**, and **pure NumPy 3D perspective mathematics**.

---

## ⚡ Key Highlights

- **3D Perspective Projection Engine:** Built from first principles in NumPy ($R_x, R_y, R_z$, depth clipping, pinhole camera projection).
- **8 Distinct Architectural Layers:** Each layer features custom 3D wireframes reflecting its exact mathematical function (sinusoidal RoPE helix, RMSNorm rings, QKV projection rays, causal attention matrix, SwiGLU expansion diamond, residual highway bypass, next-token softmax histogram).
- **CNN Hand Gesture Control:** Dual-backend MediaPipe support (Modern Tasks API `HandLandmarker` + legacy `Solutions` fallback) with Exponential Moving Average (EMA) jitter smoothing.
- **Jarvis Sci-Fi HUD:** Neon cyan bloom glow filter, animated inference token particle pulses, cybernetic pedestal rings, 21-joint hand skeleton overlay, and chamfered mathematical telemetry cards.
- **Graceful Desktop Fallback:** Full mouse and keyboard controls if running on a machine without a webcam.

---

## 🖐️ Hand Gesture Control System

| Gesture | Action | Visual Feedback |
| :--- | :--- | :--- |
| **Single-Hand Pinch** *(Thumb 4 + Index 8 < 42px)* | **3D Orbit Rotation** (Pitch & Yaw) | Dynamic tension line turns **Neon Orange**, HUD displays `[ORBIT ROTATION]` |
| **Two-Hand Distance** *(D_hands between Left & Right Index)* | **Explode / Collapse** 8 Layer Stack | Real-time `EXPANSION: XX%` segmented bar, monolithic core separates into floating plates |
| **Index Pointer** *(Index extended, others curled)* | **3D Layer Touch Inspection** | Laser reticle raycasts against layer bounding hull; touched layer glows **Neon Gold** |
| **Open Palm / Idle** | **Floating Standby** | Smooth cinematic auto-rotation, glowing knuckle beads and cybernetic bones |

---

## 🧠 The 8 Interactive LLM Layers

Each layer renders specialized 3D internal wireframe geometry and displays rich tensor math when inspected:

1. **INPUT & BPE TOKENIZER (`Text String -> [Batch, Seq_Len] (int64)`):**
   - *Geometry:* Segmented 3D token mini-blocks with a byte-pair splitting boundary line.
   - *Math:* $T = \text{BPE\_Merge}(\text{UTF8\_Bytes}(\text{text}), \text{Merge\_Table}, V)$.
2. **TOKEN EMBEDDING + RoPE (`[B, S] -> [B, S, d_model]`):**
   - *Geometry:* 2D coordinate mesh overlaid with a 3D sinusoidal spiral helix wave.
   - *Math:* $R_\Theta^d \cdot x_m$, rotating query/key 2D s

(More details coming soon...)