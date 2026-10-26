# HoloLLM: Interactive AR Transformer Architecture Visualizer

An Iron-Man / Jarvis-inspired interactive Augmented Reality (AR) visualizer of a modern **Transformer / Large Language Model (LLM)** architecture (LLaMA / GPT series) built with **OpenCV**, **MediaPipe CNN Hand Landmark Tracking**, and **pure NumPy 3D perspective mathematics**.

---

## Key Highlights

- **3D Perspective Projection Engine:** Built from first principles in NumPy ($R_x, R_y, R_z$, depth clipping, pinhole camera projection).
- **8 Distinct Architectural Layers:** Each layer features custom 3D wireframes reflecting its exact mathematical function (sinusoidal RoPE helix, RMSNorm rings, QKV projection rays, causal attention matrix, SwiGLU expansion diamond, residual highway bypass, next-token softmax histogram).
- **CNN Hand Gesture Control:** Dual-backend MediaPipe support (Modern Tasks API `HandLandmarker` + legacy `Solutions` fallback) with Exponential Moving Average (EMA) jitter smoothing.
- **Jarvis Sci-Fi HUD:** Neon cyan bloom glow filter, animated inference token particle pulses, cybernetic pedestal rings, 21-joint hand skeleton overlay, and chamfered mathematical telemetry cards.
- **Graceful Desktop Fallback:** Full mouse and keyboard controls if running on a machine without a webcam.

---

## Hand Gesture Control System

| Gesture | Action | Visual Feedback |
| :--- | :--- | :--- |
| **Single-Hand Pinch** *(Thumb 4 + Index 8 < 42px)* | **3D Orbit Rotation** (Pitch & Yaw) | Dynamic tension line turns **Neon Orange**, HUD displays `[ORBIT ROTATION]` |
| **Two-Hand Distance** *(D_hands between Left & Right Index)* | **Explode / Collapse** 8 Layer Stack | Real-time `EXPANSION: XX%` segmented bar, monolithic core separates into floating plates |
| **Index Pointer** *(Index extended, others curled)* | **3D Layer Touch Inspection** | Laser reticle raycasts against layer bounding hull; touched layer glows **Neon Gold** |
| **Open Palm / Idle** | **Floating Standby** | Smooth cinematic auto-rotation, glowing knuckle beads and cybernetic bones |

---

## The 8 Interactive LLM Layers

Each layer renders specialized 3D internal wireframe geometry and displays rich tensor math when inspected:

1. **INPUT & BPE TOKENIZER (`Text String -> [Batch, Seq_Len] (int64)`):**
   - *Geometry:* Segmented 3D token mini-blocks with a byte-pair splitting boundary line.
   - *Math:* $T = \text{BPE-Merge}(\text{UTF8-Bytes}(\text{text}), \text{Merge-Table}, V)$.
2. **TOKEN EMBEDDING + RoPE (`[B, S] -> [B, S, d_model]`):**
   - *Geometry:* 2D coordinate mesh overlaid with a 3D sinusoidal spiral helix wave.
   - *Math:* $R_\Theta^d \cdot x_m$, rotating query/key 2D slices to preserve relative token distance.
3. **RMSNorm (PRE-NORM) (`[B, S, d_model] -> [B, S, d_model]`):**
   - *Geometry:* Concentric stabilizing polygonal rings and radial crosshairs with central diamond $\Diamond$.
   - *Math:* $\bar{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})} g_i$, stabilizing gradient variance across deep stacks.
4. **QKV LINEAR PROJECTIONS (`[B, S, d_model] -> 3 x [B, n_heads, S, d_head]`):**
   - *Geometry:* 3 parallel sub-slabs (Query, Key, Value) with diverging projection rays.
   - *Math:* $Q = X W_Q, K = X W_K, V = X W_V$, updating KV-Cache in VRAM.
5. **CAUSAL MULTI-HEAD ATTENTION (`[B, n_heads, S, d_head] -> [B, S, d_model]`):**
   - *Geometry:* Lower-triangular matrix grid with masked cross-hatch cells and arched attention curves.
   - *Math:* $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T + M}{\sqrt{d_k}}\right) V$.
6. **RESIDUAL HIGHWAY + RMSNorm (`[B, S, d_model]`):**
   - *Geometry:* Twin bypass conduits looping wide around the stack into a summation node $\oplus$.
   - *Math:* $x_{l+1} = x_l + \text{Attention}(\text{RMSNorm}(x_l))$, preserving unobstructed signal flow.
7. **SwiGLU FEED-FORWARD (MLP) (`[B, S, d] -> [B, S, 4d] -> [B, S, d]`):**
   - *Geometry:* Double-hourglass diamond expansion prism ($d \to 4d \to d$).
   - *Math:* $\text{SwiGLU}(x) = (\text{Swish}(x W_{\text{gate}}) \otimes (x W_{\text{up}})) W_{\text{down}}$.
8. **LM HEAD & SOFTMAX LOGITS (`[B, S, d_model] -> [B, S, Vocab_Size]`):**
   - *Geometry:* Converging funnel pyramid topped with a 3D bar-chart histogram of top token logits.
   - *Math:* $P(w_t \mid w_{<t}) = \text{softmax}(\text{Logits} / T)$, sampling next token via Top-$p$ / Top-$k$.

---

## Desktop Fallback & Keyboard Shortcuts

If you are running without a webcam or prefer mouse/keyboard control:

- **Mouse Left-Click Drag:** Orbit rotate 3D hologram in Pitch & Yaw
- **Mouse Scroll Wheel:** Zoom camera in / out
- **Mouse Hover:** Acts as holographic pointer to inspect layers
- **`[SPACE]`**: Toggle Explode / Collapse between monolithic core and separated stack
- **`[R]`**: Reset rotation, camera zoom, and layer selection
- **`[A]`**: Toggle cinematic auto-rotation
- **`[1]` - `[8]`**: Directly select and inspect layers 1 through 8
- **`[0]`**: Clear layer selection (return to general overview)
- **`[+]` / `[-]`**: Zoom in / out
- **`[Q]` / `[ESC]`**: Exit application

