#!/usr/bin/env python3
"""
HoloLLM: Interactive AR Transformer / LLM Architecture Visualizer
=================================================================
A single-file, production-ready AR application inspired by Iron-Man / Jarvis HUDs.
Uses OpenCV for real-time 3D perspective projection and MediaPipe Hands
(CNN-based hand landmark detector) for touchless holographic interaction.

Features:
- Dual MediaPipe backend support (modern Tasks API HandLandmarker & legacy Solutions API)
- Automatic download of Google's official hand_landmarker.task model
- 8 Distinct Transformer / LLM Architectural Layers with specialized 3D wireframes
- Pure NumPy vectorized 3D perspective projection engine with depth-sorted raycasting
- Single-hand pinch gesture for 3D orbit rotation (Pitch / Yaw)
- Two-hand pinch / palm distance for smooth Explode / Collapse expansion
- Index finger laser pointer for 3D layer touch inspection & collision detection
- Dynamic forward-pass token pulse dataflow animation
- Sci-Fi Jarvis HUD with neon bloom glow, holographic pedestal, ASCII tensor math, and layer directory
- Full mouse and keyboard fallback support (SPACE, R, A, 1-8, 0, +/-, Q, mouse drag/scroll)

Author: Senior Computer Vision & Deep Learning Engineer
"""

import os
import sys
import math
import time
import urllib.request
import cv2
import numpy as np

# Suppress TensorFlow / oneDNN info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import mediapipe as mp
except ImportError:
    print("Error: MediaPipe not found. Please install via: pip install mediapipe")
    sys.exit(1)


# =============================================================================
# 1. COLOR PALETTE & VISUAL CONSTANTS (JARVIS / SCI-FI CYBERNETIC THEME)
# =============================================================================
# OpenCV uses BGR ordering
COLOR_CYAN_GLOW      = (255, 235, 50)    # Primary Neon Cyan (Electric Blue)
COLOR_CYAN_DIM       = (180, 140, 20)    # Subdued Cyan for background grid/inactive
COLOR_GOLD_HOVER     = (20, 215, 255)    # Neon Gold / Amber for active/hovered layers
COLOR_GOLD_BRIGHT    = (50, 245, 255)    # Bright Gold accent
COLOR_ORANGE_ALERT   = (0, 140, 255)     # Warm Orange for grab/pinch tension
COLOR_WHITE_PULSE    = (255, 255, 255)   # Intense core particle / highlight
COLOR_HUD_BG         = (12, 16, 24)      # Dark translucent panel background
COLOR_HUD_BORDER     = (255, 210, 40)    # Sci-Fi panel border cyan
COLOR_HUD_TEXT       = (240, 245, 250)   # Crisp white-blue text
COLOR_GREEN_OK       = (80, 240, 100)    # Tracking Online Green
COLOR_MAGENTA_ROPE   = (255, 80, 220)    # Rotary Embedding accent


# Standard 21-hand landmark bone connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17)
]


# =============================================================================
# 2. TRANSFORMER / LLM LAYER EDUCATIONAL METADATA
# =============================================================================
LLM_LAYERS_DATA = [
    {
        "id": 1,
        "name": "INPUT & BPE TOKENIZER",
        "category": "TEXT ENCODING // VOCABULARY",
        "tensor_shape": "Text String -> [Batch, Seq_Len] (int64)",
        "formula": "T = BPE_Merge(UTF8_Bytes(text), Merge_Table, Vocab_V)",
        "role": "Subword Byte-Pair Encoding Tokenizer",
        "details": [
            "Deconstructs raw text into subword token IDs from vocabulary V ~ 128,256.",
            "Byte-level BPE guarantees zero out-of-vocabulary (OOV) tokens by falling",
            "back to raw byte sequences for rare or unseen character sequences."
        ]
    },
    {
        "id": 2,
        "name": "TOKEN EMBEDDING + RoPE",
        "category": "REPRESENTATION // ROTARY EMBEDDING",
        "tensor_shape": "[B, S] -> [B, S, d_model]  (e.g., d=4096)",
        "formula": "R_Theta^d * x_m,  where theta_i = 10000^(-2(i-1)/d)",
        "role": "High-Dim Vector Lookup & Rotary Positional Embedding",
        "details": [
            "Maps discrete token indices into dense continuous vector space.",
            "Applies Rotary Positional Embeddings (RoPE) by rotating 2D query/key",
            "coordinate slices in complex space, naturally encoding relative token distance."
        ]
    },
    {
        "id": 3,
        "name": "RMSNorm (PRE-NORMALIZATION)",
        "category": "STABILIZATION // LAYER PRE-NORM",
        "tensor_shape": "[B, S, d_model] -> [B, S, d_model]",
        "formula": "a_bar_i = (a_i / RMS(a)) * g_i,  RMS(a) = sqrt(1/d sum(a_i^2) + eps)",
        "role": "Root Mean Square Pre-Layer Normalization",
        "details": [
            "Rescales activations by their root-mean-square without mean-centering overhead.",
            "Pre-Norm architecture maintains stable unit variance across deep stacks,",
            "preventing vanishing or exploding gradients in 32 to 128+ layer architectures."
        ]
    },
    {
        "id": 4,
        "name": "QKV LINEAR PROJECTIONS",
        "category": "ATTENTION SUBSPACE MAPPING",
        "tensor_shape": "[B, S, d_model] -> Q, K, V in [B, n_heads, S, d_head]",
        "formula": "Q = X * W_Q,   K = X * W_K,   V = X * W_V",
        "role": "Query, Key, Value Projections & KV-Cache Management",
        "details": [
            "Linearly projects normalized token states into specialized subspace channels.",
            "Supports Grouped-Query Attention (GQA) where multiple query heads share key/value.",
            "In autoregressive inference, new K & V projections are cached into GPU VRAM."
        ]
    },
    {
        "id": 5,
        "name": "CAUSAL MULTI-HEAD ATTENTION",
        "category": "CORE CONTEXTUAL MIXING",
        "tensor_shape": "[B, n_heads, S, d_head] -> [B, S, d_model]",
        "formula": "Attention(Q,K,V) = softmax( (Q * K^T + M) / sqrt(d_k) ) * V",
        "role": "Scaled Dot-Product Self-Attention with Causal Mask",
        "details": [
            "Computes pairwise dot-product affinity between token Queries and Keys.",
            "Lower-triangular causal mask M forbids attention to future time steps (t' > t).",
            "Values (V) are weighted by softmax attention probabilities to synthesize context."
        ]
    },
    {
        "id": 6,
        "name": "RESIDUAL HIGHWAY + RMSNorm",
        "category": "SIGNAL HIGHWAY // RESIDUAL CONNECTION",
        "tensor_shape": "[B, S, d_model] -> [B, S, d_model]",
        "formula": "x_(l+1) = x_l + Attention(RMSNorm(x_l))",
        "role": "Identity Skip Connection & Pre-FFN Normalization",
        "details": [
            "Preserves an unobstructed information highway along the entire model depth.",
            "Prevents gradient attenuation and allows individual layers to compute",
            "incremental residual updates rather than reconstructing full representations."
        ]
    },
    {
        "id": 7,
        "name": "SwiGLU FEED-FORWARD (MLP)",
        "category": "PARAMETRIC KNOWLEDGE NETWORK",
        "tensor_shape": "[B, S, d] -> [B, S, (8/3)d] -> [B, S, d]",
        "formula": "SwiGLU(x) = ( Swish(x * W_gate) (x * W_up) ) * W_down",
        "role": "Position-wise Gated Non-Linear Knowledge Store",
        "details": [
            "Expands token vectors into a wide intermediate dimension (e.g. 14,336 dims).",
            "Gated Swish non-linearity acts as key-value memory retrieving factual",
            "associations and reasoning patterns learned during large-scale pretraining."
        ]
    },
    {
        "id": 8,
        "name": "LM HEAD & SOFTMAX LOGITS",
        "category": "OUTPUT PROJECTION // NEXT-TOKEN SAMPLING",
        "tensor_shape": "[B, S, d_model] -> [B, S, Vocab_Size]",
        "formula": "P(w_t | w_<t) = softmax( Logits / Temperature )",
        "role": "Unembedding Matrix & Autoregressive Sampling",
        "details": [
            "Projects final hidden states back to vocabulary logit distribution.",
            "Applies Temperature scaling, Top-p (nucleus), and Top-k filtering to sample",
            "the next token ID, which is fed back into Layer 1 for autoregressive generation."
        ]
    }
]


# =============================================================================
# 3. 3D PERSPECTIVE PROJECTION & CAMERA ENGINE
# =============================================================================
class Camera3D:
    """Vectorized 3D perspective camera with Euler rotation, translation, and depth clipping."""

    def __init__(self, focal_length=680.0, cam_distance=620.0):
        self.focal_length = focal_length
        self.cam_dist = cam_distance
        # Initial angles for an appealing 3D isometric viewpoint
        self.pitch = 0.28   # Pitch around X-axis (radians)
        self.yaw = -0.42    # Yaw around Y-axis (radians)
        self.roll = 0.0     # Roll around Z-axis (radians)
        
        # Smooth interpolation targets
        self.target_pitch = self.pitch
        self.target_yaw = self.yaw
        self.target_dist = self.cam_dist
        
        # Translation offsets
        self.tx = 0.0
        self.ty = 0.0

    def update(self, dt=0.033):
        """Smoothly dampen / lerp current angles toward target angles."""
        alpha = min(1.0, 10.0 * dt)
        self.pitch += (self.target_pitch - self.pitch) * alpha
        self.yaw   += (self.target_yaw - self.yaw) * alpha
        self.cam_dist += (self.target_dist - self.cam_dist) * alpha

    def get_rotation_matrix(self):
        """Construct composite rotation matrix R = Ry * Rx * Rz."""
        cx, sx = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cz, sz = math.cos(self.roll), math.sin(self.roll)

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cx,  -sx],
            [0.0, sx,   cx]
        ], dtype=np.float32)

        Ry = np.array([
            [cy,  0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy]
        ], dtype=np.float32)

        Rz = np.array([
            [cz, -sz, 0.0],
            [sz,  cz, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        return Ry @ Rx @ Rz

    def project_points(self, points_3d, screen_w, screen_h):
        """
        Projects an (N, 3) array of 3D world coordinates to (N, 2) 2D pixel coordinates.
        Returns:
            points_2d: (N, 2) array of (u, v) pixel coordinates (int32)
            depths: (N,) depth array z_cam for depth sorting
            valid_mask: (N,) boolean mask indicating points in front of camera
        """
        if len(points_3d) == 0:
            return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)

        R = self.get_rotation_matrix()
        # Rotate points: (N, 3) @ (3, 3).T
        rot_pts = points_3d @ R.T

        # Camera space coordinates
        x_cam = rot_pts[:, 0] + self.tx
        y_cam = rot_pts[:, 1] + self.ty
        z_cam = rot_pts[:, 2] + self.cam_dist

        # Valid points are strictly in front of the camera plane
        valid = z_cam > 25.0
        safe_z = np.where(valid, z_cam, 1000.0)

        cx = screen_w * 0.5
        cy = screen_h * 0.5

        # Invert y because screen coordinates point downwards
        u = (self.focal_length * x_cam / safe_z) + cx
        v = (-self.focal_length * y_cam / safe_z) + cy

        points_2d = np.stack([u, v], axis=-1).astype(np.int32)
        return points_2d, z_cam, valid


# =============================================================================
# 4. 3D WIREFRAME MESH GENERATOR (8 TRANSFORMER LAYERS & PEDESTAL)
# =============================================================================
class TransformerHologram:
    """Generates and manages the 3D wireframe geometry for the 8 LLM layers."""

    def __init__(self):
        self.num_layers = 8
        self.expansion_factor = 0.0      # 0.0 = compact core, 1.0 = fully exploded
        self.target_expansion = 0.0
        self.active_layer_idx = None     # Currently hovered / selected layer (0-7)
        self.locked_layer_idx = None     # User-pinned layer via keyboard / tap
        
        # Dimensions of each layer plate (in world 3D units)
        self.plate_w = 150.0  # Width (X-axis)
        self.plate_d = 130.0  # Depth (Z-axis)
        self.plate_h = 10.0   # Thickness (Y-axis)
        
        self.collapsed_spacing = 15.0
        self.expanded_spacing = 68.0

        # Dataflow particles travelling through layers
        self.num_particles = 28
        self.particles = []
        for _ in range(self.num_particles):
            self.particles.append({
                "t": np.random.uniform(0.0, 1.0),
                "speed": np.random.uniform(0.12, 0.32),
                "rail": np.random.choice([0, 1, 2, 3, 4]), # 0=center, 1-4=corners
                "offset": np.random.uniform(-4.0, 4.0, size=2)
            })

    def update_physics(self, dt=0.033):
        """Update expansion lerp and advance dataflow particles."""
        alpha = min(1.0, 8.0 * dt)
        self.expansion_factor += (self.target_expansion - self.expansion_factor) * alpha

        # Advance particles
        for p in self.particles:
            p["t"] += p["speed"] * dt
            if p["t"] > 1.0:
                p["t"] = 0.0
                p["speed"] = np.random.uniform(0.14, 0.36)
                p["rail"] = np.random.choice([0, 1, 2, 3, 4])

    def get_layer_y(self, idx):
        """Returns the central Y coordinate for layer `idx` based on expansion."""
        spacing = self.collapsed_spacing + self.expansion_factor * (self.expanded_spacing - self.collapsed_spacing)
        # Center the stack vertically: idx 0 is bottom, idx 7 is top
        return (idx - 3.5) * spacing

    def build_layer_hull_vertices(self, idx):
        """Returns the 8 bounding corners of a layer plate for 3D drawing & 2D hit testing."""
        cy = self.get_layer_y(idx)
        hw = self.plate_w * 0.5
        hd = self.plate_d * 0.5
        hh = self.plate_h * 0.5

        # Bottom 4 corners, then Top 4 corners
        corners = np.array([
            [-hw, cy - hh, -hd],
            [ hw, cy - hh, -hd],
            [ hw, cy - hh,  hd],
            [-hw, cy - hh,  hd],
            [-hw, cy + hh, -hd],
            [ hw, cy + hh, -hd],
            [ hw, cy + hh,  hd],
            [-hw, cy + hh,  hd],
        ], dtype=np.float32)
        return corners

    def build_holographic_pedestal(self):
        """Builds concentric circular emitter rings and compass ticks at the hologram base."""
        y_base = self.get_layer_y(0) - 45.0
        segments = []
        num_segments = 24

        for radius in [90.0, 140.0, 180.0]:
            for k in range(num_segments):
                a1 = (2.0 * math.pi * k) / num_segments
                a2 = (2.0 * math.pi * (k + 1)) / num_segments
                pA = np.array([radius * math.cos(a1), y_base, radius * math.sin(a1)])
                pB = np.array([radius * math.cos(a2), y_base, radius * math.sin(a2)])
                segments.append((pA, pB, "dim"))

        # Radial compass ticks
        for a_deg in range(0, 360, 45):
            rad = math.radians(a_deg)
            p1 = np.array([140.0 * math.cos(rad), y_base, 140.0 * math.sin(rad)])
            p2 = np.array([180.0 * math.cos(rad), y_base, 180.0 * math.sin(rad)])
            segments.append((p1, p2, "accent"))

        return segments

    def build_layer_internal_geometry(self, idx):
        """
        Builds custom, mathematically descriptive wireframe lines for each layer.
        Returns a list of line segments: [ (ptA_3d, ptB_3d, style_tag), ... ]
        """
        cy = self.get_layer_y(idx)
        segments = []
        hw = self.plate_w * 0.5
        hd = self.plate_d * 0.5

        # -------------------------------------------------------------
        # LAYER 1: INPUT & BPE TOKENIZER (Grid of discrete subword blocks)
        # -------------------------------------------------------------
        if idx == 0:
            grid_cols, grid_rows = 4, 3
            bw = (self.plate_w * 0.75) / grid_cols
            bd = (self.plate_d * 0.70) / grid_rows
            start_x = - (grid_cols * bw) * 0.5
            start_z = - (grid_rows * bd) * 0.5

            for r in range(grid_rows):
                for c in range(grid_cols):
                    bx = start_x + c * bw + 2.0
                    bz = start_z + r * bd + 2.0
                    w_sub = bw - 4.0
                    d_sub = bd - 4.0
                    h_sub = 4.0 + (math.sin(c * 1.5 + r * 2.0) * 2.5)

                    p1 = [bx, cy - h_sub, bz]
                    p2 = [bx + w_sub, cy - h_sub, bz]
                    p3 = [bx + w_sub, cy - h_sub, bz + d_sub]
                    p4 = [bx, cy - h_sub, bz + d_sub]
                    p5 = [bx, cy + h_sub, bz]
                    p6 = [bx + w_sub, cy + h_sub, bz]
                    p7 = [bx + w_sub, cy + h_sub, bz + d_sub]
                    p8 = [bx, cy + h_sub, bz + d_sub]

                    for a, b in [(p1, p2), (p2, p3), (p3, p4), (p4, p1),
                                 (p5, p6), (p6, p7), (p7, p8), (p8, p5),
                                 (p1, p5), (p2, p6), (p3, p7), (p4, p8)]:
                        segments.append((np.array(a), np.array(b), "core"))

            # BPE Split Line cutting across the middle
            segments.append((np.array([-hw * 0.85, cy, 0.0]), np.array([hw * 0.85, cy, 0.0]), "accent"))

        # -------------------------------------------------------------
        # LAYER 2: TOKEN EMBEDDING + RoPE (Matrix grid + Sinusoidal Helix)
        # -------------------------------------------------------------
        elif idx == 1:
            # 2D Coordinate Grid
            for gx in np.linspace(-hw * 0.75, hw * 0.75, 5):
                segments.append((np.array([gx, cy, -hd * 0.75]), np.array([gx, cy, hd * 0.75]), "dim"))
            for gz in np.linspace(-hd * 0.75, hd * 0.75, 5):
                segments.append((np.array([-hw * 0.75, cy, gz]), np.array([hw * 0.75, cy, gz]), "dim"))

            # 3D Rotary Positional Embedding Spiral Helix
            num_helix_pts = 36
            theta_vals = np.linspace(-math.pi * 3.0, math.pi * 3.0, num_helix_pts)
            rx = hw * 0.65
            rz = hd * 0.65
            for k in range(num_helix_pts - 1):
                t1 = theta_vals[k]
                t2 = theta_vals[k + 1]
                pA = np.array([rx * (t1 / (math.pi * 3.0)), cy + math.sin(t1 * 2.0) * 8.0, rz * math.cos(t1 * 1.5)])
                pB = np.array([rx * (t2 / (math.pi * 3.0)), cy + math.sin(t2 * 2.0) * 8.0, rz * math.cos(t2 * 1.5)])
                segments.append((pA, pB, "accent"))

        # -------------------------------------------------------------
        # LAYER 3: RMSNorm (PRE-NORM) (Concentric Stabilizing Rings & Crosshair)
        # -------------------------------------------------------------
        elif idx == 2:
            radii = [18.0, 36.0, 54.0]
            num_ring_pts = 16
            for r in radii:
                for k in range(num_ring_pts):
                    a1 = (2.0 * math.pi * k) / num_ring_pts
                    a2 = (2.0 * math.pi * (k + 1)) / num_ring_pts
                    pA = np.array([r * math.cos(a1), cy, r * math.sin(a1)])
                    pB = np.array([r * math.cos(a2), cy, r * math.sin(a2)])
                    segments.append((pA, pB, "core"))

            # Crosshair stabilizing spokes
            segments.append((np.array([-58.0, cy, 0.0]), np.array([58.0, cy, 0.0]), "dim"))
            segments.append((np.array([0.0, cy, -58.0]), np.array([0.0, cy, 58.0]), "dim"))
            # Central diamond
            dm = 10.0
            segments.append((np.array([-dm, cy, 0.0]), np.array([0.0, cy, dm]), "accent"))
            segments.append((np.array([0.0, cy, dm]), np.array([dm, cy, 0.0]), "accent"))
            segments.append((np.array([dm, cy, 0.0]), np.array([0.0, cy, -dm]), "accent"))
            segments.append((np.array([0.0, cy, -dm]), np.array([-dm, cy, 0.0]), "accent"))

        # -------------------------------------------------------------
        # LAYER 4: QKV LINEAR PROJECTIONS (3 Parallel Sub-Slabs: Q, K, V)
        # -------------------------------------------------------------
        elif idx == 3:
            slab_xs = [-45.0, 0.0, 45.0]
            sw = 30.0
            sd = hd * 1.4
            sh = 6.0
            for sx in slab_xs:
                # Sub-plate box
                c1 = np.array([sx - sw * 0.5, cy - sh, -sd * 0.5])
                c2 = np.array([sx + sw * 0.5, cy - sh, -sd * 0.5])
                c3 = np.array([sx + sw * 0.5, cy - sh,  sd * 0.5])
                c4 = np.array([sx - sw * 0.5, cy - sh,  sd * 0.5])
                c5 = np.array([sx - sw * 0.5, cy + sh, -sd * 0.5])
                c6 = np.array([sx + sw * 0.5, cy + sh, -sd * 0.5])
                c7 = np.array([sx + sw * 0.5, cy + sh,  sd * 0.5])
                c8 = np.array([sx - sw * 0.5, cy + sh,  sd * 0.5])

                for a, b in [(c1, c2), (c2, c3), (c3, c4), (c4, c1),
                             (c5, c6), (c6, c7), (c7, c8), (c8, c5),
                             (c1, c5), (c2, c6), (c3, c7), (c4, c8)]:
                    segments.append((a, b, "core"))

                # Diverging ray from input origin below
                input_root = np.array([0.0, cy - self.collapsed_spacing * 0.8, 0.0])
                segments.append((input_root, np.array([sx, cy - sh, 0.0]), "accent"))

        # -------------------------------------------------------------
        # LAYER 5: CAUSAL MULTI-HEAD ATTENTION (Triangular Mask & Arcs)
        # -------------------------------------------------------------
        elif idx == 4:
            # Triangular attention matrix grid
            grid_n = 5
            spacing_x = (hw * 1.4) / grid_n
            spacing_z = (hd * 1.4) / grid_n
            x_min = - (hw * 1.4) * 0.5
            z_min = - (hd * 1.4) * 0.5

            for r in range(grid_n):
                for c in range(grid_n):
                    px = x_min + c * spacing_x
                    pz = z_min + r * spacing_z
                    if c <= r:
                        # Unmasked attention cell
                        cell_w = spacing_x * 0.8
                        cell_d = spacing_z * 0.8
                        p1 = np.array([px, cy, pz])
                        p2 = np.array([px + cell_w, cy, pz])
                        p3 = np.array([px + cell_w, cy, pz + cell_d])
                        p4 = np.array([px, cy, pz + cell_d])
                        segments.append((p1, p2, "core"))
                        segments.append((p2, p3, "core"))
                        segments.append((p3, p4, "core"))
                        segments.append((p4, p1, "core"))
                    else:
                        # Causal Masked cell (Drawn with dim 'X')
                        segments.append((np.array([px, cy, pz]), np.array([px + spacing_x * 0.5, cy, pz + spacing_z * 0.5]), "dim"))

            # Arched cross-attention curves
            for arc_k in [-35.0, 0.0, 35.0]:
                pA = np.array([arc_k, cy, -hd * 0.6])
                pB = np.array([arc_k, cy + 12.0, 0.0])
                pC = np.array([arc_k, cy, hd * 0.6])
                segments.append((pA, pB, "accent"))
                segments.append((pB, pC, "accent"))

        # -------------------------------------------------------------
        # LAYER 6: RESIDUAL STREAM + RMSNorm (Bypass Highways + Add Circle)
        # -------------------------------------------------------------
        elif idx == 5:
            # Inner plate frame
            in_w, in_d = hw * 0.7, hd * 0.7
            segments.append((np.array([-in_w, cy, -in_d]), np.array([in_w, cy, -in_d]), "dim"))
            segments.append((np.array([in_w, cy, -in_d]), np.array([in_w, cy, in_d]), "dim"))
            segments.append((np.array([in_w, cy, in_d]), np.array([-in_w, cy, in_d]), "dim"))
            segments.append((np.array([-in_w, cy, in_d]), np.array([-in_w, cy, -in_d]), "dim"))

            # Twin bypass highway arches looping wide outside the layer
            arch_w = hw * 1.35
            for sign in [-1.0, 1.0]:
                p_start = np.array([sign * hw * 0.6, cy - 22.0, 0.0])
                p_mid   = np.array([sign * arch_w, cy, 0.0])
                p_end   = np.array([sign * hw * 0.6, cy + 22.0, 0.0])
                segments.append((p_start, p_mid, "accent"))
                segments.append((p_mid, p_end, "accent"))

            # Summation (+) node circle
            sum_r = 12.0
            for k in range(8):
                a1 = (2.0 * math.pi * k) / 8.0
                a2 = (2.0 * math.pi * (k + 1)) / 8.0
                segments.append((
                    np.array([sum_r * math.cos(a1), cy, sum_r * math.sin(a1)]),
                    np.array([sum_r * math.cos(a2), cy, sum_r * math.sin(a2)]),
                    "accent"
                ))
            segments.append((np.array([-sum_r * 0.7, cy, 0.0]), np.array([sum_r * 0.7, cy, 0.0]), "accent"))
            segments.append((np.array([0.0, cy, -sum_r * 0.7]), np.array([0.0, cy, sum_r * 0.7]), "accent"))

        # -------------------------------------------------------------
        # LAYER 7: SwiGLU FEED-FORWARD (Double-Hourglass / Diamond Expansion)
        # -------------------------------------------------------------
        elif idx == 6:
            # Bottom neck (d_model)
            bn = 26.0
            # Middle expansion (4 * d_model)
            mw = hw * 1.3
            md = hd * 1.3
            # Top contraction (d_model)
            tn = 26.0

            h_half = 14.0
            bot = [
                np.array([-bn, cy - h_half, -bn]), np.array([bn, cy - h_half, -bn]),
                np.array([bn, cy - h_half, bn]), np.array([-bn, cy - h_half, bn])
            ]
            mid = [
                np.array([-mw, cy, -md]), np.array([mw, cy, -md]),
                np.array([mw, cy, md]), np.array([-mw, cy, md])
            ]
            top = [
                np.array([-tn, cy + h_half, -tn]), np.array([tn, cy + h_half, -tn]),
                np.array([tn, cy + h_half, tn]), np.array([-tn, cy + h_half, tn])
            ]

            # Bottom ring
            for i in range(4):
                segments.append((bot[i], bot[(i + 1) % 4], "core"))
            # Middle ring
            for i in range(4):
                segments.append((mid[i], mid[(i + 1) % 4], "accent"))
            # Top ring
            for i in range(4):
                segments.append((top[i], top[(i + 1) % 4], "core"))
            # Expansion & Contraction struts
            for i in range(4):
                segments.append((bot[i], mid[i], "core"))
                segments.append((mid[i], top[i], "core"))

        # -------------------------------------------------------------
        # LAYER 8: LM HEAD & SOFTMAX LOGITS (Funnel + 3D Bar-Chart Histogram)
        # -------------------------------------------------------------
        elif idx == 7:
            # Converging Funnel Lines
            fh = 8.0
            segments.append((np.array([-hw * 0.8, cy - fh, -hd * 0.8]), np.array([-25.0, cy, -25.0]), "dim"))
            segments.append((np.array([ hw * 0.8, cy - fh, -hd * 0.8]), np.array([ 25.0, cy, -25.0]), "dim"))
            segments.append((np.array([ hw * 0.8, cy - fh,  hd * 0.8]), np.array([ 25.0, cy,  25.0]), "dim"))
            segments.append((np.array([-hw * 0.8, cy - fh,  hd * 0.8]), np.array([-25.0, cy,  25.0]), "dim"))

            # 3D Bar-Chart Logits Histogram
            bar_heights = [10.0, 24.0, 8.0, 36.0, 16.0]
            num_bars = len(bar_heights)
            bar_w = 8.0
            bar_d = 8.0
            spacing_b = 14.0
            bx_start = - ((num_bars - 1) * spacing_b) * 0.5

            for b_i, bh in enumerate(bar_heights):
                bx = bx_start + b_i * spacing_b
                p1 = np.array([bx - bar_w * 0.5, cy, -bar_d * 0.5])
                p2 = np.array([bx + bar_w * 0.5, cy, -bar_d * 0.5])
                p3 = np.array([bx + bar_w * 0.5, cy,  bar_d * 0.5])
                p4 = np.array([bx - bar_w * 0.5, cy,  bar_d * 0.5])

                p5 = np.array([bx - bar_w * 0.5, cy + bh, -bar_d * 0.5])
                p6 = np.array([bx + bar_w * 0.5, cy + bh, -bar_d * 0.5])
                p7 = np.array([bx + bar_w * 0.5, cy + bh,  bar_d * 0.5])
                p8 = np.array([bx - bar_w * 0.5, cy + bh,  bar_d * 0.5])

                style = "accent" if b_i == 3 else "core" # b_i == 3 is the argmax predicted token!
                for a, b in [(p1, p2), (p2, p3), (p3, p4), (p4, p1),
                             (p5, p6), (p6, p7), (p7, p8), (p8, p5),
                             (p1, p5), (p2, p6), (p3, p7), (p4, p8)]:
                    segments.append((a, b, style))

        return segments

    def build_spine_rails(self):
        """Builds longitudinal backbone rails connecting all 8 layers."""
        y_bot = self.get_layer_y(0) - self.plate_h * 0.5
        y_top = self.get_layer_y(7) + self.plate_h * 0.5
        hw = self.plate_w * 0.5
        hd = self.plate_d * 0.5

        rails = [
            # 4 outer corner rails
            (np.array([-hw, y_bot, -hd]), np.array([-hw, y_top, -hd])),
            (np.array([ hw, y_bot, -hd]), np.array([ hw, y_top, -hd])),
            (np.array([ hw, y_bot,  hd]), np.array([ hw, y_top,  hd])),
            (np.array([-hw, y_bot,  hd]), np.array([-hw, y_top,  hd])),
            # Central dataflow core spine
            (np.array([0.0, y_bot, 0.0]), np.array([0.0, y_top, 0.0]))
        ]
        return rails

    def get_particle_positions_3d(self):
        """Returns 3D coordinates for all active dataflow particles."""
        y_bot = self.get_layer_y(0)
        y_top = self.get_layer_y(7)
        hw = self.plate_w * 0.5
        hd = self.plate_d * 0.5

        corner_offsets = [
            np.array([0.0, 0.0, 0.0]),
            np.array([-hw, 0.0, -hd]),
            np.array([ hw, 0.0, -hd]),
            np.array([ hw, 0.0,  hd]),
            np.array([-hw, 0.0,  hd]),
        ]

        pts_3d = []
        for p in self.particles:
            cur_y = y_bot + p["t"] * (y_top - y_bot)
            base_offset = corner_offsets[p["rail"]]
            pt = base_offset + np.array([p["offset"][0], cur_y, p["offset"][1]], dtype=np.float32)
            pts_3d.append(pt)

        return np.array(pts_3d, dtype=np.float32)


# =============================================================================
# 5. MEDIAPIPE CNN HAND GESTURE CONTROLLER (DUAL BACKEND)
# =============================================================================
class HandGestureTracker:
    """
    Robust MediaPipe Hands wrapper supporting both:
    1. MediaPipe Tasks API (mediapipe.tasks.python.vision.HandLandmarker)
    2. Legacy MediaPipe Solutions API (mediapipe.solutions.hands)
    Applies Exponential Moving Average (EMA) filtering for rock-solid tracking.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    def __init__(self):
        self.backend = None
        self.detector = None
        self.prev_landmarks = {}
        self.alpha_ema = 0.65  # 0.65 current, 0.35 previous

        # Gesture State Machine
        self.state_label = "IDLE / SEARCHING"
        self.is_pinching = False
        self.pinch_pos_2d = None
        self.pointer_pos_2d = None
        self.two_hand_dist = None
        self.hand_data_list = []

        self._init_mediapipe()

    def _ensure_task_model(self):
        """Downloads official hand_landmarker.task if not present locally."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        task_path = os.path.join(script_dir, "hand_landmarker.task")
        if not os.path.exists(task_path):
            print(f"[HoloLLM] Downloading hand_landmarker.task (~7.8MB) from Google Cloud...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, task_path)
                print("[HoloLLM] Model download complete!")
            except Exception as e:
                print(f"[HoloLLM] Error downloading model: {e}")
                return None
        return task_path

    def _init_mediapipe(self):
        """Initializes either Tasks API or Solutions API based on runtime availability."""
        # Try MediaPipe Tasks API first (MediaPipe 0.10.x+)
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            task_file = self._ensure_task_model()
            if task_file and os.path.exists(task_file):
                base_options = python.BaseOptions(model_asset_path=task_file)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2,
                    min_hand_detection_confidence=0.70,
                    min_hand_presence_confidence=0.65,
                    running_mode=vision.RunningMode.IMAGE
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.backend = "tasks"
                print("[HoloLLM] Initialized MediaPipe Vision Tasks HandLandmarker.")
                return
        except Exception as e:
            print(f"[HoloLLM] Tasks API initialization failed: {e}. Trying Solutions API...")

        # Try Legacy MediaPipe Solutions API (MediaPipe < 0.10.x)
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.72,
                    min_tracking_confidence=0.65
                )
                self.backend = "solutions"
                print("[HoloLLM] Initialized MediaPipe Legacy Solutions Hands.")
                return
        except Exception as e:
            print(f"[HoloLLM] Solutions API initialization failed: {e}")

        print("[HoloLLM WARNING] No hardware CNN hand tracker available. Running with Mouse & Keyboard controls.")
        self.backend = None

    def close(self):
        """Safely closes MediaPipe detector to prevent process shutdown warnings."""
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass

    def process_frame(self, frame_bgr):
        """Extracts 21 3D landmarks per hand and detects gestures."""
        h, w = frame_bgr.shape[:2]
        self.hand_data_list = []
        self.is_pinching = False
        self.pinch_pos_2d = None
        self.pointer_pos_2d = None
        self.two_hand_dist = None

        if self.backend is None:
            self.state_label = "DESKTOP MODE [MOUSE CONTROLS]"
            return

        # -------------------------------------------------------------
        # 1. RUN INFERENCE ACCORDING TO ACTIVE BACKEND
        # -------------------------------------------------------------
        detected_hands_raw = []

        if self.backend == "tasks":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = self.detector.detect(mp_image)

            if results and results.hand_landmarks:
                for idx, lms in enumerate(results.hand_landmarks):
                    raw_pts = [[lm.x * w, lm.y * h, lm.z * w] for lm in lms]
                    label = "Hand"
                    score = 0.95
                    if idx < len(results.handedness) and len(results.handedness[idx]) > 0:
                        cat = results.handedness[idx][0]
                        label = cat.category_name or cat.display_name or "Hand"
                        score = float(cat.score)
                    detected_hands_raw.append((raw_pts, label, score))

        elif self.backend == "solutions":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.detector.process(frame_rgb)

            if results and results.multi_hand_landmarks:
                for idx, (lms, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    raw_pts = [[lm.x * w, lm.y * h, lm.z * w] for lm in lms.landmark]
                    label = handedness.classification[0].label
                    score = handedness.classification[0].score
                    detected_hands_raw.append((raw_pts, label, score))

        if not detected_hands_raw:
            self.state_label = "SCANNING FOR HANDS"
            return

        # -------------------------------------------------------------
        # 2. APPLY EMA FILTERING & EXTRACT GESTURES
        # -------------------------------------------------------------
        for i, (raw_pts, label, score) in enumerate(detected_hands_raw):
            raw_pts = np.array(raw_pts, dtype=np.float32)

            hand_key = f"{label}_{i}"
            if hand_key in self.prev_landmarks:
                smoothed_pts = self.alpha_ema * raw_pts + (1.0 - self.alpha_ema) * self.prev_landmarks[hand_key]
            else:
                smoothed_pts = raw_pts
            self.prev_landmarks[hand_key] = smoothed_pts

            # 4: Thumb Tip, 8: Index Tip, 0: Wrist
            thumb_tip = smoothed_pts[4, :2]
            index_tip = smoothed_pts[8, :2]
            wrist     = smoothed_pts[0, :2]

            pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
            is_hand_pinched = pinch_dist < 44.0

            index_len = float(np.linalg.norm(index_tip - wrist))
            middle_len = float(np.linalg.norm(smoothed_pts[12, :2] - wrist))
            ring_len = float(np.linalg.norm(smoothed_pts[16, :2] - wrist))
            is_index_pointing = (not is_hand_pinched) and (index_len > middle_len * 1.15) and (index_len > ring_len * 1.25)

            self.hand_data_list.append({
                "label": label,
                "score": score,
                "landmarks": smoothed_pts,
                "thumb_tip": thumb_tip,
                "index_tip": index_tip,
                "wrist": wrist,
                "pinch_dist": pinch_dist,
                "is_pinched": is_hand_pinched,
                "is_pointing": is_index_pointing
            })

        # -------------------------------------------------------------
        # 3. INTERPRET SINGLE & TWO-HANDED GESTURES
        # -------------------------------------------------------------
        detected_count = len(self.hand_data_list)

        if detected_count == 1:
            h0 = self.hand_data_list[0]
            if h0["is_pinched"]:
                self.is_pinching = True
                self.pinch_pos_2d = h0["index_tip"]
                self.state_label = "SINGLE-HAND PINCH [ORBIT ROTATION]"
            elif h0["is_pointing"]:
                self.pointer_pos_2d = h0["index_tip"]
                self.state_label = "INDEX POINTER [TOUCH INSPECTING]"
            else:
                self.pointer_pos_2d = h0["index_tip"]
                self.state_label = "HAND DETECTED [HOVER]"

        elif detected_count >= 2:
            h0, h1 = self.hand_data_list[0], self.hand_data_list[1]
            dist_hands = float(np.linalg.norm(h0["index_tip"] - h1["index_tip"]))
            self.two_hand_dist = dist_hands

            # Allow rotation if one hand is pinched
            if h0["is_pinched"] and not h1["is_pinched"]:
                self.is_pinching = True
                self.pinch_pos_2d = h0["index_tip"]
            elif h1["is_pinched"] and not h0["is_pinched"]:
                self.is_pinching = True
                self.pinch_pos_2d = h1["index_tip"]

            if h0["is_pointing"]:
                self.pointer_pos_2d = h0["index_tip"]
            elif h1["is_pointing"]:
                self.pointer_pos_2d = h1["index_tip"]
            else:
                self.pointer_pos_2d = h0["index_tip"]

            self.state_label = f"TWO-HAND EXPAND METER [D: {int(dist_hands)}px]"


# =============================================================================
# 6. JARVIS / SCI-FI HUD & TELEMETRY RENDERER
# =============================================================================
class JarvisHUDRenderer:
    """Renders Iron-Man/Jarvis HUD graphics, holographic panels, ASCII math, and glows."""

    @staticmethod
    def draw_chamfered_panel(img, x, y, w, h, chamfer=16, color_bg=(15, 20, 28), color_border=COLOR_HUD_BORDER, alpha=0.78):
        """Draws a futuristic translucent panel with angled chamfered corners."""
        pts = np.array([
            [x + chamfer, y],
            [x + w - chamfer, y],
            [x + w, y + chamfer],
            [x + w, y + h - chamfer],
            [x + w - chamfer, y + h],
            [x + chamfer, y + h],
            [x, y + h - chamfer],
            [x, y + chamfer]
        ], dtype=np.int32)

        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color_bg)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        cv2.polylines(img, [pts], True, color_border, 1, cv2.LINE_AA)

        # High-tech corner bracket accents
        arm = 12
        cv2.line(img, (x, y + chamfer), (x, y + chamfer + arm), COLOR_CYAN_GLOW, 2, cv2.LINE_AA)
        cv2.line(img, (x + w, y + chamfer), (x + w, y + chamfer + arm), COLOR_CYAN_GLOW, 2, cv2.LINE_AA)
        cv2.line(img, (x, y + h - chamfer), (x, y + h - chamfer - arm), COLOR_CYAN_GLOW, 2, cv2.LINE_AA)
        cv2.line(img, (x + w, y + h - chamfer), (x + w, y + h - chamfer - arm), COLOR_CYAN_GLOW, 2, cv2.LINE_AA)

    @staticmethod
    def draw_top_status_bar(img, fps, state_text, expansion_pct, tracking_online=True):
        """Draws top-left system telemetry: FPS, CNN status, and expansion bar."""
        x, y, w, h = 20, 20, 420, 105
        JarvisHUDRenderer.draw_chamfered_panel(img, x, y, w, h, chamfer=12, color_bg=(10, 15, 24), color_border=COLOR_CYAN_DIM)

        # Title
        cv2.putText(img, "JARVIS NEURAL CORE // HOLO-LLM v2.4", (x + 16, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GOLD_BRIGHT, 2, cv2.LINE_AA)

        # Status & FPS
        status_color = COLOR_GREEN_OK if tracking_online else COLOR_ORANGE_ALERT
        status_str = "ONLINE (21 LNDMK)" if tracking_online else "OFFLINE // MOUSE FALLBACK"
        cv2.circle(img, (x + 22, y + 50), 4, status_color, -1, cv2.LINE_AA)
        cv2.putText(img, f"HAND CNN: {status_str}", (x + 34, y + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_HUD_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, f"FPS: {fps:4.1f}", (x + w - 85, y + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN_GLOW, 1, cv2.LINE_AA)

        # Gesture State
        cv2.putText(img, f"STATE: {state_text[:30]}", (x + 16, y + 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GOLD_HOVER, 1, cv2.LINE_AA)

        # Expansion Progress Bar
        bar_x, bar_y, bar_w, bar_h = x + 16, y + 84, w - 32, 8
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 45, 60), 1)
        fill_w = int(bar_w * (expansion_pct / 100.0))
        if fill_w > 0:
            cv2.rectangle(img, (bar_x + 1, bar_y + 1), (bar_x + fill_w, bar_y + bar_h - 1), COLOR_CYAN_GLOW, -1)
        cv2.putText(img, f"EXPANSION: {int(expansion_pct)}%", (x + w - 125, y + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_CYAN_GLOW, 1, cv2.LINE_AA)

    @staticmethod
    def draw_top_right_camera_telemetry(img, pitch_deg, yaw_deg, zoom_factor):
        """Draws top-right orientation & spatial telemetry."""
        h_img, w_img = img.shape[:2]
        w, h = 260, 68
        x = w_img - w - 20
        y = 20
        JarvisHUDRenderer.draw_chamfered_panel(img, x, y, w, h, chamfer=10, color_bg=(10, 15, 24), color_border=COLOR_CYAN_DIM)

        cv2.putText(img, "SPATIAL TELEMETRY", (x + 14, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GOLD_BRIGHT, 1, cv2.LINE_AA)
        cv2.putText(img, f"ROTATION: P {pitch_deg:+5.1f} | Y {yaw_deg:+5.1f}", (x + 14, y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_HUD_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, f"OPTICAL ZOOM: {zoom_factor:3.1f}x", (x + 14, y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_CYAN_GLOW, 1, cv2.LINE_AA)

    @staticmethod
    def draw_right_layer_directory(img, active_idx):
        """Draws the vertical directory of the 8 Transformer layers on the right screen border."""
        h_img, w_img = img.shape[:2]
        w, h = 240, 275
        x = w_img - w - 20
        y = 100
        JarvisHUDRenderer.draw_chamfered_panel(img, x, y, w, h, chamfer=12, color_bg=(10, 15, 24), color_border=COLOR_CYAN_DIM)

        cv2.putText(img, "TRANSFORMER STACK (8)", (x + 16, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_GOLD_BRIGHT, 1, cv2.LINE_AA)
        cv2.line(img, (x + 16, y + 30), (x + w - 16, y + 30), COLOR_CYAN_DIM, 1)

        start_y = y + 52
        step_y = 26
        for i in range(8):
            is_active = (active_idx == i)
            layer_title = LLM_LAYERS_DATA[i]["name"]
            ly = start_y + i * step_y

            if is_active:
                # Active highlighted background strip
                cv2.rectangle(img, (x + 10, ly - 16), (x + w - 10, ly + 6), (40, 60, 80), -1)
                cv2.rectangle(img, (x + 10, ly - 16), (x + w - 10, ly + 6), COLOR_GOLD_HOVER, 1)
                dot_color = COLOR_GOLD_BRIGHT
                text_color = COLOR_GOLD_BRIGHT
                prefix = ">"
            else:
                dot_color = COLOR_CYAN_DIM
                text_color = COLOR_HUD_TEXT
                prefix = " "

            cv2.circle(img, (x + 20, ly - 5), 3, dot_color, -1, cv2.LINE_AA)
            cv2.putText(img, f"{prefix} {i+1}. {layer_title[:17]}", (x + 30, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1 if not is_active else 2, cv2.LINE_AA)

    @staticmethod
    def draw_bottom_telemetry_card(img, layer_idx):
        """Draws the detailed mathematical and architectural telemetry card for selected layer."""
        h_img, w_img = img.shape[:2]
        w, h = 760, 175
        x = (w_img - w) // 2
        y = h_img - h - 45

        if layer_idx is None:
            # Default Overview Card
            JarvisHUDRenderer.draw_chamfered_panel(img, x, y, w, h, chamfer=16, color_bg=(10, 14, 22), color_border=COLOR_CYAN_DIM)
            cv2.putText(img, "LLM NEURAL OVERVIEW // NO LAYER PINNED", (x + 24, y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_GOLD_BRIGHT, 2, cv2.LINE_AA)
            cv2.putText(img, "AUTOREGRESSIVE TRANSFORMER DECODER STACK (LLAMA / GPT ARCHITECTURE)", (x + 24, y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_CYAN_GLOW, 1, cv2.LINE_AA)
            cv2.putText(img, "Interact via Hand Gestures or Keyboard to inspect tensor mathematics:", (x + 24, y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_HUD_TEXT, 1, cv2.LINE_AA)
            cv2.putText(img, "  * Single-Hand Pinch = 3D Orbit Rotate  |  Two-Hand Distance = Explode Layers", (x + 24, y + 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GOLD_HOVER, 1, cv2.LINE_AA)
            cv2.putText(img, "  * Index Finger Pointer = Touch Inspect Layer  |  Keys [1-8] = Select Layer", (x + 24, y + 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GOLD_HOVER, 1, cv2.LINE_AA)
            return

        data = LLM_LAYERS_DATA[layer_idx]
        JarvisHUDRenderer.draw_chamfered_panel(img, x, y, w, h, chamfer=16, color_bg=(12, 18, 28), color_border=COLOR_GOLD_HOVER)

        # Header Title & Category
        title_str = f"LAYER {data['id']}/8: {data['name']}"
        cv2.putText(img, title_str, (x + 24, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_GOLD_BRIGHT, 2, cv2.LINE_AA)
        cv2.putText(img, f"[{data['category']}]", (x + w - 310, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_CYAN_GLOW, 1, cv2.LINE_AA)

        # Separator line
        cv2.line(img, (x + 24, y + 40), (x + w - 24, y + 40), COLOR_GOLD_HOVER, 1)

        # Tensor Shape & Formula
        cv2.putText(img, f"TENSOR SHAPE: {data['tensor_shape']}", (x + 24, y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(img, f"MATH FORMULA: {data['formula']}", (x + 24, y + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_GOLD_HOVER, 1, cv2.LINE_AA)

        # Architectural Role & Details
        cv2.putText(img, f"ROLE: {data['role']}", (x + 24, y + 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_HUD_TEXT, 1, cv2.LINE_AA)
        for d_i, line in enumerate(data['details'][:2]):
            cv2.putText(img, line, (x + 24, y + 134 + d_i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 205, 220), 1, cv2.LINE_AA)

    @staticmethod
    def draw_bottom_command_bar(img):
        """Draws bottom keyboard & interaction shortcuts helper bar."""
        h_img, w_img = img.shape[:2]
        commands = "[SPACE] Explode  |  [R] Reset  |  [A] Auto-Rotate  |  [1-8] Direct Select  |  [0] Clear  |  [+/-] Zoom  |  [Q] Quit"
        cv2.putText(img, commands, ((w_img - 840) // 2, h_img - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 180), 1, cv2.LINE_AA)

    @staticmethod
    def draw_hand_skeleton_hud(img, hand_data):
        """Draws futuristic glowing target reticles, tension meters, and skeleton lines."""
        landmarks = hand_data["landmarks"]
        thumb = hand_data["thumb_tip"].astype(int)
        index = hand_data["index_tip"].astype(int)
        wrist = hand_data["wrist"].astype(int)

        # Draw glowing cybernetic bone lines
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            p1 = landmarks[p1_idx, :2].astype(int)
            p2 = landmarks[p2_idx, :2].astype(int)
            cv2.line(img, tuple(p1), tuple(p2), (180, 120, 20), 1, cv2.LINE_AA)

        # Draw knuckle joint beads
        for j_idx in range(21):
            j_pt = landmarks[j_idx, :2].astype(int)
            cv2.circle(img, tuple(j_pt), 2, COLOR_CYAN_GLOW, -1, cv2.LINE_AA)

        # Futuristic fingertip reticle on Index tip
        reticle_color = COLOR_GOLD_BRIGHT if hand_data["is_pinched"] else COLOR_CYAN_GLOW
        cv2.circle(img, tuple(index), 11, reticle_color, 1, cv2.LINE_AA)
        cv2.circle(img, tuple(index), 3, reticle_color, -1, cv2.LINE_AA)
        # 4 Crosshair tick marks
        cv2.line(img, (index[0] - 16, index[1]), (index[0] - 8, index[1]), reticle_color, 1, cv2.LINE_AA)
        cv2.line(img, (index[0] + 8, index[1]), (index[0] + 16, index[1]), reticle_color, 1, cv2.LINE_AA)
        cv2.line(img, (index[0], index[1] - 16), (index[0], index[1] - 8), reticle_color, 1, cv2.LINE_AA)
        cv2.line(img, (index[0], index[1] + 8), (index[0], index[1] + 16), reticle_color, 1, cv2.LINE_AA)

        # Thumb tip target reticle
        cv2.circle(img, tuple(thumb), 6, reticle_color, 1, cv2.LINE_AA)
        cv2.circle(img, tuple(thumb), 2, reticle_color, -1, cv2.LINE_AA)

        # Dynamic Tension Line between Thumb and Index
        p_dist = hand_data["pinch_dist"]
        tension_color = COLOR_ORANGE_ALERT if hand_data["is_pinched"] else COLOR_CYAN_DIM
        cv2.line(img, tuple(thumb), tuple(index), tension_color, 1, cv2.LINE_AA)

        # Live Distance Readout
        mid_pt = ((thumb[0] + index[0]) // 2 + 10, (thumb[1] + index[1]) // 2 - 10)
        label_text = f"PINCH: {int(p_dist)}px" if not hand_data["is_pinched"] else "GRABBED [LOCKED]"
        cv2.putText(img, label_text, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, tension_color, 1, cv2.LINE_AA)

        # Wrist Telemetry
        cv2.putText(img, f"{hand_data['label'][0]}-HAND [{int(hand_data['score']*100)}%]",
                    (wrist[0] - 30, wrist[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_CYAN_DIM, 1, cv2.LINE_AA)


# =============================================================================
# 7. MAIN AR APPLICATION CONTROLLER (HOLO-LLM)
# =============================================================================
class HoloLLMApp:
    """Master Application Class orchestrating Video Capture, MediaPipe, 3D Engine, and HUD."""

    def __init__(self):
        self.camera_3d = Camera3D()
        self.hologram = TransformerHologram()
        self.tracker = HandGestureTracker()

        # Interaction State
        self.auto_rotate = True
        self.last_pinch_pos = None
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        self.mouse_cursor_pos = (0, 0)

        # Performance Monitoring
        self.prev_time = time.time()
        self.fps = 30.0

    def handle_mouse_events(self, event, x, y, flags, param):
        """OpenCV Mouse callback for smooth desktop interaction fallback."""
        self.mouse_cursor_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_dragging = True
            self.last_mouse_pos = (x, y)
            self.auto_rotate = False
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_dragging:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            self.camera_3d.target_yaw += dx * 0.008
            self.camera_3d.target_pitch -= dy * 0.008
            self.last_mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Scroll to zoom
            if flags > 0:
                self.camera_3d.target_dist = max(350.0, self.camera_3d.target_dist - 40.0)
            else:
                self.camera_3d.target_dist = min(1200.0, self.camera_3d.target_dist + 40.0)

    def run(self):
        """Main execution loop."""
        print("=" * 70)
        print("  HoloLLM: AR Transformer Architecture Visualizer (Iron-Man / Jarvis)")
        print("=" * 70)
        print("Connecting to Webcam and Initializing Hand CNN...")

        cap = cv2.VideoCapture(0)
        use_synthetic = False
        if not cap.isOpened():
            print("[WARNING] Default webcam (ID 0) not accessible! Falling back to Synthetic Cyber Grid.")
            use_synthetic = True
        else:
            # Request 1280x720 16:9 HD resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        window_name = "HoloLLM // AR Transformer Architecture Visualizer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.handle_mouse_events)

        print("\nControls:")
        print(" - Hand Pinch (1 Hand): Orbit rotate 3D hologram in Pitch & Yaw")
        print(" - Hand Distance (2 Hands): Explode / Collapse layers")
        print(" - Index Finger Pointing: Touch inspect layer to view tensor math")
        print(" - Mouse Left-Drag: Orbit rotate  |  Scroll Wheel: Zoom in/out")
        print(" - Keys: [SPACE] Explode  |  [R] Reset  |  [A] Auto-Rotate  |  [1-8] Layers  |  [Q] Quit")
        print("=" * 70)

        try:
            while True:
                t_start = time.time()
                dt = max(0.001, t_start - self.prev_time)
                self.prev_time = t_start
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

                # -------------------------------------------------------------
                # 1. ACQUIRE FRAME
                # -------------------------------------------------------------
                if not use_synthetic:
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    else:
                        # Mirror image horizontally for natural user interaction
                        frame = cv2.flip(frame, 1)
                else:
                    # Generate synthetic animated cyber grid background
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    for gy in range(0, 720, 40):
                        cv2.line(frame, (0, gy), (1280, gy), (20, 25, 35), 1)
                    for gx in range(0, 1280, 40):
                        cv2.line(frame, (gx, 0), (gx, 720), (20, 25, 35), 1)

                h_frame, w_frame = frame.shape[:2]

                # -------------------------------------------------------------
                # 2. PROCESS HAND TRACKING & GESTURES
                # -------------------------------------------------------------
                self.tracker.process_frame(frame)

                # Gesture: Single-hand pinch drag for 3D rotation
                if self.tracker.is_pinching and self.tracker.pinch_pos_2d is not None:
                    self.auto_rotate = False
                    cur_pinch = self.tracker.pinch_pos_2d
                    if self.last_pinch_pos is not None:
                        dx = cur_pinch[0] - self.last_pinch_pos[0]
                        dy = cur_pinch[1] - self.last_pinch_pos[1]
                        self.camera_3d.target_yaw += dx * 0.009
                        self.camera_3d.target_pitch -= dy * 0.009
                    self.last_pinch_pos = cur_pinch
                else:
                    self.last_pinch_pos = None

                # Gesture: Two-handed distance for explode / collapse
                if self.tracker.two_hand_dist is not None:
                    # Map distance between 100px and 550px to expansion [0.0, 1.0]
                    norm_exp = np.clip((self.tracker.two_hand_dist - 120.0) / 400.0, 0.0, 1.0)
                    self.hologram.target_expansion = float(norm_exp)

                # Auto-rotate when idle
                if self.auto_rotate:
                    self.camera_3d.target_yaw += 0.35 * dt

                # Update physics & dampening
                self.camera_3d.update(dt)
                self.hologram.update_physics(dt)

                # -------------------------------------------------------------
                # 3. DARKEN FRAME & CREATE GLOW CANVAS
                # -------------------------------------------------------------
                # Darken the real-world frame so neon cybernetic overlays pop with high contrast
                darkened_frame = cv2.convertScaleAbs(frame, alpha=0.68, beta=0)
                glow_canvas = np.zeros_like(frame)

                # -------------------------------------------------------------
                # 4. RENDER 3D TRANSFORMER HOLOGRAM WIREFRAMES
                # -------------------------------------------------------------
                # Decide pointer coordinates for layer hit-testing (Index finger or Mouse cursor)
                if self.tracker.pointer_pos_2d is not None:
                    pointer_xy = (float(self.tracker.pointer_pos_2d[0]), float(self.tracker.pointer_pos_2d[1]))
                else:
                    pointer_xy = (float(self.mouse_cursor_pos[0]), float(self.mouse_cursor_pos[1]))

                # Hit-test layers against pointer coordinate (Depth-sorted raycasting)
                layer_hit_candidates = []
                layer_projected_hulls = {}

                for l_idx in range(self.hologram.num_layers):
                    hull_corners_3d = self.hologram.build_layer_hull_vertices(l_idx)
                    pts_2d, depths, valid = self.camera_3d.project_points(hull_corners_3d, w_frame, h_frame)

                    if np.all(valid):
                        hull_2d = cv2.convexHull(pts_2d)
                        layer_projected_hulls[l_idx] = (pts_2d, hull_2d)
                        avg_depth = float(np.mean(depths))

                        # 2D Point-in-polygon collision test
                        inside = cv2.pointPolygonTest(hull_2d, pointer_xy, False) >= 0
                        if inside:
                            layer_hit_candidates.append((avg_depth, l_idx))

                # Select front-most touched layer (closest depth)
                if self.hologram.locked_layer_idx is not None:
                    self.hologram.active_layer_idx = self.hologram.locked_layer_idx
                elif layer_hit_candidates:
                    layer_hit_candidates.sort(key=lambda item: item[0])
                    self.hologram.active_layer_idx = layer_hit_candidates[0][1]
                else:
                    self.hologram.active_layer_idx = None

                # Render Holographic Pedestal Floor
                pedestal_segments = self.hologram.build_holographic_pedestal()
                for pA, pB, style in pedestal_segments:
                    seg_pts = np.stack([pA, pB], axis=0)
                    pts_2d, _, valid = self.camera_3d.project_points(seg_pts, w_frame, h_frame)
                    if np.all(valid):
                        col = COLOR_CYAN_DIM if style == "dim" else COLOR_CYAN_GLOW
                        cv2.line(glow_canvas, tuple(pts_2d[0]), tuple(pts_2d[1]), col, 1, cv2.LINE_AA)

                # Render Backbone Spine Rails
                spine_rails = self.hologram.build_spine_rails()
                for pA, pB in spine_rails:
                    rail_pts = np.stack([pA, pB], axis=0)
                    pts_2d, depths, valid = self.camera_3d.project_points(rail_pts, w_frame, h_frame)
                    if np.all(valid):
                        cv2.line(glow_canvas, tuple(pts_2d[0]), tuple(pts_2d[1]), COLOR_CYAN_DIM, 1, cv2.LINE_AA)

                # Render Each Layer (Plates & Custom Internal Wireframes)
                for l_idx in range(self.hologram.num_layers):
                    is_active = (self.hologram.active_layer_idx == l_idx)
                    plate_color = COLOR_GOLD_HOVER if is_active else COLOR_CYAN_GLOW
                    accent_color = COLOR_GOLD_BRIGHT if is_active else COLOR_CYAN_GLOW
                    dim_color = (60, 160, 220) if is_active else COLOR_CYAN_DIM

                    # Outer Plate Edges
                    if l_idx in layer_projected_hulls:
                        pts_2d, _ = layer_projected_hulls[l_idx]
                        edges = [
                            (0, 1), (1, 2), (2, 3), (3, 0),
                            (4, 5), (5, 6), (6, 7), (7, 4),
                            (0, 4), (1, 5), (2, 6), (3, 7)
                        ]
                        for e1, e2 in edges:
                            cv2.line(glow_canvas, tuple(pts_2d[e1]), tuple(pts_2d[e2]), plate_color, 2 if is_active else 1, cv2.LINE_AA)

                    # Custom Internal Wireframe Geometry
                    internals = self.hologram.build_layer_internal_geometry(l_idx)
                    for ptA, ptB, style in internals:
                        seg_pts = np.stack([ptA, ptB], axis=0)
                        pts_2d, _, valid = self.camera_3d.project_points(seg_pts, w_frame, h_frame)
                        if np.all(valid):
                            if style == "accent":
                                col = accent_color
                                thick = 2
                            elif style == "dim":
                                col = dim_color
                                thick = 1
                            else:
                                col = plate_color
                                thick = 1
                            cv2.line(glow_canvas, tuple(pts_2d[0]), tuple(pts_2d[1]), col, thick, cv2.LINE_AA)

                # Render Animated Dataflow Forward-Pass Particles
                part_pts_3d = self.hologram.get_particle_positions_3d()
                part_pts_2d, _, part_valid = self.camera_3d.project_points(part_pts_3d, w_frame, h_frame)
                for i, (pt_2d, val) in enumerate(zip(part_pts_2d, part_valid)):
                    if val:
                        cv2.circle(glow_canvas, tuple(pt_2d), 3, COLOR_WHITE_PULSE, -1, cv2.LINE_AA)
                        cv2.circle(glow_canvas, tuple(pt_2d), 6, COLOR_GOLD_BRIGHT, 1, cv2.LINE_AA)

                # -------------------------------------------------------------
                # 5. AUTHENTIC NEON GLOW BLOOM FILTER (BLUR & ADDWEIGHTED)
                # -------------------------------------------------------------
                small_w, small_h = max(1, w_frame // 4), max(1, h_frame // 4)
                glow_small = cv2.resize(glow_canvas, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                glow_blur_small = cv2.GaussianBlur(glow_small, (15, 15), 0)
                glow_blur = cv2.resize(glow_blur_small, (w_frame, h_frame), interpolation=cv2.INTER_LINEAR)

                # Composite: Darkened frame + Crisp Wireframes + Soft Luminous Bloom
                frame_out = cv2.addWeighted(darkened_frame, 1.0, glow_canvas, 1.0, 0)
                frame_out = cv2.add(frame_out, cv2.addWeighted(glow_blur, 0.75, np.zeros_like(glow_blur), 0, 0))

                # -------------------------------------------------------------
                # 6. RENDER HUD OVERLAYS & TELEMETRY PANELS
                # -------------------------------------------------------------
                # Render Hand HUD reticles and skeleton on detected hands
                for hand_data in self.tracker.hand_data_list:
                    JarvisHUDRenderer.draw_hand_skeleton_hud(frame_out, hand_data)

                # Top-Left System Status Bar
                JarvisHUDRenderer.draw_top_status_bar(
                    frame_out,
                    fps=self.fps,
                    state_text=self.tracker.state_label,
                    expansion_pct=self.hologram.expansion_factor * 100.0,
                    tracking_online=(len(self.tracker.hand_data_list) > 0)
                )

                # Top-Right Spatial Telemetry
                pitch_deg = math.degrees(self.camera_3d.pitch)
                yaw_deg = math.degrees(self.camera_3d.yaw)
                zoom_factor = 620.0 / max(100.0, self.camera_3d.cam_dist)
                JarvisHUDRenderer.draw_top_right_camera_telemetry(frame_out, pitch_deg, yaw_deg, zoom_factor)

                # Right-Side Layer Directory
                JarvisHUDRenderer.draw_right_layer_directory(frame_out, self.hologram.active_layer_idx)

                # Bottom-Center Holographic Info Card
                JarvisHUDRenderer.draw_bottom_telemetry_card(frame_out, self.hologram.active_layer_idx)

                # Bottom Command Shortcuts Bar
                JarvisHUDRenderer.draw_bottom_command_bar(frame_out)

                # -------------------------------------------------------------
