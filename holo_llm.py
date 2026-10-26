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
