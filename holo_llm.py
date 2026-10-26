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
