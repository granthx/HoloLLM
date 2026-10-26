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
