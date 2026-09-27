# ABDA — Adaptive Behaviour Drift Analysis

## Phase 1 — Multimodal Feature Extraction & First Integration

ABDA (Adaptive Behaviour Drift Analysis) is a multimodal framework designed to
measure **behavioural change relative to an individual's own baseline**.

Rather than treating a single emotion label as the final output, the system is
designed to represent behavioural patterns across multiple sessions and
eventually quantify how much a current session deviates from a person's
established behavioural norm.

> **Important scope:** ABDA is a behavioural change/deviation analysis system.
> It is not intended to diagnose depression, burnout, or any other clinical
> condition.

---

## Project Objective

The long-term ABDA pipeline combines three behavioural modalities:

- **Facial / Video**
- **Speech / Audio**
- **Text / Language**

Each modality is converted into a structured session-level representation.
These representations are subsequently normalized, quality-weighted and fused
into a unified multimodal representation.

A later stage establishes a personal behavioural baseline and computes the
**Behaviour Drift Index (BDI)** as a continuous measure of deviation from that
baseline.

The current repository focuses on **Phase 1**:

> Modality-specific feature extraction → common feature interface →
> first multimodal integration.

---

# System Architecture

The overall architecture is organized into the following stages:

```text
                    Raw Session
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
       VIDEO           AUDIO          TEXT
          │              │              │
          ▼              ▼              ▼
      Facial          Speech           NLP
     Features        Features        Features
          │              │              │
          ▼              ▼              ▼
      Quality         Quality         Quality
       Score           Score           Score
          │              │              │
          └──────────────┼──────────────┘
                         ▼
              Normalized Representations
                         │
                         ▼
              Common Session Interface
                         │
                         ▼
              Multimodal Feature Fusion
                         │
                         ▼
              Unified Session Embedding
                         │
                         ▼
              Personal Baseline Module
                         │
                         ▼
               Behaviour Drift Index
