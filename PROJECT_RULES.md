# PROJECT_RULES.md

# SurgMap3D-AI Engineering Guidelines

## Project Goal

Build a modular AI system for intelligent surgical scene understanding using:

- DINOv2
- Endo3R
- MUSt3R
- Dens3R
- OmniVGGT
- SAM2
- Confidence Fusion
- Scene Graph
- Dynamic Knowledge Graph
- Multi-Agent Surgical Assistant

---

# Architecture

Every module must be independent.

Never tightly couple modules.

Communication should happen only through well-defined inputs and outputs.

---

# Folder Rules

preprocessing/
Only preprocessing code.

models/
Only model wrappers.

fusion/
Only confidence fusion.

scene_graph/
Only graph generation.

knowledge_graph/
Only medical reasoning.

agents/
Only AI agents.

evaluation/
Only metrics and benchmarking.

visualization/
Only plotting and visualization.

app/
Only Streamlit UI.

---

# Python Version

Python 3.10

---

# Style Guide

PEP8

Maximum line length = 88

Use type hints.

Use docstrings.

---

# Naming

snake_case for variables

PascalCase for classes

UPPER_CASE for constants

---

# Logging

Never use print() except inside demo scripts.

Always use logging.

---

# Configuration

Never hardcode paths.

Use configs/config.yaml.

---

# Error Handling

Every public function must validate inputs.

Raise meaningful exceptions.

---

# Documentation

Every module should contain

Purpose

Inputs

Outputs

Dependencies

---

# Testing

Every module must be independently testable.

---

# Git

One feature per commit.

Meaningful commit messages.

---

# Code Generation Rule

Never generate monolithic code.

Generate modular reusable components.
