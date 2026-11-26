---
name: psm_wma_spatial_memory
description: Read a committed G2 scene or retained run evidence within the current authorized Episode.
version: 1.0.0
---

# PSM-WMA spatial and run evidence

Start with `scene_overview`. Use `rooms`, `entities`, `relation`, `list_views`, or `view` for specific questions. Use `initial_context` for visible initialization role candidates and `run_history` for retained run summaries or completed action windows. `describe_skill` explains the read surface.

Use the reply's publication frame, time, and source when answering. Ask for clarification on `ambiguous` results. An `unknown`, `stale`, `not_retained`, or `forbidden` outcome carries its own reason.

Report native room assignments, hierarchy, and geometric relations with their evidence basis. Semantic `on`, `inside`, and task success require evidence beyond those relations. Select one logical View when an image is needed. Renderer membership describes the View's input geometry.

The PSM-WMA host supplies the authorized reader and explicitly binds the Skill. The query surface supports existing observations and retained evidence.
