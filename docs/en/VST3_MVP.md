# ACE-Step VST3 MVP Architecture

This document defines the initial architecture for packaging ACE-Step as a VST3 plugin for DAW
workflows.

## Summary

The MVP should be a thin VST3 plugin that delegates music generation to the existing ACE-Step
backend/API. The plugin must not embed model inference or reimplement the current generation stack
inside the DAW process.

The recommended default framework for the first implementation is JUCE, because it gives us mature
cross-platform VST3 support, stable host integration, and a practical state persistence model for
DAW projects.

## MVP Scope

The first release should focus on text-to-music generation only:

- prompt and lyrics input
- duration, seed, and quality preset controls
- job submission and polling
- generated audio preview
- basic file handoff back to the DAW workflow

Explicitly deferred from the MVP:

- reference audio input
- repaint/edit workflows
- stem routing and multitrack orchestration
- host tempo or transport sync
- AU, CLAP, and AAX builds

## Process Boundaries

The plugin process should stay lightweight.

- The DAW hosts the plugin UI and project state.
- The plugin talks to a local ACE-Step backend over the existing API.
- The backend owns model loading, generation, polling, and output files.
- All long-running work must stay off the audio thread.

This split keeps the plugin responsive, reduces risk to DAW stability, and avoids duplicating the
current inference stack.

## Plugin State Model

The plugin should persist only UI and workflow state that matters to the current DAW project:

- prompt and lyrics text
- generation controls such as duration, seed, and preset
- backend base URL or connection target
- last selected result metadata
- a versioned state schema for forward compatibility

State should be restored automatically when the DAW project is reopened. If the backend is
unavailable, the plugin should still restore its local UI state and show a disconnected state rather
than failing to open.

## Offline And Error Behavior

The plugin should treat backend failures as recoverable workflow states, not fatal errors.

- If the backend is unreachable, the plugin should show a clear disconnected status.
- Generation controls should remain editable, but generation should be disabled until connectivity is
  restored.
- Job polling should happen on a background path, never on the audio thread.
- Failed jobs should surface an actionable error message and preserve the user inputs.

## Validation Hosts

The MVP should be validated in:

- Reaper on Windows
- Reaper on macOS

These hosts are the default compatibility targets for the first plugin milestone because they are
practical, widely used, and suitable for checking load, state restore, and result handoff behavior.

## Why JUCE

JUCE is the default framework choice for the MVP because it gives us the most direct route to a
cross-platform VST3 plugin with predictable editor, parameter, and state handling.

That makes it a good fit for a first implementation where the main risk is workflow integration, not
model research or audio DSP novelty.

