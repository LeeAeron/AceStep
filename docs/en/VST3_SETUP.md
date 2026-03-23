# ACE-Step VST3 Setup and Validation

This guide documents the current ACE-Step VST3 MVP workflow, including contributor build steps,
backend setup expectations, user-facing behavior, and the validation matrix for the current draft
plugin implementation.

## Current MVP Scope

The current VST3 MVP implementation is intentionally narrow:

- isolated JUCE/CMake VST3 shell
- plugin UI for prompt, lyrics, duration, seed, preset, and quality mode
- persisted plugin state across DAW reopen
- backend health check, task submission, and polling
- generated-preview download and playback
- reveal-file handoff through the host operating system

The following items are explicitly deferred:

- reference-audio workflows
- repaint/edit workflows
- drag-and-drop into the DAW timeline
- AU, CLAP, and AAX formats

## Contributor Build

Build requirements:

- CMake 3.22 or newer
- a C++20-capable compiler
- Git access during configure so JUCE can be fetched

Build commands:

```bash
cmake -S plugins/acestep_vst3 -B build/acestep_vst3
cmake --build build/acestep_vst3 --config Release
```

Platform notes:

- Windows: Visual Studio 2022 or Ninja is recommended
- macOS: Xcode or Ninja is recommended
- The plugin workspace is isolated under `plugins/acestep_vst3/` and does not modify the repo's
  Python packaging flow

## Backend Setup

The VST3 MVP architecture assumes a separate ACE-Step backend/API process.

Recommended backend startup for future integration work:

```bash
uv run acestep-api
```

Default local endpoint:

- `http://localhost:8001`

The plugin makes live backend requests for:

- `GET /health`
- `POST /release_task`
- `POST /query_result`

The backend must already be running before the plugin is opened. Auto-launch and backend lifecycle
management are not part of the MVP.

## Current User Workflow

In the current MVP implementation:

1. Open the plugin inside a supported VST3 host.
2. Enter prompt, lyrics, duration, seed, preset, and quality mode.
3. Confirm the backend URL points to a running ACE-Step backend.
4. Trigger generation from the plugin.
5. Wait for submission, queued/running, and completion states to resolve through live polling.
6. Preview the downloaded result audio inside the plugin.
7. Play, stop, clear, or reveal the preview file from the plugin UI.

The reveal-file path is intentionally file based. Drag-and-drop into the DAW timeline is deferred
for MVP stability.

## Validation Matrix

| Area | Target | Current status | Notes |
|------|--------|----------------|-------|
| Plugin shell build | Windows | Pending manual validation | Build instructions are present; no Windows validation was run in this environment |
| Plugin shell build | macOS | Implemented | Local build succeeded in this workspace after command line tools were installed |
| Host load | Reaper (Windows) | Pending manual validation | This is the default Windows validation host for the MVP |
| Host load | Reaper (macOS) | Implemented | Plugin shell was loaded in Reaper on macOS in this workspace |
| State persistence | DAW save/reopen | Implemented, pending broader host validation | Prompt, lyrics, controls, result slots, backend metadata, and preview metadata are serialized |
| Backend offline UX | Live backend health check | Implemented | Uses real `/health` requests and visible error/status messaging |
| Prompt/job workflow | Live backend task flow | Implemented | Uses `/release_task` and `/query_result` with submission, queued/running, success, and failure states |
| Preview playback | Downloaded result playback | Implemented, pending host validation | Generated output is downloaded to a local cache file and played through JUCE transport primitives |
| File handoff | Reveal file in OS | Implemented, pending host validation | Uses file-based handoff; drag-and-drop is deferred |

## Known Limitations

- The plugin depends on an already-running ACE-Step backend
- No plugin CI pipeline is active yet
- The editor timer currently drives backend polling, so generation workflow should be exercised with
  the plugin UI open
- Windows host validation is still pending
- Drag-and-drop into the DAW timeline is intentionally out of scope for this milestone

## Tracking

This guide corresponds to the VST3 umbrella issue and its current child issues:

- `#890` umbrella tracker
- `#893` plugin shell
- `#894` plugin UI and backend job state
- `#895` preview playback and file handoff
