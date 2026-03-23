# ACE-Step VST3 Plugin Backend Contract

This document defines the MVP contract between a future ACE-Step VST3 plugin and the existing
ACE-Step backend/API. The goal is to keep the plugin thin and reuse the current REST endpoints
instead of introducing a separate inference stack inside the DAW process.

## Contract Summary

For the MVP, the plugin should use this flow:

1. Check backend availability with `GET /health`.
2. Submit a generation job with `POST /release_task`.
3. Poll job status with `POST /query_result`.
4. Use `GET /v1/audio` for preview or download once a job succeeds.

This matches the current REST surface documented in [API.md](./API.md) and the browser Studio flow
in [ui/studio.html](../../ui/studio.html).

## Backend Checks

### `GET /health`

Use this as the plugin's first reachability check.

Current API behavior:

- Returns a wrapped response with `code`, `data`, `error`, `timestamp`, and `extra`
- `data.status` is `"ok"` when the service is healthy
- `data.service` identifies the API service

Plugin interpretation:

- `offline`: request fails, times out, or returns a non-200 wrapper
- `ready`: `code === 200` and `data.status === "ok"`
- `degraded`: reachable, but the backend reports a non-ideal readiness state, such as models not
  being initialized yet

The plugin should keep the UI editable when offline, but generation should remain disabled until the
backend becomes reachable again.

## Generation Submission

### `POST /release_task`

The API accepts JSON, multipart form data, and URL-encoded form data. For the VST3 MVP, the plugin
should prefer JSON because the MVP does not need file upload.

Recommended request shape for text-to-music:

```json
{
  "prompt": "a bright indie pop chorus",
  "lyrics": "...",
  "vocal_language": "en",
  "thinking": false,
  "audio_duration": 30,
  "inference_steps": 8,
  "guidance_scale": 7.0,
  "use_random_seed": false,
  "seed": 12345,
  "batch_size": 1,
  "audio_format": "wav"
}
```

Notes:

- `prompt` and `lyrics` are the primary user inputs.
- `audio_duration` maps to the plugin's duration control.
- `seed` and `use_random_seed` cover repeatability.
- `batch_size` should stay `1` for the MVP plugin UI unless a later issue explicitly adds batch
  handling.
- `audio_format` should be set to a format the plugin can preview reliably. `wav` is the safest
  default for handoff and playback.
- `model` can be added later if the plugin exposes explicit model selection, but it is not required
  for the MVP contract.

Current API response shape:

- `code === 200` indicates submission succeeded
- `data.task_id` is the identifier used for polling
- `data.status` is typically `"queued"`
- `data.queue_position` may be present

Plugin interpretation:

- Treat `task_id` as the only required field for follow-up polling
- Preserve the original prompt/lyrics locally so the user can recover context even if the backend
  later fails

## Job Polling

### `POST /query_result`

The plugin should poll using JSON with a `task_id_list` array:

```json
{
  "task_id_list": ["550e8400-e29b-41d4-a716-446655440000"]
}
```

Current API behavior:

- The request accepts JSON or URL-encoded form data
- `task_id_list` may be a real array or a JSON-encoded string
- `status` uses integer codes:
  - `0` = queued or running
  - `1` = succeeded
  - `2` = failed
- Running responses may include `progress_text`
- Successful responses include `result` as a JSON string, not a parsed object

Plugin-side normalized job states:

- `idle`: no active job
- `submitting`: request is in flight
- `queued_or_running`: API status `0`
- `succeeded`: API status `1`
- `failed`: API status `2`

The plugin should not assume a separate queued-vs-running distinction unless the backend adds one in a
future API revision.

## Result Handling

When a task succeeds, the plugin should:

1. Parse the `result` string as JSON.
2. Use the first result item as the primary preview target.
3. Read `file` or `url` from that parsed object.
4. Build a playable URL using `GET /v1/audio` when the path is server-relative.

Current Studio behavior in [ui/studio.html](../../ui/studio.html) follows the same general pattern:
poll until status `1`, parse `task.result`, and then resolve the returned audio path into a final
URL.

Plugin result fields worth keeping locally:

- `file_url`
- `prompt`
- `lyrics`
- `metas`
- `progress_text`
- `error`

## Audio Preview

### `GET /v1/audio`

Use this endpoint for preview and download of generated files.

The plugin should treat the returned file path as the source of truth for:

- in-plugin preview playback
- reveal/open-in-folder handoff
- export or copy-to-project workflows if needed later

If the `file` value is already a full URL, the plugin can use it directly. If it is server-relative,
prefix it with the configured base URL.

## Authentication

The API server can be configured with `ACESTEP_API_KEY`.

Recommended plugin behavior:

- Prefer `Authorization: Bearer <token>` when auth is enabled
- Avoid placing tokens in request bodies
- Keep the base URL configurable so the plugin can connect to a local API server or another trusted
  endpoint

## API Gaps For Follow-Up

The current API is sufficient for the MVP contract, but the plugin would benefit from a few future
enhancements:

- A clearer readiness signal than a single health wrapper
- Structured JSON for `query_result.result` instead of a JSON string
- A task-detail endpoint for single-task polling
- More task-specific progress reporting
- Explicit artifact metadata beyond a file path
- A documented cancellation flow for in-flight generation jobs

These gaps should remain follow-up items, not blockers for the first plugin milestone.
