# Legacy one-off scripts (do not run)

Kept only as a record of how the project was built. They are **not** part of the pipeline.

- `apply_completion_fixes.py` – one-time source patcher used during the first completion audit; its edits are already in `src/`, `app/`, `ui/`. Re-running it fails (expected text no longer present).
- `finalize_status.py` – wrote an earlier README text and `results/final_status.json` with stale counts (17 tests). Running it would overwrite the current README.
- `verify_browser.cjs` – earlier Playwright screenshot script; Playwright is not installed here. Superseded by `scripts/capture_screenshots.py` (headless Edge via DevTools protocol).
