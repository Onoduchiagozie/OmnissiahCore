# Archive

Historical source kept for reference. Nothing here is imported by the API,
deployed to Oracle, or treated as current runtime code.

- `legacy-build/` — old corpus builders, manifest generators, one-time
  migrations, and scratch diagnostics. These were removed from the active root
  because they no longer match the current pipeline (`Scripts/build_db.py` with
  `BAAI/bge-m3`) or have already been applied.
- `legacy-scene-pipeline/` — the old scene-generation pipeline (`battle.py`,
  `40kPhase2Build.py`, `Stage_3_Weaver.py`, and diagnostics). The current scene
  index is built by `Scripts/buildSceneIndex.py` from `Db/battle_scenes.db`.

These files keep their original root-relative paths. They are archived, not
runnable in place, and should not be re-enabled without a reviewed change.
