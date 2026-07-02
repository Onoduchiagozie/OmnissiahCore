"""
backfill_scene_keys.py — run once to add scene_key to existing scenes.
Safe to run multiple times — only updates rows where scene_key is NULL.
"""
import sqlite3
from pathlib import Path

conn = sqlite3.connect(Path(__file__).parent / 'battle_scenes.db')

# Add columns if missing
for col, definition in [('scene_key', 'TEXT'), ('woven_at', 'TEXT')]:
    try:
        conn.execute(f'ALTER TABLE scenes ADD COLUMN {col} {definition}')
        conn.commit()
        print(f'Column {col} added')
    except Exception as e:
        print(f'Column {col} already exists: {e}')

# Build book_id -> source_raw lookup from build_progress
rows = conn.execute('SELECT source_raw, book_id FROM build_progress').fetchall()
book_to_source = {book_id: source_raw for source_raw, book_id in rows}
print(f'Loaded {len(book_to_source)} book_id -> source_raw mappings')

# Backfill scene_key for every scene that doesn't have one yet
scenes = conn.execute(
    'SELECT scene_id, book_id, rank FROM scenes WHERE scene_key IS NULL'
).fetchall()
print(f'Backfilling {len(scenes)} scenes...')

updated = 0
no_source = 0
for scene_id, book_id, rank in scenes:
    source_raw = book_to_source.get(book_id)
    if source_raw:
        key = f'{source_raw}::rank{rank}'
        conn.execute(
            'UPDATE scenes SET scene_key = ?, woven_at = datetime("now") WHERE scene_id = ?',
            (key, scene_id)
        )
        updated += 1
    else:
        no_source += 1

conn.commit()
print(f'Backfilled : {updated}')
print(f'No mapping : {no_source}  (these scenes have no build_progress entry)')

nulls = conn.execute(
    'SELECT COUNT(*) FROM scenes WHERE scene_key IS NULL'
).fetchone()[0]
print(f'Still NULL : {nulls}')

if nulls == 0:
    print('\nAll scenes have a key. Safe to run phase2_build.py')
else:
    print(f'\n{nulls} scenes still have no key — these will be re-woven on next run.')

conn.close()