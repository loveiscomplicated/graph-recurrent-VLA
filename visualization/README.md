# Rollout graph visualization

Run the read-only local frontend from the repository root:

```bash
.venv/bin/python -m visualization.server
```

Open `http://127.0.0.1:8765/`. The main page lists compatible saved Task07
graph rollout episodes found under `experiments/`. It checks the first graph
frame against its matching observation before listing an episode. Older artifacts
without TCP pose and robot geometry cannot use the EE/world toggle and are
omitted. Search by run or episode name and filter
by **성공**, **실패**, or **미확인**. Click a card to open that episode's interactive
3D graph viewer, and use **← 목록** to return. A successful training sample is
listed first when available; its success does not imply held-out performance.

The viewer reads `graph_XXXX.npz` and same-timestep `observations.npz`. The
**EE 상대 / 월드 절대** toggle changes displayed node positions; the world view
uses the saved TCP world pose. The original F5 values and `edge_attr` stay in
the EE frame. Use the timeline or play button to move through graph inputs,
drag to rotate, Shift/right drag to pan, scroll to zoom, and click a node to
inspect its saved model inputs. Frames are loaded on demand.

Status is **성공** when episode metadata explicitly records `success: true`;
**실패** when a `COMPLETE` episode explicitly records `success: false`; otherwise
it is **미확인**. This keeps smoke or unfinished rollouts from being labeled as
evaluation failures.

Options: `--experiments-root PATH`, `--host HOST`, `--port PORT`. The default
host is `127.0.0.1`; point `--experiments-root` at another local artifact tree
if needed. The server only reads artifacts and does not publish them.
