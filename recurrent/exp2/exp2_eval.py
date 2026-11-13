import argparse
import copy
import gc
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import sapien
import torch

from bench_exp2 import Graph, make_env, to_np
from data_collection.collect_episodes import initial_visibility
from envs.maniskill import get_gt_state
from envs.maniskill.wrappers import peg_task
from exp2_train import build, features, q_conj, q_rot, seed_list
from hole_motion import BoxMover, HoleMotion, add_args, sample_translation

STEP_TYPES = ["mlp_step", "stack_mlp_step", "gru_step", "mamba_step", "gru_step_nohf"]
CHUNK_TYPES = ["stack_mlp_chunk", "gru_chunk", "mamba_chunk"]
MODES = ["c1", "c3", "c5", "c16", "te"]
HORIZON = 250


def all_candidates():
    return [f"{t}@step" for t in STEP_TYPES] + [f"{t}@{m}" for t in CHUNK_TYPES for m in MODES]


def detach(state):
    if state is None:
        return None
    if torch.is_tensor(state):
        return state.detach()
    return type(state)(detach(s) for s in state)


class Decoder:
    def __init__(self, ck, mode, te_m=0.01, search=False):
        self.model = build(ck["kind"], ck["obs_dim"], 7 * ck["chunk"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        st = ck["stats"]
        self.mean, self.std, self.keep = (np.asarray(st[k], np.float32) for k in ("obs_mean", "obs_std", "obs_keep"))
        self.act_mean, self.act_std = np.asarray(st["act_mean"], np.float32), np.asarray(st["act_std"], np.float32)
        self.chunk, self.hole_frame, self.mode, self.te_m = ck["chunk"], ck["hole_frame"], mode, te_m
        self.r6 = bool(ck.get("rot6", False))
        self.kf = None
        if ck.get("kin"):
            from kin_filter import KinFilter
            self.kf = KinFilter(**(ck["kin"] if isinstance(ck["kin"], dict) else {}))
        self.search = None
        if search:
            from search import SpiralSearch
            self.search = SpiralSearch()
        self.reset()

    def reset(self):
        self.state, self.plan, self.k, self.t, self.te = None, None, 0, 0, []
        if self.kf is not None:
            self.kf.reset()
        if self.search is not None:
            self.search.reset()

    def snapshot(self):
        return copy.deepcopy((self.state, self.plan, self.k, self.t, self.te, self.kf, self.search))

    def restore(self, snap):
        self.state, self.plan, self.k, self.t, self.te, self.kf, self.search = copy.deepcopy(snap)

    @torch.no_grad()
    def act(self, state36, counts=None):
        if self.kf is not None:
            state36 = self.kf.step(state36, counts)
        self.last = np.asarray(state36, np.float64)
        f = features(state36[None].astype(np.float32), self.hole_frame, self.r6)
        x = ((f - self.mean) / self.std * self.keep).astype(np.float32)
        out, state = self.model(torch.from_numpy(x)[None], self.state)
        self.state = detach(state)
        out = out[0, 0].view(self.chunk, 7).numpy()
        if self.mode in ("step", "c1"):
            a = out[0]
        elif self.mode != "te":
            n = int(self.mode[1:])
            if self.plan is None or self.k >= n:
                self.plan, self.k = out, 0
            a = self.plan[self.k]
            self.k += 1
        else:
            self.te.append((self.t, out))
            self.te = [(s, o) for s, o in self.te if self.t - s < self.chunk]
            preds = np.stack([o[self.t - s] for s, o in self.te])
            w = np.exp(-self.te_m * np.arange(len(preds)))
            a = (preds * w[:, None]).sum(0) / w.sum()
        self.t += 1
        move = a[:6] * self.act_std + self.act_mean
        action = np.concatenate([move, [1.0 if a[6] > 0 else -1.0]]).astype(np.float32)
        if self.search is not None:
            action = self.search.step(action, state36)
        return action


def robot15(env):
    s = get_gt_state(env)
    tcp = to_np(s.ee_pose)[0]
    q = tcp[3:] if tcp[3] >= 0 else -tcp[3:]
    return np.concatenate([tcp[:3], q, to_np(s.left_fingertip_pos)[0] - tcp[:3],
                           to_np(s.right_fingertip_pos)[0] - tcp[:3], to_np(s.gripper_qpos)[0]]).astype(np.float32)


def canon(q):
    return q if q[0] >= 0 else -q


def gt_object21(env):
    s = get_gt_state(env)
    tcp, peg, hole = to_np(s.ee_pose)[0], to_np(s.peg_pose)[0], to_np(s.hole_pose)[0]
    half = to_np(s.peg_half_sizes).reshape(-1)[:3]
    tip = peg[:3] + q_rot(peg[3:], np.array([half[0], 0.0, 0.0]))
    return np.concatenate([peg[:3] - tcp[:3], canon(peg[3:]), tip - hole[:3], hole[:3] - tcp[:3], canon(hole[3:]),
                           half, to_np(s.hole_radius).reshape(-1)[:1]]).astype(np.float32)


def perceive(env, graph, obs, oracle):
    if oracle:
        return gt_object21(env), to_np(get_gt_state(env).ee_pose)[0][:3]
    o21, robot9, _, _ = graph.step(obs)
    return o21, robot9


def geometry(env):
    s = get_gt_state(env)
    peg, hole = to_np(s.peg_pose)[0], to_np(s.hole_pose)[0]
    half = float(to_np(s.peg_half_sizes).reshape(-1)[0])
    tip = peg[:3] + q_rot(peg[3:], np.array([half, 0.0, 0.0]))
    rel = q_rot(q_conj(hole[3:]), tip - hole[:3])
    return tip, hole, rel, peg[2]


def shift_box(env, hole_q, metres, sign):
    d = q_rot(hole_q, np.array([0.0, sign, 0.0]))
    d[2] = 0.0
    d = d / max(np.linalg.norm(d), 1e-9) * metres
    task = peg_task(env)
    p = task.box.pose
    pose = sapien.Pose(p.p[0].cpu().numpy() + d, p.q[0].cpu().numpy())
    task.box.set_pose(pose)
    return d


def is_success(info):
    return bool(info["success"].reshape(-1)[0].item())


def reaction(lat, start):
    for k in range(start, len(lat) - 3):
        if lat[k + 1] < lat[k] and lat[k + 2] < lat[k + 1] and lat[k + 3] < lat[k + 2]:
            return k - start
    return None


def loop(env, graph, dec, obs, t0, cfg, log, motion=None, mover=None):
    success, t = False, t0
    freq = float(env.unwrapped.control_freq)
    for t in range(t0, HORIZON):
        tip, hole, rel, peg_z = geometry(env)
        if motion is not None:
            if mover.apply(motion.update(t, t / freq, tip, hole[:3])):
                obs = env.unwrapped.get_obs()
                tip, hole, rel, peg_z = geometry(env)
            log.setdefault("moff", []).append(round(float(np.linalg.norm(mover.applied)) * 1000, 1))
        if log.get("trigger") is not None and t >= 20 and peg_z > log["peg_z0"] + 0.02 \
                and np.linalg.norm(tip - hole[:3]) < cfg.trigger_m:
            log["trigger"](t, obs)
        o21, robot9 = perceive(env, graph, obs, cfg.oracle)
        log["err"].append(float(np.linalg.norm(o21[7:10] - (tip - hole[:3])) * 1000))
        log["lat"].append(float(np.linalg.norm(rel[1:]) * 1000))
        log["ghole"].append(float(np.linalg.norm(o21[10:13] + robot9[:3] - hole[:3]) * 1000))
        gt = gt_object21(env)
        log.setdefault("epeg", []).append(float(np.linalg.norm(o21[0:3] - gt[0:3]) * 1000))
        log.setdefault("ehole", []).append(float(np.linalg.norm(o21[10:13] - gt[10:13]) * 1000))
        log.setdefault("ntr", []).append(0 if cfg.oracle else len(graph.tracker.tracks))
        log["zmax"] = max(log.get("zmax", 0.0), float(peg_z - log["peg_z0"]))
        log["final_rel"] = [round(float(v) * 1000, 2) for v in rel]
        a = dec.act(np.concatenate([robot15(env), o21]), None if cfg.oracle else graph.counts)
        d = q_rot(q_conj(hole[3:]), dec.last[22:25] - (tip - hole[:3])) * 1000
        log.setdefault("fyz", []).append([round(float(d[0]), 2), round(float(d[1]), 2), round(float(d[2]), 2)])
        obs, _, _, _, info = env.step(torch.as_tensor(a[None], device=env.unwrapped.device))
        if is_success(info):
            success = True
            break
    return success, t + 1


def run_job(env, graph, dec, eseed, cfg):
    obs, _ = env.reset(seed=eseed)
    graph.reset()
    dec.reset()
    peg_z0 = geometry(env)[3]
    saved = {}

    def trigger(t, cur_obs):
        saved.update(t=t, state=env.unwrapped.get_state_dict(), obs=copy.deepcopy(cur_obs),
                     tracks=copy.deepcopy(graph.tracker.tracks), dec=dec.snapshot(),
                     err=list(log["err"]), lat=list(log["lat"]), ghole=list(log["ghole"]))
        log["trigger"] = None

    log = {"err": [], "lat": [], "ghole": [], "peg_z0": peg_z0, "trigger": trigger}
    motion = mover = None
    if cfg.motion != "none":
        motion, mover = HoleMotion(cfg, sample_translation(eseed, cfg)), BoxMover(peg_task(env))
        log["trigger"] = None
    ok, steps = loop(env, graph, dec, obs, 0, cfg, log, motion, mover)
    err = np.array(log["err"])
    none = {"success": ok, "steps": steps, "triggered": "t" in saved, "trigger_step": saved.get("t"),
            "err_mean_mm": round(float(err.mean()), 2), "err_p90_mm": round(float(np.percentile(err, 90)), 2),
            "err_mm": [round(e, 1) for e in log["err"]],
            "lifted": bool(log.get("zmax", 0.0) > 0.02),
            "epeg_mm": [round(e, 1) for e in log.get("epeg", [])],
            "ehole_mm": [round(e, 1) for e in log.get("ehole", [])],
            "ntr": log.get("ntr", []), "final_rel_mm": log.get("final_rel"), "fxyz_mm": log.get("fyz", []),
            "motion": None if motion is None else motion.summary(), "moff_mm": log.get("moff", []),
            "searches": None if dec.search is None else dec.search.searches,
            "pushes": None if dec.search is None else dec.search.pushes}
    if cfg.no_move or "t" not in saved:
        return none, {"triggered": False}
    sign = 1.0 if eseed % 2 == 0 else -1.0
    if cfg.branch:
        env.unwrapped.set_state_dict(saved["state"])
        graph.tracker.tracks = copy.deepcopy(saved["tracks"])
        dec.restore(saved["dec"])
        obs, t0 = saved["obs"], saved["t"]
        mlog = {"err": saved["err"], "lat": saved["lat"], "ghole": saved["ghole"], "peg_z0": peg_z0, "trigger": None}
    else:
        obs, _ = env.reset(seed=eseed)
        graph.reset()
        dec.reset()
        mlog = {"err": [], "lat": [], "ghole": [], "peg_z0": peg_z0, "trigger": None}
        target = saved["t"]
        for t in range(target):
            tip, hole, rel, _ = geometry(env)
            o21, robot9 = perceive(env, graph, obs, cfg.oracle)
            mlog["err"].append(float(np.linalg.norm(o21[7:10] - (tip - hole[:3])) * 1000))
            mlog["lat"].append(float(np.linalg.norm(rel[1:]) * 1000))
            mlog["ghole"].append(float(np.linalg.norm(o21[10:13] + robot9[:3] - hole[:3]) * 1000))
            a = dec.act(np.concatenate([robot15(env), o21]), None if cfg.oracle else graph.counts)
            obs, *_ = env.step(torch.as_tensor(a[None], device=env.unwrapped.device))
        t0 = target
    hole_q = geometry(env)[1][3:]
    shift_box(env, hole_q, cfg.move_mm / 1000, sign)
    ok, steps = loop(env, graph, dec, obs, t0, cfg, mlog)
    lat, gh = np.array(mlog["lat"]), np.array(mlog["ghole"])
    catch = next((k for k in range(t0, len(gh)) if gh[k] < 3.0), None)
    react = reaction(lat, t0)
    move = {"triggered": True, "success": ok, "steps": steps, "move_step": t0,
            "graph_catch_steps": None if catch is None else catch - t0,
            "reaction_steps": react,
            "reaction_after_graph_steps": None if react is None or catch is None else react - (catch - t0)}
    return none, move


def accepted_seeds(count, start, path):
    if Path(path).is_file():
        seeds = json.loads(Path(path).read_text())
        if len(seeds) >= count:
            return seeds[:count]
    env = make_env(16)
    seeds, s = [], start
    while len(seeds) < count:
        obs, _ = env.reset(seed=s)
        result, _ = initial_visibility(env, obs)
        if result["requested_pass"]:
            seeds.append(s)
        s += 1
    env.close()
    Path(path).write_text(json.dumps(seeds))
    return seeds


class Ensemble:
    def __init__(self, decs):
        self.decs = decs
        self.search = None
        self.kf = decs[0].kf

    def reset(self):
        for d in self.decs:
            d.reset()

    def snapshot(self):
        return [d.snapshot() for d in self.decs]

    def restore(self, snap):
        for d, s in zip(self.decs, snap):
            d.restore(s)

    def act(self, state36, counts=None):
        acts = np.stack([d.act(state36, counts) for d in self.decs])
        self.last = self.decs[0].last
        move = acts[:, :6].mean(0)
        return np.concatenate([move, [1.0 if acts[:, 6].mean() > 0 else -1.0]]).astype(np.float32)


def load_ck(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def worker(wid, cfg, queue, out_path):
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    env = make_env(16)
    paths = json.load(open(cfg.paths))
    graph = Graph(paths["gnn"], paths["corner"], "cuda")
    cache = {}
    with open(out_path, "a") as out:
        while True:
            job = queue.get()
            if job is None:
                break
            cand, tseed, eseed = job
            typ, mode = cand.split("@")
            if (typ, tseed) not in cache:
                cache.clear()
                if typ.startswith("ens"):
                    spec = json.load(open(cfg.ens))[typ]
                    cache[(typ, tseed)] = [load_ck(Path(d) / f"{m}_s{sd}" / "last.pt") for d, m, sd in spec]
                else:
                    cache[(typ, tseed)] = load_ck(Path(cfg.runs) / f"{typ}_s{tseed}" / "last.pt")
            ck = cache[(typ, tseed)]
            dec = Ensemble([Decoder(c, mode) for c in ck]) if isinstance(ck, list) else Decoder(ck, mode, search=cfg.search)
            t0 = time.time()
            row, peak = None, 0.0
            for attempt in range(cfg.retries + 1):
                try:
                    torch.cuda.reset_peak_memory_stats()
                    none, move = run_job(env, graph, dec, eseed, cfg)
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    row = {"cand": cand, "train_seed": tseed, "env_seed": eseed, "none": none, "move": move,
                           "sec": round(time.time() - t0, 1), "attempts": attempt + 1, "peak_gib": round(peak, 2)}
                    break
                except torch.cuda.OutOfMemoryError as e:
                    row = {"cand": cand, "train_seed": tseed, "env_seed": eseed, "error": repr(e)[:300]}
                    graph.reset()
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(5 + 10 * attempt)
                except Exception as e:
                    row = {"cand": cand, "train_seed": tseed, "env_seed": eseed, "error": repr(e)[:300]}
                    break
            graph.reset()
            gc.collect()
            torch.cuda.empty_cache()
            out.write(json.dumps(row) + "\n")
            out.flush()
    env.close()


def run(cfg):
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = accepted_seeds(cfg.env_offset + cfg.episodes, cfg.seed_start, out / "env_seeds.json")
    eseeds = seeds[cfg.env_offset:cfg.env_offset + cfg.episodes]
    cands = all_candidates() if cfg.cands == "all" else cfg.cands.split(",")
    done = set()
    for f in out.glob("results_*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["cand"], r["train_seed"], r["env_seed"]))
    jobs = [(c, s, e) for c in cands for s in seed_list(cfg.train_seeds) for e in eseeds if (c, s, e) not in done]
    jobs.sort(key=lambda j: (j[0].split("@")[0], j[1]))
    print(f"jobs {len(jobs)} done {len(done)} workers {cfg.workers} branch {cfg.branch}", flush=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for j in jobs:
        queue.put(j)
    for _ in range(cfg.workers):
        queue.put(None)
    stamp = int(time.time())
    procs = [ctx.Process(target=worker, args=(i, cfg, queue, out / f"results_{stamp}_{i}.jsonl"))
             for i in range(cfg.workers)]
    start = time.time()
    for p in procs:
        p.start()
    total = len(jobs)
    while any(p.is_alive() for p in procs):
        time.sleep(60)
        n = sum(len(f.read_text().splitlines()) for f in out.glob(f"results_{stamp}_*.jsonl"))
        el = time.time() - start
        eta = el / max(n, 1) * (total - n)
        print(f"progress {n}/{total} elapsed_min {el / 60:.1f} eta_min {eta / 60:.1f}", flush=True)
    print("DONE", flush=True)


def report(cfg):
    rows = []
    for f in Path(cfg.out).glob("results_*.jsonl"):
        rows += [json.loads(l) for l in f.read_text().splitlines()]
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    by = {}
    for r in rows:
        by.setdefault(r["cand"], {}).setdefault(r["train_seed"], []).append(r)
    table = {}
    for cand, seeds in sorted(by.items()):
        none_rates, move_rates, reacts, catches, errs, trig = [], [], [], [], [], []
        for s, rs in seeds.items():
            none_rates.append(np.mean([r["none"]["success"] for r in rs]))
            mv = [r for r in rs if r["move"].get("triggered")]
            move_rates.append(np.mean([r["move"]["success"] if r["move"].get("triggered") else r["none"]["success"] for r in rs]))
            reacts += [r["move"]["reaction_steps"] for r in mv if r["move"]["reaction_steps"] is not None]
            catches += [r["move"]["graph_catch_steps"] for r in mv if r["move"]["graph_catch_steps"] is not None]
            errs += [r["none"]["err_mean_mm"] for r in rs]
            trig += [bool(r["move"].get("triggered")) for r in rs]
        n, m = np.array(none_rates) * 100, np.array(move_rates) * 100
        score = (n + m) / 2
        table[cand] = {"seeds": len(seeds), "episodes": sum(len(v) for v in seeds.values()),
                       "none": round(float(n.mean()), 1), "none_sd": round(float(n.std()), 1),
                       "move": round(float(np.nanmean(m)), 1), "move_sd": round(float(np.nanstd(m)), 1),
                       "score": round(float(np.nanmean(score)), 1),
                       "score_ci95": round(float(1.96 * np.nanstd(score) / max(np.sqrt(len(score)), 1)), 1),
                       "reaction_med": None if not reacts else float(np.median(reacts)),
                       "graph_catch_med": None if not catches else float(np.median(catches)),
                       "err_mean_mm": round(float(np.mean(errs)), 2), "triggered": round(float(np.mean(trig)) * 100, 1)}
    print("cand  seeds  eps  none(sd)  move(sd)  score±ci  react_med  catch_med  err_mm  reach%")
    for cand, v in sorted(table.items(), key=lambda kv: -kv[1]["score"]):
        print(f"{cand}  {v['seeds']}  {v['episodes']}  {v['none']}({v['none_sd']})  {v['move']}({v['move_sd']})  "
              f"{v['score']}±{v['score_ci95']}  {v['reaction_med']}  {v['graph_catch_med']}  {v['err_mean_mm']}  {v['triggered']}")
    for axis, key in (("model", lambda c: c.split("_")[0] if not c.startswith("stack") else "stack_mlp"),
                      ("mode", lambda c: c.split("@")[1])):
        groups = {}
        for c, v in table.items():
            if "nohf" in c or c.startswith("mlp_"):
                continue
            groups.setdefault(key(c), []).append(v["score"])
        print(axis + " " + json.dumps({k: round(float(np.mean(v)), 1) for k, v in sorted(groups.items())}))
    srch = [r["none"] for r in rows if r["none"].get("searches") is not None]
    if srch:
        used = [x for x in srch if x["searches"]]
        print(f"search used {len(used)}/{len(srch)} success_when_used {100 * np.mean([x['success'] for x in used]) if used else 0:.1f}% "
              f"found {sum(1 for x in used if x['pushes'])}")
    fails = [r["none"] for r in rows if not r["none"]["success"] and r["none"].get("final_rel_mm")]
    if fails:
        fx = np.array([x["final_rel_mm"] for x in fails])
        lat = np.linalg.norm(fx[:, 1:], axis=1)
        print(f"fail_end axial_mm median {np.median(fx[:, 0]):.1f} near_entrance(<30mm,lat<10mm) {100 * np.mean((np.abs(fx[:, 0]) < 30) & (lat < 10)):.1f}% "
              f"lat_mm median {np.median(lat):.1f}")
    print(f"rows {len(rows)} errors {len(errors)}")
    Path(cfg.out).joinpath("report.json").write_text(json.dumps({"table": table, "errors": errors[:20]}, indent=1))


def debug(cfg):
    from data_collection.scripted_policy import make_policy
    seeds = accepted_seeds(1, cfg.seed_start, Path(cfg.out) / "env_seeds.json") if not cfg.env_seed else [cfg.env_seed]
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    env = make_env(16)
    paths = json.load(open(cfg.paths))
    graph = Graph(paths["gnn"], paths["corner"], "cuda")
    typ, mode = cfg.cands.split("@")
    ck = torch.load(Path(cfg.runs) / f"{typ}_s{cfg.train_seeds.split('-')[0]}" / "last.pt", map_location="cpu", weights_only=False)
    dec = Decoder(ck, mode)
    expert = make_policy()
    for who in ("expert", "decoder"):
        obs, _ = env.reset(seed=seeds[0])
        graph.reset()
        dec.reset()
        expert.reset(env, seeds[0])
        z0 = geometry(env)[3]
        print(f"== {who} seed {seeds[0]}")
        print("t  peg_dz_mm  tip_hole_mm  err_mm  tracks  dec_move  exp_move  dec_grip  exp_grip")
        for t in range(120):
            tip, hole, rel, pz = geometry(env)
            o21, robot9, _, n = (gt_object21(env), None, None, 0) if cfg.oracle else graph.step(obs)
            err = np.linalg.norm(o21[7:10] - (tip - hole[:3])) * 1000
            s36 = np.concatenate([robot15(env), o21])
            a_dec = dec.act(s36)
            a_exp = np.asarray(to_np(expert.act(obs))).reshape(-1)
            if t % 10 == 0:
                print(f"{t}  {(pz - z0) * 1000:.0f}  {np.linalg.norm(tip - hole[:3]) * 1000:.0f}  {err:.1f}  {n}  "
                      f"{np.round(a_dec[:3], 2).tolist()}  {np.round(a_exp[:3], 2).tolist()}  {a_dec[6]:.0f}  {a_exp[6]:.0f}", flush=True)
            a = a_exp if who == "expert" else a_dec
            obs, _, _, _, info = env.step(torch.as_tensor(a[None], device=env.unwrapped.device))
            if is_success(info):
                print(f"success at {t + 1}")
                break
    env.close()


def project(p, obs):
    par = obs["sensor_param"]["base_camera"]
    k, e = to_np(par["intrinsic_cv"])[0], to_np(par["extrinsic_cv"])[0]
    c = e[:, :3] @ p + e[:, 3]
    uv = k @ c
    return int(uv[0] / uv[2]), int(uv[1] / uv[2])


def record_episode(env, graph, dec, eseed, cfg, move, label):
    import cv2
    from data_collection.scripted_policy import make_policy
    expert = make_policy() if dec is None else None
    obs, _ = env.reset(seed=eseed)
    graph.reset()
    if dec is not None:
        dec.reset()
    else:
        expert.reset(env, eseed)
    z0 = geometry(env)[3]
    frames, moved, ok = [], None, False
    for t in range(HORIZON):
        tip, hole, rel, pz = geometry(env)
        if move and moved is None and t >= 20 and pz > z0 + 0.02 and np.linalg.norm(tip - hole[:3]) < cfg.trigger_m:
            shift_box(env, hole[3:], cfg.move_mm / 1000, 1.0 if eseed % 2 == 0 else -1.0)
            moved = t
            tip, hole, rel, pz = geometry(env)
        o21, robot9 = perceive(env, graph, obs, cfg.oracle)
        img = cv2.cvtColor(to_np(obs["sensor_data"]["base_camera"]["rgb"])[0].astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
        g_hole = o21[10:13] + robot9[:3]
        for pt, col in ((hole[:3], (0, 200, 0)), (tip, (0, 200, 0)), (g_hole, (0, 0, 255)), (g_hole + o21[7:10], (0, 0, 255))):
            cv2.circle(img, project(pt, obs), 5, col, 2)
        err = np.linalg.norm(o21[7:10] - (tip - hole[:3])) * 1000
        lines = [label, f"t {t}  graph err {err:.1f} mm  lateral {np.linalg.norm(rel[1:]) * 1000:.1f} mm"]
        if moved is not None:
            lines.append(f"box moved {cfg.move_mm:.0f} mm at t {moved}")
        for i, s in enumerate(lines):
            cv2.putText(img, s, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        frames.append(img)
        if dec is None:
            a = np.asarray(to_np(expert.act(obs)), np.float32).reshape(-1)
        else:
            a = dec.act(np.concatenate([robot15(env), o21]), None if cfg.oracle else graph.counts)
        obs, _, _, _, info = env.step(torch.as_tensor(a[None], device=env.unwrapped.device))
        if is_success(info):
            ok = True
            break
    for _ in range(15):
        f = frames[-1].copy()
        cv2.putText(f, "SUCCESS" if ok else "FAIL", (8, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 220, 0) if ok else (0, 0, 255), 3)
        frames.append(f)
    return frames, ok


def video(cfg):
    import cv2
    import imageio
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(16)
    paths = json.load(open(cfg.paths))
    graph = Graph(paths["gnn"], paths["corner"], "cuda")
    eseed = cfg.env_seed or accepted_seeds(1, cfg.seed_start, out / "env_seeds.json")[0]
    tseed = int(cfg.train_seeds.split("-")[0].split(",")[0])
    clips = []
    for cand in cfg.cands.split(","):
        if cand == "expert":
            dec, label = None, "expert"
        else:
            typ, mode = cand.split("@")
            if typ.startswith("ens"):
                spec = json.load(open(cfg.ens))[typ]
                dec, label = Ensemble([Decoder(load_ck(Path(d) / f"{m}_s{sd}" / "last.pt"), mode) for d, m, sd in spec]), f"{typ} ({len(spec)} models)"
            else:
                ck = torch.load(Path(cfg.runs) / f"{typ}_s{tseed}" / "last.pt", map_location="cpu", weights_only=False)
                dec, label = Decoder(ck, mode, search=cfg.search), f"{cand} seed {tseed}" + (" +search" if cfg.search else "")
        frames, ok = record_episode(env, graph, dec, eseed, cfg, cfg.move, label + (" | move" if cfg.move else ""))
        name = f"{cand.replace('@', '_')}_env{eseed}{'_move' if cfg.move else ''}.mp4"
        imageio.mimsave(out / name, [f[..., ::-1] for f in frames], fps=20)
        clips.append(frames)
        print(f"{'PASS' if ok else 'FAIL'} {name} frames {len(frames)}", flush=True)
    if len(clips) > 1:
        n = max(len(c) for c in clips)
        side = [np.concatenate([c[min(i, len(c) - 1)] for c in clips], axis=1) for i in range(n)]
        name = f"side_by_side_env{eseed}{'_move' if cfg.move else ''}.mp4"
        imageio.mimsave(out / name, [f[..., ::-1] for f in side], fps=20)
        print(f"saved {name}", flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "report", "debug", "video"])
    p.add_argument("--move", action="store_true")
    p.add_argument("--no-move", action="store_true")
    p.add_argument("--search", action="store_true")
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--out", default="/workspace/eval/stage1")
    p.add_argument("--runs", default="/workspace/exp2/runs_dart")
    p.add_argument("--paths", default="/workspace/graph_ckpt/paths.json")
    p.add_argument("--cands", default="all")
    p.add_argument("--ens", default="")
    add_args(p)
    p.add_argument("--train-seeds", default="0-4")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--env-offset", type=int, default=0)
    p.add_argument("--seed-start", type=int, default=700000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--move-mm", type=float, default=8.0)
    p.add_argument("--trigger-m", type=float, default=0.06)
    p.add_argument("--no-branch", dest="branch", action="store_false")
    cfg = p.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    {"run": run, "report": report, "debug": debug, "video": video}[cfg.cmd](cfg)


if __name__ == "__main__":
    main()
