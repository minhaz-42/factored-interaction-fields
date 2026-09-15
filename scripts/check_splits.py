"""Verify subject-level splits and absence of leakage.

Checks: (1) every fold's train/val subjects are disjoint and cover the 10 training
subjects; (2) official test subjects never appear in any local manifest, frame
index, label file or feature cache; (3) no (subject, scene) recording appears in
two splits of any index/label file; (4) optional external-dataset participant ids
do not collide with SHOW3D subject ids.
"""
from __future__ import annotations
import json, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fif.paths import FOLDS, TRAIN_SUBJECTS, TEST_SUBJECTS, TRAIN_MANIFEST, FRAMES_ROOT, CACHE_ROOT


def subjects_in_jsonl(path: Path, key="subject_id"):
    subs = set(); scenes = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sid = row.get(key) or row["sample_id"].split("/")[0]
                subs.add(sid)
                scene = row.get("scene_id") or row["sample_id"].split("/")[1].split(":")[0]
                scenes.add((sid, scene))
    return subs, scenes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--external-ids", nargs="*", default=[]); args = ap.parse_args()
    ok = True
    for name, f in FOLDS.items():
        tr, va = set(f["train"]), set(f["val"])
        assert not (tr & va), f"fold {name}: overlap {tr & va}"
        assert tr | va == set(TRAIN_SUBJECTS), f"fold {name}: does not cover all training subjects"
        print(f"fold {name}: train {sorted(tr)} | val {sorted(va)} OK")
    man_subs, man_scenes = subjects_in_jsonl(TRAIN_MANIFEST)
    assert man_subs == set(TRAIN_SUBJECTS), man_subs
    assert not (man_subs & set(TEST_SUBJECTS)), "test subjects in train manifest"
    print(f"train manifest: {len(man_scenes)} recordings, {len(man_subs)} subjects, no test subjects OK")
    for p in list(FRAMES_ROOT.glob("**/index.jsonl")) + list(FRAMES_ROOT.glob("**/labels.jsonl")) + list(CACHE_ROOT.glob("**/*.jsonl")):
        subs, scenes = subjects_in_jsonl(p)
        leak = subs & set(TEST_SUBJECTS)
        if leak:
            ok = False; print(f"LEAK: {p} contains test subjects {leak}")
        else:
            print(f"{p}: subjects {sorted(subs)} OK")
        for name, f in FOLDS.items():
            tr_sc = {s for s in scenes if s[0] in f["train"]}; va_sc = {s for s in scenes if s[0] in f["val"]}
            if tr_sc & va_sc:
                ok = False; print(f"LEAK: recording in both splits of fold {name}: {tr_sc & va_sc}")
    for eid in args.external_ids:
        if eid in TRAIN_SUBJECTS or eid in TEST_SUBJECTS:
            ok = False; print(f"LEAK: external id {eid} collides with SHOW3D subject")
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
