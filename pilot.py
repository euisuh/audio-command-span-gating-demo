"""Frozen defensive pilot. Sandbox actions are JSON only."""
from runtime import *
import argparse
from pathlib import Path
import datetime
import re
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ARMS = ["no_gate", "utterance", "word_span", "signed_chunks"]
OUT = ROOT / "out"
PLAN = json.loads((ROOT / "allocation.json").read_text())
PATTERNS = {k: v.split() for k, v in PLAN["actions"].items()}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cache_bytes():
    # Count physical files once; HF snapshot symlinks do not duplicate storage.
    return sum(p.stat().st_size for p in CACHE.rglob("*")
               if p.is_file() and not p.is_symlink())


def resources():
    n = cache_bytes()
    assert n < 4_000_000_000, f"Cache limit exceeded: {n}"
    return n


def record(filename, row):
    path = OUT / filename
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def audio_file(name, audio):
    path = CACHE / "audio" / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SR, subtype="FLOAT")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "samples": len(audio), "seconds": len(audio) / SR}


def normalize(word):
    return re.sub("[^a-z]", "", word.lower())


def matches(words, admitted):
    assert len(words) == len(admitted)
    norm = [normalize(w["word"]) for w in words]
    actions = []
    for i in range(len(norm)):
        for action, pattern in PATTERNS.items():
            j = i + len(pattern)
            if norm[i:j] == pattern and all(admitted[i:j]):
                actions.append({"action": action, "word_indices": list(range(i, j))})
    return actions


def word_score(p, word):
    start, end = word["start"], word["end"]
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= len(p) / SR:
        return None
    a = max(0, int(np.floor((start - 0.05) * SR)))
    b = min(len(p), int(np.ceil((end + 0.05) * SR)))
    return float(np.quantile(p[a:b], 0.1, method="linear")) if b > a else None


def infer(detector, asr, audio, name):
    t = time.perf_counter()
    p, bits = detect(detector, audio)
    detection_seconds = time.perf_counter() - t
    assert len(p) == len(audio), "Localization length mismatch"
    t = time.perf_counter()
    words = transcribe(asr, audio)
    asr_seconds = time.perf_counter() - t
    scores = [word_score(p, w) for w in words]
    path = CACHE / "probabilities" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, p)
    return {
        "words": words, "word_span_scores": scores, "bits": bits,
        "exact_message": bits == MESSAGE, "clip_score": float(np.mean(p > 0.5)),
        "max_word_score": max((s for s in scores if s is not None), default=0.0),
        "detection_seconds": detection_seconds, "asr_seconds": asr_seconds,
        "probabilities_path": str(path.relative_to(ROOT)), "probabilities_sha256": sha(path),
    }, p


def waveform_actions(inference, thresholds):
    words = inference["words"]
    exact = inference["exact_message"]
    clip = exact and inference["clip_score"] > thresholds["clip"]
    spans = [exact and s is not None and s > thresholds["span"]
             for s in inference["word_span_scores"]]
    return {
        "no_gate": matches(words, [True] * len(words)),
        "utterance": matches(words, [clip] * len(words)),
        "word_span": matches(words, spans),
    }


def payload(session, index, audio):
    raw = np.asarray(audio, dtype="<f4").tobytes()
    obj = {"session": session, "index": index, "sample_rate": SR,
           "samples": len(audio), "sha256": hashlib.sha256(raw).hexdigest()}
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def packet(key, session, index, audio):
    return {"session": session, "index": index, "audio": audio,
            "signature": key.sign(payload(session, index, audio))}


def reconstruct(public, session, packets):
    accepted = {}
    checks = []
    for item in packets:
        valid = False
        reason = "unsigned"
        index = item.get("index")
        if item.get("signature") is not None:
            reason = "invalid_session_or_index"
            if item.get("session") == session and type(index) is int and index >= 0:
                try:
                    public.verify(item["signature"], payload(session, index, item["audio"]))
                    reason = "duplicate" if index in accepted else "valid"
                    if index not in accepted:
                        accepted[index] = item["audio"]
                        valid = True
                except (InvalidSignature, ValueError, TypeError):
                    reason = "invalid_signature"
        checks.append({"index": index, "accepted": valid, "reason": reason,
                       "signature": item.get("signature", b"").hex()})
    audio = (np.concatenate([accepted[i] for i in sorted(accepted)])
             if accepted else np.empty(0, dtype=np.float32))
    return audio, checks


def transitions(p, threshold, truth):
    mask = p > threshold
    indices = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    found = []
    for sample, direction in truth:
        choices = indices[mask[indices] == direction]
        if len(choices):
            difference = int(np.min(np.abs(choices - sample)))
            error = difference / SR * 1000 if difference <= SR // 2 else None
        else:
            error = None
        found.append({"true_sample": sample, "to_marked": direction, "error_ms": error})
    return found


def extra_timing(words, action, interval):
    recognized = matches(words, [True] * len(words))
    out = []
    for m in recognized:
        if m["action"] == action:
            start = words[m["word_indices"][0]]["start"]
            end = words[m["word_indices"][-1]]["end"]
            out.append({"start_error_ms": abs(start - interval[0] / SR) * 1000,
                        "end_error_ms": abs(end - interval[1] / SR) * 1000})
    return out


def run_nulls(tts, detector, asr, split, thresholds=None):
    rows = []
    for spec in PLAN[split]:
        t = time.perf_counter()
        audio = synth(tts, spec["text"], spec["voice"], spec["speed"])
        tts_seconds = time.perf_counter() - t
        info = audio_file(spec["id"], audio)
        record("inputs.jsonl", {"created_at": now(), "split": split, **spec, **info})
        inference, _ = infer(detector, asr, audio, spec["id"])
        row = {"id": spec["id"], "voice": spec["voice"], **info, **inference,
               "tts_seconds": tts_seconds}
        if thresholds is not None:
            row["actions"] = waveform_actions(inference, thresholds)
            row["score_clip_admitted"] = row["clip_score"] > thresholds["clip"]
            row["score_word_admitted"] = row["max_word_score"] > thresholds["span"]
            row["exact_clip_admitted"] = row["exact_message"] and row["score_clip_admitted"]
            row["exact_word_admitted"] = row["exact_message"] and row["score_word_admitted"]
            row["actions"]["signed_chunks"] = []  # No signed packets in null input.
        record(f"{split}.jsonl", row)
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"{split}: {len(rows)}/{len(PLAN[split])}", flush=True)
            resources()
    return rows


def run_pairs(tts, generator, detector, asr, thresholds, limit):
    key = Ed25519PrivateKey.generate()
    public = key.public_key()
    dump("out/issuer-test-public-key.json",
         {"public_key_hex": public.public_bytes_raw().hex(), "purpose": "local pilot only"})
    for number, spec in enumerate(PLAN["pairs"][:limit], 1):
        ident = spec["id"]
        t = time.perf_counter()
        command = synth(tts, spec["command_text"], spec["voice"], spec["speed"])
        filler = synth(tts, spec["filler_text"], spec["voice"], spec["speed"])
        extra = synth(tts, spec["extra_text"], spec["extra_voice"], spec["speed"])
        tts_seconds = time.perf_counter() - t
        t = time.perf_counter()
        benign = mark(generator, np.concatenate([command, filler]))
        embedding_seconds = time.perf_counter() - t
        c0, c1 = benign[:len(command)], benign[len(command):]
        signed = [packet(key, ident, 0, c0), packet(key, ident, 1, c1)]
        unsigned = {"audio": extra}
        if spec["construction"] == "append":
            packets = [*signed, unsigned]
            extra_interval = [len(benign), len(benign) + len(extra)]
            truth = [(len(benign), False)]
            intervals = [[0, len(benign)]]
        else:
            packets = [signed[0], unsigned, signed[1]]
            extra_interval = [len(c0), len(c0) + len(extra)]
            truth = [(len(c0), False), (len(c0) + len(extra), True)]
            intervals = [[0, len(c0)], [len(c0) + len(extra), len(benign) + len(extra)]]
        mixed = np.concatenate([p["audio"] for p in packets])
        signed_records = []
        for i, p in enumerate(packets):
            packet_info = audio_file(f"{ident}-packet-{i}", p["audio"])
            signed_records.append({**packet_info,
                                   "payload": payload(ident, p["index"], p["audio"]).decode()
                                   if "index" in p else None,
                                   "signature": p.get("signature", b"").hex()})
        clean_info = audio_file(f"{ident}-benign", benign)
        mixed_info = audio_file(f"{ident}-mixed", mixed)
        record("inputs.jsonl", {"created_at": now(), "split": "pairs", **spec,
                                "benign": clean_info, "mixed": mixed_info,
                                "packets": signed_records, "marked_intervals": intervals,
                                "extra_interval": extra_interval})
        clean_result, _ = infer(detector, asr, benign, f"{ident}-benign")
        mixed_result, p = infer(detector, asr, mixed, f"{ident}-mixed")
        clean_actions = waveform_actions(clean_result, thresholds)
        mixed_actions = waveform_actions(mixed_result, thresholds)
        for name, received, original, inference, actions in [
            ("benign", signed, benign, clean_result, clean_actions),
            ("mixed", packets, mixed, mixed_result, mixed_actions),
        ]:
            recovered, checks = reconstruct(public, ident, received)
            # Reuse only byte-identical verified audio, never expected command text.
            if recovered.tobytes() == benign.tobytes():
                signed_words = clean_result["words"]
                reused = True
            else:
                signed_words = transcribe(asr, recovered) if len(recovered) else []
                reused = False
            actions["signed_chunks"] = matches(signed_words, [True] * len(signed_words))
            inference["signed_checks"] = checks
            inference["signed_asr_reused_identical_benign"] = reused
            inference["signed_reconstructed_sha256"] = hashlib.sha256(recovered.tobytes()).hexdigest()
        row = {**spec, "benign": clean_result, "mixed": mixed_result,
               "benign_actions": clean_actions, "mixed_actions": mixed_actions,
               "tts_seconds": tts_seconds, "embedding_seconds": embedding_seconds,
               "boundary_errors": transitions(p, thresholds["span"], truth),
               "extra_timing": extra_timing(mixed_result["words"], spec["unauthorized"], extra_interval)}
        for condition, actions in [("benign", clean_actions), ("mixed", mixed_actions)]:
            row[f"{condition}_completion"] = {
                arm: any(a["action"] == spec["authorized"] for a in actions[arm]) for arm in ARMS}
            row[f"{condition}_unauthorized"] = {
                arm: any(a["action"] != spec["authorized"] for a in actions[arm]) for arm in ARMS}
        record("pairs.jsonl", row)
        print(f"pairs: {number}/{limit} ({ident}, {spec['construction']})", flush=True)
        resources()


def selfcheck():
    words = [{"word": w, "start": i / 10, "end": (i + 1) / 10}
             for i, w in enumerate(["Please", "open", "the", "door.", "play", "music."])]
    assert [a["action"] for a in matches(words, [True] * 6)] == ["MUSIC_PLAY"]
    assert matches(words, [True, True, False, True, False, True]) == []
    assert word_score(np.ones(SR), {"start": 0, "end": 0}) is None
    assert word_score(np.ones(SR), {"start": -0.1, "end": 0.1}) is None
    assert word_score(np.ones(SR), {"start": 0.1, "end": 0.2}) == 1
    key = Ed25519PrivateKey.generate()
    audio = np.arange(100, dtype=np.float32)
    p = packet(key, "session", 0, audio)
    recovered, checks = reconstruct(key.public_key(), "session", [p, p, {"audio": audio}])
    assert np.array_equal(recovered, audio) and [x["reason"] for x in checks] == ["valid", "duplicate", "unsigned"]
    assert not len(reconstruct(key.public_key(), "other", [p])[0])
    assert not len(reconstruct(key.public_key(), "session", [{**p, "audio": audio + 1}])[0])
    p = np.r_[np.ones(100), np.zeros(100)].astype(np.float32)
    assert transitions(p, 0.5, [(100, False)])[0]["error_ms"] == 0
    assert len(PLAN["pairs"]) == 60 and len(PLAN["calibration"]) == 100 and len(PLAN["audit"]) == 300
    splits = [{s["voice"] for s in PLAN[k]} for k in ["calibration", "audit", "pairs"]]
    assert all(not (splits[i] & splits[j]) for i in range(3) for j in range(i))
    print("PASS: parser, fail-closed spans, signatures, boundaries, allocation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--pairs", type=int, default=60, help="number of pairs to run (1-60)")
    parser.add_argument("--thresholds", help="reuse a thresholds.json (skips calibration and audit)")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    for file in ["calibration.jsonl", "audit.jsonl", "pairs.jsonl", "inputs.jsonl"]:
        assert not (OUT / file).exists(), f"Refuse overwrite: {OUT / file}; delete out/ first"
    started = time.perf_counter()
    mode = "full" if args.pairs == 60 and not args.thresholds else "reduced"
    tts = tts_model()
    generator, detector, asr = models()
    if args.thresholds:
        thresholds = json.loads(Path(args.thresholds).read_text())
        dump("out/thresholds.json", thresholds)
    else:
        cal = run_nulls(tts, detector, asr, "calibration")
        thresholds = {"clip": sorted(r["clip_score"] for r in cal)[98],
                      "span": sorted(r["max_word_score"] for r in cal)[98],
                      "rule": "strict > order statistic index 98 of 100 nulls",
                      "created_at": now(), "calibration_sha256": sha(OUT / "calibration.jsonl")}
        dump("out/thresholds.json", thresholds)
        print(f"calibration done: clip={thresholds['clip']}, span={thresholds['span']}", flush=True)
        run_nulls(tts, detector, asr, "audit", thresholds)
    run_pairs(tts, generator, detector, asr, thresholds, args.pairs)
    dump("out/run-complete.json", {"finished_at": now(), "elapsed_seconds": time.perf_counter() - started,
                                   "mode": mode, "cache_bytes": resources(),
                                   "threshold_sha256": sha(OUT / "thresholds.json")})
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
