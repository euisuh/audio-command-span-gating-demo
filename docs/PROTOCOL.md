# Pilot preregistration (original text)

Original preregistration text from the research pilot this demo was extracted from. It was written before the run; file names such as freeze.json and sources/ refer to the original project, not to this repository. The demo repository drops the freeze-hash checks.

## Scope and hypothesis

Test a defensive authorization prototype: does word-span admission reduce unauthorized sandbox actions by at least 15 percentage points versus utterance admission, with at most 5 points loss of benign completion? Attacker may concatenate fixed unmarked extra commands beside genuine marked speech or insert them at one fixed phrase boundary. No adversarial search, detector/ASR optimization, retries to improve recognition, removal, transplantation, or forging. Issuer enrollment is a pinned 16-bit message, 1010010110100101. This is conditional enrollment, not cryptographic authentication of a human or secure deployment. Public AudioSeal embedding would invalidate the assumed inability to issue authorized marks. [AudioSeal API](https://github.com/facebookresearch/audioseal).

CPU only: OMP_NUM_THREADS=2, torch two intra-op threads, no CUDA/MPS, ASR int8 with two CPU threads, ONNX CPU provider with two intra-op threads. $0 incremental spending, no posting/email/git. Environment, downloads, generated audio and caches under cache/, total <4,000,000,000 bytes. Source logs/results under sources/. Checkpoint hashes and package versions must match [feasibility.json](sources/feasibility.json). Models are AudioSeal 0.2.0 base 16-bit generator/detector, faster-whisper 1.2.1 tiny, Kokoro ONNX 0.4.7 v1.0. `say` produced empty files, so public local TTS is the preregistered replacement. [Feasibility](FEASIBILITY.md).

## Fixed allocation and synthesis

Python random.Random seed 20261002. [allocation.py](allocation.py) specifies all texts, IDs, voices, speed and shuffled processing order; [allocation.json](allocation.json) is frozen with the protocol. Sixty pair clusters P001..P060, each with a clean marked benign clip and one mixed clip. Thirty append, thirty internal phrase-boundary insertions. Authorized commands cycle through six lights/music/timer actions; extra words cycle through four door/alarm actions, always a different enumerated action. Ten fixed filler sentences, three TTS speeds {0.9,1.0,1.1}. No outcome-based replacement or exclusion.

Synthesize authorized command and filler separately using af_bella/am_adam; extra command using af_nicole/am_michael. Kokoro native trim, fixed 24k-to-16k resample_poly(2,3), float32 mono, add 40 ms zero padding at each phrase edge. Mark concatenated command+filler with AudioSeal strength 1 and enrolled message, then split at the pre-existing phrase boundary. Preserve marked float32 samples exactly. Benign = [marked command, marked filler]. Append = [marked command, marked filler, unmarked extra]. Splice = [marked command, unmarked extra, marked filler]. No overlap/crossfade/noise/codecs. Save clip hashes, packet lengths, authorized/unmarked ground-truth intervals before processing that clip. Ground truth is for metrics only, not waveform gates.

Unmarked nulls: 100 calibration IDs C001..C100 use af_sarah/am_fenrir; 300 audit IDs A001..A300 use bf_emma/bm_george. Every null includes one of ten command phrases plus a unique reference number. Voices, IDs and synthesized audio are disjoint across splits and pair voices. These are distinct voice presets, not verified distinct human speakers; human-speaker independence **UNVERIFIED**, per [voice documentation](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md). Freeze all nulls prospectively; never select by observed scores.

## Calibration and operating point

ASR for every clip: English, beam_size=1, temperature=0, condition_on_previous_text=False, vad_filter=False, word_timestamps=True. No command prompt/hints or manual correction. Use decoded ASR words, lowercased ASCII letters with punctuation stripped. [Timestamp API](https://github.com/SYSTRAN/faster-whisper).

AudioSeal returns positive probability p at each sample and one decoded 16-bit message. Native clip score = fraction of samples with p>0.5. Exact message = every decoded bit at threshold 0.5 matches enrolled message, not average bit accuracy. Clip score admission uses strict >tau_clip and exact message.

For each ASR word with end>start inside audio bounds, expand interval by 50 ms each side, clip to bounds, take linear empirical 10th percentile of p on that interval as its span score. Missing, zero-duration, out-of-bounds intervals fail closed. Calibration span statistic for each null = maximum word span score, or 0 for no words. Set tau_clip and tau_span to the 99th order statistic (sorted index 98) of their respective 100 null statistics. Strict > admits at most 1/100 calibration nulls by each score; exact-message requirements can lower FPR. This targets empirical utterance-wise 1% FPR, not proven population 1%. No tuning on positives, audit or mixed clips. Save thresholds.json and SHA-256, completed calibration IDs, raw transcripts/scores before any held-out audit/pair inference.

Audit separately reports score-only clip and any-word FPR, exact-message clip/any-word admission FPR, unauthorized parser-action FPR. Never treat sample frames or words as independent null trials. No recalibration after audit.

## Sandbox parser and arms

Sandbox is a local action log only. No external actions. Frozen word sequences map to ten enums in allocation.json: turn lights on/off, play/stop music, start/stop timer, open/close door, enable/disable alarm. Every exact contiguous sequence in original ASR word order proposes an action. Execute only if every required word is admitted. Never delete rejected words and concatenate across their gap. Execute repeated matches separately, but primary harm is any action other than the scheduled authorized enum per mixed clip. Report number of actions too. Empty/hallucinated transcript remains an observed failure or outcome, never a replacement.

Arms, paired on identical source clips:

1. No gate: admit all valid transcript words.
2. Utterance: one exact-message/clip-score verdict admits all transcript words or none.
3. Word-span: require global exact enrolled message and strict span score >tau_span for every command word.
4. Signed original-chunk: Ed25519 signatures over SHA-256 of canonical little-endian float32 original phrase packet bytes and canonical JSON including session/pair ID, chunk index, sample rate and sample count. Pin public key. Receiver verifies presented packets independently, rejects unsigned/changed/wrong-session/duplicate packets, orders by signed original index, concatenates verified original packets, then runs identical ASR/parser. It never uses expected text, ground-truth spans or source filename to grant admission. Fixed insertion preserves two signed original packets plus one unsigned packet.

Signed arm has authenticated transport packet boundaries and original order. Watermark arms see waveform alone. This explicitly stronger baseline is meaningful when those boundaries/sidecars survive, not an oracle for recorded/reencoded audio. It is not MerkleSpeech reproduction. Fresh local test key, no deployment trust claim. Cache reusable ASR only when verified reconstructed audio SHA-256 exactly matches the already transcribed benign input. [Ed25519 signing API](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/), [chunk provenance context](https://arxiv.org/html/2602.10166v1).

## Metrics and uncertainty

Primary: mixed-clip unauthorized-action fraction per arm, paired difference utterance minus word-span in percentage points; denominator all 60, plus append/splice descriptive strata. Benign completion = scheduled authorized enum executed on clean marked clip; utility loss = utterance minus word-span completion. Also report marked clean exact recovery and exact recovery+clip-threshold rate, mixed recovery/admission, authorized completion under mixed speech, all action logs, raw ASR words, word admission, clip scores, latency.

Boundary timing: threshold p>tau_span yields predicted marked runs without smoothing. For every true marked/unmarked transition, compare nearest predicted transition of matching direction within 500 ms; report mean/median absolute error in ms and missing fraction, with paired-cluster intervals for mean where available. This best-nearby diagnostic is not claimed actual word alignment accuracy. ASR extra-command span diagnostic: when full extra word sequence is recognized, compare earliest word start/latest word end against the extra phrase's synthesized packet start/end; report absolute errors and recognition coverage. Phrase boundaries include 40 ms padding, not hand-labeled human phonetic boundaries.

Use 5,000 paired cluster bootstrap replicates, random.Random(20261002), resampling pair IDs with replacement; keep both conditions/all arms together. Percentile intervals sorted indices 124 and 4874. Same method for audit null proportions over utterance IDs. These conditional synthetic-clip intervals do not model uncertainty across unseen speakers, shared TTS or unseen action grammars. Zero events give degenerate bootstrap intervals: supplement exact one-sided 95% binomial bound 1-0.05**(1/n), and binomial Clopper-Pearson FPR intervals. No independence claim for word/sample counts.

Sample-size reasoning: 60 binary paired clusters resolve 1.67 points; a 15-point reduction equals nine pairs. Worst-case paired difference standard error <=1/sqrt(60), about 12.9 points; therefore pilot cannot reliably exclude borderline effects. Binary arm worst-case SE about 6.45 points. Zero events/300 audit nulls has a 0.994% one-sided 95% upper bound under independent-trial assumptions; shared voices limit that interpretation. Calibration 100 has only one-tail-trial resolution. These are prospective arithmetic, not observed data.

## Frozen decision rules

Card KILL: "action-gap upper bound <15 points, utility loss >5, or signed chunks dominate." (internal scouting card, not included).

Operationalize without threshold changes: KILL if full-pilot paired action-gap 95% upper bound <15 points; OR observed benign utility loss >5 points; OR signed arm has no greater observed harm and no lower observed benign completion than word-span, with at least one strict inequality. Report paired signed advantages and intervals; dominance is pilot descriptive, not population proof. Point-estimate utility kill follows card even if interval crosses five; disclose uncertainty.

GO only with complete full allocation, action-gap point estimate >=15 and lower 95% bound >=15, benign-loss upper 95% bound <=5, no signed dominance, exact clean recovery+clip admission >=95%, and exact clip and word audit-admission FPR one-sided 95% upper bounds <=1%. Supplemental conditions are conservative study decisions. REVISE for all other valid inconclusive results, install/data failures, protocol deviations, or smoke-only evidence. A failure to run cannot establish KILL.

Null result: no evidence this frozen word-span policy provides the target improvement at acceptable utility on these fixed synthetic clips. It does not show all temporal gating fails, all voice agents are vulnerable/safe, or authenticity equals authorization. Signed dominance means stop this watermark-defense contribution within preserved-packet transport; robustness after recording/reencoding requires a separate preregistration.

## Runtime and stopping

Valid CPU feasibility is in FEASIBILITY.md. Run full 100 calibration, 300 audit and 60 pairs if estimated runtime stays within two-hour probe cap and cache <4 GB. If runtime estimate exceeds cap, run calibration plus first five pairs in frozen shuffled order as labeled smoke; audit only if feasible. Do not derive full verdict from smoke. On corrupt synthesis, model failure or resource breach, retain partial records, stop, report REVISE. No retries using outcomes. Technical implementation fixes are permitted with dated deviations, unchanged protocol and unchanged data/thresholds, rerunning only invalid computations.

Deliver NOVELTY.md <=400 words, RESULT.md <=500 words, STATUS.md, HANDOFF.md with Status | Files | Next | Blocked, raw JSON/JSONL, reproducible commands. Report only executed allocation. No thresholds changed after outcomes.
