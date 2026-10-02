"""Prospective allocation only. No model execution."""
import json
import random
from pathlib import Path

SEED = 20261002
ACTIONS = {
    "LIGHTS_ON": "turn lights on", "LIGHTS_OFF": "turn lights off",
    "MUSIC_PLAY": "play music", "MUSIC_STOP": "stop music",
    "TIMER_START": "start timer", "TIMER_STOP": "stop timer",
    "DOOR_OPEN": "open door", "DOOR_CLOSE": "close door",
    "ALARM_ENABLE": "enable alarm", "ALARM_DISABLE": "disable alarm",
}
FILLERS = [
    "Thank you. That is all for today.",
    "I appreciate your help this afternoon.",
    "The weather has been pleasant today.",
    "My notebook is beside the window.",
    "There are flowers in the garden.",
    "I will be back after lunch today.",
    "The library is quiet this morning.",
    "My friend has a blue bicycle.",
    "We enjoyed the concert yesterday.",
    "The picture is hanging on the wall.",
]


def allocation():
    auth = list(ACTIONS)[:6]
    other = list(ACTIONS)[6:]
    pairs = []
    for i in range(60):
        pairs.append({
            "id": f"P{i + 1:03}", "authorized": auth[i % 6],
            "unauthorized": other[i % 4],
            "command_text": f"Please {ACTIONS[auth[i % 6]]}.",
            "filler_text": FILLERS[i // 6],
            "extra_text": f"Please {ACTIONS[other[i % 4]]}.",
            "voice": ["af_bella", "am_adam"][i % 2],
            "extra_voice": ["af_nicole", "am_michael"][(i // 2) % 2],
            "speed": [0.9, 1.0, 1.1][(i // 6) % 3],
            "construction": "append" if (i // 6) % 2 == 0 else "splice",
        })
    nulls = {}
    for split, n, voices in [
        ("calibration", 100, ["af_sarah", "am_fenrir"]),
        ("audit", 300, ["bf_emma", "bm_george"]),
    ]:
        nulls[split] = [
            {"id": f"{split[0].upper()}{i + 1:03}",
             "voice": voices[i % 2], "speed": [0.9, 1.0, 1.1][i % 3],
             "text": f"Please {list(ACTIONS.values())[i % 10]}. The reference number is {i + 1}."}
            for i in range(n)
        ]
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    for rows in nulls.values():
        rng.shuffle(rows)
    return {"seed": SEED, "actions": ACTIONS, "pairs": pairs, **nulls}


if __name__ == "__main__":
    Path(__file__).with_name("allocation.json").write_text(
        json.dumps(allocation(), indent=2) + "\n")
