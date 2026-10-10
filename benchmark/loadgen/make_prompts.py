"""Regenerate prompts.jsonl: 48 short and 16 long prompts, fully deterministic (fixed seed).

python benchmark/loadgen/make_prompts.py > benchmark/loadgen/prompts.jsonl
"""

import json
import random
import sys

SHORT = [
    "What is the purpose of a pressure relief valve?",
    "Explain cavitation in centrifugal pumps.",
    "What does MTBF stand for and how is it used?",
    "Describe the difference between preventive and predictive maintenance.",
    "What is a PID controller?",
    "Why do bearings fail early?",
    "List three causes of motor overheating.",
    "What is the function of a heat exchanger?",
    "Explain lockout tagout in one paragraph.",
    "What is vibration analysis used for?",
    "How does a variable frequency drive save energy?",
    "What is the role of a PLC in a factory?",
    "Define overall equipment effectiveness.",
    "What is thermal imaging used for in maintenance?",
    "Explain the difference between AC and DC motors.",
    "What is a root cause analysis?",
    "Why is lubrication important for gearboxes?",
    "What is SCADA?",
    "Describe how a compressor works.",
    "What is a failure mode and effects analysis?",
    "What does a torque wrench do?",
    "Explain what a safety integrity level is.",
    "How is corrosion prevented in steel pipes?",
    "What is the purpose of a check valve?",
    "Describe the main parts of a hydraulic system.",
    "What is a condition monitoring sensor?",
    "Explain what an interlock is.",
    "What is a Pareto chart used for in quality control?",
    "How do you size a fuse for a motor circuit?",
    "What is the difference between accuracy and precision of a sensor?",
    "Why do pipes experience water hammer?",
    "What is a work order in a CMMS?",
    "Describe what a calibration certificate contains.",
    "What is a spare parts criticality ranking?",
    "Explain the bathtub curve of failure rates.",
    "What is a thermocouple?",
    "How does an ultrasonic flow meter work?",
    "What is a confined space permit?",
    "Explain what shaft misalignment causes.",
    "What is the purpose of an emergency stop circuit?",
    "Describe what a borescope inspection is.",
    "What is oil analysis used for?",
    "How do you read a P&ID diagram?",
    "What is a hazard and operability study?",
    "Explain what a Gantt chart shows in a shutdown plan.",
    "Why is grounding important in electrical panels?",
    "What is a reliability centered maintenance program?",
    "Explain the purpose of a mean time to repair metric.",
]

ASSETS = [
    "pump P-101",
    "compressor C-7",
    "conveyor CV-3",
    "boiler B-2",
    "chiller CH-4",
    "fan F-12",
    "mixer M-5",
    "press PR-9",
]
EVENTS = [
    "vibration rose to {v} mm/s",
    "bearing temperature reached {t} C",
    "operator reported unusual noise",
    "oil level checked and topped up",
    "filter replaced during scheduled stop",
    "motor current {a} A above baseline",
    "seal leak observed, {l} drops per minute",
    "alignment checked, offset {o} mm",
]
QUESTIONS = [
    "Which asset is most at risk of failure and why?",
    "Summarise the main trends in this log.",
    "What maintenance actions would you schedule next week?",
    "List every asset with a temperature or vibration problem.",
]


def long_prompt(rng: random.Random, i: int) -> str:
    lines = []
    for day in range(1, 37):
        asset = rng.choice(ASSETS)
        event = rng.choice(EVENTS).format(
            v=round(rng.uniform(2, 12), 1),
            t=rng.randint(55, 95),
            a=rng.randint(2, 15),
            l=rng.randint(1, 30),
            o=round(rng.uniform(0.05, 0.6), 2),
        )
        lines.append(f"Day {day}, shift {rng.randint(1, 3)}: {asset}, {event}.")
    log = "\n".join(lines)
    return f"Maintenance log for line {i}:\n{log}\n\n{QUESTIONS[i % len(QUESTIONS)]}"


def main() -> None:
    rng = random.Random(1234)
    rows = [
        {"id": f"s{i:02d}", "kind": "short", "prompt": p} for i, p in enumerate(SHORT)
    ]
    rows += [
        {"id": f"l{i:02d}", "kind": "long", "prompt": long_prompt(rng, i)}
        for i in range(16)
    ]
    rng.shuffle(rows)  # fixed order mixing short and long, same for every run
    for r in rows:
        sys.stdout.buffer.write(
            (json.dumps(r, ensure_ascii=False) + "\n").encode()
        )  # LF, UTF-8 on any OS


if __name__ == "__main__":
    main()
