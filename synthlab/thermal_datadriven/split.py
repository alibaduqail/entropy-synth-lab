"""The held-out design.  BUILD trials feed the generator; TEST trials are never touched by it.

Sessions (see dataio.SESSIONS):
  hl    heat-lamp evening: hl_off = empty room / lamp off (A0), hl_on = one standing person
        holding a lit heat lamp (the dataset calls it A1; it is really "A1 + lamp").
  rj    first radiometric session, different viewpoint, one standing person, no lamp.
  w120  60 s warm-up / 120 s capture session, two seated people close to the camera.
"""
HL_OFF = ["EXP-A_A0_run%02d" % i for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                           17, 18, 19, 20, 21, 22, 23, 24, 25, 26)]
HL_ON = ["EXP-A_A1_run%02d" % i for i in range(1, 26)]

BUILD = {
    "hl_off": HL_OFF[:13],                       # runs 01..14 (04 was excluded by the owner)
    "hl_on": HL_ON[:12],                         # runs 01..12
    "rj_A0": ["EXP-A_A0_run01", "EXP-A_A0_run02"],
    "rj_A1": ["EXP-A_A1_run01"],
    "w120_A0": ["EXP-A_A0_run02", "EXP-A_A0_run03", "EXP-A_A0_run04"],
    "w120_A1": ["EXP-A_A1_run01", "EXP-A_A1_run02"],
}
TEST = {
    "hl_off": HL_OFF[13:],                       # 12 held-out empty-room trials
    "hl_on": HL_ON[12:],                         # 13 held-out person+lamp trials
    "rj_A0": ["EXP-A_A0_run03"],
    "rj_A1": ["EXP-A_A1_run02", "EXP-A_A1_run03"],
    "w120_A0": ["EXP-A_A0_run05", "EXP-A_A0_run06"],
    "w120_A1": ["EXP-A_A1_run03", "EXP-A_A1_run04", "EXP-A_A1_run05"],
}
# key -> (dataio session, model/background name, people per frame in that session's A1)
KEYS = {
    "hl_off": ("hl_off", "hl", 0), "hl_on": ("hl_on", "hl", 1),
    "rj_A0": ("rj", "rj", 0), "rj_A1": ("rj", "rj", 1),
    "w120_A0": ("w120", "w120", 0), "w120_A1": ("w120", "w120", 2),
}
assert not any(set(BUILD[k]) & set(TEST[k]) for k in BUILD)
