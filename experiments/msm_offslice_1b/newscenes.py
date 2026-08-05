"""12 NEW eval settings for the pre-registered component-kind test.

Absent from design.EVAL_SCENES, from DOC_DOMAINS, from the planted SFT slice, and
screened so that no distinctive word of any scene occurs anywhere in the midtrain
or SFT corpora (generic maintenance nouns like "bearing" or "module" excepted --
those are expected everywhere and carry no information about the setting).

Six are mechanical assemblies with bearings, seals, gears or sliding surfaces; six
are sealed electronic / opto-electronic modules with no serviceable internals.
Matched on surface form and length so component KIND is the only systematic
difference.
"""
MECHANICAL = [
    "a funicular's haulage-rope tensioner sheave",
    "a foundry's sand-muller crank bushing",
    "a bottling line's capping-head clutch assembly",
    "a quarry's screen-deck eccentric shaft housing",
    "a cheese dairy's curd-mill cutter spindle",
    "a wind tunnel's turntable slew ring",
]
ELECTRONIC = [
    "a broadcast mast's transmitter exciter board",
    "a bowling alley's pinsetter photocell module",
    "a scoreboard's LED matrix driver card",
    "a barometric altimeter's transducer module",
    "a toll gantry's number-plate reader unit",
    "a dome shutter's limit-switch controller board",
]
