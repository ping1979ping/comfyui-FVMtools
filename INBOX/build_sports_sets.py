"""Phase-3 Sports category builder. Generates 20 sports location sets.
Each set has 7 files matching the indoor_yoga_private / outdoor_hiking_spring_pennsylvania format.
"""
from __future__ import annotations
import os, sys

ROOT = "D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools/location_lists"

# ─────────────────────────────────────────────────────────────────────────
# Two atmosphere pools — one for indoor sports (climate-controlled venue
# moods, ventilation, lighting) and one for outdoor sports (sky/wind/temp).
# ─────────────────────────────────────────────────────────────────────────

TIME_OF_DAY_INDOOR = [
    ("bright competition arena floodlights", 1.0),
    ("cool morning training session light", 0.9),
    ("late afternoon practice glow", 0.85),
    ("evening prime-time broadcast lighting", 0.95),
    ("pre-game warmup hush before tipoff", 0.8),
    ("between-period dim house lights", 0.7),
    ("post-match cooldown lighting", 0.7),
    ("sharp televised arena spotlights", 0.85),
    ("quiet empty-venue early hours", 0.65),
    ("under-the-lights night session brightness", 0.85),
    ("dimmed pre-introduction tunnel hush", 0.6),
]

WEATHER_INDOOR = [
    ("climate-controlled cool dry air", 1.0),
    ("faint smell of rubber and chalk", 0.85),
    ("crisp arena ventilation hum", 0.9),
    ("warm body-heat humidity from athletes", 0.8),
    ("scent of liniment and sweat", 0.75),
    ("subtle chlorine or rubber tang", 0.6),
    ("dry conditioned hardcourt stillness", 0.85),
    ("steady whoosh of HVAC vents", 0.8),
    ("muted echo of empty-hall acoustics", 0.7),
    ("faint scent of polished hardwood", 0.55),
    ("crisp ozone of fresh ice surface", 0.55),
]

TIME_OF_DAY_OUTDOOR = [
    ("crisp early morning warmup light", 0.95),
    ("bright midday match sunshine", 1.0),
    ("warm late afternoon match glow", 0.9),
    ("golden hour tournament light", 0.85),
    ("overcast diffused practice daylight", 0.85),
    ("evening floodlit night-match brightness", 0.9),
    ("dawn pre-race start gate light", 0.7),
    ("dusk hush after the whistle", 0.7),
    ("hard high-noon shadow contrast", 0.8),
    ("amber sunset finishing-stretch glow", 0.75),
    ("twilight cooldown blue hush", 0.6),
]

WEATHER_OUTDOOR = [
    ("clear dry game-day air", 1.0),
    ("light breeze across the field", 0.9),
    ("warm dry summer competition heat", 0.8),
    ("cool crisp autumn match air", 0.85),
    ("light overcast humid stillness", 0.75),
    ("faint scent of cut grass", 0.7),
    ("dry dusty stadium air", 0.65),
    ("damp morning dew chill", 0.6),
    ("sharp cold winter sport bite", 0.7),
    ("steady tournament-day stillness", 0.85),
    ("muggy late-summer match humidity", 0.65),
]


# ─────────────────────────────────────────────────────────────────────────
# Per-slug content. Each entry: (name, prob, coverage, texture).
# 5 non-atmosphere files keyed: bg, mg, fg, arch, props.
# ─────────────────────────────────────────────────────────────────────────

SETS: dict[str, dict[str, list[tuple[str, float, str, str]]]] = {}
SLUG_KIND: dict[str, str] = {}  # "indoor" | "outdoor"


# ── 1. indoor_sports_boxing_gym_classic ──────────────────────────────────
SETS["indoor_sports_boxing_gym_classic"] = {
    "bg": [
        ("scuffed brick gym wall", 1.0, "0.7-1.0", "weathered painted masonry"),
        ("faded fight-poster covered wall", 0.95, "0.7-1.0", "layered torn promotional bills"),
        ("white painted concrete block wall", 0.9, "0.7-1.0", "matte chipped paint surface"),
        ("dim industrial corner with hanging bags", 0.85, "0.6-1.0", "shadowed metal-frame backdrop"),
        ("vintage championship belt display wall", 0.7, "0.6-0.9", "leather and gilded plate row"),
        ("worn leather speedbag wall mount", 0.7, "0.5-0.9", "scuffed pear-shaped silhouette"),
        ("hand-painted gym signage wall", 0.8, "0.6-1.0", "block-letter painted lettering"),
        ("rust-streaked metal locker wall", 0.65, "0.6-0.9", "battered enamel paint"),
        ("ring-rope corner pillar background", 0.85, "0.5-0.9", "tightly wound canvas-covered cable"),
        ("trophy-shelf cluttered backdrop", 0.65, "0.5-0.9", "brass and plastic figurine row"),
        ("hand-wrapped boxing glove rack wall", 0.7, "0.5-0.9", "rows of stitched leather mitts"),
        ("yellowed mirror wall with smudges", 0.7, "0.6-1.0", "softly tarnished reflective glass"),
        ("torn flag hung above ring", 0.55, "0.4-0.8", "frayed fabric pennant"),
    ],
    "mg": [
        ("regulation boxing ring with red ropes", 1.0, "0.3-0.6", "padded canvas square platform"),
        ("row of heavy hanging punching bags", 0.95, "0.3-0.6", "scuffed leather cylinders on chains"),
        ("speedbag rebound platform", 0.8, "0.2-0.4", "polished oval drum board"),
        ("double-end bag tethered between floor and ceiling", 0.7, "0.2-0.4", "small leather sphere on cords"),
        ("trainer pad-holding mid-floor", 0.7, "0.2-0.5", "focus-mitt drilling stance"),
        ("free-standing reflex bag column", 0.65, "0.2-0.4", "rebound mannequin form"),
        ("jump-rope station floor zone", 0.7, "0.2-0.4", "open coiled-rope marked area"),
        ("medicine-ball rack mid-room", 0.6, "0.2-0.4", "stacked weighted leather spheres"),
        ("cornerman stool with bucket", 0.6, "0.1-0.3", "battered wooden seat and pail"),
        ("upright weight bench beside ring", 0.6, "0.2-0.4", "padded vinyl flat bench"),
        ("freestanding spit bucket and towels", 0.5, "0.1-0.3", "galvanized pail with rags"),
        ("hanging timer round-bell unit", 0.7, "0.1-0.3", "wall-mounted brass bell"),
    ],
    "fg": [
        ("worn canvas ring-floor edge", 0.95, "0.2-0.5", "scuffed taut painted canvas"),
        ("hand-wrap-strewn rubber gym mat", 0.9, "0.2-0.5", "softly buckled black rubber"),
        ("scattered rosin chalk dust patch", 0.8, "0.05-0.2", "fine white floor powder"),
        ("ring-step wooden corner riser", 0.7, "0.05-0.2", "scuffed painted wood stair"),
        ("loose hand-wrap roll on floor", 0.85, "0.05-0.2", "coiled cotton bandage"),
        ("damp sweat-stained towel drape", 0.8, "0.05-0.2", "limp cotton terry cloth"),
        ("leather speedbag glove pair", 0.7, "0.05-0.15", "scuffed paired training mitt"),
        ("upturned spit bucket foreground", 0.55, "0.05-0.15", "battered metal pail"),
        ("rosin tray near ringside", 0.55, "0.05-0.15", "small wooden powder dish"),
        ("scuffed gym shoe pair on canvas", 0.7, "0.05-0.15", "low-cut leather boxing boot"),
        ("dropped focus mitts on canvas", 0.7, "0.05-0.15", "padded curved trainer pads"),
        ("stopwatch on wooden stool", 0.55, "0.02-0.08", "chrome round-faced timer"),
    ],
    "arch": [
        ("exposed steel I-beam ceiling", 0.85, "0.1-0.2", "riveted painted girder"),
        ("hanging chain-suspension bag rig", 0.95, "0.05-0.15", "heavy industrial chain links"),
        ("caged industrial pendant lamp", 0.85, "0.05-0.15", "wire-guard metal cage shade"),
        ("ring-corner turnbuckle pad", 0.95, "0.05-0.15", "padded vinyl-covered post cap"),
        ("ceiling fan slow-spin blade", 0.7, "0.05-0.15", "battered metal industrial fan"),
        ("speedbag platform under-ceiling", 0.7, "0.05-0.15", "rebound oval mounted disk"),
        ("mounted round-bell timer fixture", 0.75, "0.02-0.1", "brass-bell wall fitting"),
        ("painted square ceiling pipe run", 0.6, "0.02-0.1", "exposed riveted conduit"),
        ("weathered gym-name signage above ring", 0.7, "0.05-0.15", "old painted name plate"),
        ("rope-ringed corner post sleeve", 0.75, "0.05-0.15", "tightly wrapped canvas cover"),
    ],
    "props": [
        ("scuffed leather boxing gloves pair", 0.95, "0.02-0.1", "lace-up padded mitts"),
        ("cotton hand-wrap roll", 0.9, "0.02-0.08", "coiled stretch bandage"),
        ("padded headgear in cornerman bucket", 0.7, "0.02-0.08", "open-face leather headguard"),
        ("ring-side spit bucket", 0.7, "0.02-0.08", "battered galvanized pail"),
        ("dented water bottle on stool", 0.7, "0.02-0.08", "scuffed plastic squeeze bottle"),
        ("ringside cornerman ice pack", 0.55, "0.02-0.08", "fabric-wrapped cold compress"),
        ("vintage timing stopwatch", 0.65, "0.02-0.05", "chrome dial round timer"),
        ("rosin powder block", 0.5, "0.02-0.05", "compact white grip block"),
        ("stitched leather mouthguard case", 0.5, "0.02-0.05", "small clamshell holder"),
        ("hand-stamped gym-name towel", 0.65, "0.02-0.08", "white cotton terry towel"),
        ("worn champion belt on rack", 0.55, "0.02-0.1", "leather and gilded plate"),
        ("focus mitts pair on stool", 0.7, "0.02-0.08", "curved padded trainer pads"),
        ("rope-skipping handle pair", 0.55, "0.02-0.05", "wooden grip skip-rope"),
    ],
}
SLUG_KIND["indoor_sports_boxing_gym_classic"] = "indoor"


# ── 2. indoor_sports_weight_room_serious ─────────────────────────────────
SETS["indoor_sports_weight_room_serious"] = {
    "bg": [
        ("matte black rubber-tile gym wall", 1.0, "0.7-1.0", "studded recycled rubber sheet"),
        ("painted dark grey training-floor wall", 0.9, "0.7-1.0", "smooth matte commercial paint"),
        ("mirrored full-height training wall", 0.95, "0.7-1.0", "polished plate-glass reflection"),
        ("perforated steel weight-rack wall", 0.85, "0.6-1.0", "punched commercial steel panel"),
        ("graffiti-stenciled barbell motto wall", 0.7, "0.6-0.9", "stencil-sprayed motivational text"),
        ("exposed concrete blockwork backdrop", 0.85, "0.7-1.0", "rough-cast structural masonry"),
        ("dark slatwall display panel", 0.75, "0.6-0.9", "horizontal grooved retail board"),
        ("framed champion-lifter photo wall", 0.7, "0.6-0.9", "rows of black-framed prints"),
        ("plate-loaded barbell-rack backdrop", 0.85, "0.6-0.9", "rows of cast-iron discs"),
        ("painted record-board wall", 0.65, "0.5-0.9", "whiteboard tracking athlete records"),
        ("industrial ribbed-metal sheet wall", 0.7, "0.6-1.0", "corrugated steel cladding"),
        ("dumbbell-cabinet steel grid wall", 0.8, "0.6-0.9", "stepped cast-iron rack"),
        ("dim performance-zone painted wall", 0.65, "0.6-0.9", "matte charcoal performance paint"),
    ],
    "mg": [
        ("commercial squat-rack station", 1.0, "0.3-0.6", "heavy-gauge powder-coated frame"),
        ("loaded competition barbell on platform", 0.95, "0.2-0.5", "knurled steel bar with plates"),
        ("dumbbell rack with rubber hex weights", 0.95, "0.3-0.6", "stepped tier of black hex heads"),
        ("padded flat bench-press station", 0.9, "0.2-0.4", "vinyl-topped commercial bench"),
        ("cable-machine pulley column", 0.8, "0.2-0.5", "twin-stack adjustable cable tower"),
        ("Olympic deadlift platform with bumpers", 0.85, "0.2-0.4", "wood-and-rubber lifting square"),
        ("seated leg-press selectorized machine", 0.7, "0.2-0.5", "cushioned commercial press"),
        ("kettlebell display row", 0.8, "0.1-0.3", "graduated cast-iron bell line"),
        ("calibrated plate stack tree", 0.7, "0.1-0.3", "centered plate-storage post"),
        ("upright preacher curl bench", 0.55, "0.1-0.3", "angled padded biceps station"),
        ("Smith-machine guided barbell rig", 0.65, "0.2-0.4", "fixed-rail lifting rack"),
        ("hex-bar trap-deadlift station", 0.5, "0.1-0.3", "hexagonal lifting frame"),
    ],
    "fg": [
        ("rubber stall-mat lifting floor", 1.0, "0.3-0.6", "thick interlocking gym tile"),
        ("scattered chalk dust handprint", 0.85, "0.05-0.2", "white powdered grip residue"),
        ("loaded barbell on platform foreground", 0.9, "0.2-0.5", "knurled steel sleeve and discs"),
        ("rolled lifting belt on floor", 0.7, "0.05-0.2", "thick coiled leather strap"),
        ("dropped wrist-wrap pair", 0.7, "0.05-0.2", "elastic wrist-support strap"),
        ("scuffed weight collar pair", 0.6, "0.02-0.1", "spring-clip steel ring"),
        ("foam roller on floor", 0.7, "0.05-0.2", "high-density cylindrical roller"),
        ("set of fractional plates", 0.55, "0.02-0.1", "small calibrated discs"),
        ("dropped chalk block by platform", 0.7, "0.02-0.1", "compressed magnesium cube"),
        ("hex-dumbbell pair grounded", 0.75, "0.05-0.2", "rubber-coated hand weight pair"),
        ("athlete water-jug on floor", 0.7, "0.05-0.15", "translucent gallon training jug"),
        ("ankle-strap loop discarded", 0.5, "0.02-0.1", "padded cuff with D-ring"),
    ],
    "arch": [
        ("exposed truss training-floor ceiling", 0.85, "0.1-0.2", "open black-painted joist"),
        ("rigged pull-up monkey-bar grid", 0.95, "0.05-0.15", "ladder-grid rig structure"),
        ("track-mounted spotlight rail", 0.85, "0.05-0.15", "high-output ceiling spot row"),
        ("safety-pinned rack catch arm", 0.95, "0.02-0.1", "horizontal safety pin"),
        ("ceiling-suspended ring-and-rope rig", 0.7, "0.05-0.15", "gymnastic ring suspension"),
        ("painted lift-zone floor stripe", 0.85, "0.05-0.2", "high-visibility safety paint"),
        ("battle-rope wall anchor plate", 0.7, "0.02-0.1", "steel mounting bracket"),
        ("competition platform corner edge", 0.85, "0.05-0.15", "wood-rubber lifting edge"),
        ("ceiling-fixed industrial fan", 0.7, "0.05-0.15", "high-volume blade"),
        ("painted record-board wall mount", 0.55, "0.02-0.1", "framed whiteboard fixture"),
    ],
    "props": [
        ("calibrated competition lifting belt", 0.95, "0.02-0.1", "thick stitched leather waist belt"),
        ("magnesium chalk block", 0.95, "0.02-0.05", "compressed white grip cube"),
        ("knurled-grip barbell collar pair", 0.85, "0.02-0.05", "spring-clip locking ring"),
        ("competition lifting shoes pair", 0.85, "0.02-0.08", "raised-heel weightlifting shoe"),
        ("wrist-wrap stiffened cuff", 0.7, "0.02-0.05", "padded elastic wrist support"),
        ("knee-sleeve neoprene pair", 0.75, "0.02-0.08", "thick stretch knee support"),
        ("training journal logbook", 0.6, "0.02-0.05", "spiral-bound rep tracker"),
        ("PR-attempt smelling-salt vial", 0.5, "0.02-0.05", "small glass ammonia capsule"),
        ("athlete shaker bottle", 0.85, "0.02-0.08", "agitator-blender sports cup"),
        ("metal collar deadlift jack", 0.55, "0.02-0.05", "small lever-handle bar lifter"),
        ("hand-grip strengthener", 0.55, "0.02-0.05", "spring squeeze trainer"),
        ("athletic-tape roll", 0.7, "0.02-0.05", "white cotton finger-tape spool"),
        ("liquid-chalk applicator bottle", 0.6, "0.02-0.05", "pump-top grip-fluid bottle"),
    ],
}
SLUG_KIND["indoor_sports_weight_room_serious"] = "indoor"


# ── 3. indoor_sports_basketball_court_pro_arena ──────────────────────────
SETS["indoor_sports_basketball_court_pro_arena"] = {
    "bg": [
        ("tiered packed arena seating bowl", 1.0, "0.7-1.0", "rows of upholstered fold seats"),
        ("luxury suite glass-front balcony", 0.85, "0.6-1.0", "mirrored skybox glazing"),
        ("giant scoreboard center hang", 0.9, "0.6-1.0", "four-sided LED video display"),
        ("team championship banner wall", 0.85, "0.6-1.0", "hanging fabric title banners"),
        ("packed lower-bowl crowd backdrop", 0.95, "0.7-1.0", "dense home-jersey color block"),
        ("tunnel-mouth player entrance", 0.7, "0.5-0.9", "darkened dramatic portal"),
        ("LED ribbon-board wall sweep", 0.85, "0.6-1.0", "scrolling animated graphic strip"),
        ("retired-jersey rafter display", 0.65, "0.5-0.9", "hanging numbered jerseys"),
        ("courtside sponsor signage backdrop", 0.85, "0.6-0.9", "logo-emblazoned floor-side panel"),
        ("home-team color upholstered seat sea", 0.75, "0.6-1.0", "swathe of theme-colored seats"),
        ("center-court logo painted floor", 0.95, "0.6-1.0", "polished glossy team emblem"),
        ("press-row cluster table-line", 0.7, "0.5-0.9", "long courtside media bench"),
        ("dimmed tunnel introduction backdrop", 0.6, "0.4-0.8", "shadowed portal hush"),
    ],
    "mg": [
        ("polished hardwood basketball court", 1.0, "0.3-0.6", "glossy maple floor plank"),
        ("regulation breakaway hoop and backboard", 1.0, "0.2-0.5", "tempered-glass square backboard"),
        ("painted three-point arc line", 0.95, "0.2-0.5", "bold curved court paint"),
        ("center-jump tip-off circle", 0.85, "0.2-0.4", "painted center logo ring"),
        ("courtside player benches", 0.85, "0.2-0.4", "padded fold-down team chair row"),
        ("scorer table center-court", 0.8, "0.2-0.4", "sponsor-wrapped media console"),
        ("photographer baseline cluster", 0.7, "0.1-0.3", "long-lens crouched cluster"),
        ("referee stand-by strip", 0.7, "0.1-0.3", "striped jersey official trio"),
        ("LED court-edge video board", 0.75, "0.2-0.4", "courtside graphic ribbon"),
        ("standing front-row VIP seats", 0.7, "0.2-0.4", "leather courtside chair"),
        ("painted free-throw lane key", 0.85, "0.2-0.4", "bold rectangular paint"),
        ("dribble-zone painted half-court line", 0.8, "0.2-0.4", "bold midcourt stripe"),
    ],
    "fg": [
        ("polished hardwood plank floor edge", 1.0, "0.3-0.6", "glossy varnished maple"),
        ("painted sideline boundary stripe", 0.95, "0.1-0.3", "high-contrast court edge"),
        ("loose game basketball on floor", 0.9, "0.05-0.2", "leather pebbled grain ball"),
        ("courtside stat-sheet drift", 0.7, "0.05-0.2", "scattered printed paper"),
        ("dropped wristband on baseline", 0.55, "0.02-0.1", "elastic terry sweatband"),
        ("wooden floor-cleaner mop", 0.6, "0.05-0.15", "long flat-mop applicator"),
        ("courtside warmup jacket pile", 0.7, "0.05-0.2", "team-color tracksuit pile"),
        ("baseline photographer crouch foreground", 0.7, "0.1-0.3", "telephoto-camera shooter"),
        ("sponsor-logo floor decal", 0.75, "0.05-0.15", "vinyl-applied court decal"),
        ("athletic resin-grip rosin patch", 0.55, "0.02-0.08", "white grip residue"),
        ("rolled towel near bench", 0.7, "0.05-0.15", "team-monogrammed cotton towel"),
        ("scuffed sneaker print on hardwood", 0.6, "0.05-0.15", "rubber sole skid mark"),
    ],
    "arch": [
        ("retracted-rafter dome ceiling", 0.85, "0.1-0.2", "soaring exposed truss"),
        ("center-hung four-sided scoreboard rig", 0.95, "0.05-0.15", "suspended LED video cube"),
        ("ceiling-rafter spot truss cluster", 0.85, "0.05-0.15", "mounted high-output spot grid"),
        ("rafter-hung championship banner", 0.85, "0.05-0.2", "fabric hanging title cloth"),
        ("retracted upper-tier seating rail", 0.7, "0.05-0.15", "pulled-back gallery handrail"),
        ("ceiling-mounted broadcast rig camera", 0.75, "0.02-0.1", "robotic suspended camera"),
        ("LED ribbon-board ceiling fascia", 0.85, "0.05-0.2", "scrolling graphic edge"),
        ("painted column-buttress cap", 0.55, "0.02-0.1", "structural concrete column top"),
        ("speaker-cluster line-array hang", 0.7, "0.05-0.15", "vertical sound array stack"),
        ("dome-panel cove uplighting", 0.65, "0.02-0.1", "graded ceiling color wash"),
    ],
    "props": [
        ("regulation leather game basketball", 1.0, "0.02-0.08", "pebbled tacky orange ball"),
        ("home-team warmup jacket", 0.85, "0.02-0.1", "satin team-color tracksuit"),
        ("courtside whiteboard play sheet", 0.8, "0.02-0.05", "marker-drawn play diagram"),
        ("athletic rosin-bag pouch", 0.6, "0.02-0.05", "talc-filled grip pouch"),
        ("team-branded water bottle", 0.85, "0.02-0.05", "squeeze-top sport bottle"),
        ("padded cushioned bench towel", 0.85, "0.02-0.08", "team-monogrammed cotton wrap"),
        ("courtside laptop stat-tablet", 0.6, "0.02-0.05", "portable team-side device"),
        ("bench microphone for coach", 0.5, "0.02-0.05", "cabled handheld mic"),
        ("athletic-tape roll", 0.7, "0.02-0.05", "white cotton finger-tape spool"),
        ("substitution paddle number", 0.55, "0.02-0.05", "digit-display sub paddle"),
        ("courtside ankle-brace pair", 0.55, "0.02-0.05", "lace-up athletic brace"),
        ("game-program courtside copy", 0.5, "0.02-0.05", "glossy printed booklet"),
        ("clipboard with timeout play", 0.7, "0.02-0.05", "wipe-board with marker"),
    ],
}
SLUG_KIND["indoor_sports_basketball_court_pro_arena"] = "indoor"


# ── 4. outdoor_sports_clay_tennis_court ──────────────────────────────────
SETS["outdoor_sports_clay_tennis_court"] = {
    "bg": [
        ("packed clay-court horizon line", 1.0, "0.6-1.0", "flat ochre crushed-brick haze"),
        ("dark green windscreen perimeter", 0.95, "0.7-1.0", "tightly woven mesh fabric"),
        ("tiered open-air spectator stand", 0.85, "0.6-1.0", "rows of bench seating"),
        ("distant clubhouse cypress row", 0.8, "0.5-0.9", "tall slender evergreen line"),
        ("cluster of umpire-shaded gallery", 0.7, "0.5-0.9", "shaded crowd cluster"),
        ("broadcast tower tournament backdrop", 0.65, "0.4-0.8", "scaffold camera platform"),
        ("court-side logo windscreen panel", 0.85, "0.6-1.0", "sponsor-printed mesh skin"),
        ("player-area awning canopy", 0.75, "0.5-0.9", "stretched white tournament cloth"),
        ("outdoor scoreboard pylon", 0.85, "0.5-0.9", "freestanding match-score column"),
        ("ball-kid shaded bench backdrop", 0.7, "0.4-0.8", "small awning over shaded seat"),
        ("tournament logo backdrop banner", 0.8, "0.6-0.9", "stretched event-brand cloth"),
        ("center-court chair umpire stand", 0.8, "0.5-0.9", "tall raised official chair"),
        ("clay-warmup adjacent court strip", 0.65, "0.5-0.9", "second-court ochre strip"),
    ],
    "mg": [
        ("regulation clay-court playing surface", 1.0, "0.3-0.6", "raked crushed-brick top dressing"),
        ("white-painted singles boundary line", 0.95, "0.2-0.5", "freshly chalked court line"),
        ("center service-line stripe", 0.95, "0.2-0.4", "bold white painted divider"),
        ("regulation tennis net center", 0.95, "0.2-0.5", "tightly stretched white-band net"),
        ("doubles-alley side stripe", 0.85, "0.2-0.4", "secondary outer line"),
        ("ball-kid kneeling baseline", 0.75, "0.1-0.3", "ready stance retriever"),
        ("clay-rake court-attendant figure", 0.7, "0.1-0.3", "broom-pulling groundsman"),
        ("court-side player chair pair", 0.85, "0.2-0.4", "shaded courtside resting seat"),
        ("center-mark baseline tab", 0.85, "0.1-0.3", "small midline reference mark"),
        ("singles-stick net-tension post", 0.7, "0.1-0.3", "small adjustable side post"),
        ("ice-and-towel changeover cooler", 0.6, "0.1-0.3", "branded courtside cooler"),
        ("warmup hitting-zone rally", 0.6, "0.2-0.4", "loose pre-match volley"),
    ],
    "fg": [
        ("freshly raked ochre clay surface edge", 1.0, "0.3-0.6", "fine combed crushed-brick"),
        ("painted baseline boundary stripe", 0.95, "0.1-0.3", "white chalked court edge"),
        ("scattered clay ball-mark divot", 0.85, "0.05-0.2", "shallow round impact pit"),
        ("wooden court-line broom", 0.7, "0.05-0.2", "dragged horsehair line sweeper"),
        ("loose tournament tennis ball", 0.95, "0.02-0.1", "felted optic-yellow ball"),
        ("coach grip-tape cutoff", 0.55, "0.02-0.05", "tossed black overgrip strip"),
        ("rolled towel near baseline chair", 0.7, "0.05-0.15", "tournament-monogrammed cotton towel"),
        ("clay-stained sock cuff foreground", 0.55, "0.02-0.08", "ochre-stained athletic sock"),
        ("ball-kid rolling kneel pose", 0.7, "0.1-0.2", "crouched retriever silhouette"),
        ("scuffed court shoe sole print", 0.7, "0.05-0.15", "herringbone-tread footprint"),
        ("dropped wristband near net post", 0.55, "0.02-0.08", "elastic terry sweatband"),
        ("courtside ice-bucket condensation puddle", 0.5, "0.02-0.1", "small wet ground patch"),
    ],
    "arch": [
        ("raised umpire-chair canopy", 0.85, "0.05-0.2", "small fabric shade over chair"),
        ("tournament-logo court-edge signage", 0.95, "0.05-0.2", "sponsor printed perimeter board"),
        ("clay-court line-tape anchor edge", 0.95, "0.05-0.15", "embedded white line strip"),
        ("net-post tension cable mount", 0.85, "0.02-0.1", "geared crank net adjuster"),
        ("ball-kid covered side bench", 0.65, "0.05-0.15", "small awning-and-seat unit"),
        ("court-side speaker post mast", 0.6, "0.02-0.1", "tall PA-speaker pole"),
        ("camera-platform scaffold leg", 0.55, "0.05-0.15", "broadcast tower base"),
        ("scoreboard column base", 0.7, "0.05-0.15", "freestanding score pillar"),
        ("perimeter windscreen tie strap", 0.7, "0.02-0.1", "looped fabric fastener"),
        ("watering-system spigot riser", 0.55, "0.02-0.1", "court-side faucet pipe"),
    ],
    "props": [
        ("tournament tennis racket", 1.0, "0.02-0.1", "stringed graphite racquet"),
        ("optic-yellow tennis ball pair", 0.95, "0.02-0.05", "felt-covered seamed ball"),
        ("courtside player towel", 0.9, "0.02-0.08", "tournament-monogrammed cotton wrap"),
        ("water bottle on side chair", 0.95, "0.02-0.05", "branded sport squeeze bottle"),
        ("banana-and-energy-bar snack", 0.65, "0.02-0.05", "fresh fruit and bar"),
        ("racket-string overgrip roll", 0.7, "0.02-0.05", "tacky black handle wrap"),
        ("vibration string-dampener button", 0.55, "0.02-0.03", "small rubber string ring"),
        ("ball-kid uniform polo", 0.6, "0.02-0.08", "tournament-color short shirt"),
        ("tournament wristband pair", 0.7, "0.02-0.05", "elastic terry sweat cuff"),
        ("clay-court shoe pair", 0.85, "0.02-0.08", "herringbone-sole athletic shoe"),
        ("tournament cap visor", 0.75, "0.02-0.05", "brand-logo athletic cap"),
        ("hawk-eye review tablet", 0.5, "0.02-0.05", "courtside official screen"),
        ("string-tension racket case", 0.55, "0.02-0.08", "padded racquet carrier"),
    ],
}
SLUG_KIND["outdoor_sports_clay_tennis_court"] = "outdoor"


# ── 5. outdoor_sports_soccer_stadium_grass ───────────────────────────────
SETS["outdoor_sports_soccer_stadium_grass"] = {
    "bg": [
        ("packed multi-tier supporter stand", 1.0, "0.7-1.0", "dense crowd-color block"),
        ("home-end ultras choreography wall", 0.85, "0.6-1.0", "team-scarf banner display"),
        ("massive stadium scoreboard pylon", 0.9, "0.6-1.0", "matchday LED video board"),
        ("perimeter advertising boards sweep", 0.95, "0.5-0.9", "sponsor LED ribbon"),
        ("opposite-stand crowd silhouette", 0.85, "0.6-1.0", "far-side spectator wall"),
        ("stadium roof-overhang cantilever", 0.8, "0.5-1.0", "broad covering sweep"),
        ("massive corner flag-bearer banner", 0.65, "0.4-0.8", "huge supporter tifo cloth"),
        ("press-box mid-tier glass front", 0.7, "0.5-0.9", "long media-window strip"),
        ("home-end goal-net far backdrop", 0.95, "0.5-0.9", "tightly hung white goal mesh"),
        ("away-supporter caged section", 0.6, "0.4-0.8", "fenced visitor enclosure"),
        ("team-color seating mosaic", 0.85, "0.6-1.0", "patterned crowd-seat block"),
        ("club crest center-stand mural", 0.65, "0.4-0.8", "painted team-emblem wall"),
        ("VIP director-box parapet line", 0.6, "0.4-0.8", "elevated officials gallery"),
    ],
    "mg": [
        ("regulation grass pitch playing surface", 1.0, "0.4-0.7", "striped mowed turf field"),
        ("painted center-circle midline", 0.95, "0.2-0.4", "bold white midfield arc"),
        ("regulation soccer goal frame", 0.95, "0.2-0.5", "white goal-post crossbar rig"),
        ("white penalty-area boundary line", 0.95, "0.2-0.4", "bold rectangular paint"),
        ("corner-flag standing pole", 0.85, "0.1-0.3", "fluttering small triangle flag"),
        ("touchline-side technical area", 0.85, "0.2-0.4", "painted bench-side rectangle"),
        ("substitute bench dugout shelter", 0.85, "0.2-0.4", "covered courtside shelter"),
        ("fourth-official sideline strip", 0.6, "0.1-0.3", "midline officiating zone"),
        ("center-spot kickoff mark", 0.85, "0.1-0.3", "small painted center dot"),
        ("perimeter LED hoarding row", 0.95, "0.2-0.5", "rolling sponsor light boards"),
        ("warmup zone bib-cluster", 0.7, "0.2-0.4", "spare players in bibs"),
        ("ball-boy crouched touchline", 0.65, "0.1-0.3", "ready-throw retriever"),
    ],
    "fg": [
        ("striped mowed-grass pitch edge", 1.0, "0.3-0.6", "fresh-cut alternating turf strips"),
        ("painted touchline boundary stripe", 0.95, "0.1-0.3", "high-contrast field edge"),
        ("loose match soccer ball", 0.9, "0.05-0.15", "panel-stitched white ball"),
        ("scuffed cleat divot turf foreground", 0.85, "0.05-0.2", "torn turf-mark patch"),
        ("dropped goalkeeper glove pair", 0.55, "0.02-0.1", "padded tacky-palm glove"),
        ("captain armband on grass", 0.5, "0.02-0.05", "elastic colored captain band"),
        ("ball-boy crouched touchline foreground", 0.7, "0.1-0.2", "ready-throw retriever"),
        ("sideline speaker cable foreground", 0.5, "0.02-0.1", "loose taped cable"),
        ("player water-bottle on grass", 0.85, "0.02-0.1", "squeeze-top sport bottle"),
        ("dropped sweat headband pair", 0.55, "0.02-0.05", "elastic terry sweat cuff"),
        ("ref-coin toss-coin foreground", 0.5, "0.02-0.03", "polished match-coin"),
        ("rolled bench warmup jacket", 0.7, "0.05-0.15", "team-color tracksuit pile"),
    ],
    "arch": [
        ("stadium cantilever roof undercut", 0.85, "0.1-0.2", "vast covering sweep"),
        ("floodlight pylon mast top", 0.95, "0.05-0.2", "high-mast lamp cluster"),
        ("painted touchline-edge tartan strip", 0.7, "0.05-0.15", "warmup synthetic edge"),
        ("perimeter LED-board column base", 0.95, "0.05-0.2", "rolling sponsor light face"),
        ("goal-mouth net-anchor post cap", 0.95, "0.05-0.15", "frame-mounted mesh fastener"),
        ("dugout-shelter polycarbonate roof", 0.85, "0.05-0.15", "transparent curved cover"),
        ("press-box overhang glass-front fascia", 0.7, "0.05-0.15", "long media-window strip"),
        ("VAR review-monitor side cabinet", 0.55, "0.02-0.1", "fourth-official screen"),
        ("video-screen pylon support strut", 0.7, "0.05-0.15", "scoreboard column brace"),
        ("speaker-cluster pylon hang", 0.7, "0.02-0.1", "high-mast PA array"),
    ],
    "props": [
        ("regulation match soccer ball", 1.0, "0.02-0.05", "panel-stitched white ball"),
        ("captain elastic armband", 0.7, "0.02-0.05", "colored captain band"),
        ("goalkeeper foam-palm glove", 0.7, "0.02-0.08", "padded tacky-palm glove"),
        ("club training cone marker", 0.85, "0.02-0.05", "bright orange-and-yellow cone"),
        ("substitute warmup bib", 0.85, "0.02-0.08", "mesh team-color overshirt"),
        ("athletic-tape roll", 0.7, "0.02-0.05", "white cotton finger-tape spool"),
        ("sideline water-bottle squad rack", 0.85, "0.02-0.1", "carrier of squeeze bottles"),
        ("substitution number paddle", 0.6, "0.02-0.05", "digit-display sub paddle"),
        ("club-monogrammed match towel", 0.7, "0.02-0.08", "team-emblem cotton wrap"),
        ("shin-guard pair", 0.7, "0.02-0.05", "padded plastic leg shield"),
        ("referee whistle-and-card set", 0.7, "0.02-0.03", "lanyard with cards"),
        ("match-day lineup clipboard", 0.55, "0.02-0.05", "official tactical sheet"),
        ("captain coin-toss pendant", 0.5, "0.02-0.03", "polished match-coin"),
    ],
}
SLUG_KIND["outdoor_sports_soccer_stadium_grass"] = "outdoor"


# ── 6. indoor_sports_competition_swim_pool_blocks ───────────────────────
SETS["indoor_sports_competition_swim_pool_blocks"] = {
    "bg": [
        ("tiled aquatic-center wall mosaic", 1.0, "0.7-1.0", "large-format tile field"),
        ("tiered natatorium spectator gallery", 0.95, "0.6-1.0", "rows of fold-down seat"),
        ("federation-standard timing-pad endwall", 0.95, "0.6-1.0", "touchpad-mounted pool wall"),
        ("massive electronic results board", 0.9, "0.6-1.0", "swim-times LED panel"),
        ("opposite-deck team-warmup gallery", 0.7, "0.5-0.9", "across-pool team area"),
        ("aquatic-center skylight clerestory", 0.8, "0.5-0.9", "high-windowed daylight band"),
        ("tournament sponsor banner sweep", 0.85, "0.6-0.9", "stretched event-cloth row"),
        ("starter podium painted backdrop", 0.7, "0.5-0.9", "raised official platform wall"),
        ("flags-of-nations sweep", 0.6, "0.4-0.8", "row of overhead country flags"),
        ("painted pool-name hall-of-fame wall", 0.55, "0.4-0.8", "championship records wall"),
        ("acoustic-baffle ceiling backdrop", 0.7, "0.5-0.9", "sound-dampening hall ceiling"),
        ("federation-record split-time wall", 0.65, "0.4-0.8", "engraved record-board"),
        ("warmup-pool adjacent overflow lane", 0.7, "0.4-0.8", "secondary lane row"),
    ],
    "mg": [
        ("competition pool starting block row", 1.0, "0.3-0.6", "track-style angled diving block"),
        ("ten-lane aquatic competition pool", 1.0, "0.4-0.7", "lane-roped clear-water field"),
        ("backstroke flag overhead row", 0.85, "0.1-0.3", "small triangle hanging pennant"),
        ("center-deck starter podium", 0.85, "0.2-0.4", "raised official platform"),
        ("electronic touchpad endwall row", 0.95, "0.2-0.4", "yellow-and-black touch panel"),
        ("camera-rig poolside boom", 0.6, "0.1-0.3", "underwater-cable broadcast unit"),
        ("warm-down lane parallel area", 0.7, "0.2-0.4", "secondary recovery lane"),
        ("officials' walkway poolside line", 0.7, "0.2-0.4", "deck-side referee path"),
        ("stagger-painted lane number tab", 0.85, "0.1-0.3", "lane-numeral end mark"),
        ("starter horn-and-strobe stand", 0.55, "0.1-0.3", "audio-trigger official unit"),
        ("warmup-pool kickboard rack", 0.6, "0.1-0.3", "stacked foam training boards"),
        ("athlete pre-swim ready bench", 0.65, "0.1-0.3", "ready-room bench seat"),
    ],
    "fg": [
        ("textured pool-deck non-slip tile edge", 1.0, "0.3-0.6", "rough anti-slip ceramic"),
        ("starting-block angled face-edge", 0.95, "0.2-0.4", "track-grip blue ramp"),
        ("painted lane-number numeral", 0.95, "0.05-0.15", "bold white deck numeral"),
        ("dropped racing goggle pair", 0.7, "0.02-0.08", "low-profile silicone seal"),
        ("scuffed deck-shoe slip-on pair", 0.55, "0.02-0.08", "rubber sandal pair"),
        ("rolled chamois drying-towel", 0.7, "0.02-0.1", "small synthetic squeeze towel"),
        ("racing-cap latex foreground", 0.7, "0.02-0.05", "smooth team-color cap"),
        ("warmup-stretch foam roller", 0.5, "0.05-0.15", "high-density cylindrical roller"),
        ("athlete robe drape on bench", 0.7, "0.05-0.2", "team-monogrammed parka"),
        ("painted record-time tab on deck", 0.55, "0.02-0.08", "engraved name plaque"),
        ("water-puddle reflection patch", 0.85, "0.05-0.2", "shallow deck-water gloss"),
        ("starter-horn cable foreground", 0.5, "0.02-0.05", "loose audio cable"),
    ],
    "arch": [
        ("acoustic-baffle natatorium ceiling panel", 0.95, "0.1-0.2", "sound-dampening tile"),
        ("clerestory skylight band overhead", 0.85, "0.05-0.2", "high-window strip"),
        ("backstroke-flag suspension cable", 0.85, "0.05-0.2", "tightened transverse line"),
        ("perimeter pool-deck gutter rim", 0.95, "0.05-0.15", "overflow channel ledge"),
        ("starter strobe pole-top fixture", 0.7, "0.02-0.1", "synchronized flash unit"),
        ("camera-rig overhead boom hang", 0.55, "0.02-0.1", "broadcast suspended unit"),
        ("tile-and-grout pool-edge join", 0.85, "0.05-0.15", "ceramic-grouted edge"),
        ("dehumidifier vent-grille panel", 0.7, "0.05-0.15", "louvered ceiling vent"),
        ("pool-side handrail stainless rail", 0.85, "0.05-0.15", "tubular stair handrail"),
        ("starter podium base-plinth edge", 0.7, "0.05-0.15", "raised official base"),
    ],
    "props": [
        ("racing low-profile swim goggle", 1.0, "0.02-0.08", "silicone-seal racing eyewear"),
        ("team-color silicone swim cap", 1.0, "0.02-0.05", "smooth molded racing cap"),
        ("athlete kickboard foam slab", 0.7, "0.02-0.1", "training swim board"),
        ("split-time stopwatch lanyard", 0.85, "0.02-0.05", "digital lap-timer pendant"),
        ("synthetic chamois towel", 0.85, "0.02-0.08", "high-absorbency squeeze cloth"),
        ("nose-clip swim accessory", 0.55, "0.02-0.03", "small silicone clip"),
        ("athlete deck robe parka", 0.85, "0.02-0.1", "team-monogrammed warm parka"),
        ("pull-buoy training float", 0.55, "0.02-0.05", "thigh-grip foam float"),
        ("athlete energy-gel pouch", 0.6, "0.02-0.03", "small foil-pack supplement"),
        ("hand-paddle training pair", 0.5, "0.02-0.05", "rigid pull-paddle"),
        ("split-card lap-counter paddle", 0.6, "0.02-0.05", "numbered turn-board"),
        ("athlete water-bottle squeeze", 0.85, "0.02-0.05", "branded sport bottle"),
        ("ear-plug silicone pair", 0.5, "0.02-0.03", "small molded ear-cap"),
    ],
}
SLUG_KIND["indoor_sports_competition_swim_pool_blocks"] = "indoor"


# ── 7. indoor_sports_ice_hockey_arena ────────────────────────────────────
SETS["indoor_sports_ice_hockey_arena"] = {
    "bg": [
        ("packed hockey-arena seating bowl", 1.0, "0.7-1.0", "rows of fold-down spectator seat"),
        ("dasher-board sponsor-graphic perimeter", 1.0, "0.6-1.0", "wrapped advertising rink edge"),
        ("rink-end glass-and-net protective wall", 0.95, "0.6-1.0", "tempered safety glass band"),
        ("center-hung scoreboard video cube", 0.95, "0.6-1.0", "four-sided rink video board"),
        ("retired-jersey rafter banner display", 0.75, "0.5-0.9", "hanging numbered jerseys"),
        ("home-team championship-banner row", 0.85, "0.6-0.9", "rafter title cloth"),
        ("Zamboni-tunnel resurfacer entrance", 0.7, "0.4-0.8", "rink-corner ice-machine portal"),
        ("press-box upper-tier glass front", 0.7, "0.5-0.9", "media-window strip"),
        ("home-team penalty-box backdrop", 0.7, "0.5-0.9", "open-front sin-bin enclosure"),
        ("opposite-end goal-net backdrop", 0.95, "0.4-0.8", "white tied goal mesh"),
        ("club-emblem center-stand mural", 0.6, "0.4-0.8", "painted team-emblem wall"),
        ("organist platform mid-tier", 0.5, "0.3-0.7", "small raised musician stand"),
        ("LED ribbon-board upper-fascia sweep", 0.85, "0.6-1.0", "scrolling rink-edge graphic"),
    ],
    "mg": [
        ("regulation hockey-rink ice surface", 1.0, "0.4-0.7", "smooth resurfaced clear ice"),
        ("center-ice face-off circle", 0.95, "0.2-0.4", "painted central red dot"),
        ("painted blue-line zone divider", 0.95, "0.2-0.4", "bold rink-zone stripe"),
        ("regulation goal-frame net rig", 0.95, "0.2-0.4", "red-frame goal-mouth"),
        ("home-team bench player area", 0.85, "0.2-0.4", "open-front rink-side bench"),
        ("officials' time-keeper bench", 0.7, "0.1-0.3", "rink-side scorer table"),
        ("penalty-box rink-edge enclosure", 0.85, "0.2-0.4", "open glass sin-bin"),
        ("face-off-circle painted dot row", 0.85, "0.1-0.3", "secondary face-off mark"),
        ("dasher-board rounded rink corner", 0.95, "0.2-0.4", "curved board panel"),
        ("Zamboni resurfacer mid-ice", 0.5, "0.2-0.5", "ice-machine resurface unit"),
        ("warmup-shoot puck pile", 0.6, "0.1-0.3", "loose vulcanized rubber pucks"),
        ("center-ice referee skater stance", 0.7, "0.1-0.3", "striped jersey official"),
    ],
    "fg": [
        ("smooth ice-rink surface edge", 1.0, "0.3-0.6", "freshly resurfaced sheet"),
        ("painted goal-crease blue arc", 0.85, "0.05-0.15", "rink-end painted arc"),
        ("vulcanized rubber puck on ice", 0.9, "0.02-0.05", "small black disk"),
        ("bench-side glove-and-stick pile", 0.7, "0.05-0.2", "rink-side gear pile"),
        ("dropped tape-roll on dasher", 0.55, "0.02-0.05", "white stick-tape spool"),
        ("stainless skate-blade scrape mark", 0.8, "0.05-0.15", "fine ice-cut groove"),
        ("rink-side water-bottle row", 0.7, "0.05-0.15", "carrier of squeeze bottles"),
        ("rolled towel near bench", 0.7, "0.05-0.15", "team-monogrammed cotton wrap"),
        ("stick-blade tape edge", 0.7, "0.02-0.08", "wrapped black blade tape"),
        ("rink-edge ice-shave snow band", 0.85, "0.05-0.15", "sprayed ice particulate"),
        ("dropped helmet near bench", 0.55, "0.02-0.1", "shell with cage face guard"),
        ("rink-side scorer paper-stack", 0.5, "0.02-0.05", "printed lineup sheet"),
    ],
    "arch": [
        ("rink-arena cantilever ceiling", 0.85, "0.1-0.2", "soaring exposed roof"),
        ("center-ice video-cube hang rig", 0.95, "0.05-0.15", "suspended four-sided board"),
        ("rafter lighting truss cluster", 0.85, "0.05-0.15", "high-output ceiling spot grid"),
        ("dasher-board plexiglass top edge", 0.95, "0.05-0.2", "tempered safety panel"),
        ("rafter-hung championship banner", 0.85, "0.05-0.2", "fabric title cloth"),
        ("LED ribbon-fascia rink edge", 0.85, "0.05-0.2", "scrolling graphic strip"),
        ("ice-resurfacer ramp portal frame", 0.7, "0.05-0.15", "rink-corner machine portal"),
        ("glass-pane mounting frame top cap", 0.95, "0.05-0.15", "rink safety-glass frame"),
        ("speaker-cluster line-array hang", 0.7, "0.05-0.15", "vertical sound array"),
        ("rink-end goal-camera mount", 0.7, "0.02-0.1", "robotic mounted camera"),
    ],
    "props": [
        ("composite hockey stick", 1.0, "0.02-0.1", "tape-wrapped graphite shaft"),
        ("vulcanized rubber puck", 1.0, "0.02-0.05", "small black disk"),
        ("padded goalie blocker", 0.7, "0.02-0.08", "rectangular padded glove"),
        ("catcher trapper goalie glove", 0.7, "0.02-0.08", "webbed-pocket goalie mitt"),
        ("composite skate boot pair", 0.85, "0.02-0.1", "stainless-blade skate"),
        ("rink-side athletic-tape roll", 0.85, "0.02-0.05", "white stick-tape spool"),
        ("team-color helmet with cage", 0.85, "0.02-0.1", "shell with face guard"),
        ("rink-side water-bottle squeeze", 0.85, "0.02-0.05", "branded sport bottle"),
        ("club-monogrammed bench towel", 0.7, "0.02-0.08", "team-emblem cotton wrap"),
        ("captain elbow-pad pair", 0.55, "0.02-0.08", "padded shell elbow guard"),
        ("rink-side stick-rack holder", 0.65, "0.02-0.1", "vertical stick storage"),
        ("face-shield visor accessory", 0.55, "0.02-0.05", "clear shield half-cage"),
        ("rink-side referee whistle", 0.55, "0.02-0.03", "metal lanyard whistle"),
    ],
}
SLUG_KIND["indoor_sports_ice_hockey_arena"] = "indoor"


# ── 8. outdoor_sports_baseball_diamond_pro ───────────────────────────────
SETS["outdoor_sports_baseball_diamond_pro"] = {
    "bg": [
        ("tiered baseball-park grandstand bowl", 1.0, "0.7-1.0", "rows of fold-down stadium seat"),
        ("center-field batter-eye dark wall", 0.95, "0.5-0.9", "matte dark-green vision wall"),
        ("perimeter outfield wall sponsor sweep", 0.95, "0.5-0.9", "padded ad-wrapped fence"),
        ("massive video scoreboard pylon", 0.9, "0.5-0.9", "large LED video panel"),
        ("foul-pole vertical yellow tower", 0.85, "0.5-0.9", "tall painted boundary marker"),
        ("retired-number outfield-wall display", 0.7, "0.4-0.8", "painted commemorative numerals"),
        ("press-box mid-tier glass front", 0.7, "0.5-0.9", "media-window strip"),
        ("home-club team-flag pole row", 0.6, "0.4-0.8", "high-mast banner row"),
        ("opposite-stand crowd silhouette", 0.85, "0.6-1.0", "far-side spectator wall"),
        ("home-team championship pennant row", 0.7, "0.4-0.8", "rafter title cloth"),
        ("manual scoreboard vintage facade", 0.55, "0.4-0.8", "hand-flipped numeral wall"),
        ("bullpen-pen relief-pitcher mound area", 0.6, "0.3-0.7", "open relief-warmup pen"),
        ("perimeter pitching-bullpen gate sweep", 0.55, "0.4-0.8", "low pen-edge fence"),
    ],
    "mg": [
        ("regulation baseball-diamond infield", 1.0, "0.4-0.7", "graded packed-clay infield"),
        ("regulation pitching-mound dirt circle", 0.95, "0.2-0.4", "raised packed-clay rise"),
        ("home-plate pentagon shape mark", 0.95, "0.1-0.3", "white painted pentagon"),
        ("first-base fielder ready stance", 0.7, "0.1-0.3", "ready-glove fielder pose"),
        ("on-deck circle hitter ready zone", 0.7, "0.1-0.3", "painted ready ring"),
        ("dugout home-team bench shelter", 0.85, "0.2-0.4", "covered courtside shelter"),
        ("painted foul-line chalk stripe", 0.95, "0.2-0.4", "bold chalked field-edge stripe"),
        ("warning-track outfield-edge strip", 0.85, "0.2-0.4", "loose-cinder transition strip"),
        ("ground-crew chalk-line cart figure", 0.6, "0.1-0.3", "wheeled chalk dispenser"),
        ("infielder dirt-cutout base path", 0.85, "0.2-0.4", "packed runner pathway"),
        ("ump home-plate official squat", 0.7, "0.1-0.3", "padded plate-umpire stance"),
        ("center-field flagpole short-mast", 0.55, "0.1-0.3", "small park flag standard"),
    ],
    "fg": [
        ("packed clay infield surface edge", 1.0, "0.3-0.6", "graded fine-clay top dressing"),
        ("painted foul-line chalk edge", 0.95, "0.1-0.3", "freshly chalked field stripe"),
        ("home-plate pentagon foreground", 0.95, "0.05-0.15", "white painted pentagon"),
        ("loose baseball on infield", 0.95, "0.02-0.05", "leather-hide stitched ball"),
        ("dropped fielder mitt foreground", 0.7, "0.05-0.1", "leather pocket fielder glove"),
        ("scuffed batting helmet foreground", 0.55, "0.05-0.1", "shell with single-flap"),
        ("rosin-bag near pitching rubber", 0.7, "0.02-0.05", "small grip-talc pouch"),
        ("dropped batting donut weight", 0.55, "0.02-0.05", "rubber bat weight ring"),
        ("scuffed cleat divot dirt foreground", 0.7, "0.05-0.15", "torn clay-mark patch"),
        ("warmup pine-tar stick", 0.55, "0.02-0.05", "grip-tar applicator stick"),
        ("dropped batting-glove pair", 0.6, "0.02-0.08", "leather palm-grip glove"),
        ("foul-line chalk-cart wheel mark", 0.55, "0.02-0.08", "fresh-chalk applicator track"),
    ],
    "arch": [
        ("foul-pole vertical column edge", 0.85, "0.05-0.2", "tall yellow boundary tower"),
        ("perimeter outfield padding edge", 0.95, "0.05-0.2", "padded ad-wrapped wall edge"),
        ("dugout-shelter polycarbonate roof", 0.85, "0.05-0.15", "transparent curved cover"),
        ("press-box overhang glass-front fascia", 0.7, "0.05-0.15", "long media-window strip"),
        ("scoreboard pylon support strut", 0.7, "0.05-0.15", "large board column"),
        ("painted home-plate rim arc", 0.85, "0.05-0.15", "small rim-edge stripe"),
        ("bullpen-gate fence cap", 0.7, "0.05-0.1", "low pen-edge rail"),
        ("speaker-cluster pylon hang", 0.7, "0.02-0.1", "stadium PA cluster"),
        ("flag-pole standard mast top", 0.55, "0.02-0.1", "high-mast metal cap"),
        ("backstop-screen netting cable", 0.85, "0.05-0.2", "tightened safety-net cord"),
    ],
    "props": [
        ("regulation leather baseball", 1.0, "0.02-0.05", "stitched-hide white ball"),
        ("ash baseball bat", 0.85, "0.02-0.1", "lathed wood bat"),
        ("leather fielder mitt", 0.95, "0.02-0.08", "pocketed catch glove"),
        ("catcher chest-protector", 0.7, "0.02-0.1", "padded torso shell"),
        ("batter helmet single-flap", 0.85, "0.02-0.08", "ear-flap protective helmet"),
        ("rosin-bag grip pouch", 0.7, "0.02-0.05", "talc-filled pitcher pouch"),
        ("pine-tar stick", 0.55, "0.02-0.05", "grip-tar applicator"),
        ("catcher mask cage", 0.65, "0.02-0.08", "wired shield face guard"),
        ("batting-donut weight ring", 0.55, "0.02-0.05", "rubber on-deck weight"),
        ("club-monogrammed cap", 0.9, "0.02-0.05", "team-emblem fitted cap"),
        ("eye-black grease tube", 0.55, "0.02-0.03", "small grease applicator"),
        ("dugout sunflower-seed bag", 0.6, "0.02-0.05", "branded snack pouch"),
        ("ball-player batting glove pair", 0.85, "0.02-0.05", "leather palm-grip glove"),
    ],
}
SLUG_KIND["outdoor_sports_baseball_diamond_pro"] = "outdoor"


# ── 9. indoor_sports_locker_room_pro ─────────────────────────────────────
SETS["indoor_sports_locker_room_pro"] = {
    "bg": [
        ("painted team-logo locker-room mural wall", 1.0, "0.7-1.0", "matte painted club emblem"),
        ("hardwood-veneer locker stall row", 0.95, "0.7-1.0", "tall stall-front cabinet line"),
        ("framed championship-photo wall", 0.85, "0.6-0.9", "rows of black-framed prints"),
        ("brushed-metal stall-row backdrop", 0.85, "0.6-1.0", "brushed-aluminum cabinet skin"),
        ("etched-glass team-motto wall", 0.7, "0.5-0.9", "softly etched motivational text"),
        ("retired-number jersey wall", 0.75, "0.5-0.9", "framed numbered jerseys"),
        ("monochrome accent paint wall", 0.85, "0.7-1.0", "matte deep team-color paint"),
        ("trainer-station tile wall", 0.7, "0.6-0.9", "smooth tile sports-medical wall"),
        ("rear training-room glass wall", 0.6, "0.5-0.9", "transparent glazing partition"),
        ("low backlit locker-name plate row", 0.75, "0.5-0.9", "etched athlete-name strip"),
        ("hand-painted club-creed quote wall", 0.65, "0.5-0.9", "stencil-painted creed"),
        ("upper trophy-shelf-row backdrop", 0.7, "0.5-0.9", "row of trophy displays"),
        ("dark stained wood-cap molding wall", 0.7, "0.6-1.0", "rich stained timber edge"),
    ],
    "mg": [
        ("hardwood-front athlete locker stall", 1.0, "0.3-0.6", "open-front numbered stall"),
        ("athlete bench seating row", 0.9, "0.2-0.4", "padded stall-front bench"),
        ("stall-mounted nameplate", 0.95, "0.1-0.3", "etched athlete-name plate"),
        ("trainer rolling massage-table", 0.7, "0.2-0.4", "padded portable table"),
        ("center-room speaker-and-sound dock", 0.55, "0.2-0.4", "small portable stereo"),
        ("freestanding gear bin column", 0.7, "0.2-0.4", "open-top equipment hamper"),
        ("strategy whiteboard rolling stand", 0.7, "0.2-0.4", "wheeled tactical board"),
        ("cold-tub recovery hydro station", 0.55, "0.2-0.4", "tiled cold-water tub"),
        ("media-prep podium stand", 0.55, "0.1-0.3", "small interview podium"),
        ("trophy-display center pedestal", 0.7, "0.1-0.3", "lit display plinth"),
        ("rolling laundry-cart hamper", 0.7, "0.1-0.3", "canvas wheeled hamper"),
        ("athlete dressing-bench rack", 0.85, "0.2-0.4", "long padded bench"),
    ],
    "fg": [
        ("polished-wood locker-room floor edge", 1.0, "0.3-0.6", "matte sealed timber plank"),
        ("low-pile carpeted center-room rug", 0.7, "0.3-0.6", "looped commercial carpet"),
        ("dropped athletic-tape roll", 0.7, "0.02-0.05", "white cotton finger-tape spool"),
        ("scattered athletic-shoe pair", 0.85, "0.05-0.15", "team-issued cross-trainer"),
        ("rolled match-day towel", 0.85, "0.05-0.15", "team-monogrammed cotton wrap"),
        ("dropped warm-up tee shirt", 0.7, "0.05-0.15", "team-color cotton tee"),
        ("scuffed gear bag on floor", 0.7, "0.05-0.2", "duffel team-emblem bag"),
        ("water-bottle on bench", 0.85, "0.02-0.08", "branded sport squeeze bottle"),
        ("dropped wrist-tape strip", 0.55, "0.02-0.05", "loose stretch wrist tape"),
        ("dropped earbud pair", 0.5, "0.02-0.03", "wireless earpiece pair"),
        ("foam roller along bench", 0.6, "0.05-0.15", "high-density cylindrical roller"),
        ("dropped pre-game letter envelope", 0.5, "0.02-0.05", "personal handwritten note"),
    ],
    "arch": [
        ("recessed downlight track grid", 0.95, "0.05-0.2", "row of inset spot fixtures"),
        ("backlit nameplate cove strip", 0.85, "0.05-0.15", "warm name-plate edge glow"),
        ("locker-stall divider top cap", 0.95, "0.05-0.15", "wood-cap stall divider"),
        ("ceiling-mounted speaker grille flush", 0.7, "0.02-0.08", "circular perforated disk"),
        ("door-portal entry threshold edge", 0.7, "0.05-0.15", "transition floor strip"),
        ("trainer-area glass partition reveal", 0.7, "0.05-0.15", "frameless glazing edge"),
        ("painted wood crown-molding band", 0.7, "0.05-0.15", "rich stained timber edge"),
        ("HVAC-supply diffuser grille", 0.7, "0.05-0.15", "white louvered ceiling vent"),
        ("ceiling-edge plaster shadow line", 0.6, "0.02-0.08", "subtly graded reveal line"),
        ("locker-row LED accent strip", 0.7, "0.05-0.15", "soft hidden cabinet glow"),
    ],
    "props": [
        ("athlete cleat-and-shoe pair in stall", 0.95, "0.02-0.1", "team-issued sport shoe"),
        ("gear duffel bag", 0.95, "0.02-0.1", "team-emblem zip duffel"),
        ("rolled match-day towel", 0.95, "0.02-0.08", "team-monogrammed cotton wrap"),
        ("wireless headphone over-ear pair", 0.85, "0.02-0.05", "padded over-ear headphone"),
        ("athlete shaker-blender bottle", 0.85, "0.02-0.05", "agitator-blender sports cup"),
        ("athletic-tape roll set", 0.85, "0.02-0.05", "white cotton finger-tape spool"),
        ("foam-roller recovery tool", 0.7, "0.02-0.1", "high-density cylindrical roller"),
        ("massage-gun percussion device", 0.7, "0.02-0.05", "handheld massage gun"),
        ("cold-tub thermometer probe", 0.5, "0.02-0.03", "digital tub temp probe"),
        ("hand-grip stress squeezer", 0.55, "0.02-0.03", "spring squeeze trainer"),
        ("personal pre-game playlist player", 0.55, "0.02-0.03", "small portable music player"),
        ("trainer's strapping-tape scissors", 0.55, "0.02-0.03", "blunt-tip athletic shears"),
        ("athlete recovery slipper pair", 0.7, "0.02-0.08", "soft slip-on recovery sandal"),
    ],
}
SLUG_KIND["indoor_sports_locker_room_pro"] = "indoor"


# ── 10. indoor_sports_mma_cage_octagon ───────────────────────────────────
SETS["indoor_sports_mma_cage_octagon"] = {
    "bg": [
        ("dark fight-night arena seating bowl", 1.0, "0.7-1.0", "rows of dimmed spectator seat"),
        ("smoke-haze upper-tier crowd silhouette", 0.85, "0.6-1.0", "atmospheric crowd backdrop"),
        ("massive sponsor-logo cage backdrop", 0.95, "0.6-1.0", "branded promotional fascia"),
        ("center-hung video cube fight rig", 0.9, "0.5-0.9", "suspended LED video display"),
        ("dim media-photographer pit", 0.7, "0.4-0.8", "front-row long-lens cluster"),
        ("walk-out tunnel entrance portal", 0.7, "0.4-0.8", "darkened dramatic walk-out"),
        ("ring-girl perimeter platform", 0.6, "0.4-0.8", "round-card walking lane"),
        ("low-arena spotlit canvas glow", 0.85, "0.5-0.9", "downcast spot pool over canvas"),
        ("event-night sponsor banner sweep", 0.85, "0.6-0.9", "stretched fight-night cloth"),
        ("smoke-machine atmospheric haze", 0.7, "0.5-0.9", "soft drifting fog layer"),
        ("front-row VIP seated crowd", 0.85, "0.5-0.9", "first-row spectator block"),
        ("dimmed organist-and-DJ booth", 0.5, "0.3-0.7", "small audio-side platform"),
        ("rafter-suspended event banner", 0.7, "0.4-0.8", "fight-night rafter cloth"),
    ],
    "mg": [
        ("regulation eight-sided fighting cage", 1.0, "0.3-0.6", "metal-fence octagonal enclosure"),
        ("padded fight-canvas cage floor", 1.0, "0.3-0.6", "stretched event-cloth canvas"),
        ("cage-gate hinged-entry door", 0.85, "0.2-0.4", "fenced-mesh entry panel"),
        ("octagon-corner cornerman stool", 0.85, "0.2-0.4", "fight-corner perch seat"),
        ("ring-side judges' table row", 0.85, "0.2-0.4", "rink-side judging bench"),
        ("cutman seat near corner", 0.7, "0.1-0.3", "cornerman wound-care seat"),
        ("chain-link cage-wall fencing", 0.95, "0.3-0.5", "vinyl-coated mesh barrier"),
        ("ring-announcer center-cage stance", 0.7, "0.1-0.3", "tuxedo-clad announcer"),
        ("cage-side broadcast-camera operator", 0.7, "0.1-0.3", "shoulder-cam shooter"),
        ("octagon-corner cornerman cluster", 0.7, "0.2-0.4", "fight-corner team huddle"),
        ("center-canvas referee stance", 0.85, "0.1-0.3", "referee in cage"),
        ("ring-card girl walking line", 0.55, "0.1-0.3", "round-card walker"),
    ],
    "fg": [
        ("padded octagon-canvas floor edge", 1.0, "0.3-0.6", "stretched event-canvas surface"),
        ("painted center-cage logo decal", 0.85, "0.05-0.2", "vinyl center mat decal"),
        ("dropped athletic-tape strip", 0.7, "0.02-0.05", "loose stretch wrist tape"),
        ("cornerman water-bucket ground", 0.7, "0.05-0.15", "battered metal pail"),
        ("dropped fight-glove pair", 0.85, "0.02-0.08", "open-finger MMA glove"),
        ("scuffed mouthguard near corner", 0.55, "0.02-0.05", "small clamshell mouthguard"),
        ("blood-and-vaseline gauze square", 0.55, "0.02-0.05", "cut-care medical gauze"),
        ("rolled hand-wrap on canvas", 0.7, "0.02-0.08", "coiled cotton bandage"),
        ("athlete corner-stool foreground", 0.7, "0.05-0.15", "fight-corner perch"),
        ("dropped corner-towel on canvas", 0.7, "0.05-0.15", "fight-corner cotton wrap"),
        ("cage-floor athlete shorts foreground", 0.6, "0.02-0.08", "loose fight-trunk pair"),
        ("dropped Vaseline tin", 0.5, "0.02-0.03", "small grease tin"),
    ],
    "arch": [
        ("octagon-cage roof-canopy hang", 0.7, "0.05-0.15", "open-air fence top edge"),
        ("center-canvas cage-floor riser edge", 0.95, "0.05-0.15", "raised platform base"),
        ("rafter spot truss with arena par-cans", 0.85, "0.05-0.2", "high-output ceiling spot grid"),
        ("center-hung video-cube hang rig", 0.85, "0.05-0.15", "suspended four-sided board"),
        ("cage-side judging-table riser strip", 0.7, "0.05-0.15", "raised judging bench front"),
        ("walk-out tunnel arch frame", 0.7, "0.05-0.2", "dramatic dark portal arch"),
        ("smoke-machine cage-floor vent strip", 0.7, "0.02-0.1", "low fog dispersal port"),
        ("cage-perimeter event-LED edge", 0.85, "0.05-0.2", "rolling sponsor light fascia"),
        ("speaker-cluster line-array hang", 0.7, "0.05-0.15", "vertical sound array"),
        ("ring-side cameraman boom-arm", 0.55, "0.02-0.1", "broadcast camera boom"),
    ],
    "props": [
        ("MMA open-finger fight glove", 1.0, "0.02-0.08", "padded open-finger glove"),
        ("athlete fight-shorts pair", 0.85, "0.02-0.08", "loose fight-trunk pair"),
        ("cornerman water bottle", 0.85, "0.02-0.05", "branded sport squeeze bottle"),
        ("cornerman ice-pack compress", 0.7, "0.02-0.05", "fabric-wrapped cold pack"),
        ("cutman vaseline tin", 0.7, "0.02-0.03", "small grease tin"),
        ("athletic-tape roll set", 0.85, "0.02-0.05", "white cotton finger-tape spool"),
        ("hand-wrap cotton roll", 0.85, "0.02-0.05", "coiled stretch bandage"),
        ("mouthguard clamshell case", 0.7, "0.02-0.03", "small protective case"),
        ("cornerman wound-swab end-tip", 0.55, "0.02-0.03", "cotton-end care swab"),
        ("athlete groin-protector cup", 0.55, "0.02-0.03", "rigid groin guard"),
        ("fight-night robe drape", 0.6, "0.02-0.1", "branded entrance robe"),
        ("athletic-shears blunt scissors", 0.55, "0.02-0.03", "blunt-tip cutman shears"),
        ("smelling-salt vial pair", 0.5, "0.02-0.03", "small ammonia capsule"),
    ],
}
SLUG_KIND["indoor_sports_mma_cage_octagon"] = "indoor"


# ── 11. outdoor_sports_golf_driving_range ────────────────────────────────
SETS["outdoor_sports_golf_driving_range"] = {
    "bg": [
        ("distant range-end target green sweep", 1.0, "0.6-1.0", "manicured fairway target lawn"),
        ("yardage-target flag-cluster row", 0.95, "0.5-0.9", "spaced numbered yardage flag"),
        ("perimeter range-fence safety net", 0.95, "0.6-1.0", "tall mesh boundary net"),
        ("clubhouse façade range-side", 0.7, "0.4-0.8", "low-profile clubhouse roofline"),
        ("range-end pine-windbreak row", 0.85, "0.5-0.9", "tall conifer perimeter line"),
        ("covered teaching-bay overhang", 0.85, "0.5-0.9", "long open-front teaching shelter"),
        ("range-side pro-shop awning", 0.6, "0.4-0.8", "small low-profile shop awning"),
        ("driving-range bunker-target sand patch", 0.7, "0.4-0.8", "white-sand target hazard"),
        ("manicured range-shoulder fairway grass", 0.85, "0.5-0.9", "fine cut-grass shoulder"),
        ("range-marker yardage post sweep", 0.85, "0.4-0.8", "row of distance posts"),
        ("range-pro lesson-tee shaded gallery", 0.6, "0.3-0.7", "shaded teaching gallery"),
        ("low parkway hedge perimeter", 0.65, "0.4-0.8", "trimmed boxwood line"),
        ("range-side cart-path strip", 0.7, "0.4-0.8", "asphalt cart-path edge"),
    ],
    "mg": [
        ("range-tee artificial mat row", 1.0, "0.3-0.6", "rubber-base hitting mat"),
        ("ball-pyramid range pyramid stack", 0.95, "0.1-0.3", "pyramidal range-ball stack"),
        ("range-side ball-dispenser column", 0.85, "0.2-0.4", "automatic ball-feeder unit"),
        ("range-stretcher athlete warmup zone", 0.7, "0.2-0.4", "warmup loose-stretch area"),
        ("driving-tee yardage-flag pole", 0.85, "0.1-0.3", "small numbered yardage flag"),
        ("ball-picker tractor mid-range", 0.7, "0.1-0.3", "wheeled-cage range collector"),
        ("range-pro instructor stance", 0.7, "0.1-0.3", "pro-coach standing pose"),
        ("range-side bag-stand cluster", 0.85, "0.2-0.4", "row of stand-bag golf bags"),
        ("yardage-target green-marker zone", 0.85, "0.2-0.4", "marked target green"),
        ("range-side lesson-tee partition", 0.7, "0.1-0.3", "small bay-divider partition"),
        ("ball-washer station mid-range", 0.55, "0.1-0.3", "small cleaning station"),
        ("club-rack range-side stand", 0.7, "0.1-0.3", "vertical demo-club rack"),
    ],
    "fg": [
        ("artificial-turf range-mat surface edge", 1.0, "0.3-0.6", "rubber-base hitting mat"),
        ("painted range-tee distance line", 0.85, "0.05-0.15", "small painted yardage stripe"),
        ("loose range-ball cluster on mat", 0.95, "0.05-0.15", "yellow-and-white range balls"),
        ("dropped club-glove pair", 0.55, "0.02-0.05", "soft leather golf glove"),
        ("rubber-tee mat-peg foreground", 0.85, "0.02-0.05", "low rubber peg post"),
        ("range-towel on bag", 0.85, "0.02-0.08", "club-monogrammed cotton wrap"),
        ("dropped tee-peg short row", 0.7, "0.02-0.03", "small wooden tee peg"),
        ("yardage-flag base ground patch", 0.7, "0.05-0.15", "flag-anchor base patch"),
        ("scuffed range-shoe pair", 0.6, "0.02-0.08", "spike-sole golf shoe"),
        ("dropped scorecard-paper sheet", 0.5, "0.02-0.05", "printed lesson-track sheet"),
        ("range-mat ball-mark indent", 0.7, "0.02-0.08", "scuffed mat impact"),
        ("ball-bucket on mat", 0.95, "0.05-0.15", "round wire ball pail"),
    ],
    "arch": [
        ("teaching-bay open-front roof line", 0.85, "0.05-0.2", "long open-eaves roof"),
        ("perimeter mesh-net cable post", 0.95, "0.05-0.15", "tall safety-net standard"),
        ("range-bay divider partition top", 0.85, "0.05-0.15", "bay-side acoustic-divider"),
        ("yardage-marker post cap", 0.85, "0.05-0.15", "small numbered post top"),
        ("ball-dispenser column top", 0.7, "0.05-0.15", "auto-feeder cabinet cap"),
        ("range-edge fence panel cap", 0.7, "0.05-0.1", "low boundary-fence top"),
        ("clubhouse range-side eaves edge", 0.7, "0.05-0.15", "shop-roof eaves line"),
        ("range-mat steel-frame edge", 0.7, "0.05-0.1", "mat-frame boundary"),
        ("range-side speaker post mast", 0.55, "0.02-0.08", "tall PA-speaker pole"),
        ("low range-step concrete riser", 0.55, "0.02-0.08", "concrete tee-up step"),
    ],
    "props": [
        ("driver-club graphite shaft", 1.0, "0.02-0.1", "graphite-shaft driver club"),
        ("iron-club blade pair", 0.85, "0.02-0.1", "forged-iron club"),
        ("yellow-stripe range ball pile", 0.95, "0.02-0.05", "yellow-band range ball"),
        ("club-glove pair", 0.85, "0.02-0.05", "soft leather golf glove"),
        ("rubber range-tee peg", 0.95, "0.02-0.03", "low rubber peg post"),
        ("club-monogrammed range towel", 0.85, "0.02-0.08", "team-club cotton wrap"),
        ("range-bag stand-bag", 0.95, "0.02-0.1", "club-pocket stand bag"),
        ("ball-marker round disc", 0.55, "0.02-0.03", "small flat round marker"),
        ("club-bristle cleaning brush", 0.55, "0.02-0.03", "metal-bristle club brush"),
        ("range-pro alignment stick", 0.7, "0.02-0.08", "fiberglass aiming rod"),
        ("range-mat ball-bucket pail", 0.95, "0.02-0.1", "round wire ball pail"),
        ("ball-mark divot-tool prong", 0.55, "0.02-0.03", "small fork repair tool"),
        ("club-headcover knit hood", 0.55, "0.02-0.05", "knit driver headcover"),
    ],
}
SLUG_KIND["outdoor_sports_golf_driving_range"] = "outdoor"


# ── 12. indoor_sports_indoor_running_track ───────────────────────────────
SETS["indoor_sports_indoor_running_track"] = {
    "bg": [
        ("banked indoor-track straightaway sweep", 1.0, "0.6-1.0", "synthetic banked oval surface"),
        ("upper-tier mezzanine spectator gallery", 0.85, "0.6-1.0", "elevated track-side gallery"),
        ("federation indoor-meet results board", 0.85, "0.6-0.9", "LED meet-result panel"),
        ("track-side warmup-pole jump pit zone", 0.7, "0.4-0.8", "vault-runway target area"),
        ("padded high-jump landing pit zone", 0.7, "0.4-0.8", "thick foam landing pit"),
        ("indoor-meet sponsor-banner sweep", 0.85, "0.6-0.9", "stretched event cloth"),
        ("track-edge officials bench row", 0.7, "0.4-0.8", "row of officials bench"),
        ("aggregate finish-line photo gantry", 0.7, "0.4-0.8", "overhead finish gantry"),
        ("infield throwing-circle area", 0.6, "0.3-0.7", "concrete shot-put circle"),
        ("warmup back-straight athlete area", 0.7, "0.4-0.8", "loose-jog backstretch"),
        ("track-edge equipment cabinet row", 0.7, "0.4-0.8", "row of meet-equipment lockers"),
        ("starter-podium center-infield stand", 0.6, "0.3-0.7", "raised official platform"),
        ("clock-tower center-infield mast", 0.65, "0.4-0.8", "infield meet-clock pylon"),
    ],
    "mg": [
        ("synthetic-rubber track lane row", 1.0, "0.3-0.6", "spongy oval-track surface"),
        ("painted lane stripe boundary", 0.95, "0.2-0.4", "bold lane-divider stripe"),
        ("starter-block sprinter row", 0.95, "0.2-0.4", "stagger-pad starting block"),
        ("finish-line painted band", 0.95, "0.1-0.3", "bold black-and-white stripe"),
        ("infield-circle shot-put station", 0.7, "0.1-0.3", "concrete circle-toe board"),
        ("vault-runway carpet-strip lane", 0.6, "0.2-0.4", "infield runway carpet"),
        ("hurdle-race hurdle row", 0.7, "0.2-0.4", "evenly-spaced track hurdle"),
        ("starter strobe-and-horn pole", 0.7, "0.1-0.3", "audio-trigger starter unit"),
        ("officials' tape-measure crew", 0.55, "0.1-0.3", "throws-event measure team"),
        ("spike-warming athlete bench row", 0.7, "0.1-0.3", "warmup waiting bench"),
        ("padded high-jump landing mat", 0.7, "0.2-0.4", "thick foam landing pit"),
        ("relay-zone exchange marker tape", 0.7, "0.1-0.3", "infield exchange zone"),
    ],
    "fg": [
        ("synthetic-rubber lane surface edge", 1.0, "0.3-0.6", "spongy oval-track surface"),
        ("painted finish-line foreground", 0.95, "0.05-0.15", "bold timing-line stripe"),
        ("dropped lane-baton on track", 0.7, "0.02-0.05", "metal relay baton"),
        ("scuffed sprint-spike pair", 0.85, "0.02-0.08", "lightweight track spike shoe"),
        ("track-towel near bench", 0.7, "0.05-0.15", "athlete cotton wrap"),
        ("dropped wristband on track", 0.55, "0.02-0.05", "elastic terry sweat cuff"),
        ("starter-block ramp foreground", 0.85, "0.05-0.15", "stagger-pad starting block"),
        ("painted-inside-rail edge stripe", 0.85, "0.05-0.15", "white inner-lane stripe"),
        ("infield-circle toe-board edge", 0.7, "0.05-0.15", "concrete circle-toe band"),
        ("dropped chip-timing tag", 0.55, "0.02-0.03", "small ankle-strap chip"),
        ("rolled warmup pants pile", 0.6, "0.05-0.15", "team-issue warmup pant"),
        ("dropped energy-gel pouch", 0.5, "0.02-0.03", "foil-pack supplement"),
    ],
    "arch": [
        ("clear-span banked-track ceiling", 0.85, "0.1-0.2", "vast open-roof span"),
        ("rafter spot truss with meet lights", 0.85, "0.05-0.2", "high-output ceiling spot grid"),
        ("photo-finish gantry overhead arch", 0.85, "0.05-0.15", "finish-line camera arch"),
        ("track-edge inner-rail standard", 0.95, "0.05-0.15", "low inside-rail post"),
        ("ceiling-mounted speaker cluster", 0.7, "0.05-0.15", "PA-cluster ceiling array"),
        ("rafter-hung meet banner", 0.85, "0.05-0.2", "fabric meet-cloth banner"),
        ("painted lane-stripe edge band", 0.95, "0.05-0.15", "lane-edge boundary stripe"),
        ("infield-circle toe-board top", 0.7, "0.05-0.1", "concrete circle-toe band"),
        ("clock-tower center mast top", 0.65, "0.02-0.1", "infield meet-clock pylon top"),
        ("starter-podium base-plinth edge", 0.7, "0.05-0.15", "raised official base"),
    ],
    "props": [
        ("sprint-spike track shoe", 1.0, "0.02-0.08", "lightweight cleated spike shoe"),
        ("relay aluminum baton", 0.85, "0.02-0.05", "tubular metal baton"),
        ("starter pistol-and-shell", 0.55, "0.02-0.03", "small starter device"),
        ("athlete chip-timing ankle-tag", 0.7, "0.02-0.03", "small ankle-strap chip"),
        ("hurdle-race race hurdle", 0.85, "0.02-0.1", "tipping-bar track hurdle"),
        ("track-pole vault pole length", 0.7, "0.02-0.1", "fiberglass long pole"),
        ("shot-put metal sphere", 0.55, "0.02-0.05", "iron-sphere throws ball"),
        ("javelin throw-rod implement", 0.55, "0.02-0.08", "long aluminum javelin"),
        ("athletic-tape roll", 0.7, "0.02-0.05", "white cotton finger-tape spool"),
        ("athlete water-bottle squeeze", 0.85, "0.02-0.05", "branded sport bottle"),
        ("warmup-jacket team-color drape", 0.85, "0.02-0.1", "team-color tracksuit"),
        ("event-bib pinned number", 0.85, "0.02-0.05", "race-bib paper number"),
        ("starting-block adjustment wrench", 0.55, "0.02-0.03", "small adjuster wrench"),
    ],
}
SLUG_KIND["indoor_sports_indoor_running_track"] = "indoor"


# ── 13. outdoor_sports_marathon_street_finish_line ───────────────────────
SETS["outdoor_sports_marathon_street_finish_line"] = {
    "bg": [
        ("packed cheering-crowd street barricade", 1.0, "0.7-1.0", "dense urban race-day crowd"),
        ("city-avenue façade urban canyon", 0.95, "0.6-1.0", "tall building-front urban wall"),
        ("massive marathon finish-line gantry", 0.95, "0.5-0.9", "overhead arch-gantry sign"),
        ("press-photographer truck platform", 0.7, "0.4-0.8", "elevated media-truck platform"),
        ("course-route directional arrow signage", 0.75, "0.4-0.8", "stanchioned route signs"),
        ("event sponsor-banner backdrop sweep", 0.95, "0.5-0.9", "stretched event-cloth strip"),
        ("city skyline distant tower line", 0.7, "0.4-0.8", "far-side urban silhouette"),
        ("crowd-control police barrier line", 0.85, "0.5-0.9", "row of metal stanchion barrier"),
        ("public-park boundary tree row", 0.65, "0.4-0.8", "city-park edge canopy"),
        ("course-marshal high-vis cluster", 0.7, "0.4-0.8", "bright-vest official line"),
        ("race-clock pylon-tower mast", 0.85, "0.4-0.8", "elapsed-time finish pylon"),
        ("flag-of-nations finish-zone row", 0.6, "0.4-0.8", "row of overhead country flags"),
        ("medical-tent canopy in distance", 0.65, "0.4-0.8", "white medical-aid tent row"),
    ],
    "mg": [
        ("painted asphalt finish-zone street", 1.0, "0.4-0.7", "wet-asphalt black road surface"),
        ("painted finish-line broad band", 0.95, "0.2-0.4", "bold race-day stripe"),
        ("course-mile distance marker post", 0.85, "0.2-0.4", "freestanding distance sign"),
        ("race-bib pinned runner stride", 0.85, "0.2-0.4", "mid-stride runner pose"),
        ("course-marshal high-vis pose", 0.7, "0.1-0.3", "bright-vest race official"),
        ("medical-aid station row", 0.7, "0.2-0.4", "small first-aid bench row"),
        ("water-cup table refreshment row", 0.85, "0.2-0.4", "long aid-table cup row"),
        ("photo-finish camera-tripod cluster", 0.6, "0.1-0.3", "long-lens finish-line crouch"),
        ("crowd-cheer barricade line", 0.95, "0.3-0.5", "fan-line metal barrier"),
        ("course-route arrow pavement decal", 0.7, "0.2-0.4", "painted route arrow"),
        ("post-finish foil-blanket walk-zone", 0.7, "0.2-0.4", "post-finish recovery lane"),
        ("course-runner pace-leader cluster", 0.7, "0.2-0.4", "lead-pack runner group"),
    ],
    "fg": [
        ("wet asphalt street surface edge", 1.0, "0.3-0.6", "rain-slick blacktop road"),
        ("painted finish-line foreground", 0.95, "0.1-0.3", "bold race-day stripe"),
        ("dropped paper water-cup", 0.95, "0.02-0.05", "creased paper aid-cup"),
        ("scattered race-bib safety pins", 0.55, "0.02-0.03", "small safety-pin pair"),
        ("dropped chip-timing tag near foot", 0.55, "0.02-0.03", "small ankle-strap chip"),
        ("crumpled energy-gel pouch", 0.85, "0.02-0.05", "foil-pack supplement"),
        ("scuffed running-shoe pair", 0.85, "0.02-0.08", "racing-flat lightweight shoe"),
        ("post-finish foil-blanket scrap", 0.7, "0.02-0.1", "creased silver mylar wrap"),
        ("dropped sweat-soaked tee shirt", 0.6, "0.05-0.15", "race-day tech tee"),
        ("rolled timing-mat edge band", 0.85, "0.05-0.15", "rubberized timing-mat strip"),
        ("dropped wristband on pavement", 0.55, "0.02-0.05", "elastic terry sweat cuff"),
        ("dropped race-program flyer", 0.55, "0.02-0.05", "creased event-program page"),
    ],
    "arch": [
        ("finish-line gantry-arch top spar", 0.95, "0.05-0.2", "race-day overhead arch"),
        ("crowd-barricade barrier-cap top edge", 0.95, "0.05-0.15", "metal stanchion-rail top"),
        ("pavement timing-mat edge band", 0.85, "0.05-0.15", "rubberized timing strip"),
        ("medical-tent canopy ridge edge", 0.7, "0.05-0.15", "white tent-canopy ridge"),
        ("street-lamp post mast top", 0.7, "0.05-0.15", "ornate city street-lamp"),
        ("press-truck platform ladder edge", 0.55, "0.02-0.1", "media-platform side rung"),
        ("course-marshal flag pole base", 0.55, "0.02-0.1", "small handheld flag-pole"),
        ("race-clock pylon-tower mast top", 0.7, "0.02-0.1", "finish-clock column top"),
        ("cobblestone curb-edge stone band", 0.65, "0.05-0.15", "city-curb stone edge"),
        ("traffic-signal head dim disc", 0.55, "0.02-0.08", "darkened street-signal head"),
    ],
    "props": [
        ("racing-flat lightweight shoe", 1.0, "0.02-0.08", "minimal-cushion racing shoe"),
        ("race-bib pinned paper number", 0.95, "0.02-0.05", "race-bib paper number"),
        ("chip-timing ankle tag", 0.85, "0.02-0.03", "small ankle-strap chip"),
        ("post-finish foil-blanket wrap", 0.85, "0.02-0.1", "silver mylar recovery sheet"),
        ("paper aid-station cup", 0.85, "0.02-0.05", "creased paper aid-cup"),
        ("squeeze-pouch energy gel", 0.85, "0.02-0.03", "foil-pack supplement"),
        ("course-marshal handheld flag", 0.7, "0.02-0.05", "small triangle race flag"),
        ("finisher medal pendant", 0.85, "0.02-0.05", "ribboned bronze medal"),
        ("race-cap visor sun cap", 0.85, "0.02-0.05", "lightweight running visor"),
        ("hand-held GPS sport-watch", 0.85, "0.02-0.03", "wrist GPS pace watch"),
        ("hydration-belt small flask", 0.7, "0.02-0.05", "waist-belt small flask"),
        ("compression sleeve calf-wrap", 0.6, "0.02-0.08", "graduated compression sleeve"),
        ("pace-bracelet split-time band", 0.55, "0.02-0.03", "printed pace-band"),
    ],
}
SLUG_KIND["outdoor_sports_marathon_street_finish_line"] = "outdoor"


# ── 14. indoor_sports_velodrome_track_cycling ────────────────────────────
SETS["indoor_sports_velodrome_track_cycling"] = {
    "bg": [
        ("steeply banked velodrome curve sweep", 1.0, "0.6-1.0", "polished hardwood banked curve"),
        ("velodrome center-infield bowl", 0.85, "0.5-0.9", "open infield gathering area"),
        ("upper-tier velodrome spectator gallery", 0.95, "0.6-1.0", "rows of fold-down seat"),
        ("center-hung velodrome scoreboard cube", 0.85, "0.5-0.9", "suspended LED video board"),
        ("infield team-staging pit area", 0.7, "0.4-0.8", "rider-team prep cluster"),
        ("velodrome cantilever roof underside", 0.85, "0.5-1.0", "soaring exposed truss"),
        ("rafter-hung championship-banner row", 0.7, "0.4-0.8", "fabric title cloth"),
        ("track-edge inner-rail boundary line", 0.85, "0.5-0.9", "low-rail track-edge boundary"),
        ("home-team supporter color wedge", 0.7, "0.5-0.9", "team-color seating wedge"),
        ("velodrome-meet sponsor-banner sweep", 0.85, "0.5-0.9", "stretched event cloth"),
        ("photo-finish camera-gantry mast", 0.7, "0.4-0.8", "overhead camera mast"),
        ("opposite-bank crowd silhouette", 0.85, "0.6-1.0", "far-bank spectator block"),
        ("warm-up roller-trainer pit area", 0.6, "0.3-0.7", "stationary warm-up trainer pit"),
    ],
    "mg": [
        ("polished-wood track-cycling boards", 1.0, "0.4-0.7", "lacquered hardwood track"),
        ("painted measurement-line stripe", 0.95, "0.2-0.4", "bold blue and red track stripe"),
        ("track-cycling pursuit-line painted band", 0.95, "0.1-0.3", "painted pursuit reference line"),
        ("cycling-rider mid-banked stance", 0.85, "0.2-0.5", "leaning track-rider pose"),
        ("center-infield team pit-row", 0.85, "0.2-0.4", "rider-team pit gathering"),
        ("track-side commissaire bench", 0.7, "0.1-0.3", "official judge bench"),
        ("starter-gate-holder kneel pose", 0.7, "0.1-0.3", "rider-holder kneeling"),
        ("derny-bike pace-rider mid-track", 0.5, "0.2-0.4", "pace-motorbike rider"),
        ("painted black-line track pursuit edge", 0.85, "0.1-0.3", "low-edge race line"),
        ("pursuit-pursuit kilo-marker band", 0.7, "0.1-0.3", "kilo-marker reference band"),
        ("center-infield warmup-roller bike", 0.7, "0.1-0.3", "stationary warmup bike"),
        ("track-edge inner-rail safety post", 0.85, "0.1-0.3", "low-rail standard"),
    ],
    "fg": [
        ("polished hardwood-board track edge", 1.0, "0.3-0.6", "lacquered hardwood plank"),
        ("painted blue-band sprinter-lane edge", 0.95, "0.1-0.3", "low blue painted lane"),
        ("scuffed cycling-shoe cleat foreground", 0.7, "0.02-0.08", "carbon-sole road cleat"),
        ("dropped track-cycling helmet near bench", 0.7, "0.05-0.15", "aero teardrop helmet"),
        ("dropped tire-pump on infield", 0.55, "0.05-0.15", "track-floor pump"),
        ("painted starter-line band foreground", 0.85, "0.05-0.15", "starter-stripe paint"),
        ("scattered track-tire tube on infield", 0.55, "0.05-0.15", "spare cycling tubular"),
        ("rider-team gear-bag drape foreground", 0.7, "0.05-0.2", "team-emblem zip duffel"),
        ("dropped wristband-sweat strap", 0.55, "0.02-0.05", "elastic terry sweat cuff"),
        ("rolled cycling-shoe pair", 0.7, "0.02-0.1", "carbon-sole road cleat"),
        ("painted race-number rear decal", 0.55, "0.02-0.05", "race-number cardboard"),
        ("scuffed pedal cleat strike-mark", 0.6, "0.02-0.05", "scuff cleat-on-board mark"),
    ],
    "arch": [
        ("velodrome cantilever ceiling truss", 0.95, "0.1-0.2", "soaring exposed roof"),
        ("rafter spot truss with meet lights", 0.85, "0.05-0.2", "high-output ceiling spot grid"),
        ("center-hung video-cube hang rig", 0.85, "0.05-0.15", "suspended four-sided board"),
        ("rafter-hung velodrome-meet banner", 0.85, "0.05-0.2", "fabric meet-cloth banner"),
        ("track-edge inner-rail standard", 0.95, "0.05-0.15", "low-rail track post"),
        ("LED-fascia ribbon edge sweep", 0.85, "0.05-0.2", "rolling sponsor light edge"),
        ("speaker-cluster line-array hang", 0.7, "0.05-0.15", "vertical sound array"),
        ("photo-finish gantry overhead arch", 0.85, "0.05-0.15", "finish-line camera arch"),
        ("painted race-line edge border", 0.95, "0.05-0.15", "lane-edge boundary stripe"),
        ("ceiling-mounted broadcast camera", 0.7, "0.02-0.1", "robotic mounted camera"),
    ],
    "props": [
        ("track-cycling carbon-frame bike", 1.0, "0.05-0.2", "fixed-gear carbon frame"),
        ("aerodynamic teardrop helmet", 0.95, "0.02-0.08", "aero-shell time-trial helmet"),
        ("carbon-sole cycling-shoe cleat pair", 0.95, "0.02-0.08", "stiff carbon-sole road cleat"),
        ("skinsuit body-suit racewear", 0.85, "0.02-0.1", "lycra one-piece skinsuit"),
        ("track-cycling tubular-tire spare", 0.7, "0.02-0.05", "rolled tubular spare"),
        ("rider-team gear-bag duffel", 0.7, "0.02-0.1", "team-emblem zip duffel"),
        ("CO2 tire-inflator canister pair", 0.55, "0.02-0.03", "small inflator canister"),
        ("chain-lubricant small bottle", 0.55, "0.02-0.03", "small lubricant flask"),
        ("rider race-number safety-pinned card", 0.85, "0.02-0.05", "race-number cardboard"),
        ("rider water-bottle squeeze", 0.85, "0.02-0.05", "branded sport bottle"),
        ("starter-gate trigger-button device", 0.55, "0.02-0.03", "small trigger-button device"),
        ("rider-pace lap-counter board", 0.7, "0.02-0.05", "small lap-board"),
        ("eyewear sport-shield glasses", 0.7, "0.02-0.05", "wraparound sport eyewear"),
    ],
}
SLUG_KIND["indoor_sports_velodrome_track_cycling"] = "indoor"


# ── 15. outdoor_sports_equestrian_arena_show_jumping ─────────────────────
SETS["outdoor_sports_equestrian_arena_show_jumping"] = {
    "bg": [
        ("manicured equestrian-arena horizon line", 1.0, "0.6-1.0", "raked sand-and-fiber footing"),
        ("perimeter white-rail show-arena fence", 1.0, "0.5-0.9", "white-painted post-and-rail"),
        ("hospitality-tent VIP gallery row", 0.85, "0.5-0.9", "white VIP tent canopy"),
        ("massive equestrian-meet sponsor-banner", 0.85, "0.5-0.9", "stretched show-day cloth"),
        ("show-jumping standard wing pillar row", 0.85, "0.5-0.9", "ornate fence-jump pillar"),
        ("flag-of-nations event flag row", 0.7, "0.4-0.8", "row of overhead country flags"),
        ("steward-and-judge box pavilion", 0.7, "0.4-0.8", "raised judges' pavilion"),
        ("perimeter spectator picnic blanket lawn", 0.7, "0.4-0.8", "spread blanket-spectator lawn"),
        ("water-feature jump-pond hazard", 0.6, "0.3-0.7", "water-jump shallow pond"),
        ("show-arena warm-up paddock", 0.65, "0.4-0.8", "warm-up adjacent paddock"),
        ("perimeter-rail clubhouse pennant", 0.55, "0.3-0.7", "clubhouse pennant flag"),
        ("photo-finish camera-tower mast", 0.55, "0.3-0.7", "overhead camera-tower"),
        ("scoreboard pylon meet-display", 0.7, "0.4-0.8", "meet-result LED panel"),
    ],
    "mg": [
        ("raked sand-and-fiber arena footing", 1.0, "0.4-0.7", "graded sand-fiber arena footing"),
        ("show-jumping painted-rail fence", 1.0, "0.2-0.5", "painted-pole jump rail"),
        ("brush-fence brush-jump element", 0.85, "0.2-0.4", "brush-pile jump element"),
        ("oxer-double-rail jump element", 0.85, "0.2-0.4", "double-rail oxer jump"),
        ("liverpool-pool water-jump element", 0.6, "0.2-0.4", "shallow blue-painted pool"),
        ("painted course-numbered fence flag", 0.85, "0.1-0.3", "show-jump number plaque"),
        ("rider warm-up canter circle", 0.7, "0.2-0.4", "warm-up loose canter ring"),
        ("course-builder ground-jury team", 0.55, "0.1-0.3", "course-design team"),
        ("show-arena gate-keeper ribbon", 0.55, "0.1-0.3", "ribbon-hold attendant"),
        ("course-decorator floral fence-base", 0.85, "0.1-0.3", "potted flower fence base"),
        ("rider-walk course-walk pre-round", 0.6, "0.1-0.3", "rider-on-foot course walk"),
        ("perimeter ribbon-banner sweep", 0.85, "0.2-0.4", "show-arena perimeter banner"),
    ],
    "fg": [
        ("raked sand arena-footing surface edge", 1.0, "0.3-0.6", "graded sand-fiber arena footing"),
        ("painted standard-pillar jump base", 0.95, "0.05-0.15", "ornate jump-pillar base"),
        ("scattered hoofprint divot patch", 0.85, "0.05-0.2", "shallow hoof impact"),
        ("dropped riding-crop on sand", 0.55, "0.02-0.05", "leather-grip crop"),
        ("dropped show-jumping helmet", 0.55, "0.02-0.08", "velvet-cover riding helmet"),
        ("scuffed riding-boot pair", 0.7, "0.02-0.1", "tall leather riding boot"),
        ("course-flagged fence-numeral plaque", 0.85, "0.05-0.15", "small painted course-numeral"),
        ("painted fence-base potted-flower base", 0.85, "0.05-0.15", "decorative jump-base flower"),
        ("scattered show-arena straw flake", 0.6, "0.05-0.15", "loose-arena straw flake"),
        ("course-walking rider stride foreground", 0.6, "0.1-0.2", "rider-walk striding pose"),
        ("dropped warmup-jacket riding-coat", 0.55, "0.05-0.15", "riding-coat tweed wrap"),
        ("scuffed sand-arena rake mark", 0.6, "0.05-0.15", "fresh-rake fork pattern"),
    ],
    "arch": [
        ("perimeter-rail post-and-rail fence cap", 0.95, "0.05-0.2", "white painted post-and-rail"),
        ("show-jumping standard-pillar top cap", 0.95, "0.05-0.15", "ornate jump-pillar top"),
        ("hospitality-tent canopy ridge edge", 0.7, "0.05-0.15", "white tent-canopy ridge"),
        ("judges'-box stand-platform riser", 0.7, "0.05-0.15", "raised judges' platform"),
        ("course-numbered fence-flag pole", 0.85, "0.02-0.1", "small course-numeral flag"),
        ("perimeter sponsor-banner-frame edge", 0.85, "0.05-0.15", "show-arena banner frame"),
        ("flag-of-nations pole standard top", 0.7, "0.02-0.1", "high-mast country flag"),
        ("scoreboard-pylon column edge", 0.7, "0.05-0.15", "scoreboard column brace"),
        ("water-jump pond-edge curb band", 0.55, "0.05-0.15", "blue-painted pool edge"),
        ("clubhouse-pennant pole top mast", 0.55, "0.02-0.1", "small pennant-pole top"),
    ],
    "props": [
        ("velvet-cover riding helmet", 1.0, "0.02-0.08", "velvet-cover ASTM-rated helmet"),
        ("tall leather riding boot pair", 0.95, "0.02-0.1", "knee-high leather riding boot"),
        ("leather-grip riding crop", 0.85, "0.02-0.05", "leather-grip riding crop"),
        ("dressage white show-coat jacket", 0.7, "0.02-0.1", "tailored show-coat jacket"),
        ("white show-shirt stock tie", 0.7, "0.02-0.05", "starched show-shirt tie"),
        ("leather show-saddle close-contact", 0.85, "0.02-0.1", "close-contact jumping saddle"),
        ("braided-mane horse decorative tie", 0.7, "0.02-0.05", "braided ribbon mane tie"),
        ("brushed dandy-curry brush set", 0.65, "0.02-0.05", "wooden grooming brush"),
        ("snaffle-bit bridle headstall", 0.7, "0.02-0.08", "leather bridle headstall"),
        ("padded saddle-pad cloth", 0.7, "0.02-0.08", "fitted saddle-pad cloth"),
        ("show-jumping spurs pair", 0.55, "0.02-0.05", "small metal riding spur"),
        ("rider gloves leather pair", 0.7, "0.02-0.05", "soft show-glove pair"),
        ("show-coat rosette ribbon prize", 0.55, "0.02-0.05", "ribbon-rosette award"),
    ],
}
SLUG_KIND["outdoor_sports_equestrian_arena_show_jumping"] = "outdoor"


# ── 16. outdoor_sports_ski_race_start_gate ───────────────────────────────
SETS["outdoor_sports_ski_race_start_gate"] = {
    "bg": [
        ("alpine snow-piste race-course slope", 1.0, "0.6-1.0", "groomed icy snow-piste"),
        ("course-line gate-flag panel sweep", 1.0, "0.5-0.9", "fluttering gate-flag panel"),
        ("snow-cornice ridge horizon line", 0.85, "0.5-0.9", "wind-sculpted snow ridge"),
        ("evergreen pine-forest piste edge", 0.85, "0.5-0.9", "snow-laden pine border"),
        ("race-meet sponsor-banner sweep", 0.95, "0.5-0.9", "stretched race-day cloth"),
        ("massive race-clock pylon-tower", 0.85, "0.5-0.9", "elapsed-time start pylon"),
        ("start-house gate-shed shelter", 0.85, "0.5-0.9", "small wooden start-house"),
        ("course-marshal high-vis cluster", 0.7, "0.4-0.8", "bright-vest race official"),
        ("rope-and-fence safety-net perimeter", 0.85, "0.5-0.9", "B-net racer safety fence"),
        ("flag-of-nations meet flag row", 0.7, "0.4-0.8", "row of overhead country flags"),
        ("piste-grooming track sweep", 0.65, "0.4-0.8", "fresh-grooming snow track"),
        ("blizzard-haze far peak silhouette", 0.7, "0.4-0.8", "atmospheric distant peak"),
        ("alpine course-cliff drop edge", 0.6, "0.3-0.7", "steep terrain drop edge"),
    ],
    "mg": [
        ("painted race-start gate-line band", 1.0, "0.2-0.4", "bold start-stripe paint"),
        ("racer ski-tip start-stance pair", 1.0, "0.1-0.3", "racing-ski tip pair"),
        ("painted course-gate alternating panel", 0.95, "0.2-0.4", "blue-and-red gate-panel"),
        ("course-marshal flag-judge stance", 0.7, "0.1-0.3", "bright-vest official"),
        ("start-wand race-trigger arm", 0.85, "0.1-0.3", "lever start-trigger wand"),
        ("racer-coach last-call cluster", 0.7, "0.1-0.3", "coach-rider final-prep huddle"),
        ("starter-marshal countdown-stance", 0.85, "0.1-0.3", "race-start official stance"),
        ("piste-edge B-net safety fence", 0.85, "0.2-0.4", "racer safety B-net fence"),
        ("padded course-bumper protective pad", 0.7, "0.1-0.3", "padded gate-pad bumper"),
        ("race-helmet coach-helmet cluster", 0.6, "0.1-0.3", "ski-coach helmet cluster"),
        ("course-recon course-side judge stand", 0.55, "0.1-0.3", "small judge stand"),
        ("piste-grooming snowcat parked", 0.55, "0.1-0.3", "snow-grooming machine parked"),
    ],
    "fg": [
        ("groomed snow-piste surface edge", 1.0, "0.3-0.6", "groomed icy snow-piste"),
        ("painted start-gate band foreground", 0.95, "0.05-0.15", "bold start-stripe paint"),
        ("dropped ski-pole on snow", 0.7, "0.02-0.1", "carbon-shaft ski-pole"),
        ("scuffed ski-edge ice-shave foreground", 0.85, "0.05-0.15", "scraped ice-edge ridge"),
        ("dropped race-helmet near gate", 0.7, "0.02-0.1", "chin-bar racing helmet"),
        ("scuffed boot-clip ski-binding pair", 0.85, "0.02-0.08", "race-binding cleat"),
        ("dropped start-bib pinned number", 0.85, "0.02-0.05", "race-bib paper number"),
        ("racer-coach goggle pair on snow", 0.7, "0.02-0.05", "anti-fog ski goggle"),
        ("painted course-gate flag base", 0.85, "0.05-0.15", "embedded gate-flag base"),
        ("scuffed glove-grip pole-handle", 0.7, "0.02-0.05", "padded ski-pole grip"),
        ("dropped chip-timing band", 0.55, "0.02-0.03", "small wrist timing band"),
        ("scuffed wax-stick scrap on snow", 0.55, "0.02-0.03", "small wax-stick scrap"),
    ],
    "arch": [
        ("start-house gate-shed roof line", 0.85, "0.05-0.2", "small start-house roof"),
        ("painted start-gate threshold edge", 0.95, "0.05-0.15", "embedded start-gate band"),
        ("course-gate flag-pole standard", 0.95, "0.05-0.15", "small fluttering gate-pole"),
        ("safety-net B-net cable post", 0.85, "0.05-0.15", "racer B-net post"),
        ("race-clock pylon-tower mast top", 0.85, "0.02-0.1", "start-clock column top"),
        ("padded gate-bumper-pad top edge", 0.7, "0.05-0.15", "padded gate-pad bumper top"),
        ("course-marshal flag pole base", 0.55, "0.02-0.08", "small handheld flag-pole"),
        ("start-wand-arm trigger top", 0.7, "0.02-0.08", "lever-trigger top"),
        ("speaker-post mast top fixture", 0.55, "0.02-0.08", "tall PA-speaker pole"),
        ("flag-of-nations pole standard top", 0.7, "0.02-0.1", "high-mast country flag"),
    ],
    "props": [
        ("racing-ski cleat pair", 1.0, "0.02-0.1", "stiff racing-ski pair"),
        ("carbon-shaft ski-pole pair", 0.95, "0.02-0.08", "carbon-shaft ski-pole"),
        ("chin-bar racing helmet", 0.95, "0.02-0.08", "chin-bar racing helmet"),
        ("anti-fog ski-goggle pair", 0.95, "0.02-0.05", "anti-fog ski goggle"),
        ("race-bib pinned number", 0.95, "0.02-0.05", "race-bib paper number"),
        ("race-skinsuit one-piece racewear", 0.85, "0.02-0.1", "lycra-one-piece skinsuit"),
        ("race-glove padded pair", 0.85, "0.02-0.05", "padded grip ski glove"),
        ("ski-wax stick block", 0.7, "0.02-0.03", "small wax-stick block"),
        ("ski-binding adjustment screwdriver", 0.55, "0.02-0.03", "small precision screwdriver"),
        ("hand-warmer chemical pack", 0.55, "0.02-0.03", "small chemical hand-warmer"),
        ("chin-strap wrist timing band", 0.55, "0.02-0.03", "small wrist timing band"),
        ("warmup-suit insulated parka", 0.7, "0.02-0.1", "team-color insulated parka"),
        ("ski-coach lap-board chart", 0.55, "0.02-0.05", "small lap-chart board"),
    ],
}
SLUG_KIND["outdoor_sports_ski_race_start_gate"] = "outdoor"


# ── 17. outdoor_sports_outdoor_calisthenics_park ─────────────────────────
SETS["outdoor_sports_outdoor_calisthenics_park"] = {
    "bg": [
        ("public-park calisthenics rig sweep", 1.0, "0.6-1.0", "steel pull-up bar rig"),
        ("urban-park concrete-paved zone", 0.95, "0.5-0.9", "broad concrete training zone"),
        ("rubber-tile shock-absorption ground", 0.85, "0.5-0.9", "interlocking rubber tile field"),
        ("park-perimeter low hedge boundary", 0.7, "0.4-0.8", "trimmed boxwood line"),
        ("city-skyline distant tower line", 0.7, "0.4-0.8", "far urban tower silhouette"),
        ("park-side asphalt-pathway sweep", 0.85, "0.4-0.8", "asphalt path strip"),
        ("graffiti-stenciled park wall mural", 0.65, "0.4-0.8", "stencil-sprayed park mural"),
        ("park-bench seating row", 0.7, "0.4-0.8", "row of park bench seats"),
        ("public-park lamppost fixture row", 0.7, "0.4-0.8", "ornate park-lamppost row"),
        ("tree-canopy park-perimeter shade", 0.85, "0.5-0.9", "spreading park-canopy"),
        ("playground edge-zone equipment", 0.6, "0.3-0.7", "park-side play structure"),
        ("park-fountain center-feature plaza", 0.55, "0.3-0.7", "small park-fountain plaza"),
        ("park-side bicycle rack stand", 0.65, "0.4-0.8", "row of bike-rack stands"),
    ],
    "mg": [
        ("steel pull-up bar parallel-bar rig", 1.0, "0.3-0.6", "powder-coated steel rig"),
        ("dip-station parallel-bar zone", 0.85, "0.2-0.4", "parallel dip-bar pair"),
        ("monkey-bar suspension rig", 0.85, "0.2-0.4", "horizontal monkey-bar grid"),
        ("ring-suspension calisthenics ring rig", 0.7, "0.2-0.4", "gymnastic-ring suspension"),
        ("low-bar push-up parallette pair", 0.7, "0.1-0.3", "low parallette pair"),
        ("plyometric-jump box stack", 0.7, "0.1-0.3", "stacked plyo-box pair"),
        ("park-bench warm-up athlete pose", 0.7, "0.1-0.3", "stretching athlete on bench"),
        ("rubber-tile floor-zone training area", 0.85, "0.2-0.4", "interlocking rubber tile zone"),
        ("park-side athlete bag drop", 0.7, "0.1-0.3", "athlete gear-bag pile"),
        ("steel-rig vertical climbing-rope", 0.55, "0.1-0.3", "knotted-rope climb line"),
        ("tractor-tire flip station", 0.55, "0.1-0.3", "large training tire"),
        ("battle-rope ground-anchor station", 0.7, "0.1-0.3", "thick battle-rope length"),
    ],
    "fg": [
        ("interlocking rubber-tile ground edge", 1.0, "0.3-0.6", "shock-absorption rubber tile"),
        ("painted training-zone boundary stripe", 0.7, "0.05-0.15", "park-stripe boundary"),
        ("scuffed training-shoe pair", 0.85, "0.02-0.08", "lightweight cross-trainer"),
        ("dropped grip-chalk block", 0.7, "0.02-0.05", "compressed magnesium cube"),
        ("rolled foam-roller cylinder", 0.6, "0.05-0.15", "high-density cylindrical roller"),
        ("athlete water-jug on tile", 0.85, "0.05-0.15", "translucent training jug"),
        ("dropped wrist-wrap pair", 0.55, "0.02-0.05", "elastic wrist-support strap"),
        ("training-resistance band loop pile", 0.7, "0.05-0.15", "looped resistance band"),
        ("dropped chalk-bag pouch", 0.65, "0.02-0.05", "small grip-chalk pouch"),
        ("scattered hand-grip glove pair", 0.55, "0.02-0.05", "fingerless training glove"),
        ("park-bench gear-pile shadow", 0.6, "0.05-0.15", "gear-bag shadow pile"),
        ("dropped jumprope coil", 0.65, "0.02-0.08", "coiled wire skip-rope"),
    ],
    "arch": [
        ("steel-rig top horizontal bar", 0.95, "0.05-0.2", "powder-coated horizontal bar"),
        ("dip-station parallel bar end-cap", 0.85, "0.05-0.15", "parallel dip-bar end"),
        ("monkey-bar grid ladder edge", 0.85, "0.05-0.15", "horizontal monkey-bar end"),
        ("rubber-tile floor-zone edge band", 0.95, "0.05-0.15", "rubber-tile boundary band"),
        ("park-lamppost post-mast top", 0.7, "0.05-0.15", "ornate park-lamppost top"),
        ("park-side asphalt-path edge curb", 0.7, "0.05-0.15", "concrete curb-edge"),
        ("steel-rig vertical post column", 0.85, "0.05-0.15", "rig-leg column"),
        ("ring-suspension cable anchor top", 0.7, "0.02-0.1", "gymnastic-ring anchor"),
        ("hedge-perimeter trim-edge band", 0.7, "0.05-0.15", "trimmed boxwood edge"),
        ("park-bench backrest top edge", 0.7, "0.05-0.1", "wooden bench backrest"),
    ],
    "props": [
        ("training-shoe lightweight cross-trainer", 0.95, "0.02-0.08", "lightweight cross-trainer"),
        ("magnesium-chalk grip block", 0.85, "0.02-0.05", "compressed magnesium cube"),
        ("training-resistance loop band", 0.95, "0.02-0.08", "looped stretch resistance band"),
        ("athlete water-jug bottle", 0.95, "0.02-0.08", "translucent training jug"),
        ("training-glove fingerless pair", 0.85, "0.02-0.05", "fingerless training glove"),
        ("ankle-weight strap pair", 0.7, "0.02-0.05", "padded ankle-weight strap"),
        ("athlete gear-bag duffel", 0.85, "0.02-0.1", "athletic-duffel bag"),
        ("calisthenics-ring suspension ring", 0.55, "0.02-0.05", "wooden gymnastic ring"),
        ("park-jump rope coil", 0.85, "0.02-0.05", "coiled wire skip-rope"),
        ("training-towel cotton wrap", 0.85, "0.02-0.08", "athletic-cotton wrap towel"),
        ("phone-arm-strap fitness tracker", 0.55, "0.02-0.03", "armband phone-strap"),
        ("training-tracker wrist watch", 0.7, "0.02-0.03", "wrist GPS pace watch"),
        ("knee-sleeve neoprene pair", 0.65, "0.02-0.05", "thick stretch knee support"),
    ],
}
SLUG_KIND["outdoor_sports_outdoor_calisthenics_park"] = "outdoor"


# ── 18. outdoor_sports_rowing_crew_boathouse_dock ────────────────────────
SETS["outdoor_sports_rowing_crew_boathouse_dock"] = {
    "bg": [
        ("flat-water rowing-river horizon line", 1.0, "0.6-1.0", "calm reflective river surface"),
        ("riverbank tree-line opposite shore", 0.95, "0.5-0.9", "wooded far-bank line"),
        ("clapboard-clad boathouse façade", 0.95, "0.5-0.9", "painted-board boathouse wall"),
        ("crew-club pennant flagpole row", 0.7, "0.4-0.8", "club-pennant flag row"),
        ("riverbank stone-revetment edge", 0.7, "0.4-0.8", "stacked-stone revetment"),
        ("rowing-buoy lane-marker chain", 0.85, "0.5-0.9", "row of small course-buoy"),
        ("regatta race-tower judging mast", 0.7, "0.4-0.8", "raised race-judging tower"),
        ("rowing-crew sponsor-banner sweep", 0.85, "0.5-0.9", "stretched regatta cloth"),
        ("riverbank willow-overhang fringe", 0.65, "0.4-0.8", "drooping willow shoots"),
        ("boathouse-rack hull-storage row", 0.85, "0.5-0.9", "stacked hull-storage rack"),
        ("rowing-crew finish-line gantry", 0.7, "0.4-0.8", "finish-line race gantry"),
        ("riverbank reed-and-rush bed", 0.6, "0.3-0.7", "shallow-water reed bed"),
        ("club-launch-coach motor-tender", 0.55, "0.3-0.7", "small coach-launch boat"),
    ],
    "mg": [
        ("painted timber-decked launching dock", 1.0, "0.3-0.6", "weathered painted plank dock"),
        ("rowing-shell hull-fleet stack row", 0.85, "0.3-0.6", "stacked carbon-shell hulls"),
        ("rowing-crew oar-rack row", 0.95, "0.2-0.4", "vertical oar-rack stand"),
        ("rowing-crew stretching warmup zone", 0.7, "0.2-0.4", "athlete stretching cluster"),
        ("rowing-shell mid-water glide line", 0.85, "0.2-0.4", "mid-water shell glide"),
        ("coxswain-coach launch-boat nearshore", 0.6, "0.2-0.4", "small motor coach-launch"),
        ("riverbank sandbag dock-anchor row", 0.7, "0.1-0.3", "sandbag dock-anchor"),
        ("rowing-crew athlete-team huddle", 0.7, "0.2-0.4", "team pre-race huddle"),
        ("dock-side ergometer warmup row", 0.6, "0.2-0.4", "indoor-row ergometer line"),
        ("dock-side flag-pole standard row", 0.7, "0.1-0.3", "club-pole flag standard"),
        ("dock-edge crew-bag drop pile", 0.7, "0.2-0.4", "rower gear-bag pile"),
        ("riverbank lane-buoy course-line", 0.85, "0.2-0.4", "row of course-marker buoy"),
    ],
    "fg": [
        ("painted-timber dock-plank surface edge", 1.0, "0.3-0.6", "weathered painted plank"),
        ("painted dock-edge boundary stripe", 0.7, "0.05-0.15", "painted dock boundary"),
        ("dropped oar-blade scoop edge", 0.85, "0.05-0.2", "carbon-blade oar scoop"),
        ("scuffed unisuit body-suit drape", 0.7, "0.05-0.15", "lycra unisuit drape"),
        ("dropped boat-shoe pair", 0.55, "0.02-0.08", "rubber-grip dock shoe"),
        ("coiled dock-line mooring rope", 0.7, "0.05-0.15", "coiled mooring line"),
        ("dropped megaphone-bullhorn", 0.5, "0.02-0.05", "small handheld bullhorn"),
        ("athlete water-bottle on dock", 0.85, "0.02-0.05", "branded sport bottle"),
        ("rolled crew-blanket pile", 0.55, "0.05-0.15", "team-color cotton wrap"),
        ("dropped seat-roller hull seat", 0.55, "0.02-0.05", "rolling-shell seat"),
        ("painted club-emblem dock decal", 0.55, "0.02-0.08", "painted club-emblem decal"),
        ("dock-edge condensation puddle patch", 0.55, "0.02-0.1", "dock-water splash patch"),
    ],
    "arch": [
        ("clapboard boathouse-eaves ridge edge", 0.95, "0.05-0.2", "painted-board eaves ridge"),
        ("oar-rack vertical post-cap", 0.95, "0.05-0.15", "oar-rack post top"),
        ("dock-railing post-mast top", 0.85, "0.05-0.15", "wooden dock-rail post top"),
        ("rowing-shell hull-rack tier edge", 0.95, "0.05-0.15", "stacked hull-rack tier"),
        ("club-pennant flagpole standard top", 0.7, "0.02-0.1", "small pennant-pole top"),
        ("boathouse-door slide-rail top", 0.7, "0.05-0.15", "sliding boathouse-door rail"),
        ("dock-anchor sandbag-row top edge", 0.55, "0.05-0.15", "sandbag-row top edge"),
        ("regatta-finish gantry-arch top spar", 0.7, "0.05-0.15", "finish-arch top"),
        ("dock-edge metal cleat-bollard", 0.7, "0.02-0.08", "small mooring-cleat bollard"),
        ("dock-side speaker-post mast top", 0.55, "0.02-0.08", "PA-speaker pole top"),
    ],
    "props": [
        ("rowing-shell single-scull hull", 0.85, "0.05-0.2", "carbon-shell single hull"),
        ("rowing-oar carbon-shaft pair", 0.95, "0.05-0.2", "carbon-shaft sweep oar"),
        ("rowing unisuit body-suit racewear", 0.95, "0.02-0.1", "lycra one-piece unisuit"),
        ("rowing-crew bow-ball foam tip", 0.55, "0.02-0.03", "small foam bow-ball"),
        ("coxswain-cox-box stroke-rate device", 0.7, "0.02-0.05", "small audio stroke-rate device"),
        ("rowing-crew blade-grip wrap tape", 0.55, "0.02-0.03", "rubberized grip-wrap"),
        ("rowing-crew water-bottle squeeze", 0.85, "0.02-0.05", "branded sport bottle"),
        ("crew-team logo-monogrammed bag", 0.85, "0.02-0.1", "team-emblem zip duffel"),
        ("rowing-crew athletic-tape roll", 0.7, "0.02-0.05", "white cotton finger-tape spool"),
        ("rowing-crew wrist sport-watch", 0.7, "0.02-0.03", "wrist GPS pace watch"),
        ("rowing-crew bow-number card", 0.55, "0.02-0.03", "small bow-number card"),
        ("rowing-blade tip-protector cap", 0.55, "0.02-0.03", "blade-tip plastic cap"),
        ("crew-trifold race-program flyer", 0.5, "0.02-0.05", "creased event-program page"),
    ],
}
SLUG_KIND["outdoor_sports_rowing_crew_boathouse_dock"] = "outdoor"


# ── 19. outdoor_sports_rugby_pitch_grass ─────────────────────────────────
SETS["outdoor_sports_rugby_pitch_grass"] = {
    "bg": [
        ("packed multi-tier rugby-stand bowl", 1.0, "0.7-1.0", "dense crowd-color block"),
        ("home-end ultras color-block", 0.85, "0.6-1.0", "team-supporter color block"),
        ("rugby-end H-shaped goal-post pair", 1.0, "0.5-0.9", "tall H-shaped rugby uprights"),
        ("perimeter rugby-pitch advert hoarding", 0.95, "0.5-0.9", "sponsor hoarding strip"),
        ("opposite-stand crowd silhouette", 0.85, "0.6-1.0", "far-side spectator wall"),
        ("rugby-stadium roof-overhang cantilever", 0.7, "0.4-0.8", "broad covering sweep"),
        ("press-box mid-tier glass front", 0.7, "0.5-0.9", "long media-window strip"),
        ("home-team championship-banner row", 0.7, "0.4-0.8", "rafter title cloth"),
        ("rugby-stand club-emblem mural", 0.65, "0.4-0.8", "painted team-emblem wall"),
        ("rugby-stand giant scoreboard pylon", 0.85, "0.5-0.9", "matchday LED video panel"),
        ("rugby-stand hospitality VIP row", 0.6, "0.4-0.8", "elevated officials gallery"),
        ("rugby-stand ribbon-board sweep", 0.85, "0.5-0.9", "scrolling rink-edge graphic"),
        ("rugby-stand fan-zone tunnel mouth", 0.65, "0.4-0.8", "darkened fan-tunnel portal"),
    ],
    "mg": [
        ("regulation rugby-pitch grass surface", 1.0, "0.4-0.7", "striped mowed turf field"),
        ("painted rugby-pitch try-line stripe", 0.95, "0.2-0.4", "bold try-line paint"),
        ("painted rugby-pitch 22m line", 0.95, "0.1-0.3", "white twenty-two stripe"),
        ("painted rugby-pitch halfway line", 0.85, "0.2-0.4", "bold midfield stripe"),
        ("rugby-end H-shaped goalposts pair", 1.0, "0.2-0.4", "tall H-shaped post pair"),
        ("rugby-team scrum-down pack stance", 0.85, "0.2-0.4", "interlocked scrum cluster"),
        ("painted rugby-pitch dead-ball-line", 0.85, "0.2-0.4", "out-of-play boundary stripe"),
        ("rugby-team substitute-bench shelter", 0.85, "0.2-0.4", "covered courtside shelter"),
        ("rugby-pitch corner-flag standing pole", 0.85, "0.1-0.3", "fluttering small triangle flag"),
        ("rugby-pitch lineout-call jump cluster", 0.7, "0.2-0.4", "lineout-jumper lift cluster"),
        ("rugby-pitch ball-boy crouched pose", 0.7, "0.1-0.3", "ready-throw retriever"),
        ("rugby-pitch warmup tackle-bag pair", 0.65, "0.1-0.3", "padded tackle-bag pair"),
    ],
    "fg": [
        ("striped mowed-grass rugby-pitch edge", 1.0, "0.3-0.6", "fresh-cut alternating turf strips"),
        ("painted rugby-pitch sideline stripe", 0.95, "0.1-0.3", "high-contrast field edge"),
        ("loose rugby-ball egg-shape ground", 0.9, "0.05-0.15", "leather oval rugby ball"),
        ("scuffed rugby-cleat divot foreground", 0.85, "0.05-0.2", "torn-turf cleat mark"),
        ("dropped rugby-mouthguard gum-shield", 0.55, "0.02-0.05", "small mouthguard"),
        ("dropped rugby-scrum cap helmet", 0.55, "0.02-0.08", "padded scrum-cap"),
        ("rugby-pitch ball-boy crouched foreground", 0.7, "0.1-0.2", "ready-throw retriever"),
        ("rugby-pitch player water-bottle ground", 0.85, "0.02-0.1", "squeeze-top sport bottle"),
        ("dropped rugby-team headband pair", 0.55, "0.02-0.05", "elastic terry sweat cuff"),
        ("rolled rugby-bench warmup jacket", 0.7, "0.05-0.15", "team-color tracksuit pile"),
        ("painted rugby-pitch try-decal foreground", 0.7, "0.05-0.15", "try-zone painted decal"),
        ("scuffed rugby-cleat sole print", 0.6, "0.05-0.15", "rubber-stud cleat track"),
    ],
    "arch": [
        ("rugby-stadium cantilever roof-undercut", 0.85, "0.1-0.2", "vast covering sweep"),
        ("rugby-stand floodlight pylon-mast top", 0.95, "0.05-0.2", "high-mast lamp cluster"),
        ("rugby-pitch H-post upright top spar", 0.95, "0.05-0.2", "tall H-post upright top"),
        ("perimeter rugby-hoarding panel base", 0.95, "0.05-0.2", "rolling sponsor advert face"),
        ("rugby-end goal-post crossbar edge", 0.95, "0.05-0.15", "H-post crossbar edge"),
        ("rugby-bench shelter polycarbonate roof", 0.85, "0.05-0.15", "transparent curved cover"),
        ("rugby-press-box glass-front fascia", 0.7, "0.05-0.15", "long media-window strip"),
        ("rugby-stand video-screen pylon strut", 0.7, "0.05-0.15", "scoreboard column brace"),
        ("rugby-corner-flag pole base", 0.85, "0.02-0.1", "fluttering small triangle flag"),
        ("rugby-stand speaker-cluster pylon", 0.7, "0.02-0.1", "stadium PA cluster"),
    ],
    "props": [
        ("regulation oval rugby-ball", 1.0, "0.02-0.08", "leather oval rugby ball"),
        ("rugby-scrum scrum-cap padded helmet", 0.7, "0.02-0.08", "padded scrum-cap"),
        ("rugby-team mouthguard gum-shield", 0.7, "0.02-0.03", "small mouthguard"),
        ("rugby-cleat soft-stud boot pair", 0.95, "0.02-0.1", "soft-stud rugby boot"),
        ("rugby-team-jersey strip", 0.85, "0.02-0.1", "team-color rugby jersey"),
        ("rugby-tape strapping shoulder roll", 0.7, "0.02-0.05", "rubberized shoulder strap"),
        ("rugby-team water-bottle squeeze", 0.85, "0.02-0.05", "branded sport bottle"),
        ("rugby-bench tackle-bag pad", 0.65, "0.02-0.1", "padded tackle-bag pad"),
        ("rugby-team-monogrammed match towel", 0.7, "0.02-0.08", "team-emblem cotton wrap"),
        ("rugby-referee whistle and cards", 0.7, "0.02-0.03", "lanyard with cards"),
        ("rugby-coach clipboard tactical sheet", 0.55, "0.02-0.05", "coach tactical sheet"),
        ("rugby-team kicking-tee plastic peg", 0.65, "0.02-0.03", "kicking-tee plastic peg"),
        ("rugby-team captain-armband elastic", 0.55, "0.02-0.05", "colored captain band"),
    ],
}
SLUG_KIND["outdoor_sports_rugby_pitch_grass"] = "outdoor"


# ── 20. indoor_sports_indoor_climbing_competition_wall ───────────────────
SETS["indoor_sports_indoor_climbing_competition_wall"] = {
    "bg": [
        ("massive indoor-climbing competition wall", 1.0, "0.7-1.0", "textured-resin climbing panel"),
        ("indoor-climb upper-tier spectator gallery", 0.95, "0.6-1.0", "rows of fold-down seat"),
        ("indoor-climb sponsor-banner backdrop sweep", 0.95, "0.6-1.0", "stretched event cloth"),
        ("indoor-climb route-setter scaffold platform", 0.7, "0.5-0.9", "raised route-setter scaffold"),
        ("indoor-climb route-numbered tag-row", 0.85, "0.5-0.9", "row of route-tag plaques"),
        ("indoor-climb tournament scoreboard panel", 0.85, "0.5-0.9", "competition LED panel"),
        ("indoor-climb route-painted color stripe", 0.95, "0.6-1.0", "painted-route color stripe"),
        ("indoor-climb upper-canopy ceiling angle", 0.7, "0.4-0.8", "overhang-route ceiling"),
        ("indoor-climb home-team color wedge", 0.7, "0.5-0.9", "team-color seating wedge"),
        ("indoor-climb roped-route belay-line area", 0.85, "0.5-0.9", "lead-rope belay area"),
        ("indoor-climb bouldering-zone padded area", 0.7, "0.4-0.8", "padded bouldering area"),
        ("indoor-climb opposite-bank crowd silhouette", 0.85, "0.6-1.0", "far-bank spectator block"),
        ("indoor-climb broadcast camera-pit", 0.6, "0.4-0.8", "front-row long-lens cluster"),
    ],
    "mg": [
        ("indoor-climb competition lead-route panel", 1.0, "0.4-0.7", "textured-resin route panel"),
        ("indoor-climb competition hold-route line", 1.0, "0.2-0.5", "colored route-hold line"),
        ("indoor-climb auto-belay descent-line column", 0.85, "0.2-0.4", "auto-belay descent column"),
        ("indoor-climb belayer ground-stance pair", 0.85, "0.1-0.3", "ground-belayer pose"),
        ("indoor-climb route-setter mid-wall pose", 0.55, "0.2-0.4", "route-setter mid-wall pose"),
        ("indoor-climb bouldering-pad stack zone", 0.85, "0.2-0.4", "padded bouldering pad stack"),
        ("indoor-climb judge-and-isolation booth", 0.7, "0.2-0.4", "small judge booth"),
        ("indoor-climb chalk-bag start-line zone", 0.7, "0.1-0.3", "athlete chalk-up zone"),
        ("indoor-climb timer-clock countdown panel", 0.85, "0.1-0.3", "competition countdown clock"),
        ("indoor-climb athlete pre-climb bench row", 0.7, "0.1-0.3", "athlete waiting bench row"),
        ("indoor-climb coach-zone stand area", 0.7, "0.1-0.3", "coach-side stand"),
        ("indoor-climb route-flag tag pole", 0.7, "0.1-0.3", "route-tag flag pole"),
    ],
    "fg": [
        ("indoor-climb padded bouldering-pad surface edge", 1.0, "0.3-0.6", "thick bouldering-pad surface"),
        ("painted indoor-climb start-line tag", 0.85, "0.05-0.15", "painted start-line tag"),
        ("dropped indoor-climb chalk-bag pouch", 0.85, "0.02-0.05", "small grip-chalk pouch"),
        ("scuffed indoor-climb shoe pair", 0.85, "0.02-0.08", "rubber-sole climbing shoe"),
        ("dropped indoor-climb harness-clip carabiner", 0.55, "0.02-0.05", "auto-lock carabiner"),
        ("rolled indoor-climb belay-rope coil", 0.85, "0.05-0.15", "dynamic-rope coil"),
        ("dropped indoor-climb athletic-tape roll", 0.7, "0.02-0.05", "white cotton finger-tape spool"),
        ("dropped indoor-climb finger-brush", 0.55, "0.02-0.03", "small hold-cleaning brush"),
        ("athlete indoor-climb water-bottle ground", 0.85, "0.02-0.05", "branded sport bottle"),
        ("indoor-climb chalk-cloud dust patch", 0.85, "0.05-0.15", "hand-applied chalk dust"),
        ("dropped indoor-climb athlete-bib number", 0.7, "0.02-0.05", "race-bib paper number"),
        ("indoor-climb spectator program flyer", 0.5, "0.02-0.05", "creased event-program page"),
    ],
    "arch": [
        ("indoor-climb wall-panel top horizon edge", 0.95, "0.05-0.2", "textured-resin wall top"),
        ("indoor-climb wall-route overhang-roof shelf", 0.85, "0.05-0.2", "overhang-route ceiling"),
        ("indoor-climb wall-route bolt-anchor row", 0.95, "0.05-0.15", "route-bolt anchor line"),
        ("indoor-climb scaffold-platform top edge", 0.7, "0.05-0.15", "route-setter scaffold edge"),
        ("indoor-climb LED-fascia ribbon edge", 0.85, "0.05-0.2", "rolling sponsor light edge"),
        ("indoor-climb auto-belay device cap", 0.85, "0.02-0.1", "auto-belay device housing"),
        ("indoor-climb judge-booth canopy edge", 0.7, "0.05-0.15", "judge-booth canopy edge"),
        ("indoor-climb timer-clock panel base", 0.7, "0.05-0.15", "countdown panel base"),
        ("indoor-climb route-tag flag-post top", 0.7, "0.02-0.08", "route-tag flag pole top"),
        ("indoor-climb venue speaker-cluster hang", 0.7, "0.05-0.15", "vertical sound array"),
    ],
    "props": [
        ("indoor-climb rubber-sole climbing shoe", 1.0, "0.02-0.08", "rubber-sole climbing shoe"),
        ("indoor-climb chalk-bag grip pouch", 1.0, "0.02-0.05", "small grip-chalk pouch"),
        ("indoor-climb harness-belt safety harness", 0.95, "0.02-0.1", "padded climb-harness"),
        ("indoor-climb dynamic belay-rope length", 0.85, "0.05-0.2", "dynamic-rope length"),
        ("indoor-climb auto-lock carabiner", 0.7, "0.02-0.05", "auto-lock carabiner"),
        ("indoor-climb belay-tube device", 0.7, "0.02-0.05", "belay-tube device"),
        ("indoor-climb finger-tape athletic roll", 0.85, "0.02-0.05", "white cotton finger-tape spool"),
        ("indoor-climb hold-cleaning brush", 0.55, "0.02-0.03", "small hold-cleaning brush"),
        ("indoor-climb athlete-bib race number", 0.85, "0.02-0.05", "race-bib paper number"),
        ("indoor-climb athlete water-bottle squeeze", 0.85, "0.02-0.05", "branded sport bottle"),
        ("indoor-climb tournament-issued cap", 0.55, "0.02-0.05", "tournament-issued cap"),
        ("indoor-climb finger-strengthener tool", 0.55, "0.02-0.03", "finger-strength trainer"),
        ("indoor-climb pre-climb energy-gel pouch", 0.55, "0.02-0.03", "foil-pack supplement"),
    ],
}
SLUG_KIND["indoor_sports_indoor_climbing_competition_wall"] = "indoor"


# ─────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────

FILE_KEYS = {
    "bg": "background",
    "mg": "midground",
    "fg": "foreground_element",
    "arch": "architecture_detail",
    "props": "props",
}

INDOOR_BAN = ["trail", "ridge", "mountain", "forest canopy", "ocean", "beach",
              "appalachian", "summit", "wilderness", "switchback"]
OUTDOOR_BAN = ["hardwood floor", "sofa", "office desk", "kitchen counter",
               "duvet", "shower stall", "bath tub", "indoor pool"]


def write_file(path: str, element_id: str, entries: list[tuple]) -> None:
    lines = [
        f"# element: {element_id}",
        "# format: name | probability | coverage_range | texture",
        "",
    ]
    for entry in entries:
        if len(entry) == 2:
            name, prob = entry
            lines.append(f"{name} | {prob} | - | -")
        else:
            name, prob, cov, tex = entry
            lines.append(f"{name} | {prob} | {cov} | {tex}")
    lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> None:
    written = 0
    errors = []
    for slug, content in SETS.items():
        slug_dir = os.path.join(ROOT, slug)
        os.makedirs(slug_dir, exist_ok=True)
        kind = SLUG_KIND.get(slug, "")
        ban = INDOOR_BAN if kind == "indoor" else OUTDOOR_BAN if kind == "outdoor" else []

        for short_key, file_id in FILE_KEYS.items():
            entries = content[short_key]
            min_n = 10 if short_key == "arch" else 12
            max_n = 15 if short_key == "arch" else 18
            if len(entries) < min_n:
                errors.append(f"{slug}/{file_id}: only {len(entries)} entries (min {min_n})")
            if len(entries) > max_n:
                errors.append(f"{slug}/{file_id}: too many {len(entries)} entries (max {max_n})")
            names_lower = [e[0].strip().lower() for e in entries]
            dups = [n for n in set(names_lower) if names_lower.count(n) > 1]
            if dups:
                errors.append(f"{slug}/{file_id}: duplicates {dups}")
            for e in entries:
                if not (0.3 <= e[1] <= 1.0):
                    errors.append(f"{slug}/{file_id}: prob out of range for {e[0]!r}={e[1]}")
                if len(e[0].split()) < 2:
                    errors.append(f"{slug}/{file_id}: name has <2 words: {e[0]!r}")
                lower = e[0].lower()
                for bad in ban:
                    if bad in lower:
                        errors.append(f"{slug}/{file_id}: banlist hit {bad!r} in {e[0]!r}")
            path = os.path.join(slug_dir, f"{file_id}.txt")
            write_file(path, file_id, entries)
            written += 1

        # atmosphere files - kind-dependent
        if kind == "indoor":
            tod_pool = TIME_OF_DAY_INDOOR
            wx_pool = WEATHER_INDOOR
        else:
            tod_pool = TIME_OF_DAY_OUTDOOR
            wx_pool = WEATHER_OUTDOOR
        for file_id, pool in (("time_of_day", tod_pool), ("weather", wx_pool)):
            path = os.path.join(slug_dir, f"{file_id}.txt")
            write_file(path, file_id, pool)
            written += 1

    print(f"WROTE {written} files for {len(SETS)} sets")
    if errors:
        print("ERRORS:")
        for e in errors:
            print("  " + e)
        sys.exit(1)
    else:
        print("SELFCHECK_PASSED")


if __name__ == "__main__":
    main()
