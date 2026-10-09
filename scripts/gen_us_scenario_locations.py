"""Generator für die US/Pennsylvania-Rework-Location-Sets.

Ersetzt die archivierten Editorial-Kategorien (office, business, private,
sports, vacation, urban, hiking) durch geerdete Everyday-Versionen: Orte, wie
sie auf Amateur-Fotos aus Pennsylvania wirklich aussehen — Cubicle statt
Skyscraper-Lobby, Motel an der Interstate statt Ice Hotel, Township-Track
statt Velodrom.

Regeln (tests/unit/test_location_lists_extended.py):
- >= 10 Einträge pro Datei, Wahrscheinlichkeit in [0.3, 1.0]
- Namen mit >= 2 Wörtern, keine Duplikate pro Datei
- Indoor-Banlist matcht SUBSTRINGS: kein "fridge"/"bridge" (enthält "ridge"),
  kein "beach", "ocean", "trail", "mountain", "summit" in Indoor-Einträgen
- Wetter macht keine Lichtaussagen — Licht gehört allein zu time_of_day

Ausführen:  python scripts/gen_us_scenario_locations.py [--force]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

ELEMENTS = ("background", "midground", "architecture_detail", "props",
            "foreground_element", "time_of_day", "weather")

HEADER = {
    key: f"# element: {key}\n# format: name | probability | coverage_range | texture\n\n"
    for key in ELEMENTS
}

# ── Shared atmosphere pools ──────────────────────────────────────────────

INDOOR_HOME_TIME = """morning light through the blinds | 0.9 | - | -
flat late morning light | 0.85 | - | -
grey overcast midday | 0.85 | - | -
afternoon light across the carpet | 0.85 | - | -
late afternoon low sun | 0.8 | - | -
early evening with a lamp on | 0.9 | - | -
evening under the ceiling fan light | 0.8 | - | -
dim light before bed | 0.6 | - | -
tv glow in a dark room | 0.5 | - | -
half dark with the hall light on | 0.55 | - | -"""

INDOOR_HOME_WEATHER = """rain streaks on the window | 0.55 | - | -
window unit humming against the heat | 0.5 | - | -
dry forced-air warmth | 0.6 | - | -
humid summer air indoors | 0.5 | - | -
wind audible against the siding | 0.4 | - | -
snow visible outside the window | 0.35 | - | -
damp air after a thunderstorm | 0.45 | - | -
still muggy evening air | 0.45 | - | -
draught from a leaky window frame | 0.4 | - | -
dry winter static air | 0.4 | - | -"""

INDOOR_PUBLIC_TIME = """flat fluorescent light throughout | 1.0 | - | -
early morning before the rush | 0.7 | - | -
mid morning lull | 0.75 | - | -
lunchtime bustle | 0.7 | - | -
mid afternoon slump | 0.8 | - | -
late afternoon wind-down | 0.75 | - | -
early evening shift change | 0.6 | - | -
daylight mixing with the overheads | 0.7 | - | -
one bank of lights switched off | 0.45 | - | -
after-hours near closing | 0.5 | - | -"""

INDOOR_PUBLIC_WEATHER = """dry conditioned air | 0.9 | - | -
wet umbrella drips by the door | 0.4 | - | -
rain visible through the front glass | 0.5 | - | -
cold air pushed in by the entrance | 0.45 | - | -
humid air the AC cannot keep up with | 0.4 | - | -
slush tracked in on the mats | 0.35 | - | -
stale recirculated air | 0.6 | - | -
damp coats drying on chair backs | 0.4 | - | -
condensation on the front windows | 0.4 | - | -
no weather visible inside | 0.7 | - | -"""

OUTDOOR_PA_TIME = """early morning haze | 0.8 | - | -
bright mid morning | 0.85 | - | -
flat overcast midday | 0.9 | - | -
harsh summer noon glare | 0.6 | - | -
late afternoon long shadows | 0.9 | - | -
golden hour before sunset | 0.8 | - | -
dusk with porch lights coming on | 0.7 | - | -
blue hour after sunset | 0.55 | - | -
grey november midday | 0.7 | - | -
school-run morning light | 0.75 | - | -"""

OUTDOOR_PA_WEATHER = """flat overcast sky | 0.9 | - | -
humid summer haze | 0.6 | - | -
light drizzle in the air | 0.55 | - | -
steady grey rain | 0.45 | - | -
wet asphalt after a storm | 0.55 | - | -
clear cold air | 0.6 | - | -
gusty wind moving the trees | 0.5 | - | -
cold air with breath visible | 0.4 | - | -
first flurries not sticking | 0.35 | - | -
heavy still air before a thunderstorm | 0.4 | - | -"""

# Season-locked pools: summer sets must never draw "grey november midday",
# fall sets never "harsh summer noon glare".
OUTDOOR_PA_SUMMER_TIME = """early morning haze | 0.8 | - | -
bright mid morning | 0.85 | - | -
harsh summer noon glare | 0.8 | - | -
lazy mid afternoon | 0.85 | - | -
late afternoon long shadows | 0.9 | - | -
golden hour before sunset | 0.85 | - | -
dusk with lightning bugs rising | 0.6 | - | -
long summer evening light | 0.8 | - | -
flat overcast midday | 0.7 | - | -
early evening cookout hour | 0.75 | - | -"""

OUTDOOR_PA_SUMMER_WEATHER = """humid summer haze | 0.8 | - | -
heavy still air before a thunderstorm | 0.5 | - | -
sudden afternoon downpour passing | 0.4 | - | -
wet pavement steaming after rain | 0.45 | - | -
dry heat with a light breeze | 0.7 | - | -
sticky evening air | 0.6 | - | -
thin high cloud | 0.6 | - | -
flat overcast sky | 0.6 | - | -
gusty wind ahead of a storm front | 0.45 | - | -
cloudless glare off the water | 0.5 | - | -"""

OUTDOOR_PA_FALL_TIME = """early morning frost light | 0.7 | - | -
bright crisp mid morning | 0.85 | - | -
flat overcast midday | 0.9 | - | -
grey november midday | 0.8 | - | -
late afternoon long shadows | 0.9 | - | -
early dusk closing in | 0.75 | - | -
golden hour through bare branches | 0.8 | - | -
low sun flickering through trunks | 0.7 | - | -
overcast early evening | 0.75 | - | -
short-day afternoon fading fast | 0.7 | - | -"""

OUTDOOR_PA_FALL_WEATHER = """clear cold air | 0.8 | - | -
cold air with breath visible | 0.6 | - | -
gusty wind stripping the leaves | 0.6 | - | -
flat overcast sky | 0.8 | - | -
light drizzle in the air | 0.5 | - | -
damp leaf-mulch smell after rain | 0.55 | - | -
first flurries not sticking | 0.4 | - | -
raw wind with a wet edge | 0.45 | - | -
thin fog burning off late | 0.45 | - | -
still cold air in the hollow | 0.5 | - | -"""

INDOOR_SUMMER_WEATHER = """humid air the window unit fights | 0.6 | - | -
sticky salt air through the screens | 0.55 | - | -
thunderstorm heard rolling through | 0.45 | - | -
rain streaks on the slider glass | 0.5 | - | -
dry heat radiating from the walkway | 0.5 | - | -
box fan air moving through the room | 0.6 | - | -
still muggy evening air | 0.55 | - | -
breeze lifting the curtain | 0.55 | - | -
damp towels never quite drying | 0.5 | - | -
sunscreen and salt smell indoors | 0.5 | - | -"""


SETS: dict[str, dict[str, dict[str, str]]] = {
    # ═════════════════════════════ INDOOR ═════════════════════════════
    # indoor/american_office (cubicle_pod, break_room, plain_conference_room) was
    # rewritten by hand to the Krea 2 rules on 2026-10-09 and lives only in
    # location_lists/ now — kept out of here so --force cannot revert it.
    "indoor/american_fitness_spot": {
        "high_school_gym": {
            "background": """gym wall with painted mascot | 1.0 | 0.4-0.8 | chipped mural paint
pulled-out bleacher sections | 0.85 | 0.3-0.7 | worn wooden benches
championship banners on the wall | 0.7 | 0.2-0.5 | felt pennants by decade
basketball hoop cranked down | 0.75 | 0.2-0.4 | frayed net
stage at the gym end | 0.5 | 0.25-0.5 | heavy curtain drawn
high windows with safety grilles | 0.6 | 0.15-0.4 | caged daylight
wall pads under the hoop | 0.6 | 0.15-0.35 | taped vinyl seams
scoreboard dark on the wall | 0.6 | 0.1-0.3 | unlit bulbs
folded lunch tables against the wall | 0.45 | 0.2-0.4 | rolling frames
rope climb anchored to the ceiling | 0.4 | 0.1-0.3 | knotted end swaying
divider curtain half drawn | 0.5 | 0.25-0.5 | motorized vinyl wall""",
            "midground": """volleyball net across the court | 0.6 | 0.2-0.5 | slack center
cart of basketballs | 0.65 | 0.1-0.3 | mixed wear levels
cones set for drills | 0.55 | 0.1-0.3 | faded orange
kids in gym class scattered | 0.5 | 0.2-0.5 | motion blur
folding chairs in a row | 0.45 | 0.15-0.35 | event leftovers
mat stack by the wall | 0.55 | 0.15-0.35 | blue crash pads
teacher's rolling cart | 0.4 | 0.1-0.25 | whistle and clipboard
free throw line worn pale | 0.5 | 0.05-0.2 | repainted stripe
scorer's table with a mic | 0.4 | 0.1-0.3 | cord taped down
ball rack of playground balls | 0.5 | 0.1-0.25 | half-flat reds
pommel horse pushed to a corner | 0.35 | 0.1-0.25 | cracked leather top""",
            "architecture_detail": """hardwood court with painted lines | 0.9 | 0.4-0.7 | layered color keys
scuff marks beyond the baseline | 0.55 | 0.1-0.3 | sole streaks
caged ceiling lights | 0.6 | 0.1-0.3 | mercury-vapor glow
floor plates for net posts | 0.5 | 0.03-0.1 | brass covers
double doors with panic bars | 0.55 | 0.1-0.25 | kick-scuffed base
water fountain alcove | 0.45 | 0.05-0.15 | gum in the drain
dead spot in the floorboards | 0.35 | 0.05-0.15 | duller bounce
electrical panel painted shut | 0.35 | 0.03-0.1 | grey enamel
banner cables angled to the wall | 0.4 | 0.05-0.15 | turnbuckle ends
vent grilles up high | 0.45 | 0.05-0.15 | dust flags
threshold plate at the locker hall | 0.4 | 0.02-0.08 | polished by traffic""",
            "props": """pinnies in a mesh bag | 0.55 | 0.03-0.1 | faded scrimmage vests
whistle on a lanyard | 0.5 | 0.02-0.06 | chained clipboard
attendance clipboard on the stage lip | 0.45 | 0.02-0.08 | curled roster
lost hoodie on the bleachers | 0.55 | 0.03-0.1 | crumpled sleeve
water bottles clustered at the wall | 0.6 | 0.03-0.1 | name-taped
stray sneaker under the bleachers | 0.4 | 0.02-0.06 | missing its pair
first aid kit on the scorer's table | 0.4 | 0.02-0.06 | scuffed plastic box
jump ropes on a wall hook | 0.5 | 0.03-0.1 | tangled handles
floor tape roll left on a chair | 0.4 | 0.02-0.06 | half-used
pump with a bent needle | 0.45 | 0.02-0.06 | worked loose
poster for the friday game | 0.5 | 0.03-0.1 | handmade letters""",
            "foreground_element": """ball dribbled at the free throw line | 0.6 | 0.1-0.3 | echo off the walls
sneakers squeaking on a cut | 0.55 | 0.1-0.25 | mid-pivot
water fountain leaned into | 0.5 | 0.05-0.2 | arc of water
bleacher row climbed sideways | 0.5 | 0.1-0.3 | hand on the rail
pinny pulled over the head | 0.5 | 0.05-0.2 | static hair
laces double-knotted on the bench | 0.5 | 0.05-0.15 | bent forward
ponytail redone before the drill | 0.5 | 0.05-0.15 | elastic in teeth
net checked with a tug | 0.4 | 0.05-0.15 | slack tested
phone filmed through the doorway | 0.4 | 0.05-0.15 | game clip
clipboard consulted mid-whistle | 0.45 | 0.05-0.15 | next drill
ball tucked under one arm | 0.5 | 0.05-0.15 | waiting turn""",
            "time_of_day": INDOOR_PUBLIC_TIME,
            "weather": INDOOR_PUBLIC_WEATHER,
        },
    },
    "indoor/private_home_room": {
        "plain_master_bedroom": {
            "background": """queen bed with a plain comforter | 1.0 | 0.4-0.8 | wrinkled solid color
dresser with a wide mirror | 0.8 | 0.3-0.6 | clutter along the top
window with bent mini blinds | 0.7 | 0.2-0.5 | one slat kinked
closet with sliding doors | 0.65 | 0.25-0.5 | one door off its track
wall with framed family photos | 0.6 | 0.2-0.45 | mixed frame styles
TV on the dresser | 0.55 | 0.15-0.35 | remote lost in the sheets
laundry hamper in the corner | 0.6 | 0.1-0.3 | overflowing lid
nightstand with a lamp | 0.75 | 0.15-0.35 | water ring marks
headboard against the wall | 0.6 | 0.2-0.45 | scuffed veneer
ceiling fan overhead | 0.65 | 0.1-0.3 | dust-lined blades
bedroom door with a robe hook | 0.5 | 0.15-0.4 | robe slumped on it""",
            "midground": """bed corner with kicked-off throw | 0.8 | 0.2-0.5 | slid to the floor
phone charger draped over the nightstand | 0.7 | 0.05-0.15 | cable to the outlet
bench of folded laundry at the foot | 0.55 | 0.15-0.35 | waiting to be put away
box fan in the window | 0.5 | 0.1-0.3 | humming on high
slippers kicked apart on the carpet | 0.55 | 0.05-0.15 | flattened heels
dog bed beside the dresser | 0.4 | 0.1-0.3 | hair-covered cushion
full-length mirror against the wall | 0.55 | 0.15-0.35 | leaning unhung
book splayed on the nightstand | 0.5 | 0.03-0.1 | spine cracked
alarm clock with red digits | 0.5 | 0.03-0.1 | blinking wrong time
water glass half full | 0.5 | 0.02-0.08 | night stand ring
cat asleep in the laundry | 0.35 | 0.05-0.2 | curled deep""",
            "architecture_detail": """wall-to-wall beige carpet | 0.85 | 0.3-0.6 | vacuum lines fading
outlet crowded behind the nightstand | 0.5 | 0.02-0.08 | plug stack
closet track lint at the floor | 0.4 | 0.03-0.1 | dust ridge line
ceiling fan pull chains | 0.5 | 0.02-0.08 | mismatched fobs
window sill with a candle | 0.45 | 0.03-0.1 | burned-down jar
baseboard scuffed by the vacuum | 0.4 | 0.05-0.15 | chipped paint line
door hinge that squeaks | 0.4 | 0.02-0.06 | painted-over pin
thermostat in the hallway glow | 0.35 | 0.02-0.08 | lit disc
carpet edge fraying at the bathroom | 0.35 | 0.03-0.1 | threshold wear
nail holes from old frames | 0.4 | 0.02-0.08 | unfilled dots
vent with a sock caught | 0.3 | 0.02-0.08 | floor register""",
            "props": """laundry basket of unfolded towels | 0.6 | 0.05-0.2 | warm from the dryer
deodorant and lotion on the dresser | 0.6 | 0.03-0.1 | mixed caps
jewelry dish by the mirror | 0.55 | 0.02-0.08 | tangled chains
charging cable knot | 0.55 | 0.02-0.08 | three cords braided
receipts and change on the dresser | 0.5 | 0.02-0.08 | pocket dump
hair dryer on the bed | 0.4 | 0.03-0.1 | cord wrapped loose
paperback stack on the floor | 0.45 | 0.03-0.1 | leaning tower
throw pillows shoved aside | 0.55 | 0.05-0.2 | nightly pile
glasses case on the nightstand | 0.45 | 0.02-0.06 | snapped shut
dog leash on the door knob | 0.4 | 0.02-0.08 | hanging clip
lint roller on the dresser | 0.4 | 0.02-0.06 | half-peeled sheet""",
            "foreground_element": """comforter pulled straight | 0.6 | 0.15-0.4 | morning fix
sock balanced on one foot | 0.5 | 0.05-0.15 | getting dressed
phone checked against the pillow | 0.6 | 0.05-0.2 | face lit
laundry folded on the bed | 0.6 | 0.1-0.3 | t-shirt mid-fold
earrings put on at the mirror | 0.5 | 0.05-0.15 | head tilted
hamper lid pushed down | 0.45 | 0.05-0.15 | overfull press
blinds cracked for a look outside | 0.5 | 0.05-0.15 | two fingers
robe belt cinched | 0.5 | 0.1-0.25 | quick knot
alarm silenced with a slap | 0.45 | 0.03-0.1 | eyes still shut
lotion worked into the hands | 0.45 | 0.05-0.12 | ring set aside
dog shooed off the bed | 0.35 | 0.1-0.25 | reluctant hop""",
            "time_of_day": INDOOR_HOME_TIME,
            "weather": INDOOR_HOME_WEATHER,
        },
        "bathroom_with_tub_shower_combo": {
            "background": """tub and shower combo with vinyl curtain | 1.0 | 0.4-0.8 | printed curtain half drawn
vanity with a cultured marble top | 0.8 | 0.3-0.6 | toothpaste specks
medicine cabinet mirror | 0.75 | 0.2-0.5 | corner rust spots
towel bar with bunched towels | 0.7 | 0.15-0.4 | mismatched colors
toilet with a fabric lid cover | 0.55 | 0.2-0.4 | dated shag cover
wall art of seashells | 0.4 | 0.1-0.25 | dollar-store frame
window of frosted glass | 0.5 | 0.1-0.3 | dusty screen behind
over-toilet shelf unit | 0.55 | 0.15-0.35 | wobbly white wire
wallpaper border at the ceiling | 0.4 | 0.1-0.3 | peeling seam
hamper wedged by the vanity | 0.5 | 0.1-0.3 | lid ajar
night light in the outlet | 0.45 | 0.02-0.08 | warm glow plug""",
            "midground": """bath mat bunched by the tub | 0.7 | 0.1-0.3 | rubber-back curling
scale pushed against the wall | 0.55 | 0.05-0.15 | dusty glass face
step stool for the kids | 0.45 | 0.05-0.15 | cartoon stickers
plunger and brush in the corner | 0.5 | 0.03-0.1 | discreet caddy
trash can with a swing lid | 0.55 | 0.05-0.15 | cotton swabs visible
laundry pile by the door | 0.5 | 0.1-0.25 | towels and pajamas
shower caddy hanging from the head | 0.65 | 0.05-0.2 | crowded bottles
kids' bath toys in a net | 0.4 | 0.05-0.15 | drip-drying
towel dropped on the floor | 0.45 | 0.05-0.2 | damp heap
curling iron cord on the vanity | 0.5 | 0.03-0.1 | wrapped loose
razor on the tub ledge | 0.45 | 0.02-0.06 | soap-crusted head""",
            "architecture_detail": """vinyl floor in faux tile | 0.8 | 0.3-0.6 | worn traffic path
caulk line darkened at the tub | 0.6 | 0.03-0.1 | mildew shadow
exhaust fan rattling | 0.55 | 0.03-0.1 | dusty grille
towel hook overloaded | 0.5 | 0.03-0.1 | four on one hook
grab bar added to the tub wall | 0.35 | 0.05-0.15 | white powder coat
sink stopper that never seals | 0.4 | 0.02-0.06 | chained plug
cabinet door that swings open | 0.4 | 0.05-0.15 | loose magnet
water stain on the ceiling corner | 0.4 | 0.05-0.15 | ringed patch
threshold strip to the hallway | 0.4 | 0.02-0.08 | lifted edge
outlet with a GFCI reset | 0.45 | 0.02-0.06 | test button
shampoo ring on the tub edge | 0.45 | 0.02-0.08 | soap film circle""",
            "props": """toothbrush cup crowded | 0.7 | 0.02-0.08 | family bristle mix
hand soap pump nearly empty | 0.6 | 0.02-0.06 | last inch
bath bomb bowl by the tub | 0.35 | 0.02-0.08 | pastel spheres
makeup bag unzipped on the vanity | 0.55 | 0.03-0.1 | spilling brushes
towel animals nobody refolds | 0.3 | 0.03-0.1 | vacation habit
box of tissues on the tank | 0.5 | 0.02-0.06 | one pulled up
contact lens case by the faucet | 0.45 | 0.02-0.05 | left-right lids
kids' toothpaste with the cap off | 0.45 | 0.02-0.06 | glitter gel
air freshener spray on the shelf | 0.5 | 0.02-0.06 | linen scent
hair ties in a soap dish | 0.5 | 0.02-0.06 | stretched spirals
bathrobe on the door hook | 0.55 | 0.1-0.3 | waffle weave""",
            "foreground_element": """steam wiped off the mirror | 0.65 | 0.1-0.25 | palm streak
towel wrapped hair turban | 0.6 | 0.1-0.25 | tucked twist
mascara leaned into the mirror | 0.55 | 0.05-0.15 | mouth open slightly
curtain pulled back with rings scraping | 0.5 | 0.1-0.3 | metallic slide
toothbrush foam mid-brush | 0.55 | 0.05-0.15 | timer on the phone
lotion applied at the vanity edge | 0.5 | 0.05-0.15 | one foot on the tub
towel tucked tight under the arms | 0.55 | 0.1-0.3 | post-shower
hair dryer aimed upside down | 0.5 | 0.05-0.2 | roots first
floss wound around fingers | 0.4 | 0.03-0.1 | mirror lean
bath water tested with a wrist | 0.4 | 0.05-0.15 | faucet running
face washed over the sink | 0.5 | 0.05-0.15 | dripping cupped hands""",
            "time_of_day": INDOOR_HOME_TIME,
            "weather": INDOOR_HOME_WEATHER,
        },
        "walk_in_closet": {
            "background": """double-hung rods packed tight | 1.0 | 0.4-0.8 | hangers wedged shoulder to shoulder
wire shelving above the rods | 0.8 | 0.25-0.5 | sweater stacks slumping
shoe shelf rows near the floor | 0.7 | 0.2-0.4 | heels toed in with sneakers
back wall with a full mirror | 0.6 | 0.2-0.45 | outfit-check smudges
plastic drawer tower in the corner | 0.6 | 0.15-0.35 | labeled masking tape
hanging garment bags at the end | 0.5 | 0.15-0.3 | zipped formal wear
belt and scarf hooks inside the door | 0.55 | 0.1-0.25 | layered loops
overflow tote bins up top | 0.6 | 0.2-0.4 | seasonal swap boxes
laundry basket on the floor | 0.55 | 0.1-0.3 | try-on rejects
bare bulb or boob light overhead | 0.55 | 0.05-0.15 | pull chain
ironing board folded against the wall | 0.4 | 0.1-0.3 | scorched cover""",
            "midground": """dresses grouped at one end | 0.7 | 0.2-0.45 | casual to church order
jeans folded over hangers | 0.7 | 0.15-0.35 | denim gradient
work tops separated by color | 0.6 | 0.15-0.35 | rough rainbow
step stool for the high shelf | 0.5 | 0.05-0.15 | folded flat
empty hangers bunched at the rod end | 0.6 | 0.05-0.2 | wire and velvet mix
donation bag half full | 0.5 | 0.1-0.25 | handles tied loose
hamper of delicates | 0.45 | 0.05-0.2 | mesh wash bags
boots paired under the dresses | 0.55 | 0.1-0.25 | one pair flopped over
purse row on the shelf | 0.6 | 0.1-0.3 | stuffed with tissue
robe on a door hook | 0.5 | 0.1-0.25 | sash dragging
vacuum kept in the corner | 0.4 | 0.1-0.25 | cord wrapped""",
            "architecture_detail": """carpet flattened down the middle | 0.7 | 0.2-0.5 | traffic stripe
wire shelf shadows on the wall | 0.45 | 0.1-0.3 | grid lines
rod bracket bent from weight | 0.4 | 0.02-0.08 | sagging center
door that only opens halfway | 0.4 | 0.05-0.15 | blocked arc
outlet used for the iron | 0.4 | 0.02-0.06 | single plug
paint scuffs at hanger height | 0.45 | 0.05-0.15 | rubbed line
attic hatch in the ceiling | 0.3 | 0.05-0.15 | painted-shut frame
baseboard dusty behind shoes | 0.4 | 0.03-0.1 | lint drift
light switch with a dimmer knob | 0.35 | 0.02-0.06 | loose plate
closet system anchor holes | 0.35 | 0.02-0.08 | old configuration dots
threshold hump into the bedroom | 0.4 | 0.02-0.08 | carpet transition""",
            "props": """lint roller on the shelf lip | 0.5 | 0.02-0.06 | torn strip hanging
shoe boxes with photo labels | 0.45 | 0.05-0.15 | printed snapshots
jewelry hanger of necklaces | 0.5 | 0.03-0.1 | untangled rows
safety pins in a dish | 0.4 | 0.02-0.05 | mixed sizes
travel steamer on the floor | 0.4 | 0.03-0.08 | wrapped cord
gift bags flattened above | 0.45 | 0.05-0.12 | reuse stash
sticky lint sheets balled up | 0.35 | 0.02-0.05 | missed the basket
perfume bottles on a tray | 0.45 | 0.02-0.08 | daily two in front
tote bag of returns | 0.45 | 0.03-0.1 | receipts stapled
sewing kit tin | 0.35 | 0.02-0.06 | button assortment
dry cleaning in plastic | 0.45 | 0.1-0.25 | paper-tagged wire hangers""",
            "foreground_element": """hangers slid one by one | 0.7 | 0.1-0.3 | scrape along the rod
top held up against the chest | 0.65 | 0.1-0.3 | mirror glance
jeans tugged on mid-hop | 0.55 | 0.1-0.3 | one leg in
shoe box pulled from the stack | 0.5 | 0.05-0.2 | tower wobbling
belt threaded through loops | 0.5 | 0.05-0.15 | half-turned
dress zipped with a reach back | 0.5 | 0.05-0.2 | elbow up
sweater refolded on the shelf | 0.5 | 0.05-0.15 | sleeves tucked
heels dangled from two fingers | 0.5 | 0.05-0.15 | deciding
outfit flat-laid over the basket | 0.45 | 0.1-0.25 | combination test
phone mirror photo taken | 0.5 | 0.05-0.2 | outfit check
tags snipped with nail scissors | 0.4 | 0.03-0.1 | new top""",
            "time_of_day": INDOOR_HOME_TIME,
            "weather": INDOOR_HOME_WEATHER,
        },
    },
    "indoor/american_vacation_stay": {
        "interstate_motel_room": {
            "background": """two queen beds with floral spreads | 1.0 | 0.4-0.8 | stiff quilted polyester
AC unit under the window | 0.85 | 0.15-0.35 | rattling front grille
heavy blackout curtains | 0.75 | 0.25-0.5 | stiff rubber-backed pleats
wood-look dresser with a TV | 0.7 | 0.25-0.5 | bolted swivel base
framed print of a lighthouse | 0.5 | 0.1-0.25 | faded generic art
vanity nook outside the bathroom | 0.6 | 0.2-0.4 | bright bulb strip
luggage rack unfolded | 0.55 | 0.1-0.25 | woven straps
connecting door with a deadbolt | 0.45 | 0.1-0.3 | painted steel
mirror bolted over the dresser | 0.6 | 0.2-0.4 | beveled edge
headboards mounted to the wall | 0.6 | 0.2-0.4 | no gap to the frame
ice bucket with a plastic liner | 0.5 | 0.05-0.15 | tray of glasses""",
            "midground": """suitcase open on the second bed | 0.75 | 0.2-0.4 | clothes erupting
plastic cups in paper sleeves | 0.5 | 0.03-0.1 | sanitary wraps
mini refrigerator humming | 0.6 | 0.1-0.25 | door dented
coffee maker with two pods | 0.6 | 0.05-0.15 | paper cup stack
phone with a laminated menu card | 0.5 | 0.03-0.1 | pizza delivery ads
remote wrapped in a paper band | 0.45 | 0.02-0.08 | sanitized label
desk chair wedged by the window | 0.5 | 0.15-0.3 | rolling on carpet
takeout bags on the dresser | 0.5 | 0.05-0.2 | napkin overflow
shoes lined at the door | 0.55 | 0.05-0.15 | road trip pairs
cooler dragged in from the car | 0.45 | 0.1-0.25 | lid cracked open
kids' blanket fort on one bed | 0.35 | 0.15-0.4 | pillow walls""",
            "architecture_detail": """commercial carpet in dark swirls | 0.8 | 0.3-0.6 | stain-hiding pattern
popcorn ceiling with a sprinkler | 0.5 | 0.1-0.3 | painted head
door with a security latch | 0.6 | 0.05-0.15 | swing bar
peephole in the door | 0.45 | 0.02-0.06 | brass ring
window that opens two inches | 0.45 | 0.05-0.15 | safety stop screw
bathroom light with a fan switch | 0.5 | 0.02-0.08 | loud toggle
outlet with USB ports | 0.45 | 0.02-0.06 | added faceplate
mystery thermostat setting | 0.45 | 0.02-0.08 | taped instructions
baseboard dinged by suitcases | 0.45 | 0.03-0.1 | wheel scuffs
curtain wand hanging | 0.45 | 0.02-0.08 | plastic baton
evacuation map on the door | 0.5 | 0.03-0.1 | you-are-here star""",
            "props": """key cards on the dresser | 0.65 | 0.02-0.06 | paper sleeve with the room number
half-empty water bottles | 0.6 | 0.02-0.08 | caps lost
phone chargers in both outlets | 0.6 | 0.02-0.08 | cords crossing the bed
gas station snacks piled | 0.55 | 0.03-0.1 | chip bags and jerky
soda from the vending machine | 0.5 | 0.02-0.08 | ice-flecked can
do not disturb tag on the handle | 0.5 | 0.02-0.06 | swinging card
motel notepad and pen | 0.45 | 0.02-0.05 | logo header
travel toiletry bag by the sink | 0.55 | 0.03-0.1 | hanging hook
paper coffee cups doubled | 0.45 | 0.02-0.06 | heat sleeve improvised
road atlas splayed on the bed | 0.35 | 0.05-0.12 | highlighted route
pillow fort remote pile | 0.4 | 0.02-0.08 | all channels checked""",
            "foreground_element": """suitcase zipper strained shut | 0.6 | 0.1-0.25 | knee on the lid
curtains yanked against the daylight | 0.55 | 0.1-0.3 | gap of glare
AC dial turned to high | 0.5 | 0.05-0.15 | vents fluttering
bed flopped onto backwards | 0.55 | 0.15-0.35 | arms out
ice bucket carried in | 0.5 | 0.05-0.15 | cubes settling
takeout unpacked on the spread | 0.55 | 0.1-0.25 | foil containers
key card tapped at the reader | 0.5 | 0.05-0.15 | green light blink
shoes kicked toward the door | 0.5 | 0.05-0.15 | mid-air laces
pillow doubled under the head | 0.5 | 0.1-0.25 | tv angle
phone propped against the lamp | 0.45 | 0.03-0.1 | show queued
blackout dark checked with a peek | 0.4 | 0.05-0.15 | curtain corner lifted""",
            "time_of_day": INDOOR_HOME_TIME,
            "weather": INDOOR_HOME_WEATHER,
        },
    },
    # ═════════════════════════════ OUTDOOR ═════════════════════════════
    "outdoor/pennsylvania_suburb": {
        "driveway_basketball_hoop": {
            "background": """asphalt driveway with sealant patches | 1.0 | 0.4-0.8 | darker repair strips
portable hoop with a sand base | 0.85 | 0.2-0.4 | tilted backboard
garage door with dents at bumper height | 0.7 | 0.3-0.6 | white panels
minivan parked to one side | 0.6 | 0.25-0.5 | magnet ribbon on the gate
basketball chalk key drawn | 0.45 | 0.1-0.3 | rain-faded lines
front lawn edging the blacktop | 0.6 | 0.2-0.45 | trimmed border
street with parked cars beyond | 0.55 | 0.25-0.5 | quiet residential
neighbor's driveway hoop across | 0.4 | 0.1-0.3 | rival setup
basement window wells along the house | 0.4 | 0.1-0.25 | gravel and covers
oil stain at the parking spot | 0.5 | 0.05-0.2 | absorbent scatter
mailbox at the driveway mouth | 0.5 | 0.05-0.15 | post leaning""",
            "midground": """basketball resting against the base | 0.7 | 0.05-0.15 | worn channels
scooter dropped mid-driveway | 0.5 | 0.05-0.2 | handlebars turned
sidewalk chalk arrows and games | 0.45 | 0.1-0.3 | hopscotch grid
trash cans staged for pickup | 0.55 | 0.1-0.25 | lids numbered
car in the open garage bay | 0.5 | 0.2-0.4 | tail lights inward
work bench visible in the garage | 0.5 | 0.15-0.35 | pegboard shadowed
lawn mower parked by the bay | 0.45 | 0.1-0.25 | grass-caked deck
rebounder net beside the hoop | 0.35 | 0.1-0.25 | return chute
folding camp chairs for spectators | 0.4 | 0.1-0.25 | cup holders sagging
dog watching from the storm door | 0.4 | 0.05-0.15 | nose to the glass
skateboard upside down on the lawn | 0.4 | 0.05-0.15 | wheels still turning""",
            "architecture_detail": """driveway crack with grass tuft | 0.55 | 0.05-0.15 | expansion seam
apron joint at the street | 0.5 | 0.05-0.15 | curb transition
garage door track and springs | 0.45 | 0.05-0.15 | greased coils
motion flood over the garage | 0.45 | 0.02-0.08 | double heads
downspout crossing the driveway edge | 0.4 | 0.03-0.1 | flattened end
hoop pole padded with a noodle | 0.4 | 0.03-0.1 | zip-tied foam
paint scuff on the garage frame | 0.4 | 0.02-0.08 | mirror-height mark
salt stains from winter | 0.4 | 0.05-0.2 | white bloom
basketball net frayed to threads | 0.45 | 0.03-0.1 | half-detached loops
seam where new blacktop meets old | 0.45 | 0.05-0.15 | color break
reflector stake at the corner | 0.35 | 0.02-0.06 | snowplow guide""",
            "props": """ball pump with a stuck needle | 0.45 | 0.02-0.06 | garage-shelf resident
water bottles on the retaining wall | 0.55 | 0.02-0.08 | game-break row
bluetooth speaker in the garage bay | 0.45 | 0.02-0.08 | echoing playlist
score kept in chalk tallies | 0.4 | 0.02-0.08 | disputed marks
hose spray nozzle on the walk | 0.4 | 0.02-0.06 | trigger cracked
extra ball wedged in the shrubs | 0.4 | 0.02-0.08 | out-of-bounds refuge
sweatshirts flung on the mailbox | 0.45 | 0.03-0.1 | goalpost markers
snow shovel not yet stored | 0.35 | 0.03-0.1 | leaning since March
car wash bucket and mitt | 0.4 | 0.03-0.1 | suds drying
sponge dart stuck in the gutter | 0.35 | 0.02-0.05 | lost ammo
freeze pops handed out | 0.4 | 0.02-0.08 | dripping sleeves""",
            "foreground_element": """free throw lined up | 0.65 | 0.1-0.3 | elbow tucked
rebound chased to the lawn | 0.55 | 0.1-0.3 | bounce off the rim
crossover dribbled low | 0.5 | 0.1-0.25 | sneaker scrape
ball spun on a finger | 0.45 | 0.05-0.15 | showing off
sweat wiped on a shoulder | 0.5 | 0.05-0.15 | mid-game pause
hoop lowered for the little kids | 0.4 | 0.1-0.25 | crank handle
horse letters counted on fingers | 0.45 | 0.05-0.15 | h-o-r
water chugged between games | 0.5 | 0.05-0.15 | bottle crackle
chalk lines redrawn | 0.4 | 0.05-0.15 | crouched stroke
garage fridge raided for drinks | 0.4 | 0.05-0.2 | door glow
buzzer beater called aloud | 0.45 | 0.05-0.2 | three-two-one""",
            "time_of_day": OUTDOOR_PA_TIME,
            "weather": OUTDOOR_PA_WEATHER,
        },
    },
    # outdoor/pennsylvania_small_town sets were rewritten by hand to the Krea 2 rules on 2026-10-09
    # and live only in location_lists/ now — kept out of here so --force cannot revert them.
    # outdoor/pennsylvania_nature_spot sets were rewritten by hand to the Krea 2 rules on 2026-10-09
    # and live only in location_lists/ now — kept out of here so --force cannot revert them.
    "outdoor/american_fitness_spot": {
        "community_pool_deck": {
            "background": """rectangle pool with lane ropes half in | 1.0 | 0.4-0.8 | chlorine-blue water
lifeguard chair with an umbrella | 0.8 | 0.15-0.35 | white tower
concession window under an awning | 0.55 | 0.15-0.35 | menu board prices
chain-link around the deck | 0.65 | 0.2-0.45 | towel-draped sections
diving board over the deep end | 0.6 | 0.1-0.3 | sandpaper tread
kiddie pool fenced separately | 0.55 | 0.2-0.4 | mushroom sprinkler
locker building of painted block | 0.55 | 0.2-0.45 | men's and women's arrows
flags strung over the lap lanes | 0.5 | 0.1-0.3 | backstroke pennants
grass sunbathing bank | 0.55 | 0.25-0.5 | towel patchwork
pump house humming | 0.4 | 0.1-0.25 | chemical smell
slide into the shallow end | 0.5 | 0.1-0.3 | water sheeting""",
            "midground": """plastic loungers in rows | 0.75 | 0.25-0.5 | strap-sagging frames
family camp of towels and bags | 0.65 | 0.15-0.35 | claimed territory
kickboards stacked by the lanes | 0.55 | 0.05-0.15 | foam wedges
umbrella tables with tilted shade | 0.6 | 0.2-0.4 | crank handles
swim team bags along the fence | 0.5 | 0.1-0.3 | mesh duffels
puddles on the deck tile | 0.6 | 0.1-0.3 | footprint trails
snack table under the awning | 0.5 | 0.1-0.25 | pretzel and slush line
lost goggles bin on the counter | 0.45 | 0.03-0.1 | tangle of straps
noodles floating loose | 0.55 | 0.05-0.2 | drifting colors
lap swimmer in the far lane | 0.5 | 0.1-0.25 | steady freestyle
adirondack chairs for the members | 0.4 | 0.15-0.3 | faded plastic""",
            "architecture_detail": """deck tile with anti-slip texture | 0.6 | 0.15-0.4 | pebbled surface
gutter grate around the pool lip | 0.55 | 0.05-0.2 | overflow slots
depth numbers painted on the edge | 0.55 | 0.03-0.1 | 3FT to 9FT
ladder rails polished bright | 0.5 | 0.03-0.1 | grip-worn chrome
rope hooks on the fence posts | 0.4 | 0.02-0.08 | coiled spares
backwash hose run to the drain | 0.4 | 0.05-0.15 | flat-lay canvas
rules sign with faded lines | 0.5 | 0.05-0.15 | no-running header
shower head on the locker wall | 0.45 | 0.03-0.1 | rinse-first sign
expansion joints in the deck | 0.45 | 0.05-0.15 | tar lines
guard stand bolts rusting | 0.4 | 0.02-0.08 | streaked base
kiddie gate with a high latch | 0.45 | 0.03-0.1 | parent-height lock""",
            "props": """sunscreen bottles on every table | 0.6 | 0.02-0.08 | family arsenal
swim diapers pack by a bag | 0.4 | 0.02-0.06 | poolside supply
goggles pushed up on foreheads | 0.55 | 0.02-0.06 | red eye rings
snack-bar pretzel with mustard | 0.45 | 0.02-0.06 | paper boat
band-aid colored water bottle | 0.4 | 0.02-0.06 | team sticker layer
whistle twirled on a finger | 0.45 | 0.02-0.06 | guard habit
flip flops in a pile at the ladder | 0.55 | 0.03-0.1 | mixed sizes
pool pass lanyards on a table | 0.45 | 0.02-0.06 | laminated photos
dive rings on the bottom | 0.45 | 0.03-0.1 | wavering colors
towel snapped in a sibling war | 0.4 | 0.03-0.1 | mid-crack
paperback swollen with humidity | 0.4 | 0.02-0.06 | deck-chair read""",
            "foreground_element": """cannonball mid-air off the board | 0.55 | 0.1-0.25 | knees hugged
towel wrapped shivering at the edge | 0.55 | 0.1-0.25 | lips faintly blue
sunscreen worked onto a squirming kid | 0.5 | 0.1-0.25 | white war paint
goggles suctioned into place | 0.5 | 0.03-0.1 | strap snap
ladder climbed dripping | 0.55 | 0.05-0.2 | water sheet
lounger dragged into the sun | 0.5 | 0.05-0.2 | legs scraping deck
slush cup shared with two straws | 0.45 | 0.03-0.1 | brain-freeze pause
kickboard lap with flutter kicks | 0.5 | 0.1-0.25 | splash rhythm
adult swim groaned at | 0.45 | 0.05-0.2 | deck exodus
wet hair slicked back at the wall | 0.5 | 0.05-0.15 | breath caught
whistle pointed at a runner | 0.45 | 0.05-0.15 | walk enforced""",
            "time_of_day": OUTDOOR_PA_SUMMER_TIME,
            "weather": OUTDOOR_PA_SUMMER_WEATHER,
        },
        "rec_soccer_field": {
            "background": """rec field with portable goals | 1.0 | 0.4-0.8 | netting tied to frames
painted lines fresh over old | 0.7 | 0.2-0.5 | doubled boundaries
parking lot along the sideline | 0.6 | 0.25-0.5 | minivan row
port-a-john pair by the lot | 0.45 | 0.05-0.15 | township blue
equipment shed with a ramp | 0.45 | 0.1-0.25 | mower parked inside
adjacent diamond backstop | 0.5 | 0.15-0.35 | chain-link wedge
tree line shading one corner | 0.55 | 0.25-0.5 | afternoon relief
snack stand shuttered midweek | 0.4 | 0.1-0.25 | plywood window
scoreboard that half works | 0.4 | 0.1-0.25 | missing bulbs
municipal sign at the entrance | 0.45 | 0.1-0.25 | park hours letters
power lines skirting the field | 0.45 | 0.1-0.3 | lazy spans""",
            "midground": """parent chairs along the touchline | 0.75 | 0.15-0.35 | folding armchair row
cooler wagon parked mid-sideline | 0.55 | 0.05-0.2 | collapsible wagon
pinnied scrimmage in progress | 0.55 | 0.2-0.5 | mixed sizes
ball bag spilled at the bench | 0.55 | 0.05-0.2 | numbered balls
corner flags leaning | 0.45 | 0.03-0.1 | spring-loaded stakes
coach's whiteboard on the grass | 0.45 | 0.03-0.1 | magnet formation
toddler sibling with a spare ball | 0.45 | 0.05-0.15 | sideline dribbler
goalkeeper gloves drying on the net | 0.4 | 0.02-0.08 | velcro open
dog on a long lead watching | 0.4 | 0.05-0.15 | play-by-play tail
bikes dumped by the bench | 0.45 | 0.05-0.2 | rode-here pile
water jug with a spigot | 0.5 | 0.05-0.15 | team refills""",
            "architecture_detail": """goal mouth worn to dirt | 0.6 | 0.1-0.25 | grassless crescent
sprinkler heads flush in the turf | 0.4 | 0.02-0.08 | mow-safe caps
field crown for drainage | 0.35 | 0.1-0.3 | subtle rise
sideline rut from spectator chairs | 0.4 | 0.05-0.2 | packed strip
anchor stakes on the portable goals | 0.5 | 0.02-0.08 | safety tie-downs
mole run ridge at the corner | 0.35 | 0.03-0.12 | soft raised line
paint wand stripes overspray | 0.4 | 0.03-0.1 | green-on-white fuzz
bench of splintered planks | 0.45 | 0.05-0.15 | team roost
trash barrel chained to a post | 0.45 | 0.03-0.1 | rolled rim
gopher hole flagged with a cone | 0.35 | 0.02-0.08 | ankle warning
lot gravel migrating into the grass | 0.4 | 0.03-0.12 | edge scatter""",
            "props": """orange slice bag in the cooler | 0.5 | 0.02-0.08 | halftime rite
shin guards strapped outside socks | 0.5 | 0.02-0.06 | backwards kid
captain band on a sleeve | 0.4 | 0.02-0.05 | stretched elastic
clipboard roster in a gust | 0.4 | 0.02-0.06 | pages pinned
juice pouches in a mesh bag | 0.45 | 0.02-0.08 | straw punch-outs
spare cleats tied by the laces | 0.45 | 0.02-0.06 | grown-out pair
bug spray passed down the chairs | 0.45 | 0.02-0.06 | dusk defense
umbrella clamped to a chair | 0.4 | 0.03-0.1 | sun angle set
scorebook kept by a parent | 0.4 | 0.02-0.06 | pencil tally
ball pump needle kit | 0.4 | 0.02-0.05 | pocket rescue
sweatshirt goalposts for warmups | 0.4 | 0.03-0.1 | improvised drill""",
            "foreground_element": """throw-in taken over the head | 0.55 | 0.05-0.2 | feet planted
cleats stamped clean on the curb | 0.5 | 0.03-0.1 | mud clods
water bottle squirted over the face | 0.5 | 0.03-0.1 | cooldown blast
high five line after the final whistle | 0.5 | 0.1-0.3 | good-game train
chair unfolded with a snap | 0.5 | 0.05-0.15 | sideline setup
juice pouch stabbed with a straw | 0.45 | 0.02-0.08 | victory sip
shin guard velcro ripped open | 0.45 | 0.02-0.08 | post-game relief
ball juggled counting touches | 0.5 | 0.05-0.15 | personal best try
coach crouched for the huddle | 0.45 | 0.1-0.25 | knee in the grass
wagon loaded for the walk back | 0.45 | 0.05-0.2 | gear tetris
sunset squinted into from the lot | 0.45 | 0.05-0.2 | drive-home light""",
            "time_of_day": OUTDOOR_PA_TIME,
            "weather": OUTDOOR_PA_WEATHER,
        },
    },
}


def write_file(path: pathlib.Path, text: str, force: bool) -> str:
    if path.exists() and not force:
        return "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "write"


# Indoor banlist from tests/unit/test_location_lists_extended.py — substring
# match, so "fridge"/"bridge" trip "ridge".
INDOOR_BANNED = ["trail", "ridge", "mountain", "forest canopy", "ocean",
                 "beach", "appalachian", "summit", "wilderness", "switchback"]
OUTDOOR_BANNED = ["hardwood floor", "sofa", "office desk", "kitchen counter",
                  "duvet", "shower stall", "bath tub", "indoor pool"]


def validate(category: str, set_name: str, element: str, body: str) -> list[str]:
    problems = []
    lines = [ln for ln in body.strip().splitlines() if ln.strip()]
    if len(lines) < 10:
        problems.append(f"only {len(lines)} entries")
    names = []
    banned = INDOOR_BANNED if category.startswith("indoor") else OUTDOOR_BANNED
    atmosphere = element in ("time_of_day", "weather")
    for ln in lines:
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) != 4:
            problems.append(f"bad column count: {ln!r}")
            continue
        name, prob = parts[0], parts[1]
        names.append(name)
        try:
            p = float(prob)
            if not 0.05 <= p <= 1.0:
                problems.append(f"probability {p} out of range: {name!r}")
        except ValueError:
            problems.append(f"bad probability: {ln!r}")
        if not atmosphere and len(name.split()) < 2:
            problems.append(f"one-word name: {name!r}")
        low = name.lower()
        for b in banned:
            if b in low:
                problems.append(f"banlist '{b}' in {name!r}")
    if len(names) != len(set(names)):
        dupes = {n for n in names if names.count(n) > 1}
        problems.append(f"duplicates: {dupes}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ok = True
    written = skipped = 0
    for category, sets in SETS.items():
        for set_name, elements in sets.items():
            missing = [e for e in ELEMENTS if e not in elements]
            if missing:
                print(f"ERROR {category}/{set_name}: missing {missing}")
                ok = False
                continue
            for element, body in elements.items():
                problems = validate(category, set_name, element, body)
                if problems:
                    ok = False
                    for p in problems:
                        print(f"ERROR {category}/{set_name}/{element}: {p}")
    if not ok:
        return 1

    for category, sets in SETS.items():
        for set_name, elements in sets.items():
            target = ROOT / "location_lists" / pathlib.Path(category) / set_name
            for element, body in elements.items():
                result = write_file(target / f"{element}.txt",
                                    HEADER[element] + body.strip() + "\n",
                                    args.force)
                written += result == "write"
                skipped += result == "skip"

    total = sum(len(v) for v in SETS.values())
    print(f"locations: {written} written, {skipped} skipped ({total} sets)")
    if skipped:
        print("Use --force to overwrite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
