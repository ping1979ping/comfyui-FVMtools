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
    "indoor/office_us": {
        "cubicle_pod": {
            "background": """grey fabric cubicle walls | 1.0 | 0.5-0.9 | pin-pricked woven panels
row of cubicles down the aisle | 0.85 | 0.4-0.7 | repeating grey partitions
drop ceiling with light panels | 0.8 | 0.2-0.5 | stained acoustic tiles
office window with vertical blinds | 0.55 | 0.2-0.5 | dusty tilted slats
whiteboard wall in the distance | 0.5 | 0.2-0.4 | half-erased marker
supply cabinet against the wall | 0.5 | 0.15-0.4 | beige steel doors
printer station along the aisle | 0.55 | 0.2-0.4 | paper trays and toner boxes
motivational poster in a frame | 0.4 | 0.1-0.25 | faded stock photo
conference room glass wall beyond | 0.45 | 0.2-0.5 | fingerprinted glass
coat hooks with hanging jackets | 0.45 | 0.1-0.3 | slumped fabric
exit sign over the far door | 0.4 | 0.05-0.15 | red glowing letters""",
            "midground": """desk with two monitors | 1.0 | 0.3-0.6 | smudged dark screens
wheeled office chair | 0.9 | 0.2-0.5 | worn mesh back
neighboring desk with clutter | 0.7 | 0.2-0.5 | stacked folders and mugs
rolling file cabinet under the desk | 0.65 | 0.1-0.3 | dented beige steel
desk phone with coiled cord | 0.6 | 0.05-0.2 | dusty handset
recycling bin by the partition | 0.5 | 0.05-0.2 | overfilled paper
space heater under the desk | 0.4 | 0.05-0.15 | scorched grille
box of copy paper on the floor | 0.5 | 0.1-0.25 | half-opened carton
keyboard tray pulled out | 0.6 | 0.1-0.3 | crumb-dusted keys
second chair pulled over | 0.4 | 0.15-0.35 | borrowed from the next pod
water bottle collection on the desk | 0.5 | 0.05-0.15 | mixed plastic and steel""",
            "architecture_detail": """carpet tiles in mixed grey | 0.85 | 0.3-0.6 | coffee-stained squares
cable grommet in the desk | 0.55 | 0.03-0.1 | cords bunched through
power strip along the baseboard | 0.55 | 0.05-0.15 | daisy-chained plugs
partition rail with name plate | 0.5 | 0.03-0.12 | sliding plastic tag
vent grille in the ceiling | 0.5 | 0.05-0.15 | dust-streaked louvers
scuffed corner guard | 0.4 | 0.05-0.15 | chipped plastic edge
fluorescent tube flickering | 0.35 | 0.05-0.15 | uneven pale light
thermostat box on the column | 0.4 | 0.03-0.1 | locked plastic cover
ethernet jack plate low on the wall | 0.4 | 0.02-0.08 | numbered ports
ceiling tile with a water stain | 0.45 | 0.05-0.15 | brown ringed corner
carpet seam lifting at the aisle | 0.35 | 0.05-0.12 | frayed edge""",
            "props": """sticky notes on the monitor edge | 0.85 | 0.03-0.1 | curled yellow squares
coffee mug with a company logo | 0.75 | 0.03-0.1 | ring-stained ceramic
lanyard and badge on the desk | 0.7 | 0.02-0.08 | printed name card
takeout container in the bin | 0.5 | 0.03-0.1 | grease-marked cardboard
family photo in a cheap frame | 0.55 | 0.02-0.08 | glossy print
stress ball by the keyboard | 0.45 | 0.02-0.06 | squeezed foam
tangle of charging cables | 0.6 | 0.03-0.1 | knotted white and black
half-eaten bag of pretzels | 0.5 | 0.02-0.08 | rolled-down bag
desk calendar still on last month | 0.45 | 0.02-0.08 | doodled margins
hand sanitizer pump bottle | 0.5 | 0.02-0.06 | crusted nozzle
birthday card standing open | 0.35 | 0.02-0.08 | signed in ballpoint""",
            "foreground_element": """spinning slightly in the chair | 0.6 | 0.1-0.3 | hand on the armrest
coffee mug lifted mid-sip | 0.6 | 0.05-0.15 | steam over the rim
leaning on the cubicle wall to chat | 0.6 | 0.1-0.3 | arms folded on the rail
headset pushed back off one ear | 0.5 | 0.05-0.15 | foam pad askew
phone checked below desk level | 0.5 | 0.05-0.15 | screen glow on the lap
lanyard hanging over the shirt | 0.55 | 0.05-0.15 | swinging badge
stretching back from the keyboard | 0.5 | 0.1-0.3 | arms behind the head
stack of folders carried past | 0.45 | 0.1-0.25 | papers sliding
sticky note held up to read | 0.4 | 0.03-0.1 | scribbled reminder
cardigan pulled against the AC | 0.5 | 0.1-0.3 | sleeves pushed up
birthday cake slice on a paper plate | 0.35 | 0.05-0.12 | plastic fork""",
            "time_of_day": INDOOR_PUBLIC_TIME,
            "weather": INDOOR_PUBLIC_WEATHER,
        },
        "break_room": {
            "background": """break room wall with notices | 1.0 | 0.5-0.9 | cork board and OSHA posters
counter with coffee maker | 0.9 | 0.3-0.6 | stained laminate
vending machines side by side | 0.75 | 0.3-0.6 | lit glass fronts
cabinets over the counter | 0.7 | 0.3-0.6 | mismatched laminate doors
small window onto the parking lot | 0.5 | 0.15-0.4 | dusty mini blinds
sink with a drying rack | 0.65 | 0.15-0.4 | water-spotted steel
wall clock over the door | 0.55 | 0.05-0.15 | plain white face
white refrigerator with magnets | 0.8 | 0.2-0.5 | passive-aggressive notes taped up
microwave on the counter | 0.75 | 0.1-0.3 | splattered interior glass
bulletin board with sign-up sheets | 0.5 | 0.15-0.35 | overlapping flyers
water cooler in the corner | 0.6 | 0.1-0.3 | blue five-gallon jug""",
            "midground": """round table with mismatched chairs | 1.0 | 0.3-0.6 | wiped laminate top
second table pushed to the wall | 0.6 | 0.2-0.5 | folding legs
coffee pot half full on the burner | 0.8 | 0.05-0.2 | scorched glass carafe
paper towel roll on the counter | 0.65 | 0.05-0.15 | torn edge hanging
trash can with a swinging lid | 0.6 | 0.1-0.25 | overfull liner
box of donuts left open | 0.5 | 0.05-0.15 | one left in the corner
chair pulled out at an angle | 0.6 | 0.15-0.35 | worn vinyl seat
lost-and-found box by the door | 0.4 | 0.1-0.25 | tangled umbrellas
stack of paper cups by the machine | 0.55 | 0.03-0.1 | sleeve half gone
dish tub with abandoned mugs | 0.5 | 0.05-0.15 | soaking handles
ice machine growling in the corner | 0.4 | 0.1-0.3 | scuffed steel bin""",
            "architecture_detail": """vinyl tile floor | 0.8 | 0.3-0.6 | heel-scuffed squares
drop ceiling with a humming light | 0.7 | 0.15-0.4 | flickering corner panel
backsplash of plain white tile | 0.55 | 0.1-0.3 | coffee-splashed grout
outlet crowded with plugs | 0.55 | 0.03-0.1 | adapter stack
door with a push plate | 0.45 | 0.1-0.25 | worn steel plate
baseboard peeling at the corner | 0.4 | 0.03-0.12 | lifted vinyl strip
hand-washing sign over the sink | 0.5 | 0.03-0.1 | laminated printout
thermostat with a plastic lockbox | 0.4 | 0.03-0.1 | scratched cover
ceiling vent stained grey | 0.4 | 0.05-0.12 | dust-furred louvers
window sill with dead flies | 0.3 | 0.03-0.1 | chipped paint
fire extinguisher in a wall bracket | 0.45 | 0.03-0.1 | inspection tag hanging""",
            "props": """sugar packets spilled by the machine | 0.6 | 0.02-0.08 | torn paper packets
creamer tubs in a bowl | 0.6 | 0.02-0.08 | mixed flavors
someone's labeled lunch bag | 0.6 | 0.03-0.1 | sharpie name
crossword folded on the table | 0.45 | 0.03-0.1 | half-filled squares
birthday sheet cake remains | 0.4 | 0.05-0.15 | frosting-smeared board
mug tree with novelty mugs | 0.5 | 0.03-0.1 | mismatched handles
plastic cutlery cup | 0.55 | 0.02-0.06 | forks nearly out
salt and pepper shakers | 0.45 | 0.02-0.06 | rice grains in the salt
menu stack from the pizza place | 0.5 | 0.02-0.08 | dog-eared corners
tupperware drying upside down | 0.5 | 0.02-0.08 | warped lids
coffee sign-up sheet on the board | 0.4 | 0.02-0.08 | mostly blank lines""",
            "foreground_element": """stirring coffee with a plastic stick | 0.7 | 0.05-0.15 | swirling creamer
microwave door held open | 0.55 | 0.05-0.2 | steam rolling out
lunch container peeled open | 0.6 | 0.05-0.15 | fogged lid in hand
leaning on the counter mid-story | 0.6 | 0.1-0.3 | ankles crossed
vending machine buttons pressed | 0.5 | 0.05-0.2 | coil turning
paper plate carried to the table | 0.5 | 0.05-0.15 | sagging under a slice
phone scrolled while chewing | 0.55 | 0.05-0.15 | elbow on the table
mug rinsed at the sink | 0.5 | 0.05-0.15 | water running
donut box lid lifted for a look | 0.45 | 0.05-0.12 | glaze stuck to cardboard
chair tipped back on two legs | 0.4 | 0.1-0.25 | hand on the table edge
ice rattled into a cup | 0.4 | 0.03-0.1 | scoop in the bin""",
            "time_of_day": INDOOR_PUBLIC_TIME,
            "weather": INDOOR_PUBLIC_WEATHER,
        },
        "conference_room_plain": {
            "background": """long laminate conference table | 1.0 | 0.4-0.8 | wood-look top with cable box
whiteboard covered in old notes | 0.85 | 0.3-0.6 | ghosted marker layers
projection screen pulled halfway | 0.6 | 0.25-0.5 | wrinkled matte surface
glass wall to the hallway | 0.6 | 0.3-0.6 | frosted privacy band
wall-mounted flatscreen | 0.65 | 0.2-0.45 | cable dangling below
credenza along the back wall | 0.5 | 0.2-0.45 | stacked handouts
window with vertical blinds | 0.55 | 0.2-0.5 | half-tilted slats
company plaque wall | 0.4 | 0.15-0.35 | dusty framed awards
speakerphone in the table center | 0.6 | 0.05-0.15 | grey three-legged unit
flip chart easel in the corner | 0.45 | 0.1-0.3 | curled used pages
clock above the door | 0.5 | 0.05-0.12 | five minutes fast""",
            "midground": """rolling chairs around the table | 1.0 | 0.3-0.6 | mismatched heights
chair left turned from the table | 0.6 | 0.15-0.35 | seat still dented
laptop open at one seat | 0.6 | 0.1-0.25 | lid stickers
power strip snake across the table | 0.55 | 0.05-0.2 | taped-down cord
water pitcher on a tray | 0.45 | 0.05-0.15 | sweating glass
notepad stack at the center | 0.5 | 0.05-0.15 | company letterhead
markers in the whiteboard tray | 0.6 | 0.03-0.1 | most dried out
trash can by the credenza | 0.45 | 0.05-0.15 | coffee cups inside
spare chair against the wall | 0.5 | 0.15-0.35 | odd one out
projector cart wheeled aside | 0.4 | 0.15-0.3 | cable coiled on top
box of pastries on the credenza | 0.4 | 0.05-0.15 | picked-over selection""",
            "architecture_detail": """carpet with a dark spill patch | 0.6 | 0.1-0.3 | blotted stain
drop ceiling with a projector mount | 0.6 | 0.1-0.3 | empty bracket
light switch bank by the door | 0.5 | 0.02-0.08 | taped labels
floor box with flip-up lid | 0.5 | 0.03-0.1 | tangled cords inside
scuff marks from chair backs | 0.5 | 0.05-0.2 | rubbed wall paint
air vent whistling faintly | 0.4 | 0.03-0.1 | bent louver
door stop wedge on the floor | 0.4 | 0.02-0.06 | worn rubber
window sill with a dead plant | 0.35 | 0.05-0.15 | crisp brown leaves
cable cover strip along the floor | 0.45 | 0.05-0.15 | yellowed plastic ramp
thermostat argued over daily | 0.4 | 0.02-0.08 | fingerprinted cover
sign-out sheet holder on the door | 0.35 | 0.02-0.08 | plastic wall pocket""",
            "props": """dry erase eraser on the tray | 0.55 | 0.02-0.06 | marker-blackened felt
abandoned coffee cups at seats | 0.6 | 0.03-0.1 | varying fill levels
agenda printouts at each place | 0.5 | 0.05-0.15 | corner-stapled pages
remote for the flatscreen | 0.5 | 0.02-0.06 | taped battery cover
sticky notes stuck to the table | 0.45 | 0.02-0.08 | action items scrawled
hdmi dongle bowl | 0.45 | 0.02-0.08 | adapter tangle
name tent cards half collapsed | 0.4 | 0.03-0.1 | folded cardstock
phone face down mid-meeting | 0.5 | 0.02-0.06 | buzzing against wood
laser pointer without batteries | 0.35 | 0.02-0.05 | chewed cap
pizza boxes from a lunch meeting | 0.4 | 0.05-0.15 | grease-spotted lids
umbrella hooked on a chair | 0.35 | 0.03-0.1 | still damp""",
            "foreground_element": """marker uncapped at the whiteboard | 0.65 | 0.05-0.15 | mid-diagram
laptop turned to show the screen | 0.55 | 0.1-0.25 | glare on the display
chair swiveled toward the window | 0.5 | 0.1-0.3 | listening posture
handout passed across the table | 0.5 | 0.05-0.15 | reaching arms
water poured into a paper cup | 0.45 | 0.03-0.1 | pitcher tilted
notes taken on a legal pad | 0.55 | 0.05-0.15 | pen mid-line
sleeve pushed up checking a watch | 0.45 | 0.03-0.1 | subtle glance
phone held under the table edge | 0.45 | 0.03-0.1 | thumb scrolling
leaning back with hands laced | 0.5 | 0.1-0.3 | chair tilted
donut balanced on a napkin | 0.4 | 0.03-0.1 | sugar dusting
cord untangled for the projector | 0.4 | 0.05-0.15 | adapter hunt""",
            "time_of_day": INDOOR_PUBLIC_TIME,
            "weather": INDOOR_PUBLIC_WEATHER,
        },
    },
    "indoor/fitness_us": {
        "strip_mall_gym": {
            "background": """mirrored wall over the dumbbell rack | 1.0 | 0.4-0.8 | smudged floor-length mirrors
row of treadmills facing TVs | 0.85 | 0.4-0.7 | worn belts and consoles
motivational decal on the wall | 0.6 | 0.2-0.4 | peeling vinyl lettering
cable machine tower | 0.7 | 0.25-0.5 | worn pulleys and pins
front desk with a check-in scanner | 0.5 | 0.2-0.4 | laminate counter
storefront glass papered halfway | 0.5 | 0.25-0.5 | sun-faded posters
wall-mounted fans high up | 0.5 | 0.1-0.25 | dusty cages
group class room behind glass | 0.45 | 0.2-0.45 | stacked step platforms
rack of fixed barbells | 0.6 | 0.2-0.4 | chipped chrome sleeves
TVs on a sports channel | 0.55 | 0.1-0.3 | captions on
water fountain with bottle filler | 0.5 | 0.1-0.25 | counter ticking""",
            "midground": """dumbbell rack with gaps | 0.9 | 0.25-0.5 | mismatched pairs
flat bench with torn vinyl | 0.75 | 0.2-0.4 | taped corner rip
squat rack with plates loaded | 0.65 | 0.25-0.5 | rusted collar pins
plate tree half empty | 0.6 | 0.15-0.35 | scattered change plates
rowing machine pushed to the wall | 0.5 | 0.2-0.4 | dusty rail
stability balls in a corral | 0.45 | 0.15-0.3 | half-deflated one
someone mid-set at the cables | 0.5 | 0.15-0.35 | blurred motion
mat area with stretching space | 0.6 | 0.25-0.5 | worn blue mats
spray bottle and towel station | 0.55 | 0.05-0.15 | drip-stained floor
elliptical with an out-of-order sign | 0.4 | 0.15-0.3 | taped paper note
kettlebells lined by the mirror | 0.5 | 0.1-0.25 | chipped paint handles""",
            "architecture_detail": """rubber tile flooring | 0.85 | 0.3-0.6 | chalk-dusted black tiles
drop ceiling with new and old panels | 0.55 | 0.15-0.35 | mismatched whites
exposed conduit along the wall | 0.45 | 0.05-0.2 | painted-over runs
floor-to-ceiling column padded | 0.45 | 0.1-0.25 | taped foam wrap
AC vent blowing on one machine | 0.4 | 0.05-0.15 | ribbon fluttering
baseboard scuffed by plates | 0.45 | 0.05-0.15 | dented drywall above
mirror seam slightly misaligned | 0.4 | 0.05-0.2 | doubled reflection line
emergency exit with alarm bar | 0.4 | 0.05-0.15 | warning sticker
ceiling fan on a long downrod | 0.4 | 0.05-0.15 | wobbling slowly
platform edge with worn tape | 0.4 | 0.05-0.15 | frayed gaffer layers
clock cage over the class room | 0.35 | 0.03-0.1 | wire guard""",
            "props": """chalk bucket by the racks | 0.5 | 0.03-0.1 | dusted rim
lifting belt hung on a peg | 0.5 | 0.03-0.1 | cracked leather
resistance bands on a hook | 0.55 | 0.03-0.1 | faded colors
someone's keys and phone on the floor | 0.55 | 0.02-0.08 | screen face up
shaker bottle on a bench | 0.6 | 0.02-0.08 | protein residue
gym wipes canister empty | 0.45 | 0.02-0.08 | flap hanging open
lost earbud case on the desk | 0.35 | 0.02-0.05 | taped label
foam roller against the wall | 0.55 | 0.03-0.12 | dented ridges
jump rope coiled on a peg | 0.45 | 0.02-0.08 | worn handles
clip collars scattered | 0.5 | 0.02-0.06 | spring-loaded pairs
sweat towel draped on a rack | 0.5 | 0.03-0.1 | gym logo faded""",
            "foreground_element": """dumbbells re-racked with a clank | 0.6 | 0.1-0.25 | knurled handles
water bottle squeezed mid-gulp | 0.6 | 0.05-0.15 | condensation ring
phone timer checked between sets | 0.6 | 0.05-0.15 | rest countdown
towel wiped across the forehead | 0.55 | 0.05-0.15 | damp terry
mirror check of the form | 0.55 | 0.1-0.3 | focused stare
headphones adjusted over a cap | 0.5 | 0.05-0.15 | cord tucked
bench wiped down with spray | 0.5 | 0.05-0.15 | paper towel pass
plates slid onto the bar | 0.5 | 0.1-0.25 | collar spun tight
laces retied on one knee | 0.45 | 0.05-0.15 | double knot
gym bag unzipped on the floor | 0.5 | 0.05-0.2 | straps splayed
chalk clapped off the palms | 0.4 | 0.05-0.15 | drifting dust""",
            "time_of_day": INDOOR_PUBLIC_TIME,
            "weather": INDOOR_PUBLIC_WEATHER,
        },
        "community_yoga_studio": {
            "background": """plain studio wall in warm grey | 1.0 | 0.5-0.9 | matte paint with scuffs
wall of mirrors on one side | 0.75 | 0.3-0.6 | streak-cleaned glass
shelf of rolled mats | 0.7 | 0.2-0.45 | mixed colors and wear
big front window with paper shade | 0.55 | 0.2-0.5 | strip-mall daylight
corner altar shelf with a plant | 0.5 | 0.1-0.3 | pothos trailing down
coat hooks by the door | 0.55 | 0.1-0.3 | jackets and tote bags
cubby shelf full of shoes | 0.6 | 0.15-0.35 | sneakers toed in
string lights along the ceiling edge | 0.45 | 0.1-0.3 | warm dots
class schedule board by the door | 0.5 | 0.1-0.25 | dry-erase grid
folded blankets in a stack | 0.55 | 0.1-0.3 | wool edges aligned
speaker on a stand in the corner | 0.45 | 0.05-0.15 | cable taped down""",
            "midground": """mats unrolled in loose rows | 1.0 | 0.4-0.7 | staggered spacing
cork blocks paired at mat tops | 0.7 | 0.1-0.3 | dented corners
straps coiled beside mats | 0.6 | 0.05-0.15 | buckle ends
bolster propped against the wall | 0.55 | 0.1-0.3 | flattened middle
instructor mat facing the room | 0.6 | 0.15-0.3 | centered alone
water bottles at mat corners | 0.65 | 0.05-0.15 | assorted steel and plastic
folding chair for a modified pose | 0.35 | 0.1-0.25 | paint-chipped seat
fan on a stand oscillating | 0.45 | 0.1-0.25 | ribbon tied to the cage
diffuser puffing in the corner | 0.4 | 0.05-0.15 | mist curl
someone settled early in child's pose | 0.45 | 0.1-0.3 | still shape
towels folded at the mat ends | 0.5 | 0.05-0.15 | rental stack""",
            "architecture_detail": """laminate floor in pale wood-look | 0.85 | 0.3-0.6 | sock-polished sheen
baseboard heater along the wall | 0.5 | 0.05-0.2 | ticking metal fins
dimmer switch bank by the door | 0.5 | 0.02-0.08 | taped preset marks
ceiling fan turning slowly | 0.5 | 0.05-0.15 | wobble at speed one
drop ceiling painted out black | 0.4 | 0.15-0.4 | sprayed grid
door with a quiet-close arm | 0.4 | 0.05-0.15 | slow hinge
window film bubbling at a corner | 0.35 | 0.05-0.15 | peeling frost layer
column wrapped in rope light | 0.35 | 0.05-0.15 | warm coil
vent hushed to a whisper | 0.4 | 0.03-0.1 | filter taped over
threshold strip at the entry | 0.4 | 0.02-0.08 | worn aluminum
wall clock with a silent sweep | 0.45 | 0.03-0.08 | no ticking""",
            "props": """essential oil bottle on the shelf | 0.45 | 0.02-0.06 | lavender label
tissue box at the room edge | 0.45 | 0.02-0.06 | one sheet up
hair ties looped on a wrist basket | 0.4 | 0.02-0.05 | mixed elastics
eye pillows in a bowl | 0.45 | 0.02-0.08 | flaxseed sachets
sign-in tablet on a stand | 0.5 | 0.02-0.08 | fingerprinted screen
hand weights in a corner basket | 0.4 | 0.05-0.12 | one and two pounders
lost-and-found bin of water bottles | 0.4 | 0.03-0.1 | orphaned lids
chalkboard with the day's intention | 0.4 | 0.03-0.1 | hand-lettered word
spray bottles of mat cleaner | 0.5 | 0.02-0.08 | tea tree scent
donation basket by the door | 0.35 | 0.02-0.08 | folded bills
phone basket asking for silence | 0.4 | 0.03-0.08 | screens face down""",
            "foreground_element": """mat unrolled with a flick | 0.6 | 0.1-0.3 | settling flat
socks pulled off at the ankle | 0.55 | 0.05-0.15 | balanced on one foot
strap looped around a foot | 0.5 | 0.05-0.2 | stretch in progress
water bottle set down quietly | 0.5 | 0.03-0.1 | careful placement
hair twisted up into a knot | 0.55 | 0.05-0.15 | elastic between teeth
block reached for mid-pose | 0.5 | 0.05-0.15 | fingertips on cork
blanket folded over the knees | 0.45 | 0.1-0.25 | settling for savasana
hoodie unzipped before class | 0.5 | 0.1-0.25 | layer coming off
phone silenced and tucked in a bag | 0.45 | 0.03-0.1 | last check
wrists stretched against the floor | 0.45 | 0.05-0.15 | fingers spread
mat wiped down after class | 0.5 | 0.05-0.15 | spray and cloth pass""",
            "time_of_day": INDOOR_PUBLIC_TIME,
            "weather": INDOOR_PUBLIC_WEATHER,
        },
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
    "indoor/private_us": {
        "master_bedroom_plain": {
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
        "bathroom_tub_combo": {
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
        "walk_in_closet_everyday": {
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
    "indoor/vacation_us": {
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
        "poconos_cabin_living": {
            "background": """knotty pine walls | 1.0 | 0.5-0.9 | orange-toned tongue and groove
wood stove on a slate pad | 0.8 | 0.2-0.4 | black steel with a glass door
plaid couch with sagging cushions | 0.8 | 0.3-0.6 | flattened arms
antler decor over the mantel | 0.5 | 0.1-0.3 | dusty mount
picture window to the trees | 0.7 | 0.3-0.6 | wavy old glass
board game shelf | 0.6 | 0.15-0.35 | taped box corners
braided rug on plank floor | 0.65 | 0.25-0.5 | concentric ovals
kitchenette along the back wall | 0.55 | 0.25-0.5 | two-burner stove
ladder to a sleeping loft | 0.5 | 0.15-0.35 | worn rungs
mismatched recliners | 0.55 | 0.2-0.4 | duct-taped corner
firewood rack inside the door | 0.6 | 0.1-0.3 | split logs and kindling""",
            "midground": """coffee table with puzzle in progress | 0.7 | 0.2-0.4 | edge pieces done
quilt thrown over the couch back | 0.65 | 0.15-0.35 | hand-stitched squares
space heater as backup | 0.45 | 0.05-0.15 | glowing coils
card table set for a game | 0.5 | 0.15-0.35 | folding chairs pulled up
cooler by the kitchenette | 0.5 | 0.1-0.25 | melted ice sloshing
duffel bags along the wall | 0.55 | 0.1-0.3 | weekend packing
lantern on the windowsill | 0.45 | 0.03-0.1 | battery powered
dog stretched by the stove | 0.4 | 0.1-0.3 | twitching asleep
snow boots drying on a mat | 0.45 | 0.05-0.15 | puddle ring
marshmallow bag on the counter | 0.4 | 0.02-0.08 | clip sealed
guitar leaned in the corner | 0.4 | 0.1-0.25 | missing a string""",
            "architecture_detail": """exposed beam ceiling | 0.7 | 0.25-0.5 | hand-hewn marks
stone chimney face | 0.6 | 0.2-0.4 | mortar smears
plank floor gaps | 0.55 | 0.15-0.4 | draft lines
gas lamp fixture converted | 0.4 | 0.03-0.1 | wired sconce
window frames swollen shut | 0.4 | 0.05-0.15 | painted sashes
stovepipe through the ceiling plate | 0.55 | 0.05-0.15 | heat-discolored ring
door with a wooden latch | 0.4 | 0.05-0.12 | leather pull
chinking cracked between logs | 0.4 | 0.05-0.2 | patched seams
loft rail of peeled poles | 0.45 | 0.1-0.25 | varnish worn
thermometer nailed by the door | 0.45 | 0.02-0.06 | analog dial
outlet strip added along the base | 0.4 | 0.02-0.08 | modern retrofit""",
            "props": """flannel shirt on a peg | 0.55 | 0.05-0.15 | buffalo check
percolator on the burner | 0.5 | 0.03-0.1 | blue enamel
deck of cards mid-shuffle | 0.5 | 0.02-0.08 | worn corners
firewood carrier of canvas | 0.5 | 0.03-0.1 | bark crumbs
matches in a mason jar | 0.45 | 0.02-0.06 | strike strip glued
paperback westerns on the shelf | 0.4 | 0.03-0.1 | yellowed pages
bug spray by the door | 0.45 | 0.02-0.06 | greasy nozzle
s'more sticks bundled | 0.45 | 0.02-0.08 | whittled points
crossword book with a pen | 0.4 | 0.02-0.08 | half done
board game money sorted | 0.4 | 0.03-0.1 | banker's tray
wool socks drying on the stove rail | 0.45 | 0.02-0.08 | steaming faintly""",
            "foreground_element": """log settled into the stove | 0.6 | 0.1-0.25 | sparks rising
quilt pulled over the knees | 0.6 | 0.1-0.3 | couch corner claim
cocoa mug held in both hands | 0.55 | 0.05-0.15 | marshmallow melt
puzzle piece tried and rejected | 0.5 | 0.05-0.15 | close but wrong
cards dealt around the table | 0.5 | 0.1-0.25 | flicked precisely
boots unlaced at the door | 0.5 | 0.05-0.2 | heel wedged off
window fog wiped for a look | 0.45 | 0.05-0.15 | palm circle
guitar strummed quietly | 0.4 | 0.1-0.25 | half-remembered song
flannel sleeves rolled up | 0.5 | 0.05-0.2 | stove-side warmth
dog ears scratched absently | 0.45 | 0.05-0.15 | thumping tail
phone raised for a no-signal check | 0.45 | 0.03-0.1 | one bar hunt""",
            "time_of_day": INDOOR_HOME_TIME,
            "weather": INDOOR_HOME_WEATHER,
        },
        "shore_rental_kitchen": {
            "background": """white cabinets with worn pulls | 1.0 | 0.4-0.8 | repainted layers
window over the sink with gauzy curtain | 0.8 | 0.2-0.45 | shell-print fabric
laminate counters in speckled grey | 0.8 | 0.3-0.6 | sun-faded patches
shell decor on the sill | 0.55 | 0.05-0.15 | collected jar
seafood restaurant magnets on the icebox | 0.5 | 0.1-0.3 | crab-shaped clips
open shelf of mixed mugs | 0.55 | 0.15-0.3 | rental assortment
sliding door to the deck | 0.65 | 0.3-0.6 | salt-hazed glass
paper towel holder bolted under a cabinet | 0.5 | 0.03-0.1 | half roll
tide clock on the wall | 0.45 | 0.05-0.12 | novelty dial
rattan bar stools at the counter | 0.55 | 0.2-0.4 | unravelling wraps
ceiling fan with wicker blades | 0.5 | 0.1-0.3 | wobbling pull chain""",
            "midground": """drying rack of sandy sneakers by the door | 0.5 | 0.1-0.25 | grit trail
cooler propped open to dry | 0.55 | 0.1-0.3 | drain plug out
grocery bags half unpacked | 0.6 | 0.1-0.3 | vacation stock-up
blender out for smoothies | 0.5 | 0.05-0.15 | rinsed pitcher
corn ears stacked for dinner | 0.45 | 0.05-0.15 | husks on
sunscreen bottles on the counter | 0.6 | 0.03-0.1 | greasy caps
beat-up toaster with a bent lever | 0.5 | 0.03-0.1 | crumb tray out
folding chairs leaned by the slider | 0.5 | 0.1-0.3 | webbed seats
bucket of shells rinsed in the sink | 0.45 | 0.05-0.15 | saltwater cloudy
board shorts drying on a chair | 0.45 | 0.05-0.15 | drip line
pizza boxes from the first night | 0.45 | 0.05-0.15 | stacked flat""",
            "architecture_detail": """sand in the floor grout lines | 0.6 | 0.05-0.2 | swept but back again
slider track gritty underfoot | 0.5 | 0.03-0.1 | crunching roll
painted beadboard backsplash | 0.5 | 0.1-0.3 | chipped ridges
outlet covers hand-painted | 0.35 | 0.02-0.06 | tiny lighthouse motif
cabinet doors that swell shut | 0.4 | 0.05-0.15 | humid stick
screen door with a torn corner | 0.45 | 0.05-0.15 | patched square
rust freckles on the range hood | 0.4 | 0.03-0.1 | salt-air pitting
window crank missing its handle | 0.35 | 0.02-0.06 | pliers nearby
vinyl floor bubbled near the slider | 0.4 | 0.05-0.15 | sun-lifted seam
hooks by the door labeled with names | 0.4 | 0.03-0.1 | sharpie tags
renter binder on the counter | 0.45 | 0.02-0.08 | laminated house rules""",
            "props": """boogie board leaned in the corner | 0.5 | 0.05-0.2 | wax-scraped deck
sunglasses collection by the sink | 0.55 | 0.02-0.08 | family assortment
saltwater taffy box open | 0.45 | 0.02-0.08 | wrappers twisted
citronella candle tin | 0.45 | 0.02-0.06 | blackened wick
paper plates in a holder | 0.5 | 0.02-0.08 | wicker frame
koozies scattered on the counter | 0.5 | 0.02-0.08 | souvenir foam
crab mallet and picks in a cup | 0.4 | 0.02-0.06 | seasoned wood
ice pop wrappers balled up | 0.45 | 0.02-0.06 | sticky twist
disposable camera someone brought | 0.3 | 0.02-0.05 | wind wheel
lost flip flop under a stool | 0.45 | 0.02-0.06 | orphaned single
hose-rinsed goggles on the rack | 0.45 | 0.02-0.06 | strap tangle""",
            "foreground_element": """watermelon halved on the counter | 0.5 | 0.1-0.25 | knife mid-slice
smoothie poured into mixed cups | 0.5 | 0.05-0.15 | pastel pour
sandy feet rinsed at the door | 0.5 | 0.05-0.2 | hose through the slider
towel snapped off a shoulder | 0.45 | 0.1-0.25 | heading to the outdoor shower
sunscreen worked into a shoulder | 0.55 | 0.05-0.15 | white streaks
cooler packed with ice bags | 0.55 | 0.1-0.25 | cans buried
hat brim pushed up for a drink | 0.45 | 0.05-0.15 | tan line reveal
corn shucked into a paper bag | 0.45 | 0.05-0.15 | silk strands flying
screen door caught before the slam | 0.45 | 0.05-0.15 | practiced reflex
taffy offered around | 0.4 | 0.03-0.1 | flavor debate
phone dried against a shirt hem | 0.4 | 0.03-0.1 | splash recovery""",
            "time_of_day": INDOOR_HOME_TIME,
            "weather": INDOOR_SUMMER_WEATHER,
        },
    },
    # ═════════════════════════════ OUTDOOR ═════════════════════════════
    "outdoor/suburb_pa": {
        "front_porch_vinyl": {
            "background": """vinyl-sided house front | 1.0 | 0.5-0.9 | pale siding with seam lines
porch posts wrapped in aluminum | 0.8 | 0.2-0.45 | white column sleeves
front door with a storm door | 0.75 | 0.2-0.4 | glass and screen combo
black shutters flanking windows | 0.6 | 0.2-0.4 | plastic snap-on pairs
american flag off a bracket | 0.55 | 0.1-0.3 | folding in the breeze
street of similar houses beyond | 0.65 | 0.3-0.6 | repeating rooflines
maple tree in the front yard | 0.6 | 0.2-0.5 | shade over the walk
concrete walk to the driveway | 0.6 | 0.2-0.4 | expansion joint cracks
foundation shrubs trimmed round | 0.55 | 0.15-0.35 | mulch bed edging
porch ceiling painted white | 0.5 | 0.15-0.35 | bare bulb fixture
house numbers by the door | 0.5 | 0.03-0.1 | brushed metal digits""",
            "midground": """two rocking chairs on the porch | 0.7 | 0.2-0.45 | weathered resin
welcome mat at the door | 0.65 | 0.05-0.15 | bristle letters worn
planters of seasonal flowers | 0.6 | 0.1-0.3 | petunias spilling
package left by the door | 0.55 | 0.05-0.15 | taped brown box
porch swing on chains | 0.45 | 0.2-0.4 | drifting slightly
kids' chalk buckets on the step | 0.45 | 0.05-0.15 | rainbow stubs
garden hose coiled by the spigot | 0.55 | 0.1-0.25 | kinked green loops
recycling bin at the curb | 0.5 | 0.1-0.25 | lid flipped back
mail sticking from the box | 0.5 | 0.03-0.1 | folded flyers
cat watching from the rail | 0.4 | 0.05-0.15 | tail flicking
wind chime in the corner | 0.45 | 0.03-0.1 | tangled strings""",
            "architecture_detail": """porch floorboards painted grey | 0.7 | 0.25-0.5 | worn traffic path
downspout at the porch corner | 0.55 | 0.05-0.15 | splash block below
storm door closer arm | 0.45 | 0.02-0.08 | hissing cylinder
doorbell with a camera | 0.5 | 0.02-0.06 | glowing ring
porch light with a bug halo | 0.45 | 0.03-0.1 | yellowed globe
step edge painted with grit strip | 0.45 | 0.05-0.12 | anti-slip band
lattice skirt under the porch | 0.45 | 0.1-0.25 | diamond gaps
gutter guard sticking up | 0.35 | 0.03-0.1 | lifted mesh
hose spigot dripping | 0.4 | 0.02-0.06 | rust streak below
settlement crack in the walk | 0.4 | 0.05-0.15 | ant hill in the joint
storm door sticker faded | 0.35 | 0.02-0.06 | security company logo""",
            "props": """amazon box tucked behind a planter | 0.5 | 0.03-0.1 | driver's hiding spot
citronella candle on the rail | 0.45 | 0.02-0.08 | melted center
kid's bike dropped on the walk | 0.5 | 0.1-0.25 | wheel still spinning
watering can by the planters | 0.5 | 0.03-0.1 | faded plastic
flag football left on the lawn | 0.45 | 0.03-0.1 | dew-damp
car keys and sunglasses on the rail | 0.4 | 0.02-0.06 | quick errand pause
halloween or seasonal decor bin | 0.35 | 0.05-0.15 | between-holidays limbo
iced tea glass sweating on the arm | 0.45 | 0.02-0.08 | lemon wedge
newspaper in a plastic sleeve | 0.4 | 0.02-0.06 | driveway toss
dog leash hooked by the door | 0.45 | 0.02-0.08 | walk-ready
solar path lights along the walk | 0.5 | 0.05-0.15 | tilted stakes""",
            "foreground_element": """screen door held with a hip | 0.6 | 0.1-0.25 | groceries balanced
rocking chair set moving | 0.55 | 0.1-0.3 | heel push
wave to a passing neighbor | 0.55 | 0.05-0.2 | raised hand
package scooped off the mat | 0.5 | 0.05-0.2 | label glance
mail flipped through | 0.5 | 0.05-0.15 | junk sorted out
hose aimed at the planters | 0.5 | 0.1-0.25 | thumb spray
sandals kicked off at the door | 0.5 | 0.05-0.15 | toe hook
chalk drawing admired | 0.4 | 0.1-0.25 | crouched down
flag straightened on its pole | 0.4 | 0.05-0.15 | bracket adjust
iced tea sipped on the step | 0.5 | 0.05-0.2 | elbow on knee
porch light bulb swapped | 0.35 | 0.05-0.15 | tiptoe reach""",
            "time_of_day": OUTDOOR_PA_TIME,
            "weather": OUTDOOR_PA_WEATHER,
        },
        "backyard_deck_grill": {
            "background": """pressure-treated deck boards | 1.0 | 0.4-0.8 | greyed grain and pops
vinyl privacy fence line | 0.75 | 0.3-0.6 | white panels with green algae streak
back of the house with a slider | 0.7 | 0.3-0.6 | siding and screen door
neighbor's above-ground pool over the fence | 0.45 | 0.15-0.35 | blue wall arc
lawn stretching to the swing set | 0.6 | 0.3-0.6 | mowed stripes fading
propane grill with a worn cover half off | 0.75 | 0.15-0.35 | grease-stained shelf
deck rail with solar caps | 0.6 | 0.15-0.35 | dim post lights
umbrella table with chairs | 0.65 | 0.25-0.5 | crank umbrella tilted
shed at the yard corner | 0.5 | 0.15-0.35 | plastic ramp
utility lines crossing the sky | 0.5 | 0.1-0.3 | sagging spans
flower boxes on the rail | 0.45 | 0.1-0.25 | marigolds leggy""",
            "midground": """grill lid open with smoke | 0.7 | 0.15-0.35 | flare-up haze
cooler with the lid ajar | 0.6 | 0.1-0.25 | ice and cans
citronella torches staked | 0.45 | 0.05-0.2 | bamboo poles
kiddie pool on the lawn | 0.45 | 0.15-0.3 | grass clippings floating
lawn chairs unfolded in a loose circle | 0.6 | 0.2-0.4 | woven straps
bird feeder on a shepherd hook | 0.5 | 0.05-0.15 | squirrel-bent pole
tomato plants in buckets | 0.5 | 0.1-0.25 | staked and tied
sprinkler mid-arc | 0.45 | 0.1-0.3 | ticking spray
dog patrolling the fence line | 0.45 | 0.1-0.25 | nose down
bug zapper hanging from the eave | 0.4 | 0.03-0.1 | violet glow
trampoline with a safety net | 0.45 | 0.2-0.4 | anchor straps""",
            "architecture_detail": """deck screws backing out | 0.5 | 0.03-0.1 | proud heads
stairs to the lawn with a wobbly rail | 0.55 | 0.1-0.25 | loose baluster
hose reel bolted to the house | 0.5 | 0.05-0.15 | crank handle
dryer vent flap on the wall | 0.4 | 0.02-0.08 | lint beard
gutter downspout extension | 0.45 | 0.05-0.15 | corrugated black pipe
deck board replaced unstained | 0.45 | 0.05-0.15 | bright new plank
hose bib with a splitter | 0.4 | 0.02-0.08 | two-valve brass
paver pad under the grill | 0.45 | 0.05-0.15 | grease shadow
lattice under the deck | 0.45 | 0.1-0.25 | stored kayak behind
motion light at the corner | 0.4 | 0.02-0.08 | angled twin heads
anti-slip tape on the steps | 0.35 | 0.03-0.1 | peeling strips""",
            "props": """spatula and tongs on the grill shelf | 0.65 | 0.03-0.1 | grease sheen
paper plate stack weighted with a rock | 0.5 | 0.02-0.08 | breeze insurance
ketchup and mustard bottles | 0.55 | 0.02-0.08 | picnic table pair
bag of buns clipped shut | 0.5 | 0.02-0.08 | squished corner
wiffle bat in the grass | 0.5 | 0.03-0.1 | cracked barrel
sunscreen and bug spray on the table | 0.55 | 0.02-0.08 | family defense line
bluetooth speaker on the rail | 0.5 | 0.02-0.08 | playlist going
corn hole boards mid-lawn | 0.5 | 0.1-0.25 | bags scattered
sidewalk chalk on the patio edge | 0.4 | 0.02-0.08 | rain-melted stubs
fly swatter on a nail | 0.35 | 0.02-0.06 | bent mesh
garden gloves on the rail | 0.45 | 0.02-0.06 | dirt-crusted palms""",
            "foreground_element": """burgers flipped with a press | 0.65 | 0.1-0.25 | sizzle and drip
can cracked from the cooler | 0.6 | 0.05-0.15 | ice water shake-off
corn hole bag mid-toss | 0.5 | 0.05-0.2 | arc to the board
lawn chair settled into | 0.55 | 0.1-0.3 | strap stretch
plate shielded from a bug | 0.45 | 0.05-0.15 | wave-off
kid caught before the pool edge | 0.4 | 0.1-0.25 | scoop grab
grill lid checked with a peek | 0.55 | 0.1-0.25 | smoke release
umbrella cranked open | 0.5 | 0.05-0.2 | ratchet clicks
hose kink walked out | 0.45 | 0.05-0.15 | flow restored
tomatoes checked for ripeness | 0.45 | 0.05-0.15 | gentle squeeze
sparkler traced in the dusk | 0.35 | 0.05-0.15 | light trail""",
            "time_of_day": OUTDOOR_PA_TIME,
            "weather": OUTDOOR_PA_WEATHER,
        },
        "driveway_hoop": {
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
    "outdoor/town_pa": {
        "main_street_small_town": {
            "background": """brick storefronts in a row | 1.0 | 0.5-0.9 | painted lintels and cornices
angled parking along the curb | 0.8 | 0.3-0.6 | faded stall lines
awnings over the shop windows | 0.7 | 0.2-0.45 | striped canvas
church steeple over the roofline | 0.5 | 0.1-0.3 | white spire
banner across the street for a fair | 0.45 | 0.1-0.3 | vinyl and rope
lamp posts with hanging baskets | 0.6 | 0.1-0.3 | petunia balls
pizza shop neon sign | 0.55 | 0.1-0.25 | half-lit letters
hardware store window display | 0.5 | 0.15-0.35 | seasonal stack
crosswalk to the square | 0.55 | 0.15-0.35 | brick-inlaid stripes
war memorial on the corner green | 0.45 | 0.1-0.3 | bronze plaque and wreath
fire company sign board | 0.5 | 0.1-0.25 | bingo night letters""",
            "midground": """diagonal-parked pickups | 0.65 | 0.2-0.45 | tailgates to the walk
bench outside the bakery | 0.6 | 0.1-0.3 | dedication plaque
sandwich board on the sidewalk | 0.6 | 0.05-0.2 | daily special chalk
bike rack with two bikes | 0.5 | 0.1-0.25 | one helmet clipped
parking meters at intervals | 0.55 | 0.05-0.2 | coin-only heads
planter tubs along the curb | 0.55 | 0.1-0.25 | township-maintained
dog tied outside the coffee shop | 0.45 | 0.05-0.15 | patient sit
stroller parked by the bench | 0.4 | 0.1-0.25 | brake locked
newspaper box chained to a post | 0.45 | 0.05-0.12 | county weekly
mail truck making its loop | 0.4 | 0.1-0.25 | flashers on
teens clustered by the pizza door | 0.45 | 0.15-0.3 | backpacks dropped""",
            "architecture_detail": """sidewalk slabs heaved by roots | 0.6 | 0.1-0.3 | tilted joints
curb painted for the fire zone | 0.5 | 0.05-0.15 | yellow chipped stripe
cornerstone dated 1912 | 0.4 | 0.02-0.08 | carved numerals
transom windows over the doors | 0.45 | 0.05-0.15 | gold-leaf remnants
cellar doors in the sidewalk | 0.4 | 0.05-0.15 | steel diamond plate
brick sidewalk section from restoration | 0.45 | 0.1-0.25 | herringbone patch
utility pole layered with staples | 0.5 | 0.05-0.15 | flyer archaeology
storm drain with a painted fish | 0.4 | 0.02-0.08 | watershed stencil
alley gap between buildings | 0.45 | 0.1-0.3 | dumpster glimpse
second-story apartments over shops | 0.5 | 0.2-0.4 | AC units in windows
flag bracket on every other pole | 0.45 | 0.03-0.1 | veterans banners""",
            "props": """free little library on a post | 0.45 | 0.03-0.1 | glass-door cabinet
water bowl outside the shop door | 0.5 | 0.02-0.06 | for the dogs
chalkboard easel with ice cream flavors | 0.5 | 0.03-0.1 | hand-drawn cone
lost cat flyer taped to a pole | 0.45 | 0.02-0.06 | phone tabs cut
scout troop popcorn table | 0.35 | 0.1-0.25 | folding table and sash
bike leaned unlocked at the curb | 0.4 | 0.05-0.15 | small-town trust
to-go cup on the bench arm | 0.45 | 0.02-0.06 | sleeve doubled
event flyers in shop windows | 0.5 | 0.03-0.1 | tape-cornered pages
quarters fed from a cup holder | 0.4 | 0.02-0.06 | meter ritual
hose watering the curb planters | 0.4 | 0.03-0.1 | morning shopkeeper
sidewalk sale rack outside | 0.45 | 0.1-0.25 | clearance tags flipping""",
            "foreground_element": """shop door held for a stranger | 0.55 | 0.05-0.2 | bell over the frame
window display browsed slowly | 0.55 | 0.1-0.3 | reflection layered
meter fed with found change | 0.5 | 0.05-0.15 | coin pinch
coffee sipped walking the block | 0.55 | 0.05-0.2 | unhurried pace
dog greeted outside the bakery | 0.5 | 0.05-0.2 | ear scratch crouch
crosswalk waited at the light | 0.5 | 0.1-0.25 | button pressed twice
pizza slice folded on the walk | 0.45 | 0.05-0.15 | paper plate flex
flyer read on the pole | 0.45 | 0.05-0.15 | head tilted
bench sat with a paper | 0.45 | 0.1-0.3 | county news spread
bakery bag pinched closed | 0.5 | 0.03-0.1 | still warm
wave through a shop window | 0.45 | 0.05-0.15 | familiar face""",
            "time_of_day": OUTDOOR_PA_TIME,
            "weather": OUTDOOR_PA_WEATHER,
        },
        "strip_mall_sidewalk": {
            "background": """strip mall storefront row | 1.0 | 0.5-0.9 | plate glass and sign band
anchor grocery at the far end | 0.6 | 0.25-0.5 | cart corral out front
nail salon and pizza signage | 0.65 | 0.2-0.4 | backlit letter boxes
parking lot stretching out | 0.8 | 0.3-0.6 | faded stall grid
cart corral mid-lot | 0.6 | 0.1-0.25 | nested carts
dollar store window posters | 0.55 | 0.15-0.35 | neon price stars
covered walkway with columns | 0.7 | 0.3-0.6 | painted steel posts
vacant unit with paper on glass | 0.5 | 0.15-0.35 | for-lease sign
gumball machines by a door | 0.45 | 0.05-0.15 | quarter-turn row
laundromat window with folding tables | 0.45 | 0.15-0.35 | fluorescent interior
chinese takeout menu taped up | 0.5 | 0.05-0.15 | laminated sheet""",
            "midground": """carts drifted against a curb | 0.55 | 0.1-0.25 | wheels turned
newspaper boxes in a row | 0.45 | 0.05-0.15 | mixed publications
handicap ramp with yellow bumps | 0.55 | 0.1-0.2 | truncated domes
minivan idling for a pickup | 0.5 | 0.15-0.35 | hazards blinking
folding sign for a phone repair | 0.45 | 0.05-0.15 | arrow flipping
bike locked to a column | 0.45 | 0.05-0.15 | front wheel gone once
delivery hand truck by a door | 0.45 | 0.05-0.15 | strapped boxes
kids waiting outside the karate school | 0.4 | 0.15-0.3 | white gis
trash can by each column | 0.55 | 0.05-0.15 | dome lids
puddle mirroring the sign band | 0.45 | 0.1-0.25 | oily sheen
sale rack rolled outside | 0.45 | 0.1-0.25 | discount tags""",
            "architecture_detail": """sidewalk expansion joints | 0.6 | 0.05-0.2 | tar-filled lines
bollards painted safety yellow | 0.55 | 0.05-0.15 | chipped domes
sign band conduit runs | 0.4 | 0.05-0.15 | exposed feeds
column base rusted at the bolts | 0.45 | 0.03-0.1 | powdered orange
automatic door mat sensor | 0.45 | 0.03-0.1 | rubber threshold
roofline parapet with a gap | 0.35 | 0.1-0.25 | pigeon perch
sprinkler riser against the wall | 0.4 | 0.02-0.08 | caged valve
cigarette urn by the vacancy | 0.4 | 0.02-0.06 | sand-topped
painted curb numbers | 0.45 | 0.02-0.08 | suite markers
window tint bubbling in a pane | 0.4 | 0.05-0.15 | purple blister
gum constellation on the concrete | 0.45 | 0.03-0.1 | black dots""",
            "props": """abandoned receipt tumbleweed | 0.45 | 0.02-0.06 | wind-rolled curl
quarters pressed into a gumball turn | 0.4 | 0.02-0.06 | kid's ritual
pizza boxes carried flat | 0.5 | 0.03-0.1 | steady stack
laundry basket on a hip | 0.45 | 0.05-0.15 | detergent on top
helium balloon tied to a stroller | 0.4 | 0.03-0.1 | party-store exit
flyer stack weighted by a rock | 0.35 | 0.02-0.06 | car-wash fundraiser
lotto ticket scratched at the trash can | 0.4 | 0.02-0.06 | key-edge shavings
takeout bag with chopstick flags | 0.45 | 0.02-0.08 | stapled receipt
energy drink cans on a window sill | 0.4 | 0.02-0.06 | after-practice row
lost glove on a bollard | 0.35 | 0.02-0.05 | winter orphan
cart with one bad wheel | 0.5 | 0.05-0.15 | rhythmic clatter""",
            "foreground_element": """cart wrangled from the corral | 0.55 | 0.1-0.25 | nested yank
door leaned open with a shoulder | 0.55 | 0.05-0.2 | bags in both hands
pizza boxes balanced to the car | 0.5 | 0.05-0.2 | chin assist
kid steered by the hood | 0.45 | 0.05-0.2 | parking lot protocol
receipt checked against the bags | 0.5 | 0.05-0.15 | walk-and-scan
gumball cranked and caught | 0.4 | 0.03-0.1 | palm at the chute
karate belt retied outside | 0.4 | 0.05-0.15 | bow adjusted
laundry bag hoisted higher | 0.45 | 0.05-0.15 | grip shift
puddle stepped around | 0.5 | 0.05-0.15 | tightrope curb
keys fished out at the curb | 0.5 | 0.03-0.1 | pocket dig
takeout sniffed through the bag | 0.45 | 0.03-0.1 | anticipation""",
            "time_of_day": OUTDOOR_PA_TIME,
            "weather": OUTDOOR_PA_WEATHER,
        },
        "diner_parking_lot": {
            "background": """chrome-edged diner facade | 1.0 | 0.4-0.8 | stainless bands and glass block
neon OPEN sign buzzing | 0.7 | 0.1-0.25 | red script glow
gravel lot edge past the asphalt | 0.6 | 0.2-0.45 | potholed transition
route highway beyond the lot | 0.6 | 0.25-0.5 | passing trucks
diner windows with booth silhouettes | 0.7 | 0.25-0.5 | venetian blinds half up
reader board sign on a pole | 0.6 | 0.1-0.3 | scrapple special letters
dumpster corral at the back | 0.45 | 0.1-0.25 | wooden fence screen
flag pole by the entrance | 0.45 | 0.05-0.2 | halyard clanking
propane tank cage at the side | 0.4 | 0.05-0.15 | painted white cage
farm field across the road | 0.5 | 0.25-0.5 | corn rows to the treeline
milk crate stack by the kitchen door | 0.45 | 0.05-0.15 | plastic tower""",
            "midground": """pickups nosed to the building | 0.65 | 0.25-0.5 | work-mud flanks
handicap spots by the ramp | 0.55 | 0.1-0.25 | faded blue paint
motorcycle pair parked together | 0.45 | 0.1-0.25 | helmets on mirrors
entrance steps with a railing | 0.6 | 0.1-0.25 | astroturf treads
newspaper boxes by the door | 0.5 | 0.05-0.15 | auto-trader and county news
smokers' bench at the corner | 0.4 | 0.05-0.15 | coffee-can ashtray
delivery van at the side door | 0.45 | 0.15-0.3 | ramp down
puddles in the low spots | 0.5 | 0.1-0.25 | sky reflections
state trooper cruiser parked | 0.35 | 0.15-0.3 | coffee stop
tour bus taking up the back row | 0.3 | 0.2-0.4 | casino day trip
gull working the lot | 0.4 | 0.05-0.15 | fry scout""",
            "architecture_detail": """glass block corner detail | 0.5 | 0.1-0.25 | lit from inside
stainless siding seams | 0.55 | 0.1-0.3 | riveted panels
concrete wheel stops | 0.55 | 0.05-0.15 | rebar showing
downspout splashing the walk | 0.4 | 0.03-0.1 | painted shut strap
window AC units at the back | 0.4 | 0.05-0.15 | drip stains
door with a period handle | 0.45 | 0.03-0.1 | polished dull
lot lights on wooden poles | 0.5 | 0.05-0.2 | mismatched fixtures
vestibule with a second door | 0.45 | 0.05-0.15 | draft lock
painted parking arrows worn | 0.45 | 0.05-0.15 | ghost strokes
step edge in safety yellow | 0.45 | 0.03-0.1 | fresh coat
kitchen exhaust fan housing | 0.4 | 0.05-0.12 | grease shadow""",
            "props": """to-go cup left on a bumper | 0.45 | 0.02-0.06 | forgotten at the door
mint toothpicks dropped | 0.35 | 0.02-0.05 | scattered pair
newspaper tucked under an arm | 0.45 | 0.02-0.08 | folded sports page
doggie bag carried out | 0.5 | 0.02-0.08 | foil swan maybe
pie box tied with string | 0.45 | 0.02-0.08 | bakery counter treat
umbrella dripping in a truck bed | 0.35 | 0.02-0.08 | quick storm
receipt spike visible in the window | 0.35 | 0.02-0.06 | counter relic
crumbs tossed to the gull | 0.35 | 0.02-0.08 | against better judgment
motorcycle gloves on a seat | 0.4 | 0.02-0.06 | fingers curled
coffee thermos refilled to go | 0.45 | 0.02-0.06 | road ritual
quarters left in the paper box | 0.35 | 0.02-0.05 | honor system""",
            "foreground_element": """door held against the vestibule spring | 0.55 | 0.05-0.2 | bell chime
step down mid-conversation | 0.5 | 0.1-0.25 | keys already out
to-go coffee shielded from wind | 0.5 | 0.03-0.1 | lid tested
truck door swung wide | 0.5 | 0.1-0.3 | one boot up
pie box carried level | 0.5 | 0.05-0.15 | two-hand care
stretch after the big breakfast | 0.5 | 0.05-0.2 | belt adjust
lot gravel crunched across | 0.5 | 0.1-0.25 | unhurried diagonal
menu board read from the car | 0.4 | 0.05-0.15 | window down
helmet buckled by the bikes | 0.4 | 0.05-0.15 | strap tug
wave to the waitress through glass | 0.45 | 0.05-0.15 | regulars' code
leftover fry stolen from the bag | 0.45 | 0.03-0.1 | passenger tax""",
            "time_of_day": OUTDOOR_PA_TIME,
            "weather": OUTDOOR_PA_WEATHER,
        },
    },
    "outdoor/nature_pa": {
        "state_park_trailhead": {
            "background": """gravel parking lot at the trailhead | 1.0 | 0.4-0.8 | packed limestone chips
wooden signboard with a trail map | 0.85 | 0.15-0.35 | plexiglass over paper
trail opening into the woods | 0.8 | 0.3-0.6 | worn dirt mouth
hardwood hillside rising behind | 0.7 | 0.3-0.6 | oak and maple canopy
split-rail fence along the lot | 0.6 | 0.15-0.4 | greyed cedar rails
pit toilet building | 0.45 | 0.1-0.25 | brown board-and-batten
state park entrance sign | 0.5 | 0.1-0.25 | routed yellow letters
picnic pavilion in the trees | 0.5 | 0.2-0.4 | green steel roof
creek heard below the lot | 0.45 | 0.1-0.3 | glimpsed riffle
blaze marks on the first trees | 0.55 | 0.05-0.15 | painted rectangles
gate arm for after-hours | 0.45 | 0.05-0.15 | padlocked pipe""",
            "midground": """cars nosed against the rail | 0.6 | 0.2-0.45 | dusty hatchbacks
kiosk with permit envelopes | 0.5 | 0.05-0.2 | pencil on a string
dog watered from a bottle | 0.5 | 0.05-0.15 | cupped-hand bowl
hikers checking the map board | 0.5 | 0.1-0.3 | finger on the route
bear-proof trash can | 0.55 | 0.05-0.15 | latched steel lid
bench made from a split log | 0.45 | 0.1-0.25 | eagle scout plaque
boot-brush station at the trail mouth | 0.4 | 0.03-0.1 | invasive-species sign
firewood bundle stack for campers | 0.4 | 0.1-0.25 | shrink-wrapped
horse rig parked long-ways | 0.3 | 0.2-0.4 | trailer with hay bag
kayaks strapped on a roof | 0.4 | 0.1-0.25 | foam block rig
fallen log at the lot edge | 0.45 | 0.1-0.25 | shelf fungus""",
            "architecture_detail": """culvert pipe under the entrance road | 0.4 | 0.05-0.15 | corrugated mouth
lot edge timbers bolted down | 0.5 | 0.05-0.2 | rebar pinned
signpost with mileage arrows | 0.55 | 0.03-0.1 | routed digits
trail register box on a post | 0.5 | 0.02-0.08 | hinged lid notebook
erosion bars across the first climb | 0.5 | 0.05-0.2 | half-buried timbers
lichen crust on the fence rails | 0.45 | 0.05-0.15 | pale green patches
carsonite post with rules decals | 0.4 | 0.02-0.08 | flexible marker
gravel washboard at the entrance | 0.4 | 0.1-0.25 | rippled surface
pavilion post carved with initials | 0.4 | 0.03-0.1 | decades of layers
drainage swale along the lot | 0.4 | 0.05-0.2 | leaf-packed channel
bulletin board staple constellation | 0.45 | 0.03-0.1 | flyer corners""",
            "props": """trekking poles leaned on a bumper | 0.5 | 0.02-0.08 | cork grips
water bottles lined on a tailgate | 0.55 | 0.02-0.08 | pre-hike ritual
trail mix bag passed around | 0.5 | 0.02-0.08 | m&m mining
paper map folded to the section | 0.45 | 0.02-0.08 | crease-worn
dog leash clipped to a harness | 0.5 | 0.02-0.06 | excited strain
muddy boots swapped at the car | 0.5 | 0.03-0.1 | sock-footed hop
granola wrapper pocketed | 0.4 | 0.02-0.05 | pack-it-out habit
first aid kit in a dry bag | 0.4 | 0.02-0.06 | zipped pouch
camera with a zoom slung across | 0.4 | 0.02-0.08 | bird hopes
tick spray applied at the ankles | 0.45 | 0.02-0.08 | sock cuff spritz
thermos of coffee on the rail | 0.4 | 0.02-0.06 | steam curl""",
            "foreground_element": """bootlaces cinched on the bumper | 0.6 | 0.05-0.2 | heel-locked wrap
pack shrugged onto the shoulders | 0.6 | 0.1-0.25 | strap settle
map photo taken at the board | 0.5 | 0.05-0.15 | phone raised
first blaze pointed out | 0.5 | 0.05-0.2 | route confirmed
dog's harness double-checked | 0.45 | 0.05-0.15 | buckle tug
register signed with the stub pencil | 0.45 | 0.03-0.1 | date and initials
stretch against the fence rail | 0.5 | 0.05-0.2 | calf lengthened
water sip before the climb | 0.5 | 0.03-0.1 | bottle tipped
jacket tied around the waist | 0.5 | 0.05-0.2 | sleeves knotted
gravel kicked from a tread | 0.4 | 0.03-0.1 | boot tapped on the timber
trail mouth sized up | 0.5 | 0.1-0.3 | hands on hips""",
            "time_of_day": OUTDOOR_PA_TIME,
            "weather": OUTDOOR_PA_WEATHER,
        },
        "creek_bank_summer": {
            "background": """shallow creek over flat shale | 1.0 | 0.4-0.8 | amber water and riffles
sycamore leaning over the water | 0.7 | 0.2-0.5 | mottled bark
far bank of rhododendron | 0.6 | 0.25-0.5 | glossy tangled wall
swimming hole below the riffle | 0.6 | 0.25-0.5 | green-dark pool
gravel bar mid-creek | 0.6 | 0.2-0.4 | sorted stones
old stone abutment remnant | 0.4 | 0.1-0.3 | mossy block wall
hemlock shade on the far side | 0.5 | 0.2-0.45 | deep green dark
riffle catching the light | 0.6 | 0.15-0.35 | broken sparkle
path worn down the bank | 0.6 | 0.15-0.35 | root staircase
railroad grade above the far bank | 0.35 | 0.15-0.35 | ballast line
cliff swallows working the water | 0.35 | 0.05-0.2 | looping flights""",
            "midground": """towels spread on warm rocks | 0.6 | 0.1-0.3 | drying flat
cooler wedged in the shallows | 0.5 | 0.05-0.2 | cans cooling in the current
kids stacking a stone dam | 0.45 | 0.1-0.3 | busy engineering
water shoes drying on a log | 0.5 | 0.05-0.15 | neoprene pairs
folding chair in the shallows | 0.45 | 0.1-0.25 | legs sunk in gravel
minnow bucket at the edge | 0.35 | 0.03-0.1 | dip net across
dog shaking off mid-bank | 0.45 | 0.05-0.2 | spray halo
inner tube beached on the bar | 0.4 | 0.1-0.25 | patched vinyl
fishing rod propped in a forked stick | 0.4 | 0.05-0.15 | line in the pool
rope swing over the hole | 0.4 | 0.05-0.2 | knotted polypro
driftwood pile at the bend | 0.45 | 0.1-0.3 | bleached tangle""",
            "architecture_detail": """shale ledges stepping into the water | 0.6 | 0.15-0.4 | flat cleaved sheets
undercut bank with exposed roots | 0.5 | 0.1-0.3 | woven overhang
water line stain on the rocks | 0.5 | 0.05-0.2 | high-water mark
crayfish holes in the mud edge | 0.4 | 0.03-0.1 | chimney burrows
flood debris high in a branch | 0.45 | 0.03-0.12 | plastic flag
gravel sorted by the current | 0.5 | 0.1-0.3 | fined-to-coarse bands
moss carpet on the north rocks | 0.5 | 0.05-0.2 | spongy pads
dragonfly perch stick | 0.4 | 0.02-0.08 | repeat landings
foam line at the eddy | 0.4 | 0.03-0.12 | slow rotation
sun-warmed slab at the pool edge | 0.5 | 0.1-0.25 | lizard-flat rock
old cable anchor in a boulder | 0.3 | 0.02-0.08 | rusted eye bolt""",
            "props": """mesh bag of creek toys | 0.45 | 0.02-0.08 | cups and scoops
sunscreen tube on a towel | 0.55 | 0.02-0.06 | sand-crusted cap
sandwich cooler opened on a rock | 0.5 | 0.05-0.12 | wax-paper stack
bug net leaned on a log | 0.35 | 0.02-0.08 | kid-sized handle
polarized sunglasses pushed up | 0.4 | 0.02-0.06 | fish-spotting tool
creek-found brick shard | 0.35 | 0.02-0.05 | rounded edges
jar of caught minnows | 0.4 | 0.02-0.06 | studied closely
flip flops parked toe-out | 0.5 | 0.02-0.08 | bank lineup
walking stick claimed for the day | 0.45 | 0.02-0.08 | perfect height
snack wrappers pinned by a rock | 0.4 | 0.02-0.06 | wind insurance
towel cape on a shivering kid | 0.4 | 0.05-0.15 | superhero drape""",
            "foreground_element": """flat stone skipped down the pool | 0.6 | 0.05-0.2 | four-hop count
toes tested in the current | 0.6 | 0.05-0.2 | cold verdict
crayfish lifted for inspection | 0.5 | 0.05-0.15 | pinch-safe grip
splash war mid-riffle | 0.5 | 0.1-0.3 | double-arm spray
rope swing arc at the top | 0.4 | 0.1-0.3 | release moment
towel wrapped against the shade chill | 0.5 | 0.1-0.25 | shoulder bundle
can fished from the cooler current | 0.45 | 0.03-0.1 | dripping retrieve
water shoes emptied of gravel | 0.45 | 0.03-0.1 | one-leg balance
minnow chased with a cup | 0.45 | 0.05-0.15 | quick scoop
wet hair wrung to one side | 0.5 | 0.05-0.15 | twist and drip
balance tested rock to rock | 0.55 | 0.1-0.25 | arms out crossing""",
            "time_of_day": OUTDOOR_PA_SUMMER_TIME,
            "weather": OUTDOOR_PA_SUMMER_WEATHER,
        },
        "fall_woods_walk": {
            "background": """leaf-covered path through hardwoods | 1.0 | 0.4-0.8 | russet and gold drift
oaks holding brown leaves | 0.7 | 0.3-0.6 | rattling canopy
maples gone bare at the crown | 0.65 | 0.25-0.5 | grey branch lace
understory of witch hazel | 0.45 | 0.15-0.35 | late yellow ribbons
stone wall running into the woods | 0.5 | 0.15-0.35 | farm-era line
fog hanging in the hollow | 0.45 | 0.2-0.5 | soft grey pool
deer stand in a distant tree | 0.35 | 0.05-0.15 | plywood platform
laurel thicket staying green | 0.5 | 0.2-0.4 | leathery leaves
old woods road grade | 0.5 | 0.2-0.45 | double-rut memory
posted signs along one edge | 0.4 | 0.05-0.15 | purple paint blazes
squirrel nest high in a fork | 0.4 | 0.05-0.12 | leaf ball""",
            "midground": """fallen oak across the path | 0.55 | 0.15-0.35 | step-over trunk
leaf drifts against the deadfall | 0.6 | 0.15-0.35 | knee-deep pockets
shagbark hickory trunk | 0.5 | 0.1-0.3 | peeling plates
grapevine tangle in a gap | 0.45 | 0.1-0.3 | hanging loops
boulder field patch | 0.45 | 0.15-0.35 | glacial scatter
spring seep crossing the trail | 0.4 | 0.05-0.2 | black muck strip
dog ranging ahead | 0.45 | 0.05-0.2 | nose-driven zigzag
stump shelf of turkey tail fungus | 0.45 | 0.05-0.15 | banded fans
buck rub on a sapling | 0.4 | 0.03-0.1 | barked scar
squirrel diggings in the leaves | 0.45 | 0.05-0.15 | fresh excavations
another walker's red jacket far off | 0.35 | 0.05-0.15 | color through trunks""",
            "architecture_detail": """leaf layer over the path stones | 0.6 | 0.15-0.4 | slick hidden cobbles
water bar log across the grade | 0.45 | 0.05-0.15 | half-buried diverter
wall gap where a gate stood | 0.4 | 0.05-0.2 | rotted post stub
barbed wire grown into a tree | 0.4 | 0.02-0.08 | swallowed strand
survey marker on a corner tree | 0.35 | 0.02-0.06 | yellow tag nail
frost heave in the muck crossing | 0.35 | 0.05-0.15 | ice needle columns
root flare washed bare | 0.45 | 0.05-0.15 | knuckled grip
old apple tree from a homestead | 0.4 | 0.1-0.25 | wolf-tree spread
charcoal flat from a hearth | 0.3 | 0.05-0.2 | black soil circle
blaze repainted over an old one | 0.4 | 0.02-0.06 | doubled rectangle
culvert of dry-laid stone | 0.35 | 0.05-0.15 | keystone arch""",
            "props": """acorn cap collection in a pocket | 0.4 | 0.02-0.05 | kid treasure
thermos shared at the wall | 0.45 | 0.02-0.08 | cap-cup steam
binoculars raised at a knock | 0.4 | 0.02-0.08 | woodpecker hunt
orange beanie for the season | 0.5 | 0.02-0.08 | visibility choice
walking stick from the deadfall | 0.45 | 0.02-0.08 | thumb-worn knot
leaf pressed into a paperback | 0.35 | 0.02-0.05 | maple keeper
trail bar unwrapped one-handed | 0.45 | 0.02-0.06 | glove in teeth
phone camera aimed at the canopy | 0.45 | 0.02-0.08 | color panorama
dog's found stick oversized | 0.45 | 0.03-0.1 | proud carry
seed heads picked off socks | 0.4 | 0.02-0.06 | hitchhiker burrs
compass checked out of habit | 0.3 | 0.02-0.05 | baseline bearing""",
            "foreground_element": """leaves shuffled ankle-deep | 0.65 | 0.1-0.3 | wading strides
breath fogged on the cold air | 0.55 | 0.05-0.15 | brief plume
log crossed with a hand down | 0.5 | 0.1-0.25 | three-point step
beanie tugged over the ears | 0.5 | 0.05-0.15 | wind response
leaf caught midair | 0.45 | 0.05-0.15 | lucky snatch
stone wall sat for a break | 0.5 | 0.1-0.3 | flat capstone seat
dog called back from a scent | 0.45 | 0.05-0.2 | name and clap
jacket zipped to the chin | 0.5 | 0.05-0.15 | hollow chill
photo crouch for the fungus shelf | 0.4 | 0.05-0.15 | low angle hunt
trail snack split two ways | 0.45 | 0.03-0.1 | half offered
hands warmed in armpits | 0.45 | 0.05-0.15 | glove regret""",
            "time_of_day": OUTDOOR_PA_FALL_TIME,
            "weather": OUTDOOR_PA_FALL_WEATHER,
        },
    },
    "outdoor/fitness_us": {
        "township_track": {
            "background": """rubberized track in faded red | 1.0 | 0.4-0.8 | patched lanes
chain-link fence around the oval | 0.75 | 0.2-0.5 | galvanized diamond mesh
aluminum bleachers on one side | 0.7 | 0.2-0.45 | rattling risers
football field inside the oval | 0.7 | 0.3-0.6 | worn hash marks
press box on stilts | 0.45 | 0.1-0.25 | plywood booth
scoreboard on poles | 0.55 | 0.1-0.3 | sponsor panels below
equipment shed by the gate | 0.5 | 0.1-0.25 | padlocked double door
long jump pit at the straightaway | 0.5 | 0.1-0.25 | sand raked in ridges
school building beyond the fence | 0.55 | 0.25-0.5 | brick and banks of windows
flag at the field entrance | 0.45 | 0.05-0.2 | snapping halyard
tree line past the far curve | 0.55 | 0.25-0.5 | township woods""",
            "midground": """lane markers painted at the start | 0.6 | 0.05-0.2 | numbered blocks
hurdles stacked on the infield | 0.55 | 0.1-0.3 | staggered heights
walkers in the outside lanes | 0.6 | 0.1-0.3 | steady laps
water bottles on the bottom bleacher | 0.55 | 0.03-0.1 | team row
soccer goals folded against the fence | 0.45 | 0.1-0.3 | net bundled
stroller parked track-side | 0.4 | 0.05-0.2 | lap-counting parent
cones marking a workout ladder | 0.5 | 0.05-0.2 | interval spacing
kettlebell left by the bleachers | 0.35 | 0.03-0.1 | boot-camp leftovers
dog tied in the shade | 0.4 | 0.05-0.15 | water bowl set out
maintenance cart on the infield | 0.4 | 0.1-0.25 | line-painting rig
high jump mat under a tarp | 0.45 | 0.1-0.3 | bungeed cover""",
            "architecture_detail": """lane lines repainted off-register | 0.5 | 0.05-0.2 | doubled edges
track surface bubbled at a seam | 0.45 | 0.05-0.15 | patch peeling
gate with a coded padlock | 0.5 | 0.03-0.1 | chain wrap
bleacher footboards flexing | 0.45 | 0.05-0.15 | oil-canning steel
curb ring around the infield | 0.5 | 0.05-0.15 | concrete lip
drainage grate at the low curve | 0.45 | 0.03-0.1 | puddle magnet
starting line embedded brass | 0.35 | 0.02-0.08 | worn insert
fence topped with bare tension wire | 0.45 | 0.05-0.15 | no barbs
light poles dark until friday | 0.5 | 0.05-0.2 | stadium heads
ticket booth shuttered | 0.4 | 0.05-0.15 | plywood window
ramp to the bleachers | 0.45 | 0.05-0.15 | painted grip strips""",
            "props": """stopwatch on a lanyard | 0.5 | 0.02-0.06 | thumb-worn buttons
interval workout on a sticky note | 0.4 | 0.02-0.05 | sweat-curled paper
resistance bands looped on the fence | 0.45 | 0.02-0.08 | color set
foam roller on the bottom riser | 0.4 | 0.02-0.08 | post-run ritual
spikes bag by the start | 0.4 | 0.02-0.08 | drawstring sack
orange slices in a cooler | 0.4 | 0.03-0.1 | halftime tradition
pinnies drying on the fence | 0.45 | 0.03-0.1 | mesh row
starting blocks carried out | 0.4 | 0.03-0.1 | awkward armful
chalk line for the relay zone | 0.4 | 0.02-0.08 | powder stripe
headphones case on a towel | 0.45 | 0.02-0.06 | lap soundtrack
whistle heard across the field | 0.45 | 0.02-0.08 | practice cadence""",
            "foreground_element": """laces double-knotted at the line | 0.6 | 0.05-0.15 | pre-run ritual
watch started into the first stride | 0.6 | 0.05-0.15 | beep and go
stride opened on the back straight | 0.55 | 0.1-0.3 | relaxed form
hands on knees after the interval | 0.55 | 0.05-0.2 | recovery fold
water grabbed on the walk lap | 0.5 | 0.03-0.1 | bottle swing
hurdle height adjusted | 0.45 | 0.05-0.15 | pin and slide
baton passed in the exchange zone | 0.4 | 0.05-0.2 | reach and grip
bleacher step-ups counted | 0.45 | 0.05-0.2 | rhythm climb
sunset squinted at mid-cooldown | 0.45 | 0.05-0.2 | golden lap
calf stretched on the curb ring | 0.5 | 0.05-0.15 | heel drop
splits read off the wrist | 0.5 | 0.03-0.1 | pace math""",
            "time_of_day": OUTDOOR_PA_TIME,
            "weather": OUTDOOR_PA_WEATHER,
        },
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
            if not 0.3 <= p <= 1.0:
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
