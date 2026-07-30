"""Generator für die Alltags-Location-Sets (deutscher/europäischer Zuschnitt).

Die bestehenden `everyday_us`-Sets sind gut, aber US-zentriert (Walmart, CVS,
Little League). Diese hier sind das europäische Gegenstück: Orte, an denen man
täglich wirklich steht.

Die Einträge erfüllen die Regeln aus ``tests/unit/test_location_lists_extended.py``:
mindestens 10 Einträge pro Datei, Wahrscheinlichkeit in [0.3, 1.0], Namen mit
mindestens zwei Wörtern, keine Duplikate, keine Indoor/Outdoor-Verwechslungen.

Ausführen:  python scripts/gen_everyday_locations.py [--force]
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

# Shared atmosphere blocks — 10 entries each, as the tests require.
INDOOR_TIME = """morning light through the window | 0.9 | - | -
flat late morning light | 0.85 | - | -
grey overcast midday | 0.9 | - | -
early afternoon with lights off | 0.8 | - | -
late afternoon low sun | 0.85 | - | -
early evening with a lamp on | 0.9 | - | -
evening under the ceiling light | 0.85 | - | -
dim light before bed | 0.6 | - | -
harsh overhead light | 0.7 | - | -
half dark with one light on | 0.55 | - | -"""

# Weather must not make any light-level claim — that is time_of_day's job.
# Otherwise the two combine into "evening under the ceiling light,
# bright sun through the window".
INDOOR_WEATHER = """rain streaks on the glass | 0.55 | - | -
misted-up window pane | 0.45 | - | -
rain drumming on the window | 0.45 | - | -
wind audible outside | 0.4 | - | -
damp air after rain | 0.5 | - | -
dry still air indoors | 0.7 | - | -
draught from the window frame | 0.4 | - | -
snow visible outside the window | 0.35 | - | -
warm close air indoors | 0.5 | - | -
wet marks tracked in by the door | 0.35 | - | -"""

OUTDOOR_TIME = """early morning light | 0.85 | - | -
bright mid morning | 0.85 | - | -
flat overcast midday | 0.9 | - | -
early afternoon glare | 0.7 | - | -
late afternoon long shadows | 0.9 | - | -
golden hour before sunset | 0.8 | - | -
blue hour after sunset | 0.6 | - | -
dusk with street lights coming on | 0.8 | - | -
grey commuter morning | 0.85 | - | -
overcast early evening | 0.8 | - | -"""

OUTDOOR_WEATHER = """flat overcast sky | 0.9 | - | -
light drizzle in the air | 0.55 | - | -
steady grey rain | 0.5 | - | -
wet ground after rain | 0.6 | - | -
clear cold air | 0.6 | - | -
hazy warm air | 0.55 | - | -
gusty wind moving the trees | 0.55 | - | -
cold air with breath visible | 0.4 | - | -
thin high cloud | 0.6 | - | -
damp mild air | 0.7 | - | -"""


SETS: dict[str, dict[str, dict[str, str]]] = {
    "indoor/everyday_de": {
        "kitchen_cooking": {
            "background": """plain white kitchen wall tiles | 1.0 | 0.5-0.9 | glossy square tiles with grout
kitchen wall cupboards in light wood | 0.9 | 0.4-0.8 | laminate fronts with simple handles
window above the sink | 0.8 | 0.2-0.5 | plain glass with a roller blind
tiled splashback behind the hob | 0.85 | 0.2-0.5 | wiped-down tiles
open shelf with jars | 0.6 | 0.2-0.5 | mismatched screw-top jars
pinboard with notes and photos | 0.55 | 0.15-0.4 | overlapping paper scraps
plain painted wall above the tiles | 0.75 | 0.3-0.6 | flat emulsion with grease marks
extractor hood over the hob | 0.7 | 0.15-0.4 | brushed steel with a filter grille
kitchen door left ajar | 0.5 | 0.2-0.5 | painted timber with a worn handle
calendar hanging by the door | 0.45 | 0.1-0.3 | printed paper months
row of hanging tea towels | 0.5 | 0.1-0.3 | faded checked cotton""",
            "midground": """laminate worktop with clutter | 1.0 | 0.4-0.8 | scratched light laminate
gas hob with used pots | 0.9 | 0.2-0.6 | steel grates with cooking marks
refrigerator covered in magnets | 0.75 | 0.2-0.5 | white enamel under paper notes
sink full of washing up | 0.8 | 0.2-0.5 | soap film on stacked plates
small kitchen table with crumbs | 0.7 | 0.3-0.6 | wiped wooden top
dishwasher door half open | 0.5 | 0.15-0.4 | steel front with a loaded rack
kettle beside the toaster | 0.75 | 0.1-0.3 | limescaled plastic and chrome
bin with the lid propped open | 0.55 | 0.15-0.35 | pedal bin with a stretched bag
microwave on the worktop | 0.6 | 0.15-0.35 | scratched black casing
drying rack of clean dishes | 0.65 | 0.15-0.35 | plastic-coated wire
kitchen chair pulled out | 0.55 | 0.2-0.45 | worn seat pad""",
            "architecture_detail": """practical ceiling light | 0.8 | 0.1-0.3 | plain glass shade
vinyl kitchen flooring | 0.8 | 0.3-0.6 | wood-look vinyl planks
skirting board along the wall | 0.5 | 0.1-0.25 | painted with scuff marks
radiator under the window | 0.55 | 0.1-0.3 | painted steel ribs
socket strip along the tiles | 0.6 | 0.05-0.2 | yellowed plastic sockets
grouted tile joints | 0.7 | 0.15-0.4 | darkened grout lines
kitchen door frame | 0.5 | 0.1-0.3 | chipped painted timber
window sill with clutter | 0.65 | 0.1-0.3 | painted sill with rings
under-cupboard strip light | 0.5 | 0.05-0.2 | small fluorescent tube
worn threshold strip at the door | 0.4 | 0.05-0.15 | scratched aluminium
ceiling corner with a cobweb | 0.35 | 0.05-0.15 | dusty plaster""",
            "props": """chopping board with vegetable scraps | 0.85 | 0.05-0.2 | knife-marked wood
open spice jars | 0.7 | 0.05-0.15 | mismatched screw-top jars
dish towel over the oven handle | 0.75 | 0.05-0.15 | faded checked cotton
half-drunk mug of coffee | 0.7 | 0.05-0.15 | stained ceramic
wooden spoon resting on the hob | 0.65 | 0.05-0.12 | scorched spoon end
packet of pasta left open | 0.55 | 0.05-0.15 | torn plastic packaging
bottle of oil beside the hob | 0.6 | 0.05-0.12 | greasy label
knife block by the worktop | 0.5 | 0.05-0.15 | wooden slots
shopping list under a magnet | 0.5 | 0.03-0.1 | biro on torn paper
salt and pepper mills | 0.55 | 0.03-0.1 | worn wooden bodies
sponge in the sink corner | 0.6 | 0.02-0.08 | frayed yellow sponge""",
            "foreground_element": """pot steaming on the hob | 0.9 | 0.1-0.3 | condensation on the lid
hands stirring in the pan | 0.7 | 0.1-0.3 | steam over the wrist
open cookbook propped up | 0.5 | 0.05-0.2 | splashed page
dirty plates stacked in the sink | 0.6 | 0.1-0.25 | soap film
cutting into vegetables | 0.65 | 0.1-0.25 | wet blade
mug lifted towards the camera | 0.5 | 0.05-0.15 | chipped rim
tea towel over the shoulder | 0.5 | 0.05-0.15 | damp cotton
phone propped against the tiles | 0.45 | 0.05-0.15 | smeared screen
oven glove on the worktop | 0.45 | 0.05-0.15 | scorched quilted fabric
packet being opened over the pan | 0.4 | 0.05-0.15 | crinkled foil
steam catching the window light | 0.55 | 0.1-0.25 | drifting vapour""",
            "time_of_day": INDOOR_TIME,
            "weather": INDOOR_WEATHER,
        },
        "living_room_sofa": {
            "background": """plain painted living room wall | 1.0 | 0.5-0.9 | flat emulsion in off-white
shelf with books and photos | 0.85 | 0.3-0.6 | mismatched spines
window with curtains half open | 0.8 | 0.2-0.5 | plain lined curtains
television on the wall | 0.75 | 0.2-0.5 | dusty dark screen
framed prints hung slightly crooked | 0.6 | 0.2-0.5 | cheap poster frames
patterned wallpaper on one wall | 0.45 | 0.3-0.6 | dated repeating pattern
doorway through to the hall | 0.55 | 0.2-0.5 | painted frame
sideboard against the wall | 0.6 | 0.25-0.55 | veneer with ring marks
houseplant in the corner | 0.65 | 0.15-0.4 | dusty leaves
radiator below the window | 0.6 | 0.1-0.3 | painted steel panel
curtain pole with rings | 0.4 | 0.1-0.3 | painted timber pole""",
            "midground": """worn fabric sofa with cushions | 1.0 | 0.4-0.8 | flattened woven upholstery
low coffee table with rings | 0.85 | 0.2-0.5 | marked veneer top
armchair beside the sofa | 0.6 | 0.25-0.55 | sagging seat cushion
television stand with clutter | 0.7 | 0.2-0.5 | dark wood-look laminate
laundry basket waiting to be folded | 0.6 | 0.15-0.4 | plastic weave
rug in front of the sofa | 0.7 | 0.3-0.6 | flattened pile
side table with a lamp | 0.6 | 0.15-0.4 | scratched top
stack of magazines on the floor | 0.5 | 0.1-0.3 | curled covers
child's toys pushed to one side | 0.5 | 0.1-0.35 | scuffed plastic
folded blanket over the sofa back | 0.7 | 0.15-0.4 | pilled fleece
footstool used as a table | 0.45 | 0.1-0.3 | worn fabric top""",
            "architecture_detail": """laminate living room floor | 0.8 | 0.3-0.6 | scuffed plank laminate
plain ceiling lamp | 0.7 | 0.05-0.2 | fabric shade
skirting board along the wall | 0.55 | 0.1-0.25 | painted with scuffs
light switch by the doorway | 0.5 | 0.03-0.1 | yellowed plastic plate
socket with cables plugged in | 0.6 | 0.05-0.15 | tangled leads
window reveal with a deep sill | 0.5 | 0.1-0.3 | painted plaster
ceiling rose around the light | 0.35 | 0.05-0.15 | moulded plaster
door architrave by the hall | 0.45 | 0.1-0.25 | chipped paint
carpet edge meeting the laminate | 0.4 | 0.05-0.2 | metal threshold strip
picture hook marks on the wall | 0.35 | 0.05-0.15 | filled nail holes
scuffed corner where the sofa touches | 0.4 | 0.05-0.15 | rubbed paint""",
            "props": """remote controls on the table | 0.85 | 0.05-0.15 | worn rubber buttons
mug on a coaster | 0.75 | 0.05-0.12 | tea-stained rim
charging cable across the floor | 0.55 | 0.05-0.15 | tangled white cable
open book left face down | 0.55 | 0.03-0.1 | bent spine
bowl with a few crumbs | 0.5 | 0.03-0.1 | smeared glaze
tissue box on the side table | 0.5 | 0.03-0.1 | dented cardboard
child's drawing on the table | 0.45 | 0.03-0.1 | felt tip on paper
headphones left on the arm | 0.45 | 0.03-0.1 | cracked ear pad
glasses folded on the book | 0.5 | 0.02-0.08 | smudged lenses
half-eaten packet of biscuits | 0.45 | 0.03-0.1 | rolled-down wrapper
phone face down on the cushion | 0.55 | 0.02-0.08 | scratched case""",
            "foreground_element": """blanket crumpled on the sofa arm | 0.9 | 0.15-0.35 | pilled fleece
open laptop on the sofa | 0.6 | 0.1-0.25 | fingerprinted lid
feet up on the coffee table | 0.55 | 0.1-0.25 | socked feet
cushion pulled onto the lap | 0.6 | 0.1-0.3 | squashed filling
mug held in both hands | 0.55 | 0.05-0.2 | steam rising
remote held towards the television | 0.5 | 0.05-0.15 | worn buttons
knees drawn up on the seat | 0.5 | 0.1-0.3 | creased trouser fabric
phone raised for a photo | 0.5 | 0.05-0.15 | reflection in the screen
cat curled on the cushion | 0.4 | 0.1-0.25 | soft fur
plate balanced on the arm | 0.4 | 0.05-0.15 | crumbs on the rim
book held open on the lap | 0.45 | 0.05-0.2 | thumbed pages""",
            "time_of_day": INDOOR_TIME,
            "weather": INDOOR_WEATHER,
        },
        "bathroom_mirror": {
            "background": """bathroom mirror over the basin | 1.0 | 0.4-0.8 | slightly spotted glass
plain tiled bathroom wall | 0.95 | 0.5-0.9 | white tiles with grey grout
shower curtain half drawn | 0.55 | 0.2-0.5 | creased plastic
tiled shower recess behind | 0.6 | 0.25-0.55 | grouted wall tiles
mirrored cabinet door | 0.6 | 0.25-0.55 | smeared mirror front
towel rail with used towels | 0.8 | 0.15-0.4 | worn terry cloth
small frosted window | 0.6 | 0.1-0.3 | textured glass
painted wall above the tiles | 0.65 | 0.2-0.5 | flat paint with steam marks
door with a hook and dressing gown | 0.5 | 0.2-0.5 | painted timber
shelf of bottles along the wall | 0.7 | 0.15-0.4 | crowded plastic bottles
radiator towel warmer | 0.45 | 0.1-0.3 | chrome bars""",
            "midground": """ceramic basin with taps | 1.0 | 0.2-0.5 | chipped white ceramic
narrow shelf under the mirror | 0.8 | 0.15-0.4 | crowded glass shelf
toilet beside the basin | 0.55 | 0.2-0.5 | white ceramic with a worn seat
washing machine squeezed in | 0.4 | 0.2-0.45 | scuffed white enamel
laundry basket in the corner | 0.6 | 0.15-0.35 | cracked plastic weave
bath edge in the frame | 0.5 | 0.2-0.5 | stained enamel rim
bin beside the basin | 0.5 | 0.05-0.2 | pedal bin with a liner
stool holding folded towels | 0.4 | 0.1-0.3 | painted wooden top
scales pushed against the wall | 0.4 | 0.05-0.2 | scratched glass top
cabinet under the basin | 0.5 | 0.15-0.4 | swollen chipboard door
bottle of cleaner behind the toilet | 0.35 | 0.05-0.15 | dusty trigger spray""",
            "architecture_detail": """strip light above the mirror | 0.8 | 0.05-0.2 | plain fluorescent tube
tiled bathroom floor | 0.75 | 0.2-0.5 | small square tiles
sealant line around the basin | 0.6 | 0.05-0.15 | discoloured silicone
extractor grille in the wall | 0.5 | 0.03-0.12 | dusty plastic vent
tile edge trim | 0.45 | 0.05-0.15 | plastic edging strip
skirting tiles at the floor | 0.45 | 0.05-0.2 | grouted border
pull cord for the light | 0.4 | 0.03-0.1 | greyed cord
grout darkened in the corners | 0.55 | 0.05-0.2 | mildew-marked grout
door lock with a worn catch | 0.35 | 0.03-0.1 | scratched chrome
pipe boxing beside the basin | 0.4 | 0.1-0.25 | painted plywood
limescale ring in the basin | 0.5 | 0.03-0.12 | chalky deposit""",
            "props": """toothbrushes in a cup | 0.9 | 0.05-0.15 | plastic mug with limescale
hairbrush with loose hair | 0.7 | 0.05-0.12 | plastic bristles
half-used tubes and bottles | 0.85 | 0.05-0.2 | squeezed plastic
hair tie left on the shelf | 0.55 | 0.02-0.08 | stretched elastic
soap bar in a dish | 0.6 | 0.02-0.08 | softened edges
folded flannel over the tap | 0.5 | 0.02-0.08 | damp cotton
deodorant can on the shelf | 0.55 | 0.03-0.1 | scuffed label
comb beside the basin | 0.45 | 0.02-0.08 | dusty teeth
cotton buds in a tub | 0.45 | 0.02-0.08 | clear plastic tub
razor on the basin edge | 0.45 | 0.02-0.08 | water spots
hand cream with the lid off | 0.4 | 0.02-0.08 | greasy tube""",
            "foreground_element": """phone raised for a mirror photo | 0.8 | 0.05-0.2 | fingerprint-smeared screen
toothpaste splashes on the mirror | 0.5 | 0.05-0.15 | dried white specks
hand adjusting hair in the mirror | 0.65 | 0.1-0.25 | strands between fingers
towel held against the chest | 0.5 | 0.1-0.3 | damp terry cloth
water running from the tap | 0.55 | 0.05-0.15 | thin stream
steam fogging the mirror edge | 0.5 | 0.1-0.3 | soft condensation
toothbrush raised to the mouth | 0.45 | 0.05-0.15 | wet bristles
hairdryer held up | 0.4 | 0.05-0.2 | tangled cable
tube squeezed in one hand | 0.4 | 0.03-0.12 | creased tube
mirror reflection slightly off-centre | 0.5 | 0.1-0.3 | doubled edge
hair clip held between the lips | 0.35 | 0.02-0.08 | worn plastic clip""",
            "time_of_day": INDOOR_TIME,
            "weather": INDOOR_WEATHER,
        },
        "supermarket_aisle_de": {
            "background": """long supermarket shelving aisle | 1.0 | 0.6-1.0 | packed shelves with price rails
chiller cabinets along the wall | 0.8 | 0.4-0.8 | glass doors with condensation
promotional signs overhead | 0.7 | 0.2-0.5 | printed cardboard
end-of-aisle display stack | 0.7 | 0.3-0.6 | shrink-wrapped cartons
aisle number sign hanging down | 0.6 | 0.1-0.3 | printed plastic panel
freezer island in the next aisle | 0.5 | 0.25-0.55 | frosted glass lids
back wall with staff door | 0.4 | 0.2-0.5 | scuffed swing door
bakery counter further along | 0.45 | 0.2-0.5 | lit glass front
shelf gap where stock ran out | 0.55 | 0.15-0.4 | empty metal shelf
drinks shelving wall | 0.6 | 0.3-0.6 | rows of bottles
checkout lights in the distance | 0.4 | 0.15-0.4 | glowing number signs""",
            "midground": """shopping trolley half full | 0.85 | 0.2-0.5 | scuffed steel basket
pallet of boxed goods mid-aisle | 0.75 | 0.2-0.5 | shrink-wrapped cardboard
stacked crates of drinks | 0.7 | 0.2-0.5 | plastic crates
hand basket on the floor | 0.5 | 0.05-0.2 | red plastic handles
roll cage left by the shelves | 0.5 | 0.2-0.45 | steel mesh sides
shelf edge with price rails | 0.8 | 0.2-0.5 | printed paper strips
another shopper further down | 0.55 | 0.15-0.4 | blurred coat
pallet truck parked to one side | 0.4 | 0.15-0.35 | chipped yellow paint
cardboard tray of tins | 0.6 | 0.1-0.3 | torn shelf-ready packaging
wet floor sign mid-aisle | 0.35 | 0.1-0.3 | yellow folding plastic
special offer bin | 0.5 | 0.15-0.4 | dented metal bin""",
            "architecture_detail": """speckled supermarket floor | 0.9 | 0.3-0.6 | polished terrazzo-look vinyl
strip lighting overhead | 0.85 | 0.1-0.3 | bare fluorescent tubes
low suspended ceiling grid | 0.6 | 0.15-0.4 | plain white panels
scuff marks along the shelf base | 0.6 | 0.05-0.2 | rubbed metal kickplate
floor joint running down the aisle | 0.45 | 0.1-0.3 | sealed vinyl seam
ceiling ducting above the aisle | 0.45 | 0.1-0.3 | painted steel trunking
sprinkler heads in the ceiling | 0.35 | 0.03-0.12 | brass fittings
security mirror in the corner | 0.35 | 0.05-0.15 | convex plastic
shelf uprights with slots | 0.55 | 0.1-0.3 | perforated steel
worn floor near the chiller | 0.4 | 0.1-0.3 | dulled vinyl
cable tray above the shelving | 0.35 | 0.05-0.2 | galvanised tray""",
            "props": """paper price tags on the rail | 0.85 | 0.05-0.15 | printed yellow labels
empty cardboard on the floor | 0.55 | 0.05-0.15 | torn box
promotional wobbler on a shelf | 0.5 | 0.03-0.1 | curled printed card
receipt dropped in the aisle | 0.4 | 0.02-0.08 | curled thermal paper
carrier bags stuffed in the trolley | 0.6 | 0.05-0.15 | crumpled plastic
tin left on the wrong shelf | 0.45 | 0.03-0.1 | dusty lid
shelf gap label for a missing item | 0.45 | 0.02-0.08 | orange printed strip
loyalty card held in hand | 0.4 | 0.02-0.08 | scratched plastic
shopping list on the trolley handle | 0.5 | 0.02-0.08 | biro on paper
bottle cap on the floor | 0.35 | 0.02-0.06 | scratched plastic cap
weighing scales at the aisle end | 0.35 | 0.05-0.15 | worn keypad""",
            "foreground_element": """trolley handle in the near frame | 0.8 | 0.1-0.3 | worn plastic grip
product held up to read the label | 0.65 | 0.05-0.2 | glossy packaging
hand reaching towards a shelf | 0.6 | 0.1-0.25 | sleeve edge in frame
basket carried at the hip | 0.5 | 0.1-0.3 | plastic handle
phone with a list open | 0.5 | 0.05-0.15 | lit screen
coat sleeve crossing the frame | 0.45 | 0.1-0.25 | creased shell fabric
items piled in the trolley | 0.6 | 0.15-0.35 | mixed packaging
purse held in one hand | 0.4 | 0.05-0.15 | worn zip
shelf edge close to the camera | 0.5 | 0.1-0.3 | metal lip
carrier bag hooked on the arm | 0.4 | 0.05-0.2 | stretched handles
child's hand on the trolley | 0.35 | 0.05-0.15 | small fingers""",
            "time_of_day": """flat store lighting throughout | 1.0 | - | -
quiet early opening hour | 0.7 | - | -
busy late afternoon | 0.85 | - | -
after-work rush | 0.8 | - | -
mid morning lull | 0.75 | - | -
saturday crowding | 0.7 | - | -
near closing time | 0.6 | - | -
lunchtime queueing | 0.65 | - | -
restocking in progress | 0.55 | - | -
just after opening | 0.6 | - | -""",
            "weather": """wet floor by the entrance | 0.4 | - | -
no weather visible inside | 0.9 | - | -
draught from the sliding doors | 0.55 | - | -
cold draught from the chilled shelves | 0.5 | - | -
damp air near the entrance | 0.6 | - | -
umbrellas dripping in the aisle | 0.35 | - | -
warm air near the bakery | 0.45 | - | -
condensation on the chiller glass | 0.6 | - | -
muddy footprints on the floor | 0.35 | - | -
bright sun at the sliding doors | 0.4 | - | -""",
        },
    },
    "outdoor/everyday_de": {
        "apartment_balcony": {
            "background": """facade of the opposite block | 1.0 | 0.5-0.9 | rendered wall with small windows
balcony railing across the view | 0.95 | 0.3-0.7 | painted steel bars
rooftops beyond the courtyard | 0.65 | 0.2-0.5 | tiled roofs and aerials
neighbouring balconies in a row | 0.7 | 0.3-0.6 | mixed clutter behind railings
bare trees in the courtyard | 0.6 | 0.2-0.5 | thin branches
parked cars below | 0.5 | 0.15-0.4 | dusty roofs
sky above the roofline | 0.8 | 0.3-0.6 | flat pale cloud
satellite dishes on the wall | 0.4 | 0.1-0.3 | greyed plastic
window with a roller shutter | 0.55 | 0.2-0.5 | slatted plastic shutter
line of washing on a neighbour balcony | 0.45 | 0.1-0.35 | hanging fabric
distant tram wires | 0.35 | 0.1-0.3 | thin overhead cables""",
            "midground": """drying rack on the balcony | 0.85 | 0.2-0.5 | folding aluminium frame
plastic balcony chair | 0.8 | 0.15-0.4 | weathered white plastic
planters along the railing | 0.75 | 0.15-0.4 | cracked terracotta
small folding table | 0.6 | 0.15-0.35 | scratched plastic top
storage box against the wall | 0.5 | 0.15-0.35 | sun-faded lid
bicycle wedged in the corner | 0.4 | 0.15-0.4 | rusted chain
watering can beside the planters | 0.6 | 0.05-0.2 | faded plastic
laundry basket on the floor | 0.55 | 0.1-0.3 | cracked plastic weave
parasol folded against the wall | 0.35 | 0.1-0.3 | dusty canvas
herb pots on the ledge | 0.55 | 0.05-0.2 | dry soil surface
broom leaning in the corner | 0.4 | 0.1-0.25 | worn bristles""",
            "architecture_detail": """concrete balcony floor | 0.85 | 0.3-0.6 | stained concrete slab
downpipe along the wall | 0.55 | 0.05-0.2 | painted metal pipe
railing fixings in the concrete | 0.5 | 0.05-0.15 | rust-streaked bolts
threshold at the balcony door | 0.55 | 0.05-0.2 | worn aluminium strip
drainage hole in the floor | 0.4 | 0.03-0.1 | stained opening
render cracked at the corner | 0.45 | 0.05-0.2 | hairline cracks
ceiling of the balcony above | 0.5 | 0.15-0.4 | painted soffit
wall light beside the door | 0.4 | 0.03-0.12 | dusty bulkhead fitting
moss in the floor joints | 0.4 | 0.05-0.15 | dark green growth
paint peeling on the railing | 0.5 | 0.05-0.2 | flaking coats
balcony door frame | 0.5 | 0.1-0.3 | weathered timber""",
            "props": """watering can in the corner | 0.6 | 0.05-0.15 | faded plastic
ashtray on the railing | 0.4 | 0.02-0.08 | glass with ash
laundry pegs on the rack | 0.6 | 0.02-0.1 | mixed plastic pegs
mug left on the ledge | 0.55 | 0.03-0.1 | chipped ceramic
plant saucer with rainwater | 0.5 | 0.03-0.1 | algae ring
empty bottle beside the chair | 0.35 | 0.03-0.1 | dusty glass
folded towel over the railing | 0.5 | 0.05-0.15 | sun-stiffened cotton
seed packet on the table | 0.35 | 0.02-0.06 | faded paper
small trowel in a pot | 0.4 | 0.02-0.08 | soil-crusted blade
doormat at the balcony door | 0.45 | 0.05-0.15 | bristly worn mat
cigarette packet on the table | 0.3 | 0.02-0.06 | crumpled card""",
            "foreground_element": """hands hanging up washing | 0.7 | 0.1-0.3 | damp fabric
mug set on the railing | 0.6 | 0.05-0.15 | chipped ceramic
elbows resting on the railing | 0.6 | 0.1-0.3 | sleeve against metal
washing basket held at the hip | 0.5 | 0.1-0.3 | plastic rim
watering a pot with the can | 0.5 | 0.1-0.25 | thin stream of water
phone held over the railing | 0.45 | 0.05-0.15 | lit screen
cardigan pulled around the shoulders | 0.5 | 0.1-0.3 | loose knit
bare feet on the concrete | 0.35 | 0.05-0.2 | cool grey slab
laundry peg held in the mouth | 0.35 | 0.02-0.08 | worn plastic peg
leaning over to look down | 0.4 | 0.1-0.3 | hair falling forward
towel shaken out over the railing | 0.4 | 0.1-0.3 | snapping fabric""",
            "time_of_day": OUTDOOR_TIME,
            "weather": OUTDOOR_WEATHER,
        },
        "bus_stop_de": {
            "background": """bus shelter with glass panels | 1.0 | 0.4-0.8 | scratched perspex
street with parked cars | 0.9 | 0.4-0.8 | tarmac with kerbs
row of houses across the road | 0.75 | 0.3-0.6 | rendered facades
hedge behind the shelter | 0.6 | 0.2-0.5 | untrimmed foliage
shop fronts further along | 0.55 | 0.2-0.5 | mixed signage
bare street trees along the kerb | 0.6 | 0.2-0.5 | thin trunks
zebra crossing down the road | 0.45 | 0.15-0.4 | worn white stripes
apartment block behind | 0.55 | 0.3-0.6 | balconies in rows
road sign on a post | 0.5 | 0.1-0.3 | reflective face
bus approaching in the distance | 0.4 | 0.15-0.4 | blurred front panel
overhead tram wires | 0.35 | 0.1-0.3 | thin cables""",
            "midground": """timetable case on the post | 0.85 | 0.1-0.3 | scratched plastic cover
metal bench in the shelter | 0.8 | 0.15-0.4 | perforated painted steel
bin beside the shelter | 0.65 | 0.05-0.2 | dented metal
bus stop flag on the pole | 0.8 | 0.1-0.3 | faded green sign
another person waiting | 0.55 | 0.15-0.4 | dark coat
bicycle locked to the post | 0.45 | 0.15-0.35 | rusted frame
advertising panel on the side | 0.6 | 0.2-0.5 | backlit poster
kerb edge along the stop | 0.7 | 0.1-0.3 | chipped concrete
parked car at the kerb | 0.6 | 0.2-0.5 | rain-spotted paint
puddle at the kerb line | 0.45 | 0.05-0.2 | still brown water
push bike leaning on the shelter | 0.35 | 0.15-0.35 | mud on the tyres""",
            "architecture_detail": """paving slabs at the stop | 0.85 | 0.2-0.5 | uneven concrete slabs
kerb with faded markings | 0.6 | 0.1-0.3 | chipped painted kerb
shelter roof edge | 0.6 | 0.1-0.3 | streaked plastic sheet
drain grate by the kerb | 0.5 | 0.03-0.12 | rusted cast iron
tactile paving strip | 0.45 | 0.05-0.2 | studded slabs
bolt plates at the shelter base | 0.4 | 0.03-0.1 | rust-stained steel
tarmac patch in the road | 0.5 | 0.1-0.3 | darker repair
cracked slab underfoot | 0.5 | 0.05-0.2 | hairline cracks
lamp post beside the shelter | 0.5 | 0.1-0.3 | painted steel column
weeds in the paving joints | 0.45 | 0.03-0.12 | thin green growth
white line along the bus bay | 0.4 | 0.05-0.2 | worn road paint""",
            "props": """stickers on the shelter glass | 0.55 | 0.02-0.1 | half-peeled stickers
cigarette ends by the bench | 0.45 | 0.02-0.08 | trodden filters
advertising poster in a frame | 0.6 | 0.1-0.3 | sun-bleached print
timetable page behind glass | 0.65 | 0.05-0.2 | curled paper
coffee cup left on the bench | 0.45 | 0.03-0.1 | stained lid
free newspaper on the seat | 0.5 | 0.05-0.15 | creased newsprint
chewing gum marks on the paving | 0.4 | 0.02-0.08 | flattened grey spots
graffiti tag on the panel | 0.4 | 0.05-0.15 | marker scrawl
ticket in a gloved hand | 0.35 | 0.02-0.06 | printed slip
bottle in the bin opening | 0.35 | 0.02-0.08 | scratched plastic
leaflet stuck to the wet slab | 0.3 | 0.02-0.06 | soaked paper""",
            "foreground_element": """phone checked for the departure | 0.7 | 0.05-0.2 | lit screen
shopping bag set down on the paving | 0.6 | 0.05-0.2 | slumped carrier bag
hand gripping the shelter frame | 0.45 | 0.05-0.15 | cold metal
coat collar pulled up | 0.55 | 0.1-0.3 | creased fabric
breath visible in the cold | 0.4 | 0.05-0.2 | faint vapour
bag strap across the chest | 0.5 | 0.1-0.3 | worn webbing
looking down the road for the bus | 0.6 | 0.1-0.3 | turned shoulder
umbrella held closed | 0.45 | 0.05-0.2 | wet fabric
foot resting on the kerb | 0.4 | 0.05-0.2 | scuffed shoe
ticket held ready | 0.35 | 0.02-0.08 | folded slip
hood pulled forward against the wind | 0.45 | 0.1-0.3 | flapping edge""",
            "time_of_day": OUTDOOR_TIME,
            "weather": OUTDOOR_WEATHER,
        },
        "pedestrian_zone": {
            "background": """row of shop fronts | 1.0 | 0.5-0.9 | mixed signage and awnings
paved pedestrian street | 0.95 | 0.4-0.8 | patterned block paving
church tower further down | 0.45 | 0.15-0.4 | weathered stone
upper floors above the shops | 0.7 | 0.3-0.6 | rendered facades with windows
bakery window with trays | 0.5 | 0.2-0.5 | lit glass front
pharmacy sign projecting out | 0.45 | 0.1-0.3 | illuminated green cross
crowd further along the street | 0.6 | 0.2-0.5 | blurred coats
department store entrance | 0.5 | 0.25-0.55 | glass doors
street narrowing into an alley | 0.4 | 0.2-0.5 | shaded gap
awnings pulled out over the paving | 0.5 | 0.2-0.5 | striped canvas
banner strung across the street | 0.35 | 0.1-0.3 | printed vinyl""",
            "midground": """bike racks with locked bikes | 0.75 | 0.2-0.5 | rusted frames
café tables outside a shop | 0.7 | 0.2-0.5 | folding metal furniture
planted tub in the middle of the street | 0.65 | 0.1-0.3 | concrete tub
bench along the paving | 0.6 | 0.15-0.4 | slatted timber
market stall at the corner | 0.45 | 0.2-0.5 | striped canopy
bin beside the bench | 0.6 | 0.05-0.2 | dented metal
A-board with the day's offer | 0.6 | 0.05-0.2 | chalked board
pram pushed along the street | 0.4 | 0.1-0.3 | worn fabric hood
delivery trolley by a doorway | 0.4 | 0.1-0.3 | steel frame
fountain surround in the square | 0.35 | 0.2-0.5 | wet stone rim
street musician with a case open | 0.3 | 0.1-0.3 | worn instrument case""",
            "architecture_detail": """drainage channel in the paving | 0.5 | 0.05-0.2 | worn stone channel
bollards at the street entrance | 0.55 | 0.05-0.2 | scuffed steel posts
block paving pattern underfoot | 0.8 | 0.25-0.55 | interlocking pavers
kerbless flush surface | 0.5 | 0.15-0.4 | level paving
manhole cover in the paving | 0.45 | 0.03-0.12 | worn cast iron
shop threshold step | 0.5 | 0.05-0.2 | worn stone lip
lamp column with a bracket | 0.5 | 0.1-0.3 | painted steel
tree grille around a trunk | 0.4 | 0.05-0.2 | cast metal grid
paving sunken near a drain | 0.4 | 0.05-0.2 | dished surface
downpipe on a shop wall | 0.4 | 0.05-0.2 | painted metal
weeds between the pavers | 0.4 | 0.03-0.12 | thin green shoots""",
            "props": """pigeons near a bin | 0.5 | 0.05-0.15 | scruffy feathers
bin overflowing slightly | 0.5 | 0.05-0.15 | dented metal
chalk price board outside a café | 0.55 | 0.05-0.2 | smudged chalk
flyer trodden into the paving | 0.4 | 0.02-0.08 | dirty paper
bicycle basket with a bag in it | 0.4 | 0.03-0.12 | wire basket
menu stand by the tables | 0.45 | 0.05-0.15 | laminated card
coins in a busker's case | 0.3 | 0.02-0.06 | dull metal
shopping bags leaning on a bench | 0.5 | 0.05-0.2 | printed carrier bags
cigarette bin on a wall | 0.35 | 0.02-0.08 | stained steel tube
sale stickers in a window | 0.45 | 0.05-0.15 | bright printed circles
dog lead tied to a rail | 0.3 | 0.02-0.08 | worn strap""",
            "foreground_element": """shopping bag in each hand | 0.7 | 0.1-0.3 | printed carrier bags
phone held up for directions | 0.5 | 0.05-0.2 | lit screen
coat over the forearm | 0.5 | 0.1-0.3 | creased fabric
pram handle in the near frame | 0.4 | 0.1-0.3 | worn grip
paper bag from the bakery | 0.5 | 0.05-0.2 | crumpled paper
sunglasses pushed up into the hair | 0.45 | 0.03-0.12 | scratched lenses
purse held open in one hand | 0.4 | 0.05-0.15 | worn lining
bag strap slipping off the shoulder | 0.45 | 0.1-0.3 | webbing strap
takeaway cup in hand | 0.5 | 0.05-0.15 | printed sleeve
stepping over the drainage channel | 0.35 | 0.1-0.25 | shoe mid-stride
hair blown across the face | 0.4 | 0.05-0.2 | loose strands""",
            "time_of_day": OUTDOOR_TIME,
            "weather": OUTDOOR_WEATHER,
        },
        "playground_de": {
            "background": """playground with a climbing frame | 1.0 | 0.5-0.9 | weathered timber and rope
hedge and trees around the edge | 0.85 | 0.3-0.6 | patchy foliage
apartment blocks beyond | 0.65 | 0.2-0.5 | rendered facades
football goal at the far end | 0.45 | 0.15-0.4 | bent metal frame
path leading into the park | 0.6 | 0.2-0.5 | patched asphalt
low fence around the area | 0.65 | 0.15-0.4 | painted timber rails
sandpit at the far side | 0.55 | 0.2-0.5 | damp sand
bare trees behind the frame | 0.6 | 0.25-0.55 | thin branches
parked cars beyond the hedge | 0.4 | 0.15-0.4 | dusty roofs
notice board at the entrance | 0.4 | 0.1-0.3 | faded park rules
open grass area beside | 0.6 | 0.25-0.55 | worn patchy grass""",
            "midground": """swings on a metal frame | 0.9 | 0.2-0.5 | chipped paint and chains
sandpit with toys left behind | 0.75 | 0.2-0.5 | damp sand
bench facing the playground | 0.8 | 0.15-0.4 | slatted wood
slide with a worn chute | 0.7 | 0.2-0.5 | scratched plastic
seesaw at rest | 0.5 | 0.15-0.4 | faded painted timber
spring rider on a base | 0.45 | 0.1-0.3 | cracked plastic seat
bin beside the bench | 0.55 | 0.05-0.2 | dented metal
pram parked by the bench | 0.5 | 0.1-0.3 | worn fabric hood
scooters left on the path | 0.5 | 0.05-0.2 | scuffed decks
climbing net between posts | 0.45 | 0.15-0.4 | frayed rope
picnic table to one side | 0.35 | 0.15-0.4 | weathered planks""",
            "architecture_detail": """rubber safety matting | 0.7 | 0.2-0.5 | worn black tiles
bark chippings under the frame | 0.6 | 0.2-0.5 | damp wood chips
concrete path edge | 0.5 | 0.1-0.3 | chipped kerb
posts set into concrete collars | 0.5 | 0.05-0.2 | cracked concrete
worn dirt patch under the swings | 0.65 | 0.1-0.3 | compacted bare earth
gate in the low fence | 0.45 | 0.1-0.3 | sprung metal latch
drain grate near the path | 0.35 | 0.03-0.12 | leaf-clogged grate
bolt heads on the climbing frame | 0.45 | 0.03-0.12 | rounded steel caps
paint worn from the handrails | 0.5 | 0.05-0.2 | bare metal patches
lamp post at the path corner | 0.4 | 0.1-0.3 | painted steel column
kerb between grass and path | 0.4 | 0.05-0.2 | mossy concrete""",
            "props": """plastic bucket and spade | 0.7 | 0.02-0.1 | faded plastic
scooter parked against the bench | 0.55 | 0.05-0.15 | scuffed deck
drinks bottle on the bench | 0.5 | 0.02-0.08 | condensation on plastic
jacket left on the bench arm | 0.55 | 0.05-0.15 | crumpled shell fabric
ball resting in the grass | 0.5 | 0.03-0.12 | scuffed panels
pram bag hooked on the handle | 0.45 | 0.03-0.12 | bulging fabric
snack packet on the bench | 0.45 | 0.02-0.08 | rolled-down wrapper
chalk drawing on the path | 0.35 | 0.05-0.15 | faint chalk lines
lost glove on the fence post | 0.35 | 0.02-0.06 | damp knit
sand toys scattered in the pit | 0.5 | 0.05-0.15 | sun-bleached plastic
wet wipes packet half open | 0.3 | 0.02-0.06 | creased plastic""",
            "foreground_element": """pushing a swing | 0.65 | 0.1-0.3 | chain in hand
jacket left on the bench | 0.55 | 0.05-0.2 | crumpled shell fabric
crouching by the sandpit | 0.5 | 0.1-0.3 | knees in the sand
holding a child's hand | 0.5 | 0.05-0.2 | small fingers
coffee cup held while watching | 0.5 | 0.05-0.15 | printed sleeve
phone held low in one hand | 0.5 | 0.05-0.15 | lit screen
bag on the shoulder while standing | 0.45 | 0.1-0.3 | worn strap
leaning on the fence rail | 0.45 | 0.1-0.3 | flaking paint
brushing sand off a knee | 0.4 | 0.05-0.2 | gritty palm
scooter held by the handlebars | 0.4 | 0.05-0.2 | worn grips
hair tied back while bending down | 0.35 | 0.05-0.2 | loose strands""",
            "time_of_day": OUTDOOR_TIME,
            "weather": OUTDOOR_WEATHER,
        },
    },
}


def write_file(path: pathlib.Path, text: str, force: bool) -> str:
    if path.exists() and not force:
        return "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "write"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    written = skipped = 0
    for category, sets in SETS.items():
        for set_name, elements in sets.items():
            target = ROOT / "location_lists" / pathlib.Path(category) / set_name
            missing = [e for e in ELEMENTS if e not in elements]
            if missing:
                print(f"ERROR {category}/{set_name}: missing {missing}")
                return 1
            for element, body in elements.items():
                lines = [ln for ln in body.strip().splitlines() if ln.strip()]
                if len(lines) < 10:
                    print(f"ERROR {category}/{set_name}/{element}: "
                          f"only {len(lines)} entries, need >= 10")
                    return 1
                result = write_file(target / f"{element}.txt",
                                    HEADER[element] + body.strip() + "\n", args.force)
                written += result == "write"
                skipped += result == "skip"

    total = sum(len(v) for v in SETS.values())
    print(f"locations: {written} written, {skipped} skipped ({total} sets)")
    if skipped:
        print("Use --force to overwrite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
