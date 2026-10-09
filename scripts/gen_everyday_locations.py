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
    "indoor/german_everyday_scene": {
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
    },
    "outdoor/german_everyday_scene": {
        "german_bus_stop": {
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
        "german_playground": {
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
