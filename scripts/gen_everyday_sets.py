"""Generator für die Alltags-Sets (Outfits + Locations).

Die bestehenden Ordner sind stark thematisch (opera, startrek, burlesque) oder
US-zentriert. Was fehlte, war unspektakulärer Alltag: Kleidung, die man zum
Einkaufen anzieht, und Orte, an denen man tatsächlich täglich steht — hier mit
deutschem/europäischem Zuschnitt.

Leitlinie für die Inhalte: nichts, was nach Styling aussieht. Getragene, einfache
Sachen, Supermarkt-Marken, verwaschene Farben. Für Krea 2 heißt das auch: keine
Slogans (die rendert das Modell als echten Text) und keine Designer-Farbwörter.

Ausführen:  python scripts/gen_everyday_sets.py [--force]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

OUTFIT_HEADER = {
    "top": "# slot: top\n# format: garment_name | probability | formality_min-formality_max | fabric1,fabric2,...\n\n",
    "bottom": "# slot: bottom\n# format: garment_name | probability | formality_min-formality_max | fabric1,fabric2,...\n\n",
    "footwear": "# slot: footwear\n# format: garment_name | probability | formality_min-formality_max | fabric1,fabric2,...\n\n",
    "outerwear": "# slot: outerwear\n# format: garment_name | probability | formality_min-formality_max | fabric1,fabric2,...\n\n",
    "headwear": "# slot: headwear\n# format: garment_name | probability | formality_min-formality_max | fabric1,fabric2,...\n\n",
    "accessories": "# slot: accessories\n# format: garment_name | probability | formality_min-formality_max | fabric1,fabric2,...\n\n",
    "bag": "# slot: bag\n# format: garment_name | probability | formality_min-formality_max | fabric1,fabric2,...\n\n",
    "fabrics": "# fabric database\n# format: fabric_name | formality | family | weight\n\n",
    "prints": "# prints/patterns\n# format: print_name | probability | compatible_slots | formality_min-formality_max\n\n",
    "texts": "# texts/slogans\n# format: text_value | probability | compatible_slots | font_or_descriptor\n\n",
}

LOCATION_HEADER = {
    key: f"# element: {key}\n# format: name | probability | coverage_range | texture\n\n"
    for key in ("background", "midground", "architecture_detail", "props",
                "foreground_element", "time_of_day", "weather")
}

# Shared fabric table — plain everyday materials only.
FABRICS = """cotton | 0.3 | natural | medium
jersey | 0.2 | natural | light
denim | 0.3 | natural | heavy
fleece | 0.15 | synthetic | medium
polyester | 0.25 | synthetic | light
knit | 0.35 | natural | medium
canvas | 0.2 | natural | medium
corduroy | 0.35 | natural | heavy
flannel | 0.3 | natural | medium
softshell | 0.25 | synthetic | medium
terry cloth | 0.15 | natural | medium
ribbed cotton | 0.25 | natural | light
"""

# Everyday clothes carry almost no pattern, and never a slogan — Krea 2 would
# render the words onto the garment.
PRINTS_PLAIN = """solid color | 0.90 | top,bottom,outerwear | 0.1-0.6
thin stripe pattern | 0.20 | top | 0.1-0.5
small checked pattern | 0.12 | top,outerwear | 0.2-0.5
faded washed-out look | 0.18 | top,bottom | 0.1-0.4
"""
TEXTS_NONE = "# intentionally empty: printed slogans get rendered as real text\n"


# ── Outfit sets ──────────────────────────────────────────────────────────
# slot -> list of "name | prob | form_min-form_max | fabrics"

OUTFIT_SETS: dict[str, dict[str, str]] = {
    "grocery_run": {
        "top": """plain long sleeve tee | 0.90 | 0.1-0.4 | cotton,jersey
simple crew sweatshirt | 0.85 | 0.1-0.3 | cotton,fleece
basic cotton t-shirt | 0.80 | 0.1-0.3 | cotton,jersey
thin knit jumper | 0.70 | 0.2-0.4 | knit
striped long sleeve top | 0.55 | 0.1-0.4 | cotton,jersey""",
        "bottom": """straight-leg jeans | 0.90 | 0.2-0.4 | denim
plain cotton leggings | 0.70 | 0.1-0.3 | cotton,jersey
loose chino trousers | 0.55 | 0.3-0.5 | cotton
comfortable jogger trousers | 0.60 | 0.1-0.3 | cotton,fleece""",
        "footwear": """worn white sneakers | 0.90 | 0.1-0.4 | canvas,leather
simple slip-on trainers | 0.70 | 0.1-0.3 | canvas
flat ankle boots | 0.50 | 0.2-0.5 | leather""",
        "outerwear": """zip-up fleece jacket | 0.75 | 0.1-0.3 | fleece
plain quilted jacket | 0.65 | 0.2-0.4 | polyester
simple denim jacket | 0.55 | 0.2-0.4 | denim
thin rain shell | 0.45 | 0.1-0.3 | softshell""",
        "headwear": """plain knit beanie | 0.35 | 0.1-0.3 | knit
simple baseball cap | 0.25 | 0.1-0.3 | cotton""",
        "accessories": """hair tie on wrist | 0.55 | 0.1-0.4 | cotton
small stud earrings | 0.45 | 0.1-0.4 | metal
plain wristwatch | 0.35 | 0.1-0.5 | metal""",
        "bag": """reusable shopping bag | 0.80 | 0.1-0.4 | canvas,cotton
worn shoulder bag | 0.60 | 0.2-0.4 | canvas,leather
small crossbody bag | 0.50 | 0.2-0.5 | polyester""",
    },
    "home_lounge": {
        "top": """oversized cotton t-shirt | 0.90 | 0.1-0.2 | cotton,jersey
soft hoodie | 0.85 | 0.1-0.2 | cotton,fleece
long sleeve pyjama top | 0.65 | 0.1-0.2 | cotton,flannel
thin ribbed tank top | 0.55 | 0.1-0.2 | ribbed cotton""",
        "bottom": """soft jogging bottoms | 0.90 | 0.1-0.2 | cotton,fleece
cotton pyjama shorts | 0.60 | 0.1-0.2 | cotton
plain cotton leggings | 0.70 | 0.1-0.2 | cotton,jersey
loose lounge trousers | 0.65 | 0.1-0.2 | cotton,terry cloth""",
        "footwear": """thick house socks | 0.85 | 0.1-0.2 | knit,cotton
worn slippers | 0.70 | 0.1-0.2 | fleece
bare feet | 0.45 | 0.1-0.2 | -""",
        "outerwear": """open cardigan | 0.55 | 0.1-0.3 | knit
zip-up fleece | 0.45 | 0.1-0.2 | fleece""",
        "headwear": """hair clipped up loosely | 0.45 | 0.1-0.2 | -
messy bun | 0.55 | 0.1-0.2 | -""",
        "accessories": """scrunchie on wrist | 0.50 | 0.1-0.2 | cotton
reading glasses pushed up | 0.30 | 0.1-0.3 | plastic""",
        "bag": """no bag | 0.90 | 0.1-0.2 | -""",
    },
    "school_run": {
        "top": """plain sweatshirt | 0.90 | 0.1-0.3 | cotton,fleece
long sleeve cotton top | 0.85 | 0.1-0.3 | cotton,jersey
thin knit pullover | 0.65 | 0.2-0.4 | knit""",
        "bottom": """straight-cut jeans | 0.85 | 0.2-0.4 | denim
plain leggings | 0.80 | 0.1-0.3 | cotton,jersey
practical cargo trousers | 0.45 | 0.1-0.3 | cotton,canvas""",
        "footwear": """everyday trainers | 0.90 | 0.1-0.3 | canvas,polyester
flat winter boots | 0.55 | 0.2-0.4 | leather,softshell""",
        "outerwear": """padded winter coat | 0.70 | 0.2-0.4 | polyester
zip fleece jacket | 0.70 | 0.1-0.3 | fleece
light rain jacket | 0.55 | 0.1-0.3 | softshell""",
        "headwear": """knit beanie | 0.45 | 0.1-0.3 | knit
hood pulled up | 0.30 | 0.1-0.3 | -""",
        "accessories": """simple scarf | 0.50 | 0.1-0.4 | knit
car keys in hand | 0.55 | 0.1-0.4 | metal
phone in hand | 0.65 | 0.1-0.4 | -""",
        "bag": """large shoulder tote | 0.70 | 0.1-0.4 | canvas
small backpack | 0.55 | 0.1-0.3 | polyester""",
    },
    "dog_walk": {
        "top": """plain fleece pullover | 0.85 | 0.1-0.3 | fleece
long sleeve cotton shirt | 0.80 | 0.1-0.3 | cotton
thermal long sleeve top | 0.60 | 0.1-0.3 | ribbed cotton""",
        "bottom": """worn jeans | 0.85 | 0.1-0.4 | denim
practical outdoor trousers | 0.65 | 0.1-0.3 | softshell,polyester
warm leggings | 0.55 | 0.1-0.3 | cotton,fleece""",
        "footwear": """muddy walking shoes | 0.85 | 0.1-0.3 | leather,canvas
rubber wellington boots | 0.55 | 0.1-0.3 | rubber
worn trainers | 0.65 | 0.1-0.3 | canvas""",
        "outerwear": """practical rain jacket | 0.80 | 0.1-0.3 | softshell
padded gilet | 0.60 | 0.1-0.3 | polyester
old quilted coat | 0.55 | 0.1-0.3 | polyester""",
        "headwear": """knit hat | 0.50 | 0.1-0.3 | knit
cap against drizzle | 0.30 | 0.1-0.3 | cotton""",
        "accessories": """dog lead over shoulder | 0.85 | 0.1-0.3 | -
gloves half tucked in pocket | 0.40 | 0.1-0.3 | knit""",
        "bag": """small pouch for treats | 0.55 | 0.1-0.3 | polyester""",
    },
    "summer_errands": {
        "top": """simple cotton t-shirt | 0.90 | 0.1-0.3 | cotton,jersey
loose linen-look blouse | 0.60 | 0.2-0.4 | cotton
plain tank top | 0.70 | 0.1-0.3 | cotton,jersey""",
        "bottom": """denim shorts | 0.75 | 0.1-0.3 | denim
plain cotton skirt | 0.55 | 0.2-0.4 | cotton
light summer trousers | 0.60 | 0.2-0.4 | cotton""",
        "footwear": """flat sandals | 0.80 | 0.1-0.3 | leather
canvas trainers | 0.70 | 0.1-0.3 | canvas
simple flip-flops | 0.45 | 0.1-0.2 | rubber""",
        "outerwear": """thin cotton overshirt | 0.35 | 0.1-0.3 | cotton""",
        "headwear": """sunglasses pushed into hair | 0.55 | 0.1-0.3 | plastic
plain cap | 0.30 | 0.1-0.3 | cotton""",
        "accessories": """simple sunglasses | 0.65 | 0.1-0.4 | plastic
thin bracelet | 0.35 | 0.1-0.4 | metal""",
        "bag": """woven shopping bag | 0.70 | 0.1-0.3 | canvas
small crossbody bag | 0.55 | 0.2-0.4 | leather""",
    },
    "winter_errands": {
        "top": """thick knit jumper | 0.90 | 0.2-0.4 | knit
fleece-lined long sleeve | 0.75 | 0.1-0.3 | fleece
plain thermal top | 0.60 | 0.1-0.3 | ribbed cotton""",
        "bottom": """everyday jeans | 0.85 | 0.2-0.4 | denim
warm lined trousers | 0.65 | 0.2-0.4 | corduroy,polyester
thick leggings | 0.55 | 0.1-0.3 | cotton,fleece""",
        "footwear": """flat winter boots | 0.85 | 0.2-0.4 | leather
lined ankle boots | 0.65 | 0.2-0.5 | leather
chunky trainers | 0.45 | 0.1-0.3 | polyester""",
        "outerwear": """long padded winter coat | 0.90 | 0.2-0.5 | polyester
puffer jacket | 0.75 | 0.1-0.4 | polyester
wool-look coat | 0.50 | 0.3-0.6 | knit""",
        "headwear": """knit beanie | 0.70 | 0.1-0.3 | knit
hood up over hat | 0.30 | 0.1-0.3 | -""",
        "accessories": """thick scarf | 0.80 | 0.1-0.4 | knit
knit gloves | 0.65 | 0.1-0.4 | knit""",
        "bag": """shoulder bag under coat | 0.60 | 0.2-0.4 | leather,canvas
reusable bag in hand | 0.55 | 0.1-0.3 | canvas""",
    },
    "office_plain": {
        "top": """simple button blouse | 0.85 | 0.4-0.7 | cotton
plain fine-knit top | 0.80 | 0.3-0.6 | knit
unfussy shirt | 0.65 | 0.4-0.7 | cotton""",
        "bottom": """straight office trousers | 0.85 | 0.4-0.7 | polyester,cotton
plain smart jeans | 0.55 | 0.3-0.5 | denim
simple knee-length skirt | 0.50 | 0.4-0.7 | polyester""",
        "footwear": """flat leather shoes | 0.80 | 0.3-0.6 | leather
plain low ankle boots | 0.60 | 0.3-0.6 | leather
clean white trainers | 0.45 | 0.2-0.4 | leather""",
        "outerwear": """unstructured blazer | 0.60 | 0.4-0.7 | polyester
plain cardigan | 0.65 | 0.3-0.5 | knit""",
        "headwear": """hair tied back neatly | 0.40 | 0.3-0.6 | -""",
        "accessories": """small stud earrings | 0.55 | 0.3-0.6 | metal
plain watch | 0.45 | 0.3-0.6 | metal
lanyard with badge | 0.35 | 0.3-0.6 | polyester""",
        "bag": """plain work tote | 0.75 | 0.3-0.6 | leather,canvas
laptop shoulder bag | 0.55 | 0.3-0.6 | polyester""",
    },
    "rainy_errands": {
        "top": """plain long sleeve top | 0.85 | 0.1-0.4 | cotton,jersey
thin knit jumper | 0.70 | 0.2-0.4 | knit""",
        "bottom": """jeans with damp hems | 0.80 | 0.1-0.4 | denim
practical trousers | 0.65 | 0.1-0.3 | softshell""",
        "footwear": """rubber ankle boots | 0.80 | 0.1-0.3 | rubber
waterproof trainers | 0.60 | 0.1-0.3 | softshell""",
        "outerwear": """rain jacket with hood | 0.95 | 0.1-0.4 | softshell
transparent rain cape | 0.30 | 0.1-0.3 | plastic""",
        "headwear": """hood pulled up | 0.75 | 0.1-0.3 | -""",
        "accessories": """folded umbrella in hand | 0.70 | 0.1-0.4 | polyester
phone kept under sleeve | 0.35 | 0.1-0.3 | -""",
        "bag": """shoulder bag held close | 0.60 | 0.1-0.4 | polyester""",
    },
}


# ── Location sets ────────────────────────────────────────────────────────

LOCATION_SETS: dict[str, dict[str, dict[str, str]]] = {
    "indoor/everyday_de": {
        "kitchen_cooking": {
            "background": """plain white kitchen wall tiles | 1.0 | 0.5-0.9 | glossy square tiles with grout
kitchen wall cupboards in light wood | 0.9 | 0.4-0.8 | laminate fronts with simple handles
window over the sink | 0.75 | 0.2-0.5 | plain glass with roller blind""",
            "midground": """laminate worktop with clutter | 1.0 | 0.4-0.8 | scratched light laminate
gas hob with used pots | 0.9 | 0.2-0.5 | steel grates with cooking marks
fridge covered in magnets | 0.7 | 0.2-0.5 | white enamel with paper notes""",
            "architecture_detail": """practical ceiling light | 0.8 | 0.1-0.25 | plain glass shade
tiled splashback behind the hob | 0.85 | 0.2-0.4 | wiped-down tiles
vinyl kitchen flooring | 0.75 | 0.3-0.6 | wood-look vinyl planks""",
            "props": """chopping board with vegetable scraps | 0.85 | 0.05-0.15 | knife-marked wood
open spice jars | 0.7 | 0.05-0.1 | mismatched screw-top jars
dish towel over the oven handle | 0.75 | 0.05-0.1 | faded checked cotton
half-drunk mug of coffee | 0.65 | 0.05-0.1 | stained ceramic""",
            "foreground_element": """pot steaming on the hob | 0.9 | 0.1-0.25 | condensation on the lid
open cookbook propped up | 0.5 | 0.05-0.15 | splashed page
dirty plates stacked in the sink | 0.6 | 0.1-0.2 | soap film""",
            "time_of_day": """late afternoon kitchen light | 0.9 | - | -
evening with the ceiling light on | 0.85 | - | -
morning light through the blind | 0.8 | - | -""",
            "weather": """grey daylight outside the window | 0.7 | - | -
rain streaks on the kitchen window | 0.45 | - | -""",
        },
        "bathroom_mirror": {
            "background": """bathroom mirror over the basin | 1.0 | 0.4-0.8 | slightly spotted glass
plain tiled bathroom wall | 0.95 | 0.5-0.9 | white tiles with grey grout
shower curtain half drawn | 0.5 | 0.2-0.5 | creased plastic""",
            "midground": """ceramic basin with taps | 1.0 | 0.2-0.5 | chipped white ceramic
narrow shelf under the mirror | 0.8 | 0.15-0.35 | crowded glass shelf
towel rail with used towels | 0.75 | 0.15-0.35 | worn terry cloth""",
            "architecture_detail": """small frosted bathroom window | 0.6 | 0.1-0.25 | textured glass
strip light above the mirror | 0.8 | 0.05-0.15 | plain fluorescent tube
tiled bathroom floor | 0.7 | 0.2-0.5 | small square tiles""",
            "props": """toothbrushes in a cup | 0.9 | 0.05-0.1 | plastic mug with limescale
hairbrush with loose hair | 0.7 | 0.05-0.1 | plastic bristles
half-used tubes and bottles | 0.85 | 0.05-0.15 | squeezed plastic
hair tie left on the shelf | 0.55 | 0.02-0.05 | stretched elastic""",
            "foreground_element": """phone raised for a mirror photo | 0.8 | 0.05-0.15 | fingerprint-smeared screen
toothpaste splashes on the mirror | 0.5 | 0.05-0.1 | dried white specks""",
            "time_of_day": """harsh bathroom light | 0.9 | - | -
morning with the light on | 0.85 | - | -
evening before bed | 0.6 | - | -""",
            "weather": """no weather visible indoors | 0.9 | - | -""",
        },
        "living_room_sofa": {
            "background": """plain painted living room wall | 1.0 | 0.5-0.9 | flat emulsion in off-white
shelf with books and photos | 0.85 | 0.3-0.6 | mismatched spines
window with curtains half open | 0.7 | 0.2-0.5 | plain lined curtains""",
            "midground": """worn fabric sofa with cushions | 1.0 | 0.4-0.8 | flattened woven upholstery
low coffee table with rings | 0.85 | 0.2-0.5 | marked veneer top
television on a low unit | 0.8 | 0.2-0.5 | dusty black screen""",
            "architecture_detail": """laminate living room floor | 0.8 | 0.3-0.6 | scuffed plank laminate
radiator under the window | 0.6 | 0.1-0.25 | painted steel ribs
plain ceiling lamp | 0.7 | 0.05-0.15 | fabric shade""",
            "props": """remote controls on the table | 0.85 | 0.05-0.1 | worn rubber buttons
laundry basket waiting to be folded | 0.6 | 0.1-0.2 | plastic weave
mug on a coaster | 0.7 | 0.05-0.1 | tea-stained rim
charging cable across the floor | 0.5 | 0.05-0.1 | tangled white cable""",
            "foreground_element": """blanket crumpled on the sofa arm | 0.9 | 0.15-0.3 | pilled fleece
open laptop on the sofa | 0.55 | 0.1-0.2 | fingerprinted lid""",
            "time_of_day": """grey afternoon light | 0.9 | - | -
evening with one lamp on | 0.9 | - | -
morning light across the floor | 0.7 | - | -""",
            "weather": """rain against the window | 0.5 | - | -
overcast light outside | 0.7 | - | -""",
        },
        "supermarket_aisle_de": {
            "background": """long supermarket shelving aisle | 1.0 | 0.6-1.0 | packed shelves with price rails
chiller cabinets along the wall | 0.8 | 0.4-0.8 | glass doors with condensation
promotional signs overhead | 0.7 | 0.2-0.5 | printed cardboard""",
            "midground": """pallet of boxed goods mid-aisle | 0.75 | 0.2-0.5 | shrink-wrapped cardboard
shopping trolley half full | 0.85 | 0.2-0.4 | scuffed steel basket
stacked crates of drinks | 0.65 | 0.2-0.5 | plastic crates""",
            "architecture_detail": """speckled supermarket floor | 0.9 | 0.3-0.6 | polished terrazzo-look vinyl
strip lighting overhead | 0.85 | 0.1-0.3 | bare fluorescent tubes
low suspended ceiling grid | 0.6 | 0.15-0.35 | plain white panels""",
            "props": """paper price tags on the rail | 0.85 | 0.05-0.1 | printed yellow labels
empty cardboard on the floor | 0.55 | 0.05-0.1 | torn box
hand basket left on a shelf | 0.5 | 0.05-0.1 | red plastic""",
            "foreground_element": """trolley handle in the near frame | 0.8 | 0.1-0.25 | worn plastic grip
product held up to read the label | 0.6 | 0.05-0.15 | glossy packaging""",
            "time_of_day": """flat store lighting, no daylight | 1.0 | - | -""",
            "weather": """wet floor sign near the entrance | 0.35 | - | -""",
        },
        "bakery_counter": {
            "background": """glass bakery counter with trays | 1.0 | 0.5-0.9 | fingerprinted glass
shelves of loaves behind the counter | 0.9 | 0.3-0.7 | wooden bread racks
chalk price board on the wall | 0.6 | 0.15-0.35 | handwritten chalk""",
            "midground": """counter top with a till | 0.9 | 0.2-0.5 | wiped stainless steel
paper bag dispenser | 0.6 | 0.05-0.15 | stacked paper bags
small standing table by the window | 0.5 | 0.2-0.4 | laminate top""",
            "architecture_detail": """tiled bakery floor | 0.7 | 0.2-0.5 | worn quarry tiles
warm downlights over the counter | 0.8 | 0.1-0.25 | small halogen spots""",
            "props": """crumbs on the counter | 0.7 | 0.02-0.05 | scattered crumbs
tongs resting in a tray | 0.65 | 0.02-0.05 | steel tongs
coffee machine steaming | 0.55 | 0.05-0.15 | stained chrome""",
            "foreground_element": """paper bag being handed over | 0.7 | 0.05-0.15 | crumpled paper
coins counted out on the counter | 0.5 | 0.02-0.08 | worn coins""",
            "time_of_day": """early morning bakery light | 0.9 | - | -
late morning, half-empty trays | 0.8 | - | -""",
            "weather": """condensation on the shop window | 0.4 | - | -""",
        },
        "tram_interior": {
            "background": """tram window with the city sliding past | 1.0 | 0.4-0.8 | scratched safety glass
row of moulded seats | 0.95 | 0.4-0.8 | patterned worn fabric
grab poles down the aisle | 0.8 | 0.2-0.5 | scuffed yellow metal""",
            "midground": """seat backs in front | 0.9 | 0.3-0.6 | flattened seat fabric
ticket validator by the door | 0.6 | 0.1-0.25 | plastic housing
route map above the window | 0.5 | 0.1-0.25 | printed strip""",
            "architecture_detail": """ribbed rubber tram floor | 0.75 | 0.2-0.5 | studded rubber
ceiling strip lights | 0.7 | 0.1-0.25 | plain diffuser
stop-request buttons | 0.5 | 0.02-0.08 | worn red buttons""",
            "props": """chewing gum marks on the floor | 0.4 | 0.02-0.05 | flattened grey spots
free newspaper left on a seat | 0.5 | 0.05-0.1 | creased newsprint
backpack on the seat beside | 0.6 | 0.1-0.2 | worn nylon""",
            "foreground_element": """hand holding the grab pole | 0.7 | 0.05-0.15 | metal with fingerprints
phone held low in the lap | 0.65 | 0.05-0.15 | lit screen""",
            "time_of_day": """grey commuter morning | 0.9 | - | -
early evening with interior lights on | 0.85 | - | -""",
            "weather": """rain streaking the tram window | 0.5 | - | -
misted-up windows | 0.4 | - | -""",
        },
        "hallway_shoes": {
            "background": """narrow flat hallway wall | 1.0 | 0.5-0.9 | scuffed painted plaster
coat hooks with jackets | 0.9 | 0.3-0.6 | overloaded hooks
front door with a spyhole | 0.8 | 0.3-0.6 | painted steel door""",
            "midground": """shoe rack by the door | 0.95 | 0.2-0.5 | crowded plastic rack
small console with post on it | 0.6 | 0.15-0.35 | cheap veneer
mirror by the door | 0.55 | 0.15-0.35 | smudged glass""",
            "architecture_detail": """doormat inside the door | 0.8 | 0.1-0.25 | bristly worn mat
laminate hallway floor | 0.75 | 0.3-0.6 | scratched planks
hallway ceiling light | 0.65 | 0.05-0.15 | plain shade""",
            "props": """keys hanging on a hook | 0.7 | 0.02-0.05 | mixed keyring
umbrella leaning in the corner | 0.5 | 0.05-0.1 | still damp
post pile on the console | 0.6 | 0.05-0.1 | advertising leaflets""",
            "foreground_element": """shoes half kicked off | 0.85 | 0.1-0.2 | worn heels
jacket over the arm | 0.55 | 0.1-0.25 | creased shell fabric""",
            "time_of_day": """hallway light on | 0.9 | - | -
daylight through the open door | 0.6 | - | -""",
            "weather": """wet shoe prints on the mat | 0.45 | - | -""",
        },
        "laundry_room": {
            "background": """plain cellar laundry wall | 1.0 | 0.5-0.9 | painted breeze block
washing machines side by side | 0.95 | 0.3-0.7 | scuffed white enamel
drying rack with laundry | 0.75 | 0.3-0.6 | plastic-coated wire""",
            "midground": """laundry baskets on the floor | 0.9 | 0.2-0.4 | cracked plastic weave
small sink in the corner | 0.5 | 0.15-0.3 | stained ceramic
shelf with detergent | 0.7 | 0.15-0.3 | sticky plastic bottles""",
            "architecture_detail": """bare concrete floor with a drain | 0.8 | 0.3-0.6 | painted concrete
small high cellar window | 0.55 | 0.1-0.25 | dusty wired glass
strip light on the ceiling | 0.75 | 0.1-0.25 | bare tube""",
            "props": """detergent scoop left in the box | 0.6 | 0.02-0.05 | powder residue
lint caught on the machine | 0.45 | 0.02-0.05 | grey fluff
clothes pegs in a bag | 0.55 | 0.02-0.08 | mixed plastic pegs""",
            "foreground_element": """damp washing being lifted out | 0.8 | 0.1-0.25 | heavy wet fabric
open machine door | 0.6 | 0.1-0.2 | rubber seal""",
            "time_of_day": """flat cellar light | 0.95 | - | -""",
            "weather": """no weather visible | 0.9 | - | -""",
        },
    },
    "outdoor/everyday_de": {
        "apartment_balcony": {
            "background": """facade of the opposite block | 1.0 | 0.5-0.9 | rendered wall with small windows
balcony railing | 0.95 | 0.3-0.7 | painted steel bars
rooftops beyond the courtyard | 0.6 | 0.2-0.5 | tiled roofs and aerials""",
            "midground": """drying rack on the balcony | 0.8 | 0.2-0.4 | folding aluminium frame
plastic balcony chair | 0.75 | 0.15-0.35 | weathered white plastic
planters along the railing | 0.7 | 0.15-0.35 | cracked terracotta""",
            "architecture_detail": """concrete balcony floor | 0.85 | 0.3-0.6 | stained concrete slab
downpipe along the wall | 0.5 | 0.05-0.15 | painted metal pipe""",
            "props": """watering can in the corner | 0.6 | 0.05-0.1 | faded plastic
ashtray on the railing | 0.4 | 0.02-0.05 | glass with ash
laundry pegs on the rack | 0.55 | 0.02-0.08 | mixed pegs""",
            "foreground_element": """hands hanging up washing | 0.7 | 0.1-0.25 | damp fabric
mug set on the railing | 0.55 | 0.05-0.1 | chipped ceramic""",
            "time_of_day": """flat overcast midday | 0.9 | - | -
low evening sun between the blocks | 0.8 | - | -
early morning shade | 0.7 | - | -""",
            "weather": """overcast grey sky | 0.85 | - | -
light drizzle | 0.4 | - | -
cold clear air | 0.5 | - | -""",
        },
        "bus_stop_de": {
            "background": """bus shelter with glass panels | 1.0 | 0.4-0.8 | scratched perspex
street with parked cars | 0.9 | 0.4-0.8 | tarmac with kerbs
row of houses across the road | 0.7 | 0.3-0.6 | rendered facades""",
            "midground": """timetable case on the post | 0.85 | 0.1-0.3 | scratched plastic cover
metal bench in the shelter | 0.8 | 0.15-0.35 | perforated painted steel
bin beside the shelter | 0.6 | 0.05-0.15 | dented metal""",
            "architecture_detail": """paving slabs at the stop | 0.85 | 0.2-0.5 | uneven concrete slabs
kerb with faded markings | 0.6 | 0.1-0.25 | chipped painted kerb""",
            "props": """stickers on the shelter glass | 0.55 | 0.02-0.08 | half-peeled stickers
cigarette ends by the bench | 0.45 | 0.02-0.05 | trodden filters
advertising poster in a frame | 0.6 | 0.1-0.25 | sun-bleached print""",
            "foreground_element": """phone checked for the departure | 0.7 | 0.05-0.15 | lit screen
shopping bag set down on the paving | 0.55 | 0.05-0.15 | slumped carrier bag""",
            "time_of_day": """grey commuter morning | 0.9 | - | -
dusk with the street lights coming on | 0.8 | - | -""",
            "weather": """steady drizzle | 0.5 | - | -
overcast and windy | 0.7 | - | -
cold with breath visible | 0.35 | - | -""",
        },
        "pedestrian_zone": {
            "background": """row of shop fronts | 1.0 | 0.5-0.9 | mixed signage and awnings
paved pedestrian street | 0.95 | 0.4-0.8 | patterned block paving
church tower further down | 0.4 | 0.15-0.35 | weathered stone""",
            "midground": """bike racks with locked bikes | 0.75 | 0.2-0.4 | rusted frames
café tables outside a shop | 0.7 | 0.2-0.4 | folding metal furniture
planted tub in the middle of the street | 0.6 | 0.1-0.3 | concrete tub""",
            "architecture_detail": """drainage channel in the paving | 0.5 | 0.05-0.15 | worn stone channel
bollards at the street entrance | 0.55 | 0.05-0.15 | scuffed steel posts""",
            "props": """A-board with the day's offer | 0.65 | 0.05-0.15 | chalked board
pigeons near a bin | 0.5 | 0.05-0.1 | scruffy feathers
bin overflowing slightly | 0.5 | 0.05-0.1 | dented metal""",
            "foreground_element": """shopping bag in each hand | 0.7 | 0.1-0.25 | printed carrier bags
pram pushed along the paving | 0.4 | 0.1-0.25 | worn fabric hood""",
            "time_of_day": """busy midday | 0.9 | - | -
late afternoon with long shadows | 0.85 | - | -""",
            "weather": """overcast | 0.8 | - | -
wet paving after rain | 0.5 | - | -
bright cold sunshine | 0.55 | - | -""",
        },
        "allotment_garden": {
            "background": """allotment plots with sheds | 1.0 | 0.5-0.9 | mismatched timber huts
hedge along the path | 0.8 | 0.3-0.6 | dense untrimmed hedge
neighbouring fruit trees | 0.6 | 0.2-0.5 | gnarled branches""",
            "midground": """vegetable beds with canes | 0.9 | 0.3-0.6 | dug soil with bamboo canes
wheelbarrow beside the path | 0.65 | 0.1-0.3 | rusted tray
compost heap in the corner | 0.55 | 0.1-0.3 | layered garden waste""",
            "architecture_detail": """gravel path between the plots | 0.8 | 0.2-0.5 | loose grey gravel
low wire fence | 0.6 | 0.1-0.3 | sagging wire mesh""",
            "props": """watering can by the bed | 0.7 | 0.05-0.1 | dented galvanised metal
garden gloves on the fence post | 0.5 | 0.02-0.05 | soil-stained fabric
seed packets on a crate | 0.4 | 0.02-0.05 | faded paper""",
            "foreground_element": """hands in the soil | 0.7 | 0.1-0.25 | dirt under the nails
harvest basket half full | 0.6 | 0.05-0.15 | woven basket""",
            "time_of_day": """warm late afternoon | 0.9 | - | -
overcast morning | 0.8 | - | -""",
            "weather": """mild and overcast | 0.8 | - | -
sunny with a light breeze | 0.7 | - | -""",
        },
        "playground_de": {
            "background": """playground with climbing frame | 1.0 | 0.5-0.9 | weathered timber and rope
hedge and trees around the edge | 0.85 | 0.3-0.6 | patchy foliage
apartment blocks beyond | 0.6 | 0.2-0.5 | rendered facades""",
            "midground": """swings on a metal frame | 0.9 | 0.2-0.5 | chipped paint and chains
sandpit with toys left behind | 0.75 | 0.2-0.4 | damp sand
bench facing the playground | 0.8 | 0.15-0.35 | slatted wood""",
            "architecture_detail": """rubber safety matting | 0.7 | 0.2-0.5 | worn black tiles
low fence around the area | 0.6 | 0.1-0.3 | painted timber""",
            "props": """plastic bucket and spade | 0.7 | 0.02-0.08 | faded plastic
scooter parked against the bench | 0.55 | 0.05-0.1 | scuffed deck
drinks bottle on the bench | 0.5 | 0.02-0.05 | condensation on plastic""",
            "foreground_element": """pushing a swing | 0.65 | 0.1-0.25 | chain in hand
jacket left on the bench | 0.55 | 0.05-0.15 | crumpled shell fabric""",
            "time_of_day": """after-school afternoon | 0.9 | - | -
low sun before dinner | 0.8 | - | -""",
            "weather": """overcast and mild | 0.8 | - | -
damp after rain | 0.5 | - | -
crisp autumn air | 0.5 | - | -""",
        },
        "supermarket_carpark_de": {
            "background": """supermarket facade with a logo sign | 1.0 | 0.4-0.8 | corrugated cladding
car park with marked bays | 0.95 | 0.5-0.9 | faded white lines on tarmac
trolley shelter | 0.8 | 0.15-0.35 | steel frame with a roof""",
            "midground": """parked cars in a row | 0.9 | 0.3-0.7 | dusty paintwork
trolleys nested together | 0.8 | 0.15-0.35 | scuffed steel
bottle return machine by the wall | 0.5 | 0.1-0.25 | stickered housing""",
            "architecture_detail": """cracked tarmac with weeds | 0.7 | 0.2-0.5 | patched asphalt
kerb around the trolley bay | 0.5 | 0.05-0.15 | chipped concrete""",
            "props": """receipt on the ground | 0.45 | 0.02-0.05 | curled thermal paper
abandoned trolley between bays | 0.55 | 0.05-0.15 | one wheel turned
coin slot on the trolley handle | 0.4 | 0.02-0.05 | worn plastic""",
            "foreground_element": """boot open with bags going in | 0.75 | 0.1-0.25 | full carrier bags
trolley pushed towards the car | 0.65 | 0.1-0.25 | loaded basket""",
            "time_of_day": """flat overcast midday | 0.9 | - | -
dusk with the car park lights on | 0.8 | - | -""",
            "weather": """overcast grey | 0.85 | - | -
wet tarmac reflecting the signs | 0.5 | - | -""",
        },
        "bike_path": {
            "background": """asphalt cycle path along a field | 1.0 | 0.5-0.9 | patched asphalt with a centre line
hedgerow beside the path | 0.85 | 0.3-0.6 | untrimmed bushes
distant village rooftops | 0.5 | 0.15-0.35 | small tiled roofs""",
            "midground": """wooden bench at the path side | 0.6 | 0.1-0.3 | weathered slats
signpost with cycle routes | 0.65 | 0.1-0.25 | faded green signs
bicycle leaning on the sign | 0.7 | 0.15-0.35 | mud-spattered frame""",
            "architecture_detail": """grass verge along the path | 0.8 | 0.2-0.5 | mown rough grass
small culvert under the path | 0.35 | 0.05-0.15 | concrete pipe""",
            "props": """water bottle in the bike cage | 0.6 | 0.02-0.05 | scratched plastic
helmet hung on the handlebars | 0.55 | 0.02-0.08 | scuffed shell""",
            "foreground_element": """handlebars in the near frame | 0.7 | 0.1-0.25 | worn grips
jacket tied around the waist | 0.5 | 0.05-0.15 | creased shell fabric""",
            "time_of_day": """bright morning | 0.85 | - | -
golden late afternoon | 0.85 | - | -""",
            "weather": """light breeze under a clear sky | 0.7 | - | -
overcast and cool | 0.7 | - | -""",
        },
        "courtyard_bins": {
            "background": """rear courtyard of an apartment block | 1.0 | 0.5-0.9 | rendered wall with downpipes
row of wheelie bins | 0.95 | 0.3-0.6 | scuffed coloured lids
bike stands against the wall | 0.6 | 0.15-0.35 | rusted metal hoops""",
            "midground": """bin enclosure with a gate | 0.8 | 0.2-0.5 | timber slats
carpet beater frame | 0.45 | 0.1-0.25 | painted steel bar
basement window at ground level | 0.5 | 0.1-0.25 | dusty wired glass""",
            "architecture_detail": """concrete courtyard slabs | 0.85 | 0.3-0.6 | weathered slabs with moss
drain grate in the middle | 0.5 | 0.05-0.15 | rusted cast iron""",
            "props": """flattened cardboard beside the bins | 0.65 | 0.05-0.15 | rain-soft cardboard
crate of empty bottles | 0.6 | 0.05-0.15 | scratched plastic crate
bin bag not quite in the bin | 0.5 | 0.05-0.1 | stretched black plastic""",
            "foreground_element": """bin lid lifted open | 0.7 | 0.1-0.25 | grubby plastic
bag carried out at arm's length | 0.6 | 0.1-0.2 | bulging bin bag""",
            "time_of_day": """dull morning in the shade | 0.9 | - | -
evening with a wall light on | 0.7 | - | -""",
            "weather": """overcast | 0.85 | - | -
wet slabs after rain | 0.5 | - | -""",
        },
    },
}


def write_file(path: pathlib.Path, text: str, force: bool) -> str:
    if path.exists() and not force:
        return "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "write"


def build_outfits(force: bool) -> tuple[int, int]:
    written = skipped = 0
    base = ROOT / "outfit_lists" / "female" / "everyday"
    for set_name, slots in OUTFIT_SETS.items():
        target = base / set_name
        for slot, body in slots.items():
            text = OUTFIT_HEADER[slot] + body.strip() + "\n"
            result = write_file(target / f"{slot}.txt", text, force)
            written += result == "write"
            skipped += result == "skip"
        for slot, text in (("fabrics", OUTFIT_HEADER["fabrics"] + FABRICS),
                           ("prints", OUTFIT_HEADER["prints"] + PRINTS_PLAIN),
                           ("texts", OUTFIT_HEADER["texts"] + TEXTS_NONE)):
            result = write_file(target / f"{slot}.txt", text, force)
            written += result == "write"
            skipped += result == "skip"
    return written, skipped


def build_locations(force: bool) -> tuple[int, int]:
    written = skipped = 0
    for category, sets in LOCATION_SETS.items():
        for set_name, elements in sets.items():
            target = ROOT / "location_lists" / pathlib.Path(category) / set_name
            for element, body in elements.items():
                text = LOCATION_HEADER[element] + body.strip() + "\n"
                result = write_file(target / f"{element}.txt", text, force)
                written += result == "write"
                skipped += result == "skip"
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="overwrite files that already exist")
    args = parser.parse_args()

    ow, os_ = build_outfits(args.force)
    lw, ls_ = build_locations(args.force)
    print(f"outfits:   {ow} written, {os_} skipped  ({len(OUTFIT_SETS)} sets)")
    print(f"locations: {lw} written, {ls_} skipped  "
          f"({sum(len(v) for v in LOCATION_SETS.values())} sets)")
    if os_ or ls_:
        print("Use --force to overwrite existing files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
