"""Generator für die Kleider/Röcke- und Underwear-Kategorien (US/PA, weiblich).

Drei neue Kategorien, alle im Everyday-Register statt Editorial:

female/dresses_heels/  — Kleider und Röcke, die Bein zeigen, mit Absatz.
                          Nicht nur fancy: Büro, Kirche, Diner, Mall, BBQ.
female/dresses_flats/  — Kleider und Röcke ohne Absatz: bedeckte Beine,
                          Strumpfhosen, Sneaker, Ballerinas. Zuhause, Einkauf,
                          School-Run, Büro in flach.
female/underwear/      — Alltagsunterwäsche, bewusst überwiegend non-fancy:
                          Baumwollsets, T-Shirt-BHs, Sport, Schlaf, Laundry-Day.
                          Ein einziges leicht hübscheres Set (simple_lace_touch).

Konventionen wie bei den bestehenden Sets:
- Kleider liegen im bottom-Slot, top.txt ist ein "none"-Stub (Engine filtert
  Platzhalter seit dem _is_none_garment-Fix).
- Kein #color#-Präfix in den Namen — die Engine setzt die Farbrolle davor.
- Namen mit Eigenfarbe ("white tennis shoes") bekommen keine zweite Farbe.

Ausführen:  python scripts/gen_dresses_underwear_sets.py [--force]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

SLOT_FILES = ("headwear", "top", "outerwear", "bottom", "footwear",
              "accessories", "bag")

SLOT_HEADER = ("# slot: {slot}\n"
               "# format: garment_name | probability | "
               "formality_min-formality_max | fabric1,fabric2,...\n\n")

FABRICS_HEADER = ("# fabric database\n"
                  "# format: fabric_name | formality | family | weight\n\n")

PRINTS_HEADER = ("# prints/patterns\n"
                 "# format: print_name | probability | compatible_slots | "
                 "formality_min-formality_max\n\n")

TEXTS_HEADER = ("# texts/slogans\n"
                "# format: text_value | probability | compatible_slots | "
                "font_or_descriptor\n\n"
                "# intentionally empty: printed slogans get rendered as real text\n")

NONE_STUB = "none | 0.95 | 0.0-1.0 | -"

# ── Shared blocks ────────────────────────────────────────────────────────

DRESS_FABRICS = """cotton | 0.3 | natural | medium
jersey | 0.2 | natural | light
denim | 0.3 | natural | heavy
knit | 0.3 | natural | medium
polyester | 0.25 | synthetic | light
rayon | 0.35 | synthetic | light
viscose | 0.4 | synthetic | light
linen | 0.45 | natural | light
ponte | 0.45 | synthetic | medium
chiffon | 0.55 | synthetic | light
leather | 0.4 | natural | heavy
suede | 0.45 | natural | medium
lace | 0.5 | delicate | light
satin | 0.55 | delicate | light
corduroy | 0.3 | natural | heavy
tweed | 0.5 | natural | heavy"""

DRESS_PRINTS = """solid color | 0.85 | top,bottom,outerwear | 0.0-1.0
small ditsy floral | 0.3 | bottom | 0.1-0.6
soft watercolor floral | 0.2 | bottom | 0.2-0.7
classic polka dot | 0.18 | bottom,top | 0.2-0.7
thin vertical stripe | 0.15 | bottom,top | 0.2-0.7
muted gingham check | 0.15 | bottom | 0.1-0.5
subtle leopard print | 0.1 | bottom | 0.2-0.6
paisley in faded tones | 0.08 | bottom | 0.2-0.6"""

UNDERWEAR_FABRICS = """cotton | 0.2 | natural | light
modal | 0.25 | synthetic | light
jersey | 0.2 | natural | light
microfiber | 0.3 | synthetic | light
seamless knit | 0.3 | synthetic | light
ribbed cotton | 0.2 | natural | light
spandex blend | 0.25 | synthetic | light
lace | 0.5 | delicate | light
satin | 0.55 | delicate | light
mesh | 0.45 | delicate | light
fleece | 0.1 | synthetic | medium
flannel | 0.15 | natural | medium
waffle knit | 0.2 | natural | light"""

UNDERWEAR_PRINTS = """solid color | 0.85 | top,bottom | 0.0-1.0
small heart print | 0.15 | bottom | 0.0-0.4
tiny star pattern | 0.12 | bottom | 0.0-0.4
thin stripe | 0.15 | top,bottom | 0.0-0.5
small floral sprig | 0.15 | bottom | 0.0-0.5
classic polka dot | 0.12 | top,bottom | 0.0-0.5"""

HEELS_HEADWEAR = """sunglasses pushed up into the hair | 0.5 | 0.1-0.7 | -
hair down in loose waves | 0.6 | 0.1-0.8 | -
low ponytail with a ribbon | 0.35 | 0.2-0.7 | -
claw clip half-up twist | 0.45 | 0.1-0.6 | -
soft headband | 0.3 | 0.2-0.7 | knit
hair tucked behind both ears | 0.4 | 0.1-0.8 | -"""

HEELS_ACCESSORIES = """small stud earrings | 0.6 | 0.1-0.8 | -
thin pendant necklace | 0.5 | 0.2-0.8 | -
simple bracelet watch | 0.4 | 0.2-0.8 | -
thin waist belt | 0.4 | 0.2-0.7 | leather
gold hoop earrings | 0.4 | 0.2-0.8 | -
delicate layered necklaces | 0.3 | 0.2-0.7 | -
silk neck scarf | 0.2 | 0.3-0.8 | satin"""

HEELS_BAG = """small crossbody bag | 0.6 | 0.1-0.7 | leather
structured shoulder bag | 0.5 | 0.3-0.8 | leather
compact top-handle bag | 0.4 | 0.3-0.8 | leather
soft clutch under the arm | 0.3 | 0.4-0.9 | suede
canvas tote over the shoulder | 0.3 | 0.1-0.5 | canvas"""

FLATS_HEADWEAR = """messy bun with a claw clip | 0.55 | 0.0-0.5 | -
low ponytail | 0.5 | 0.0-0.6 | -
hair down and unstyled | 0.5 | 0.0-0.6 | -
soft knit headband | 0.3 | 0.0-0.5 | knit
baseball cap | 0.3 | 0.0-0.4 | cotton
sunglasses pushed up into the hair | 0.4 | 0.1-0.6 | -"""

FLATS_ACCESSORIES = """small stud earrings | 0.55 | 0.0-0.7 | -
hair tie around the wrist | 0.4 | 0.0-0.4 | -
simple watch | 0.4 | 0.1-0.7 | -
thin pendant necklace | 0.4 | 0.1-0.7 | -
crossbody phone strap | 0.3 | 0.0-0.5 | polyester
knotted cotton scarf | 0.2 | 0.1-0.6 | cotton"""

FLATS_BAG = """canvas tote over the shoulder | 0.55 | 0.0-0.5 | canvas
small crossbody bag | 0.55 | 0.0-0.6 | leather
backpack worn on one shoulder | 0.35 | 0.0-0.4 | polyester
reusable shopping bag | 0.3 | 0.0-0.4 | canvas
diaper bag packed full | 0.25 | 0.0-0.4 | polyester"""

UNDERWEAR_HEADWEAR = """messy bun with loose strands | 0.55 | 0.0-0.5 | -
hair down and sleep-tangled | 0.45 | 0.0-0.5 | -
low ponytail | 0.4 | 0.0-0.5 | -
claw clip holding hair up | 0.4 | 0.0-0.5 | -
scrunchie on a loose bun | 0.35 | 0.0-0.5 | -
headband pushing hair back | 0.25 | 0.0-0.5 | knit"""

UNDERWEAR_ACCESSORIES = """small stud earrings | 0.35 | 0.0-0.6 | -
hair tie around the wrist | 0.4 | 0.0-0.5 | -
thin everyday necklace kept on | 0.3 | 0.0-0.6 | -
glasses instead of contacts | 0.35 | 0.0-0.6 | -
no jewelry at all | 0.4 | 0.0-0.6 | -"""

UNDERWEAR_BAG = NONE_STUB


def _set(top, bottom, footwear, outerwear, headwear, accessories, bag,
         fabrics, prints):
    return {
        "top": top, "bottom": bottom, "footwear": footwear,
        "outerwear": outerwear, "headwear": headwear,
        "accessories": accessories, "bag": bag,
        "fabrics": fabrics, "prints": prints,
    }


SETS: dict[str, dict[str, dict[str, str]]] = {
    # ═══════════════ zeigt Bein + High Heels ═══════════════
    "female/dresses_heels": {
        "office_dress_heels": _set(
            top=NONE_STUB,
            bottom="""knee-length sheath dress | 0.8 | 0.4-0.8 | ponte,polyester
knee-length wrap dress | 0.75 | 0.3-0.7 | jersey,viscose
belted shirt dress above the knee | 0.6 | 0.3-0.7 | cotton,rayon
pencil skirt with a plain blouse tucked in | 0.6 | 0.4-0.8 | polyester,tweed
fit-and-flare dress to the knee | 0.55 | 0.3-0.7 | cotton,ponte
sleeveless shift dress | 0.5 | 0.4-0.8 | ponte,linen
a-line skirt with a fine knit top | 0.5 | 0.3-0.7 | viscose,knit""",
            footwear="""block heel pumps | 0.8 | 0.3-0.8 | leather
kitten heel slingbacks | 0.6 | 0.4-0.8 | leather
low heeled mules | 0.5 | 0.3-0.7 | suede
classic mid-heel pumps | 0.55 | 0.4-0.9 | leather
heeled loafers | 0.4 | 0.3-0.7 | leather""",
            outerwear="""unstructured blazer | 0.55 | 0.4-0.8 | polyester,tweed
fine knit cardigan kept at the desk | 0.5 | 0.2-0.6 | knit
trench coat over the arm | 0.35 | 0.4-0.8 | cotton
cropped jacket | 0.3 | 0.3-0.7 | ponte""",
            headwear=HEELS_HEADWEAR, accessories=HEELS_ACCESSORIES,
            bag=HEELS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
        "church_sunday_heels": _set(
            top=NONE_STUB,
            bottom="""floral knee-length dress | 0.75 | 0.3-0.7 | viscose,cotton
modest wrap dress | 0.65 | 0.3-0.7 | jersey,viscose
pleated midi skirt with a tucked blouse | 0.55 | 0.3-0.7 | chiffon,polyester
fit-and-flare dress with cap sleeves | 0.6 | 0.3-0.7 | cotton,ponte
soft a-line dress to the knee | 0.55 | 0.3-0.7 | rayon,linen
lace-trim shift dress | 0.4 | 0.4-0.8 | lace,polyester""",
            footwear="""low block heels | 0.75 | 0.3-0.7 | leather
kitten heel pumps | 0.6 | 0.3-0.8 | leather
wedge heels with an ankle strap | 0.5 | 0.2-0.6 | suede
heeled mary janes | 0.4 | 0.3-0.7 | leather""",
            outerwear="""light cardigan over the shoulders | 0.6 | 0.2-0.6 | knit
short dressy coat | 0.35 | 0.4-0.8 | tweed
thin wrap shawl | 0.3 | 0.3-0.7 | chiffon""",
            headwear=HEELS_HEADWEAR, accessories=HEELS_ACCESSORIES,
            bag=HEELS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
        "dinner_out_heels": _set(
            top=NONE_STUB,
            bottom="""slip skirt with a fitted tee | 0.65 | 0.3-0.7 | satin,jersey
midi dress with a thigh-high slit | 0.55 | 0.4-0.8 | viscose,satin
little black dress above the knee | 0.6 | 0.4-0.9 | ponte,polyester
ribbed knit dress | 0.6 | 0.3-0.7 | knit
denim skirt with a dressy blouse | 0.5 | 0.2-0.5 | denim,chiffon
satin wrap dress | 0.45 | 0.4-0.8 | satin""",
            footwear="""strappy heeled sandals | 0.7 | 0.4-0.9 | leather
block heel ankle boots | 0.55 | 0.3-0.7 | suede
classic pumps | 0.5 | 0.4-0.9 | leather
heeled mules | 0.5 | 0.3-0.7 | suede""",
            outerwear="""cropped leather jacket | 0.45 | 0.3-0.7 | leather
oversized blazer | 0.45 | 0.3-0.8 | polyester
denim jacket dressed up | 0.35 | 0.2-0.5 | denim""",
            headwear=HEELS_HEADWEAR, accessories=HEELS_ACCESSORIES,
            bag=HEELS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
        "mall_afternoon_heels": _set(
            top=NONE_STUB,
            bottom="""casual skater dress | 0.7 | 0.1-0.5 | jersey,cotton
t-shirt dress with a knotted waist | 0.6 | 0.1-0.4 | jersey
denim mini skirt with a soft top | 0.55 | 0.1-0.4 | denim,jersey
button-front a-line skirt with a tee | 0.5 | 0.1-0.5 | corduroy,cotton
swing dress above the knee | 0.5 | 0.1-0.5 | rayon,viscose""",
            footwear="""low wedge sandals | 0.65 | 0.1-0.5 | suede
block heel sandals | 0.6 | 0.2-0.6 | leather
platform espadrilles | 0.5 | 0.1-0.5 | canvas
heeled ankle boots | 0.45 | 0.2-0.6 | suede""",
            outerwear="""cropped denim jacket | 0.5 | 0.1-0.4 | denim
lightweight cardigan | 0.4 | 0.1-0.5 | knit
zip hoodie thrown over | 0.3 | 0.0-0.3 | cotton""",
            headwear=HEELS_HEADWEAR, accessories=HEELS_ACCESSORIES,
            bag=HEELS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
        "summer_bbq_heels": _set(
            top=NONE_STUB,
            bottom="""floral sundress above the knee | 0.75 | 0.1-0.5 | cotton,rayon
smocked-bodice summer dress | 0.6 | 0.1-0.5 | cotton,viscose
tiered mini sundress | 0.5 | 0.1-0.4 | cotton,chiffon
linen wrap skirt with a tank | 0.5 | 0.1-0.5 | linen,jersey
halter-neck summer dress | 0.45 | 0.1-0.5 | rayon,cotton""",
            footwear="""low wedge sandals | 0.7 | 0.1-0.5 | suede
block heel slides | 0.55 | 0.1-0.5 | leather
espadrille wedges with ankle ties | 0.5 | 0.1-0.5 | canvas
strappy low heels | 0.45 | 0.2-0.6 | leather""",
            outerwear="""light denim jacket for the evening | 0.4 | 0.1-0.4 | denim
crochet cover-up cardigan | 0.3 | 0.1-0.4 | knit
thin linen overshirt | 0.3 | 0.1-0.5 | linen""",
            headwear=HEELS_HEADWEAR, accessories=HEELS_ACCESSORIES,
            bag=HEELS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
        "girls_night_heels": _set(
            top=NONE_STUB,
            bottom="""bodycon mini dress | 0.6 | 0.3-0.7 | ponte,jersey
satin slip dress | 0.5 | 0.4-0.8 | satin
denim skirt with a going-out top | 0.55 | 0.2-0.5 | denim,chiffon
wrap mini dress | 0.5 | 0.3-0.7 | viscose,jersey
sequin-trim skirt with a plain tee | 0.35 | 0.3-0.7 | polyester,jersey
off-shoulder fitted dress | 0.45 | 0.3-0.7 | ponte""",
            footwear="""strappy stiletto heels | 0.6 | 0.4-0.9 | leather
platform heeled sandals | 0.5 | 0.3-0.7 | suede
heeled ankle boots | 0.5 | 0.2-0.6 | leather
block heel mary janes | 0.4 | 0.3-0.7 | leather""",
            outerwear="""cropped moto jacket | 0.45 | 0.2-0.6 | leather
blazer worn open | 0.4 | 0.3-0.7 | polyester
faux fur short jacket | 0.25 | 0.3-0.7 | polyester""",
            headwear=HEELS_HEADWEAR, accessories=HEELS_ACCESSORIES,
            bag=HEELS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
    },
    # ═══════════════ Kleider/Röcke ohne Absatz ═══════════════
    "female/dresses_flats": {
        "office_dress_flats": _set(
            top=NONE_STUB,
            bottom="""midi shirt dress with sheer tights | 0.7 | 0.3-0.7 | cotton,viscose
knee-length dress over opaque tights | 0.65 | 0.3-0.7 | ponte,jersey
pleated midi skirt with a fine knit | 0.6 | 0.3-0.7 | polyester,knit
long a-line skirt with a tucked blouse | 0.5 | 0.3-0.7 | viscose,cotton
sweater dress over tights | 0.55 | 0.2-0.6 | knit
column midi dress | 0.45 | 0.3-0.7 | ponte""",
            footwear="""pointed-toe ballet flats | 0.7 | 0.3-0.7 | leather
leather loafers | 0.65 | 0.3-0.7 | leather
low chelsea boots | 0.5 | 0.2-0.6 | leather
white leather sneakers kept clean | 0.45 | 0.1-0.5 | leather
mary jane flats | 0.4 | 0.3-0.7 | leather""",
            outerwear="""long cardigan | 0.55 | 0.2-0.6 | knit
unstructured blazer | 0.5 | 0.3-0.8 | polyester
wool coat for the commute | 0.35 | 0.3-0.8 | tweed""",
            headwear=FLATS_HEADWEAR, accessories=FLATS_ACCESSORIES,
            bag=FLATS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
        "home_house_dress": _set(
            top=NONE_STUB,
            bottom="""soft t-shirt dress | 0.75 | 0.0-0.3 | jersey,cotton
oversized shirt dress unbuttoned low | 0.55 | 0.0-0.3 | cotton
knit lounge dress | 0.6 | 0.0-0.3 | knit,modal
tank dress to mid-thigh | 0.5 | 0.0-0.3 | jersey
button-front house dress | 0.5 | 0.0-0.3 | cotton,rayon
long jersey maxi dress | 0.45 | 0.0-0.3 | jersey""",
            footwear="""bare feet | 0.6 | 0.0-0.3 | -
thick house socks | 0.5 | 0.0-0.3 | knit
soft slippers | 0.5 | 0.0-0.3 | fleece
slide sandals by the door | 0.35 | 0.0-0.3 | -""",
            outerwear="""oversized cardigan | 0.5 | 0.0-0.3 | knit
boyfriend flannel worn open | 0.4 | 0.0-0.3 | cotton
zip hoodie half on | 0.35 | 0.0-0.3 | fleece""",
            headwear=FLATS_HEADWEAR, accessories=FLATS_ACCESSORIES,
            bag=NONE_STUB, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
        "grocery_sundress": _set(
            top=NONE_STUB,
            bottom="""relaxed cotton sundress | 0.75 | 0.1-0.4 | cotton
t-shirt dress with a denim jacket tied at the waist | 0.55 | 0.1-0.4 | jersey,denim
button-front midi dress | 0.55 | 0.1-0.5 | rayon,cotton
soft jersey midi skirt with a tee | 0.5 | 0.1-0.4 | jersey
smocked summer dress | 0.5 | 0.1-0.4 | cotton,viscose""",
            footwear="""white tennis shoes | 0.7 | 0.1-0.4 | canvas
low-top canvas sneakers | 0.6 | 0.1-0.4 | canvas
flat slide sandals | 0.55 | 0.1-0.4 | leather
ballet flats worn soft | 0.4 | 0.1-0.5 | leather
birkenstock-style sandals | 0.45 | 0.1-0.4 | suede""",
            outerwear="""light denim jacket | 0.45 | 0.1-0.4 | denim
thin cardigan against the AC | 0.4 | 0.1-0.4 | knit
zip-up fleece from the car | 0.3 | 0.0-0.3 | fleece""",
            headwear=FLATS_HEADWEAR, accessories=FLATS_ACCESSORIES,
            bag=FLATS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
        "school_run_skirt": _set(
            top=NONE_STUB,
            bottom="""soft midi skirt with a plain tee | 0.65 | 0.1-0.4 | jersey,viscose
knee-length denim skirt with a sweatshirt | 0.55 | 0.1-0.4 | denim,cotton
button-front corduroy skirt with a knit top | 0.5 | 0.1-0.5 | corduroy,knit
casual shirt dress with the sleeves rolled | 0.55 | 0.1-0.4 | cotton
tiered maxi skirt with a fitted tee | 0.45 | 0.1-0.4 | viscose,cotton""",
            footwear="""white leather sneakers | 0.7 | 0.1-0.4 | leather
low canvas sneakers | 0.6 | 0.1-0.4 | canvas
ballet flats by the door | 0.45 | 0.1-0.5 | leather
low chelsea boots | 0.4 | 0.1-0.5 | leather""",
            outerwear="""quilted vest | 0.45 | 0.1-0.4 | polyester
denim jacket | 0.45 | 0.1-0.4 | denim
long open cardigan | 0.45 | 0.1-0.4 | knit
rain shell grabbed off the hook | 0.3 | 0.0-0.3 | polyester""",
            headwear=FLATS_HEADWEAR, accessories=FLATS_ACCESSORIES,
            bag=FLATS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
        "fall_skirt_boots": _set(
            top=NONE_STUB,
            bottom="""midi skirt over opaque black tights | 0.7 | 0.2-0.6 | viscose,polyester
sweater dress with thick tights | 0.65 | 0.2-0.6 | knit
corduroy skirt with ribbed tights | 0.55 | 0.2-0.5 | corduroy
plaid a-line skirt with tights | 0.45 | 0.2-0.6 | tweed,polyester
long knit dress over leggings | 0.5 | 0.1-0.5 | knit,jersey""",
            footwear="""flat knee-high boots | 0.65 | 0.2-0.6 | leather
chelsea boots | 0.6 | 0.1-0.5 | leather
lace-up ankle boots | 0.55 | 0.1-0.5 | suede
lug-sole boots | 0.5 | 0.1-0.5 | leather""",
            outerwear="""long wool coat | 0.5 | 0.3-0.7 | tweed
chunky knit cardigan | 0.55 | 0.1-0.5 | knit
quilted barn jacket | 0.45 | 0.1-0.5 | cotton
puffer vest over a turtleneck | 0.4 | 0.1-0.4 | polyester""",
            headwear=FLATS_HEADWEAR, accessories=FLATS_ACCESSORIES,
            bag=FLATS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
        "church_modest_flats": _set(
            top=NONE_STUB,
            bottom="""knee-length floral dress with a modest neckline | 0.7 | 0.3-0.7 | viscose,cotton
midi wrap dress | 0.6 | 0.3-0.7 | jersey,viscose
pleated midi skirt with a soft blouse | 0.55 | 0.3-0.7 | chiffon,polyester
long a-line dress with short sleeves | 0.5 | 0.3-0.7 | rayon,linen
shirt dress belted at the waist | 0.5 | 0.3-0.7 | cotton""",
            footwear="""ballet flats | 0.7 | 0.2-0.7 | leather
low wedge pumps | 0.5 | 0.3-0.7 | leather
mary jane flats | 0.5 | 0.2-0.6 | leather
flat slingbacks | 0.4 | 0.3-0.7 | leather""",
            outerwear="""light cardigan over the shoulders | 0.6 | 0.2-0.6 | knit
thin wrap shawl | 0.3 | 0.3-0.7 | chiffon
short spring coat | 0.35 | 0.3-0.7 | cotton""",
            headwear=FLATS_HEADWEAR, accessories=FLATS_ACCESSORIES,
            bag=FLATS_BAG, fabrics=DRESS_FABRICS, prints=DRESS_PRINTS),
    },
    # ═══════════════ Underwear, überwiegend non-fancy ═══════════════
    "female/underwear": {
        "everyday_cotton": _set(
            top="""plain cotton bra | 0.8 | 0.0-0.4 | cotton
soft cotton bralette | 0.65 | 0.0-0.4 | ribbed cotton
simple wireless bra | 0.6 | 0.0-0.4 | cotton,modal
cotton crop cami | 0.45 | 0.0-0.4 | ribbed cotton""",
            bottom="""plain cotton briefs | 0.8 | 0.0-0.4 | cotton
cotton hipster panties | 0.7 | 0.0-0.4 | cotton
high-waist cotton briefs | 0.5 | 0.0-0.4 | cotton
cotton bikini panties | 0.55 | 0.0-0.4 | cotton""",
            footwear="""bare feet | 0.65 | 0.0-0.4 | -
plain ankle socks | 0.5 | 0.0-0.4 | cotton
crew socks slouched | 0.35 | 0.0-0.4 | cotton""",
            outerwear="""oversized worn-in tee half on | 0.35 | 0.0-0.3 | jersey
open flannel shirt | 0.3 | 0.0-0.3 | flannel
waffle knit robe untied | 0.3 | 0.0-0.4 | waffle knit""",
            headwear=UNDERWEAR_HEADWEAR, accessories=UNDERWEAR_ACCESSORIES,
            bag=UNDERWEAR_BAG, fabrics=UNDERWEAR_FABRICS,
            prints=UNDERWEAR_PRINTS),
        "tshirt_bra_basics": _set(
            top="""smooth t-shirt bra | 0.85 | 0.0-0.5 | microfiber
seamless wireless bra | 0.6 | 0.0-0.5 | seamless knit
molded-cup everyday bra | 0.55 | 0.0-0.5 | microfiber,modal
plain nude bra | 0.5 | 0.0-0.5 | microfiber""",
            bottom="""seamless hipster panties | 0.75 | 0.0-0.5 | seamless knit
no-show briefs | 0.6 | 0.0-0.5 | microfiber
laser-cut bikini panties | 0.5 | 0.0-0.5 | microfiber
smooth high-cut briefs | 0.45 | 0.0-0.5 | modal""",
            footwear="""bare feet | 0.6 | 0.0-0.4 | -
sheer ankle socks | 0.35 | 0.0-0.4 | -
no-show liner socks | 0.4 | 0.0-0.4 | cotton""",
            outerwear="""unbuttoned work blouse | 0.35 | 0.1-0.5 | cotton
cardigan pulled on for the hallway | 0.3 | 0.0-0.4 | knit""",
            headwear=UNDERWEAR_HEADWEAR, accessories=UNDERWEAR_ACCESSORIES,
            bag=UNDERWEAR_BAG, fabrics=UNDERWEAR_FABRICS,
            prints=UNDERWEAR_PRINTS),
        "sports_everyday": _set(
            top="""racerback sports bra | 0.8 | 0.0-0.4 | spandex blend
medium-support sports bra | 0.65 | 0.0-0.4 | spandex blend
longline sports bra | 0.5 | 0.0-0.4 | seamless knit
zip-front sports bra | 0.4 | 0.0-0.4 | spandex blend""",
            bottom="""seamless boyshorts | 0.7 | 0.0-0.4 | seamless knit
athletic bikini briefs | 0.6 | 0.0-0.4 | microfiber
compression shorts worn as underwear | 0.5 | 0.0-0.4 | spandex blend
high-waist seamless briefs | 0.45 | 0.0-0.4 | seamless knit""",
            footwear="""white ankle socks | 0.6 | 0.0-0.4 | cotton
cushioned running socks | 0.5 | 0.0-0.4 | cotton
bare feet on the mat | 0.45 | 0.0-0.4 | -""",
            outerwear="""oversized gym hoodie half zipped | 0.4 | 0.0-0.3 | fleece
loose tank thrown over | 0.35 | 0.0-0.3 | jersey""",
            headwear=UNDERWEAR_HEADWEAR, accessories=UNDERWEAR_ACCESSORIES,
            bag=UNDERWEAR_BAG, fabrics=UNDERWEAR_FABRICS,
            prints=UNDERWEAR_PRINTS),
        "sleep_set": _set(
            top="""soft sleep cami | 0.7 | 0.0-0.3 | modal
oversized sleep tee slipping off one shoulder | 0.6 | 0.0-0.3 | jersey
ribbed tank worn to bed | 0.55 | 0.0-0.3 | ribbed cotton
long-sleeve pajama top unbuttoned at the collar | 0.4 | 0.0-0.3 | flannel""",
            bottom="""cotton sleep shorts | 0.7 | 0.0-0.3 | cotton
plaid pajama shorts | 0.55 | 0.0-0.3 | flannel
soft modal sleep pants rolled at the waist | 0.45 | 0.0-0.3 | modal
briefs under the sleep tee | 0.5 | 0.0-0.3 | cotton""",
            footwear="""bare feet | 0.7 | 0.0-0.3 | -
fuzzy socks | 0.45 | 0.0-0.3 | fleece
thick house socks | 0.4 | 0.0-0.3 | knit""",
            outerwear="""waffle knit robe hanging open | 0.4 | 0.0-0.3 | waffle knit
comforter pulled around the shoulders | 0.3 | 0.0-0.3 | -
long cardigan over the pajamas | 0.3 | 0.0-0.3 | knit""",
            headwear=UNDERWEAR_HEADWEAR, accessories=UNDERWEAR_ACCESSORIES,
            bag=UNDERWEAR_BAG, fabrics=UNDERWEAR_FABRICS,
            prints=UNDERWEAR_PRINTS),
        "laundry_day_mismatch": _set(
            top="""faded sports bra washed soft | 0.6 | 0.0-0.3 | spandex blend
old comfortable bra with stretched straps | 0.55 | 0.0-0.3 | cotton
cami with a built-in shelf bra | 0.5 | 0.0-0.3 | jersey
band tee knotted above the waist | 0.4 | 0.0-0.3 | jersey""",
            bottom="""mismatched cotton panties | 0.7 | 0.0-0.3 | cotton
boyshorts from the back of the drawer | 0.55 | 0.0-0.3 | modal
striped briefs that lost their set | 0.5 | 0.0-0.3 | cotton
period-safe dark briefs | 0.45 | 0.0-0.3 | microfiber""",
            footwear="""one sock missing its pair | 0.5 | 0.0-0.3 | cotton
bare feet on the cool floor | 0.55 | 0.0-0.3 | -
old slippers flattened at the heel | 0.4 | 0.0-0.3 | fleece""",
            outerwear="""boyfriend flannel with rolled sleeves | 0.4 | 0.0-0.3 | flannel
towel worn as a cape between loads | 0.25 | 0.0-0.3 | -
zip hoodie with nothing matching | 0.35 | 0.0-0.3 | fleece""",
            headwear=UNDERWEAR_HEADWEAR, accessories=UNDERWEAR_ACCESSORIES,
            bag=UNDERWEAR_BAG, fabrics=UNDERWEAR_FABRICS,
            prints=UNDERWEAR_PRINTS),
        "simple_lace_touch": _set(
            top="""lace-trim bralette | 0.7 | 0.1-0.6 | lace,modal
satin bralette with thin straps | 0.55 | 0.1-0.6 | satin
soft cup bra with a lace band | 0.5 | 0.1-0.6 | lace,microfiber
mesh-panel wireless bra | 0.4 | 0.1-0.6 | mesh,modal""",
            bottom="""lace-waist hipster panties | 0.65 | 0.1-0.6 | lace,cotton
satin bikini panties | 0.5 | 0.1-0.6 | satin
mesh-side briefs | 0.45 | 0.1-0.6 | mesh,microfiber
lace-back cheeky panties | 0.4 | 0.1-0.6 | lace""",
            footwear="""bare feet | 0.65 | 0.0-0.5 | -
sheer ankle socks with a lace cuff | 0.3 | 0.0-0.5 | -""",
            outerwear="""short satin robe loosely tied | 0.45 | 0.1-0.6 | satin
long cardigan worn off one shoulder | 0.3 | 0.0-0.5 | knit
oversized button-down barely buttoned | 0.35 | 0.0-0.5 | cotton""",
            headwear=UNDERWEAR_HEADWEAR, accessories=UNDERWEAR_ACCESSORIES,
            bag=UNDERWEAR_BAG, fabrics=UNDERWEAR_FABRICS,
            prints=UNDERWEAR_PRINTS),
    },
}


def write_file(path: pathlib.Path, text: str, force: bool) -> str:
    if path.exists() and not force:
        return "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "write"


def validate(category: str, set_name: str) -> list[str]:
    problems = []
    spec = SETS[category][set_name]
    for slot in SLOT_FILES:
        body = spec.get(slot)
        if body is None:
            problems.append(f"missing slot {slot}")
            continue
        for ln in [l for l in body.strip().splitlines() if l.strip()]:
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) != 4:
                problems.append(f"{slot}: bad column count {ln!r}")
                continue
            try:
                prob = float(parts[1])
                lo, hi = parts[2].split("-")
                float(lo), float(hi)
            except ValueError:
                problems.append(f"{slot}: bad numbers in {ln!r}")
                continue
            if not 0.0 < prob <= 1.0:
                problems.append(f"{slot}: probability {prob} out of range")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ok = True
    for category, sets in SETS.items():
        for set_name in sets:
            for p in validate(category, set_name):
                print(f"ERROR {category}/{set_name}: {p}")
                ok = False
    if not ok:
        return 1

    written = skipped = 0
    for category, sets in SETS.items():
        for set_name, spec in sets.items():
            target = ROOT / "outfit_lists" / pathlib.Path(category) / set_name
            for slot in SLOT_FILES:
                text = SLOT_HEADER.format(slot=slot) + spec[slot].strip() + "\n"
                r = write_file(target / f"{slot}.txt", text, args.force)
                written += r == "write"
                skipped += r == "skip"
            r = write_file(target / "fabrics.txt",
                           FABRICS_HEADER + spec["fabrics"].strip() + "\n",
                           args.force)
            written += r == "write"
            skipped += r == "skip"
            r = write_file(target / "prints.txt",
                           PRINTS_HEADER + spec["prints"].strip() + "\n",
                           args.force)
            written += r == "write"
            skipped += r == "skip"
            r = write_file(target / "texts.txt", TEXTS_HEADER, args.force)
            written += r == "write"
            skipped += r == "skip"

    total = sum(len(v) for v in SETS.values())
    print(f"outfits: {written} written, {skipped} skipped ({total} sets)")
    if skipped:
        print("Use --force to overwrite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
