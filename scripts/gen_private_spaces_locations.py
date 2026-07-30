"""Generator für die Private-Spaces-Sets: ruhige Orte ohne Publikumsverkehr.

Kernidee: Räume und Ecken, in denen niemand durchläuft — und in denen es
möglichst NICHTS mit Aufschrift gibt. Text-Encoder wie Krea 2 rendern jedes
erwähnte Schild, Poster oder Etikett als lesbaren (meist kaputten) Text ins
Bild; diese Sets vermeiden deshalb Beschriftungen schon auf Datenebene.

indoor/private_spaces_us   parking_garage_level, home_garage_interior,
                           spare_room_boxes, office_stairwell,
                           office_supply_room, store_stockroom,
                           school_gym_storage
outdoor/private_spaces_us  mall_parking_lot_far_corner,
                           parking_garage_roof_deck, side_yard_between_houses,
                           backyard_behind_the_shed, office_courtyard_quiet,
                           loading_dock_after_hours, school_back_field

Der Validator prüft zusätzlich zur Standard-Kuration eine Signage-Banlist
(Substring!): "sign" fängt auch "assigned", "ridge" auch "cartridge" — solche
Wörter kommen hier schlicht nicht vor.

Ausführen:  python scripts/gen_private_spaces_locations.py [--force]
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

# ── Atmosphären-Pools (ohne Lichtaussagen im Wetter, ohne Text-Objekte) ──

INDOOR_UTILITY_TIME = """bare fluorescent tube light | 0.9 | - | -
daylight through a small high window | 0.7 | - | -
dim light from the door left ajar | 0.6 | - | -
one tube flickering at the far end | 0.5 | - | -
single bulb on a pull cord | 0.6 | - | -
grey daylight not reaching the corners | 0.7 | - | -
after-hours half darkness | 0.5 | - | -
morning light in a thin stripe | 0.6 | - | -
late afternoon slant through the doorway | 0.6 | - | -
even shadowless work light | 0.7 | - | -"""

INDOOR_UTILITY_WEATHER = """dusty still air | 0.8 | - | -
cold air off the concrete | 0.7 | - | -
damp mineral basement air | 0.5 | - | -
draught under the door | 0.5 | - | -
rain heard on a metal roof | 0.45 | - | -
dry heated air from a duct | 0.5 | - | -
musty cardboard smell | 0.5 | - | -
humid summer air trapped inside | 0.45 | - | -
cool air pooling at the floor | 0.5 | - | -
still air with drifting dust motes | 0.6 | - | -"""

OUTDOOR_QUIET_TIME = """early morning before anyone is up | 0.8 | - | -
bright mid morning | 0.8 | - | -
flat overcast midday | 0.9 | - | -
late afternoon long shadows | 0.9 | - | -
golden hour on the walls | 0.8 | - | -
dusk settling in | 0.7 | - | -
blue hour after sunset | 0.55 | - | -
first lights coming on nearby | 0.5 | - | -
grey november midday | 0.6 | - | -
last light on the upper edges | 0.6 | - | -"""

OUTDOOR_QUIET_WEATHER = """flat overcast sky | 0.9 | - | -
light drizzle in the air | 0.5 | - | -
wet asphalt after a shower | 0.55 | - | -
clear cold air | 0.6 | - | -
humid summer haze | 0.55 | - | -
gusty wind funneled between walls | 0.5 | - | -
cold air with breath visible | 0.4 | - | -
thin high cloud | 0.6 | - | -
heavy still air before a storm | 0.4 | - | -
puddles shivering in the wind | 0.45 | - | -"""


SETS: dict[str, dict[str, dict[str, str]]] = {
    # ═════════════════════════════ INDOOR ═════════════════════════════
    "indoor/private_spaces_us": {
        "parking_garage_level": {
            "background": """rows of square concrete columns | 1.0 | 0.5-0.9 | board-formed grey concrete
low concrete ceiling with pipe runs | 0.9 | 0.3-0.6 | sprinkler and conduit lines
ramp curving up to the next level | 0.7 | 0.25-0.5 | sloped broom-finished concrete
half-empty parking bays | 0.8 | 0.3-0.6 | oil-shadowed stalls
open side letting in a strip of daylight | 0.65 | 0.2-0.5 | bright band on the deck
concrete block stair core | 0.6 | 0.2-0.4 | painted grey masonry
elevator lobby recess | 0.45 | 0.15-0.35 | brushed steel doors
far wall fading into gloom | 0.6 | 0.3-0.6 | dark unlit depth
caged ceiling lights in rows | 0.7 | 0.1-0.3 | wire guards
ventilation fan set into the wall | 0.5 | 0.1-0.25 | dusty blades behind mesh
cars clustered near the core | 0.55 | 0.25-0.5 | dusty rooflines""",
            "midground": """lone car parked nose-in | 0.7 | 0.2-0.4 | dust-filmed paint
concrete wheel stops in a row | 0.7 | 0.1-0.3 | chipped rebar ends
shopping cart drifted between bays | 0.5 | 0.1-0.25 | nested wire basket
puddle under a dripping pipe | 0.5 | 0.05-0.2 | slow ring ripples
motorcycle tucked beside the core | 0.4 | 0.1-0.25 | cover half slipped off
column wrapped with a rubber guard | 0.55 | 0.05-0.2 | scuffed black collar
hose reel cabinet on the wall | 0.4 | 0.05-0.15 | steel box with a glass front
pallet of salt bags for winter | 0.35 | 0.1-0.25 | shrink-wrapped stack
sweeper machine parked in a corner | 0.35 | 0.1-0.25 | dusty brushes
bicycle chained to a rail | 0.45 | 0.05-0.2 | rusted frame
row of empty bays receding | 0.6 | 0.25-0.5 | repeating stalls""",
            "architecture_detail": """oil stains in the parking bays | 0.7 | 0.05-0.2 | dark soaked patches
expansion joint crossing the deck | 0.55 | 0.05-0.2 | rubber-filled gap
drain grate at the low point | 0.5 | 0.03-0.1 | rusted slots
chipped column corner | 0.55 | 0.03-0.12 | exposed aggregate
tire scuff arcs at the turn | 0.5 | 0.05-0.2 | black rubber sweeps
conduit junction boxes overhead | 0.45 | 0.05-0.15 | grey steel fittings
water stain running down a column | 0.5 | 0.05-0.15 | mineral streak
patched core hole in the ceiling | 0.35 | 0.02-0.08 | grout circle
kick plate on the stair door | 0.4 | 0.02-0.08 | dented steel
hairline cracks in the deck | 0.5 | 0.05-0.2 | meandering lines
low clearance bar at the ramp | 0.4 | 0.05-0.15 | scraped yellow pipe""",
            "props": """lost hubcap against the wall | 0.45 | 0.02-0.08 | scratched silver dish
flattened cardboard in a corner | 0.45 | 0.03-0.1 | rain-warped sheet
coffee cup left on a wheel stop | 0.4 | 0.02-0.06 | dried ring inside
dropped glove near the stairs | 0.4 | 0.02-0.05 | flattened knit
loose gravel swept into a pile | 0.4 | 0.03-0.1 | broom lines
bungee cord hanging off a pipe | 0.35 | 0.02-0.06 | stretched hooks
crushed water bottle in a bay | 0.4 | 0.02-0.05 | cloudy plastic
stray cone knocked on its side | 0.45 | 0.03-0.1 | faded orange
cigarette ends by the stair door | 0.4 | 0.02-0.06 | trodden filters
snapped umbrella in a corner | 0.35 | 0.02-0.06 | bent ribs
dusty tarp over a stored boat | 0.3 | 0.1-0.25 | bungeed corners""",
            "foreground_element": """keys fished out mid-stride | 0.65 | 0.05-0.15 | fob in hand
trunk lifted for the bags | 0.6 | 0.1-0.25 | struts hissing
footsteps echoing off the deck | 0.55 | 0.1-0.3 | long reverb
car door held open with a hip | 0.55 | 0.1-0.25 | bags in both hands
reverse lights flicking on nearby | 0.45 | 0.05-0.2 | white glow on concrete
phone light swept along the bays | 0.4 | 0.05-0.15 | hunting for the car
cart pushed toward the elevator | 0.45 | 0.1-0.25 | wheels clattering
jacket pulled tighter in the chill | 0.5 | 0.05-0.2 | concrete cold
mirror checked before backing out | 0.45 | 0.05-0.15 | head turned
parking level counted on fingers | 0.4 | 0.03-0.1 | which floor again
child's hand held crossing the deck | 0.4 | 0.05-0.15 | short quick steps""",
            "time_of_day": INDOOR_UTILITY_TIME,
            "weather": INDOOR_UTILITY_WEATHER,
        },
        "home_garage_interior": {
            "background": """workbench along the back wall | 1.0 | 0.4-0.7 | scarred plywood top
pegboard of hanging hand tools | 0.85 | 0.25-0.5 | outlines of missing tools
parked car under a film of dust | 0.6 | 0.3-0.6 | dulled paint
garage door tracks and coiled springs | 0.7 | 0.2-0.45 | greased steel
shelving of stacked plastic bins | 0.75 | 0.25-0.5 | lidded tubs in columns
water heater in the corner | 0.5 | 0.1-0.3 | pipe-wrapped tank
bikes hung from ceiling hooks | 0.55 | 0.15-0.35 | wheels overhead
chest freezer against the side wall | 0.5 | 0.15-0.3 | humming white box
step up to the kitchen door | 0.55 | 0.1-0.3 | worn wooden tread
window filmed with cobwebs | 0.5 | 0.1-0.25 | grey gauzy corners
ladder laid across the rafters | 0.5 | 0.15-0.35 | aluminum rails""",
            "midground": """lawn mower parked mid-floor | 0.6 | 0.15-0.3 | grass-caked deck
rolling tool chest with drawers ajar | 0.6 | 0.15-0.3 | sockets glinting
snow shovels and rakes in a barrel | 0.55 | 0.1-0.25 | handles fanned out
sawhorses with a half-cut board | 0.45 | 0.15-0.3 | fresh sawdust
extension cord coiled on a nail | 0.55 | 0.05-0.15 | orange loops
shop vacuum with a kinked hose | 0.5 | 0.1-0.25 | dust-caked canister
bag of softener salt slumped open | 0.4 | 0.05-0.15 | white pellets spilling
folding chairs leaned in a rank | 0.45 | 0.1-0.25 | webbed seats
cooler stack by the door | 0.45 | 0.1-0.25 | nested lids
kid's bike with training wheels | 0.45 | 0.1-0.25 | streamers on the grips
oil pan slid under the car | 0.35 | 0.05-0.15 | black-slicked steel""",
            "architecture_detail": """bare stud wall with wiring runs | 0.6 | 0.2-0.45 | exposed framing
concrete floor with a painted patch | 0.55 | 0.15-0.4 | flaking grey coat
floor drain in the center slope | 0.4 | 0.02-0.08 | rusted round grate
single bulb on a pull chain | 0.55 | 0.03-0.1 | swinging cone of light
door closer on the kitchen door | 0.4 | 0.02-0.08 | hissing arm
crack radiating from an anchor bolt | 0.4 | 0.03-0.1 | hairline star
weather strip shredded at the big door | 0.45 | 0.05-0.15 | torn rubber flap
mouse trap set along the wall | 0.35 | 0.02-0.06 | wooden snap bar
stained ceiling patch under the bathroom | 0.35 | 0.05-0.15 | ringed drywall
hose bib poking through the wall | 0.4 | 0.02-0.06 | dripping spigot
paint drips fossilized on the floor | 0.45 | 0.03-0.1 | speckled history""",
            "props": """jar of loose screws on the bench | 0.55 | 0.02-0.06 | mixed hardware
paint cans stacked by shade of use | 0.5 | 0.05-0.15 | drip-crusted rims
sports gear tub overflowing | 0.5 | 0.05-0.15 | gloves and balls
car wash bucket with a stiff mitt | 0.45 | 0.03-0.1 | soap-dried wool
trickle charger clipped to a battery | 0.35 | 0.02-0.08 | quiet red glow
work gloves molded to hand shapes | 0.5 | 0.02-0.06 | oil-stiffened leather
birdseed bag rolled shut | 0.4 | 0.03-0.08 | clipped top
spare tire leaned on the wall | 0.45 | 0.05-0.15 | deep tread shadow
fishing rods bundled in a corner | 0.4 | 0.05-0.15 | tangled tips
radio caked in sawdust | 0.4 | 0.02-0.08 | bent antenna
holiday bins taped shut | 0.45 | 0.1-0.2 | bulging lids""",
            "foreground_element": """bike tire pumped with a thumb check | 0.55 | 0.05-0.2 | pressure squeeze
bin dragged down from the shelf | 0.55 | 0.1-0.25 | dust avalanche
cobweb waved away from the face | 0.5 | 0.05-0.15 | backhand swipe
tool lifted off the pegboard | 0.5 | 0.05-0.15 | outline left behind
big door rolled up by hand | 0.5 | 0.1-0.3 | daylight flooding in
freezer lid held open for a dig | 0.45 | 0.1-0.2 | frost breath
sawdust brushed off the bench | 0.45 | 0.05-0.15 | palm sweep
cord unwound arm over elbow | 0.45 | 0.05-0.15 | practiced loops
car hood propped for a look | 0.4 | 0.1-0.3 | rod in the catch
knuckle sucked after a slip | 0.4 | 0.03-0.1 | wrench bite
paint can pried with a flathead | 0.4 | 0.05-0.12 | rim crack""",
            "time_of_day": INDOOR_UTILITY_TIME,
            "weather": INDOOR_UTILITY_WEATHER,
        },
        "spare_room_boxes": {
            "background": """plain cardboard boxes stacked to shoulder height | 1.0 | 0.4-0.7 | taped seams
spare bed under a dust sheet | 0.7 | 0.3-0.55 | draped ghost shape
closet with sliding doors half open | 0.6 | 0.25-0.5 | crammed rail visible
curtains drawn against the day | 0.6 | 0.2-0.45 | glowing fabric edge
garment rack of off-season coats | 0.6 | 0.2-0.4 | shoulder line of plastic
ironing board leaned by the wall | 0.5 | 0.1-0.3 | scorched cover
bookshelf of mismatched spines | 0.5 | 0.2-0.4 | double-stacked rows
exercise bike used as a hanger | 0.5 | 0.15-0.35 | towel over the bars
bare walls with pale frame ghosts | 0.5 | 0.25-0.5 | sun-faded outlines
window with the blind at half | 0.55 | 0.15-0.35 | slatted light
crib frame waiting in pieces | 0.35 | 0.15-0.3 | bundled rails""",
            "midground": """suitcase pile by size | 0.6 | 0.15-0.35 | nested hard shells
vacuum parked mid-room | 0.5 | 0.1-0.25 | cord half wound
sewing machine in a hard case | 0.45 | 0.1-0.2 | latched handle
folded card table against the bed | 0.45 | 0.15-0.3 | chipped edge
box overflowing with cables | 0.5 | 0.05-0.15 | grey spaghetti
mirror leaned face to the wall | 0.45 | 0.15-0.3 | brown paper back
rolled rugs standing in a corner | 0.45 | 0.15-0.3 | twine-tied columns
fan waiting for summer | 0.45 | 0.1-0.2 | dust-furred blades
big stuffed animal on the box stack | 0.4 | 0.1-0.2 | outgrown bear
picture frames bundled with a strap | 0.4 | 0.1-0.2 | corner protectors
laundry drying rack half folded | 0.45 | 0.1-0.25 | clipped hinges""",
            "architecture_detail": """carpet with furniture dents | 0.6 | 0.2-0.45 | deep round footprints
closet track off its runner | 0.4 | 0.03-0.1 | leaning door
ceiling light without a shade | 0.5 | 0.03-0.1 | bare bulb glare
heat register half blocked by a box | 0.4 | 0.03-0.1 | muffled airflow
door that bumps the bed frame | 0.4 | 0.05-0.15 | limited swing
window latch painted stuck | 0.35 | 0.02-0.06 | sealed sash
scuffed paint at box-carrying height | 0.45 | 0.05-0.15 | cardboard burns
outlet hidden behind the rack | 0.35 | 0.02-0.06 | plug at a stretch
closet shelf bowed in the middle | 0.4 | 0.05-0.12 | sagging plank
dust line where boxes stood before | 0.45 | 0.05-0.15 | clean rectangles
blind cord wrapped on a cleat | 0.35 | 0.02-0.06 | figure eights""",
            "props": """tape gun resting on a box | 0.5 | 0.02-0.08 | dried tape tongue
bubble wrap roll leaning | 0.45 | 0.05-0.15 | popped patches
old lamp waiting for a new home | 0.45 | 0.05-0.12 | shade askew
photo albums stacked chin high | 0.4 | 0.05-0.12 | leather spines
board games wedged sideways | 0.45 | 0.05-0.12 | crushed corners
yoga mat rolled with a strap | 0.4 | 0.03-0.1 | flaking foam
gift wrap tubes in a bin | 0.4 | 0.03-0.1 | crushed ends
winter blanket bag zipped tight | 0.45 | 0.05-0.15 | vacuum-shrunk brick
loose curtain rings in a bowl | 0.35 | 0.02-0.06 | brass circles
humidifier with its cord wrapped | 0.4 | 0.03-0.1 | white tank
sheet music in a crate | 0.3 | 0.03-0.1 | yellowed pages""",
            "foreground_element": """box flaps torn open for a search | 0.65 | 0.1-0.25 | tape ripped back
dust sheet lifted for a peek | 0.5 | 0.1-0.25 | slow reveal
box hefted with a knee assist | 0.55 | 0.1-0.25 | weight test
sneeze caught in a sleeve | 0.45 | 0.05-0.15 | dust cloud
coat tried on from the rack | 0.45 | 0.1-0.25 | mirror-less guess
suitcase zipped around the corner | 0.45 | 0.05-0.2 | knee on the lid
found photo studied too long | 0.45 | 0.05-0.15 | task forgotten
stack steadied with a chin | 0.45 | 0.05-0.2 | wobbling tower
bubble wrap popped absent-mindedly | 0.4 | 0.03-0.1 | double pinch
rug roll walked corner to corner | 0.4 | 0.1-0.25 | end-over-end
blind raised for working light | 0.45 | 0.05-0.15 | dust in the beam""",
            "time_of_day": INDOOR_UTILITY_TIME,
            "weather": INDOOR_UTILITY_WEATHER,
        },
        "office_stairwell": {
            "background": """painted concrete stairs doubling back up | 1.0 | 0.4-0.8 | grey treads with worn noses
steel handrails in gloss paint | 0.85 | 0.2-0.45 | chipped tube rails
landing with a small wired-glass window | 0.7 | 0.15-0.35 | mesh-embedded pane
fire door with a push bar | 0.7 | 0.15-0.35 | dented steel face
block walls glossy to shoulder height | 0.75 | 0.3-0.6 | two-tone paint line
stair core rising out of sight | 0.6 | 0.3-0.6 | receding zigzag
pipe runs clamped to the wall | 0.5 | 0.1-0.3 | painted-over conduit
underside of the flight above | 0.6 | 0.2-0.4 | shadowed soffit
caged light on each landing | 0.6 | 0.05-0.15 | wire basket glow
dusty corner behind the door swing | 0.5 | 0.1-0.25 | untouched triangle
roof access ladder at the top | 0.35 | 0.1-0.25 | rungs into shadow""",
            "midground": """handrail post anchored in the tread | 0.55 | 0.05-0.15 | bolted base
mop bucket parked on a landing | 0.4 | 0.05-0.15 | grey wringer
stack of ceiling tiles waiting | 0.35 | 0.1-0.2 | chalky corners
door wedge holding one level open | 0.45 | 0.03-0.1 | rubber cheese
old chair exiled to a landing | 0.4 | 0.1-0.2 | one wheel missing
ladder chained under the flight | 0.35 | 0.1-0.2 | padlocked rails
dust broom leaned in the corner | 0.4 | 0.03-0.1 | flattened bristles
paint tray dried mid-project | 0.3 | 0.05-0.12 | skinned roller
window ledge with a dead plant | 0.35 | 0.03-0.1 | crisp leaves
draft rolling dust at the base | 0.4 | 0.05-0.15 | drifting fluff
handprints on the push bar | 0.45 | 0.02-0.08 | polished patch""",
            "architecture_detail": """worn nosing strips on the treads | 0.6 | 0.05-0.2 | smoothed grit tape
paint drips on the stringer | 0.45 | 0.03-0.1 | ancient runs
hairline crack chasing the landing | 0.45 | 0.05-0.15 | wandering line
rail bracket loose at one joint | 0.4 | 0.02-0.06 | rattling collar
scuffed arc where the door drags | 0.45 | 0.03-0.1 | quarter-circle groove
wired glass with a stress crack | 0.4 | 0.03-0.1 | starred corner
riser heights not quite equal | 0.35 | 0.05-0.15 | stumble step
echoing acoustics off hard faces | 0.5 | 0.1-0.3 | slap-back sound
hinge painted into stiffness | 0.4 | 0.02-0.06 | groaning swing
floor level change at the door | 0.4 | 0.02-0.08 | steel threshold
dust settled on the pipe tops | 0.45 | 0.03-0.1 | grey felt line""",
            "props": """forgotten coffee cup on a step | 0.45 | 0.02-0.06 | cold half inch
rubber band dropped on a landing | 0.35 | 0.01-0.04 | dusty loop
hair tie by the rail base | 0.35 | 0.01-0.04 | stretched spiral
paperclip in a tread corner | 0.3 | 0.01-0.03 | bent wire
umbrella hooked on the rail | 0.4 | 0.02-0.08 | drip line below
badge reel clipped and abandoned | 0.3 | 0.01-0.04 | retracted cord
crumpled tissue near the door | 0.35 | 0.01-0.04 | missed pocket
salt crystals tracked up in winter | 0.4 | 0.03-0.1 | white scatter
lanyard draped on a rail post | 0.35 | 0.02-0.06 | dangling clip
energy bar wrapper folded small | 0.35 | 0.01-0.04 | guilty crease
single glove on the window ledge | 0.35 | 0.02-0.05 | waiting for its owner""",
            "foreground_element": """two steps taken at a time | 0.6 | 0.1-0.3 | hand skimming the rail
breather taken on a landing | 0.55 | 0.1-0.25 | hand on the hip
phone call taken in the echo | 0.5 | 0.05-0.2 | lowered voice
push bar hit with a forearm | 0.5 | 0.05-0.15 | door swinging wide
heels carried for the climb | 0.4 | 0.05-0.15 | stocking steps
coffee carried level on the turns | 0.5 | 0.05-0.15 | careful wrist
stretch against the rail | 0.45 | 0.05-0.2 | calf pulled long
floor counted under the breath | 0.45 | 0.03-0.1 | three more to go
jacket shrugged on mid-flight | 0.45 | 0.05-0.2 | sleeve chase
window glanced through on the turn | 0.45 | 0.05-0.15 | roofline check
railing drummed absent-mindedly | 0.4 | 0.03-0.1 | ring of the tube""",
            "time_of_day": INDOOR_UTILITY_TIME,
            "weather": INDOOR_UTILITY_WEATHER,
        },
        "office_supply_room": {
            "background": """steel shelving stacked with paper reams | 1.0 | 0.4-0.7 | white bricks in rows
copier against the wall mid-cycle | 0.7 | 0.25-0.5 | warm grey bulk
cabinets with narrow drawers | 0.6 | 0.25-0.45 | dented beige steel
storage boxes squared on high shelves | 0.7 | 0.25-0.5 | lidded cardboard
door with a wired-glass slit | 0.5 | 0.1-0.3 | hallway glow through mesh
folding table for collating | 0.55 | 0.2-0.4 | scratched laminate
broken chairs gathered in a corner | 0.45 | 0.15-0.3 | stacked seat pans
window into a dark light-well | 0.4 | 0.1-0.25 | dusty pane
shelf of mismatched binders | 0.55 | 0.2-0.4 | leaning spines
old monitors facing the wall | 0.45 | 0.15-0.3 | grey screens
coat rack of forgotten layers | 0.4 | 0.1-0.25 | abandoned cardigans""",
            "midground": """cart with squeaking casters | 0.55 | 0.1-0.25 | wobbling shelf
stack of paper boxes as a step | 0.5 | 0.1-0.25 | crushed corner
shredder with a full bin | 0.5 | 0.1-0.2 | confetti drift inside
tub of tangled cables | 0.55 | 0.05-0.15 | grey spaghetti
spare keyboards shingled in a box | 0.5 | 0.05-0.15 | key rows overlapping
water cooler jugs lined empty | 0.45 | 0.1-0.2 | blue plastic ranks
ladder folded against the shelf end | 0.45 | 0.1-0.25 | paint-freckled rungs
box of holiday decorations | 0.4 | 0.1-0.2 | tinsel escaping
banker's lamp missing its chain | 0.3 | 0.03-0.1 | brass and green glass
crate of misdelivered supplies | 0.4 | 0.1-0.2 | still strapped
fan for the copier heat | 0.4 | 0.05-0.15 | oscillating slowly""",
            "architecture_detail": """vinyl tile floor gone amber | 0.6 | 0.2-0.45 | waxed layers
ceiling tile bowed with age | 0.45 | 0.05-0.15 | sagging center
shelf uprights bolted to the wall | 0.5 | 0.05-0.2 | anti-tip brackets
outlet strip mounted at waist height | 0.45 | 0.02-0.08 | switch glowing
paint shadow of removed shelving | 0.4 | 0.05-0.15 | pale rectangles
door closer that slams anyway | 0.4 | 0.02-0.08 | worn damper
flicker in the far tube | 0.4 | 0.03-0.1 | strobe corner
dust felt on the cabinet tops | 0.45 | 0.03-0.1 | untouched years
floor scuffs from the cart wheels | 0.45 | 0.05-0.15 | grey arcs
vent breathing paper-dry air | 0.4 | 0.02-0.08 | warm exhale
keyhole worn bright in the cabinet | 0.35 | 0.01-0.05 | brass ring""",
            "props": """stapler graveyard in a drawer tray | 0.45 | 0.02-0.08 | jammed veterans
box of loose pens tested and mixed | 0.5 | 0.02-0.08 | chewed caps
tape rolls threaded on a spindle | 0.45 | 0.02-0.06 | shrinking stack
scissors tethered with string | 0.4 | 0.01-0.05 | communal pair
rubber band ball started years ago | 0.4 | 0.01-0.05 | layered sphere
pushpins in a magnet dish | 0.4 | 0.01-0.05 | bristling colors
envelope stacks sorted by size | 0.5 | 0.05-0.12 | banded bundles
toner boxes sealed in plastic | 0.45 | 0.05-0.12 | heavy little bricks
mug of dried-out highlighters | 0.4 | 0.02-0.06 | capless casualties
staple remover like a tiny jaw | 0.35 | 0.01-0.04 | sprung teeth
first aid tin restocked halfway | 0.4 | 0.02-0.06 | bandage ends""",
            "foreground_element": """ream carried under one arm | 0.6 | 0.05-0.15 | dense little load
copier lid lifted for a jam hunt | 0.55 | 0.1-0.25 | green glow escaping
drawer eased past its squeak | 0.45 | 0.05-0.15 | slow pull
cable tub dug through elbow deep | 0.5 | 0.1-0.2 | connector hunt
box lid pried without the cutter | 0.45 | 0.05-0.15 | tape stretched white
shelf reached on tiptoe | 0.5 | 0.05-0.2 | fingertip slide
cart steered around the table | 0.45 | 0.1-0.25 | caster squeal
pen tested on a palm | 0.45 | 0.02-0.08 | spiral scribble
armload balanced with a chin | 0.5 | 0.05-0.2 | tower management
shredder fed one guilty page | 0.4 | 0.05-0.15 | satisfying pull
light waited out mid-flicker | 0.4 | 0.05-0.15 | patient pause""",
            "time_of_day": INDOOR_UTILITY_TIME,
            "weather": INDOOR_UTILITY_WEATHER,
        },
        "store_stockroom": {
            "background": """tall back-room shelving to the ceiling | 1.0 | 0.5-0.8 | boltless steel bays
shrink-wrapped pallets waiting | 0.8 | 0.3-0.55 | cloudy plastic towers
roll cages parked in a rank | 0.7 | 0.25-0.5 | folded mesh walls
swinging double doors to the floor | 0.65 | 0.15-0.35 | scuffed port-hole doors
baler of flattened cardboard | 0.55 | 0.2-0.4 | strapped brown cube
concrete floor worn smooth | 0.7 | 0.3-0.6 | polished traffic lanes
racking bay of overstock tubs | 0.6 | 0.25-0.5 | lidded grey totes
dock door sealed with a rubber skirt | 0.5 | 0.2-0.4 | daylight at the edges
mezzanine steps to upper storage | 0.45 | 0.15-0.35 | checker-plate treads
ceiling of open joists and ducting | 0.6 | 0.25-0.5 | silver insulation
cage of compressed gas bottles | 0.35 | 0.1-0.25 | chained cylinders""",
            "midground": """pallet jack resting under a load | 0.65 | 0.1-0.3 | forks sunk in
hand truck leaned on a bay post | 0.55 | 0.05-0.2 | bent toe plate
ladder rolling on its track | 0.45 | 0.15-0.35 | hooked rail wheels
stack of empty totes swaying tall | 0.55 | 0.15-0.3 | nested tower
apron hung on a bay end | 0.45 | 0.03-0.1 | pocket sagging
broken pallet set aside | 0.45 | 0.1-0.2 | splintered slats
floor fan moving the stale air | 0.45 | 0.1-0.2 | ribbons on the cage
spill kit bucket in its corner | 0.4 | 0.05-0.15 | sealed red lid
shrink wrap roll on a spindle | 0.5 | 0.05-0.15 | stretched tail
returns cart of orphaned goods | 0.45 | 0.1-0.25 | mixed jumble
step stool worn to bare metal | 0.45 | 0.05-0.15 | ribbed platform""",
            "architecture_detail": """rack legs in dented guards | 0.55 | 0.05-0.15 | scraped yellow steel
floor joints filled and refilled | 0.5 | 0.05-0.2 | dark tar lines
column base ringed with scuffs | 0.5 | 0.05-0.15 | forklift history
sprinkler heads among the joists | 0.45 | 0.03-0.1 | brass drops
walls of bare painted block | 0.55 | 0.25-0.5 | grey masonry
skylight panel of milky plastic | 0.4 | 0.05-0.2 | diffused glow
door sweep shedding bristles | 0.4 | 0.02-0.08 | brush strip
chill near the dock seal | 0.45 | 0.1-0.25 | leaking edge
dust on the top-bay stock | 0.5 | 0.05-0.2 | long-stay layer
wheel marks curving to the doors | 0.5 | 0.1-0.25 | polished arcs
emergency light with twin heads | 0.4 | 0.02-0.08 | dusty frog eyes""",
            "props": """box cutter parked on a rail | 0.5 | 0.01-0.05 | taped handle
packing tape roll on a wrist | 0.45 | 0.02-0.06 | quick-draw loop
gloves tucked in an apron pocket | 0.45 | 0.02-0.06 | molded fingers
strapping offcuts curled on the floor | 0.45 | 0.02-0.08 | plastic ribbons
broom and long-handled pan set | 0.45 | 0.03-0.1 | leaning pair
radio murmuring from a shelf | 0.4 | 0.02-0.06 | dusty speaker
water bottle squadron on a beam | 0.45 | 0.02-0.08 | shift hydration
banana box repurposed again | 0.45 | 0.05-0.12 | vented ends
zip ties in a coffee can | 0.4 | 0.01-0.05 | bristling loops
knee pads dropped at a bay | 0.35 | 0.02-0.06 | scarred shells
dolly strap coiled like a snake | 0.4 | 0.02-0.06 | hooked ends""",
            "foreground_element": """pallet jack pumped to lift | 0.6 | 0.1-0.25 | handle strokes
tote slid off at chest height | 0.55 | 0.1-0.2 | controlled drop
wrap slashed along the seam | 0.5 | 0.05-0.15 | plastic sighing open
cage rolled leaning into the turn | 0.5 | 0.1-0.3 | mesh rattle
armful steadied with the chin | 0.5 | 0.05-0.2 | tower discipline
double doors backed through | 0.5 | 0.1-0.25 | hip push
top bay scouted from the ladder | 0.45 | 0.1-0.3 | neck craned
tape gun run around a box | 0.5 | 0.05-0.15 | squealing pass
gloves snapped on for the glass load | 0.4 | 0.03-0.1 | finger wiggle
count kept on raised fingers | 0.45 | 0.03-0.1 | silent tally
pallet corner nudged square | 0.45 | 0.05-0.15 | boot tap""",
            "time_of_day": INDOOR_UTILITY_TIME,
            "weather": INDOOR_UTILITY_WEATHER,
        },
        "school_gym_storage": {
            "background": """crash mats stacked chest high | 1.0 | 0.4-0.7 | blue vinyl slabs
ball bins of every code | 0.85 | 0.25-0.5 | rubber and leather spheres
cone stacks leaning like towers | 0.7 | 0.15-0.35 | faded orange spirals
folded net posts on a rack | 0.6 | 0.2-0.4 | padded uprights
caged window high in the wall | 0.55 | 0.1-0.25 | wire-hatched daylight
hurdles nested in a row | 0.5 | 0.2-0.4 | staggered frames
gym floor visible through the door | 0.5 | 0.15-0.35 | pale sprung boards
rope coils on wall pegs | 0.5 | 0.1-0.25 | thick climbing hemp
beam sections parked on trolleys | 0.45 | 0.2-0.4 | chalk-smudged suede
block wall painted glossy | 0.6 | 0.3-0.55 | rolled enamel
stack of relay batons in a crate | 0.45 | 0.05-0.15 | worn aluminum""",
            "midground": """cart of playground balls half flat | 0.6 | 0.15-0.3 | soft red casualties
parachute bag slumped in a corner | 0.5 | 0.1-0.25 | rainbow spilling out
pommel horse under a dust sheet | 0.45 | 0.15-0.3 | draped hulk
spring board on its side | 0.45 | 0.1-0.25 | carpeted wedge
bag of pinnies knotted shut | 0.55 | 0.05-0.15 | mesh bundle
badminton racquets in a barrel | 0.5 | 0.1-0.2 | tangled strings
floor tape rolls on a spindle | 0.45 | 0.03-0.1 | bright rings
stack of hoops leaning tall | 0.5 | 0.1-0.25 | nested circles
scooter boards piled seat to seat | 0.45 | 0.1-0.2 | caster clatter waiting
first aid bum bag on a hook | 0.4 | 0.02-0.08 | half-zipped
bases and a pitching rubber in a crate | 0.4 | 0.05-0.15 | scuffed slabs""",
            "architecture_detail": """concrete floor cool underfoot | 0.6 | 0.25-0.5 | sealed grey
door wide enough for the trolleys | 0.5 | 0.1-0.3 | double leaf
wall pegs bent from rope weight | 0.45 | 0.03-0.1 | angled hooks
ceiling grille over a heater | 0.4 | 0.03-0.1 | ticking fins
chalk dust settled along the base | 0.45 | 0.05-0.15 | pale drift line
scuff constellation at cart height | 0.5 | 0.05-0.15 | black nebulae
hinge squeal known to every class | 0.45 | 0.02-0.08 | two-tone creak
mesh over the light fittings | 0.45 | 0.03-0.1 | ball-proof cages
floor plate for the net posts | 0.4 | 0.02-0.08 | brass cover
paint worn where the mats slide | 0.45 | 0.05-0.15 | bare patch
draft under the outside door | 0.4 | 0.03-0.1 | leaf-fed gap""",
            "props": """whistle on a nail by the door | 0.5 | 0.01-0.05 | chrome pea
pump with a chewed needle valve | 0.5 | 0.02-0.06 | duct-taped shaft
stopwatch in a velvet box | 0.35 | 0.01-0.04 | teacher's treasure
bean bags faded to pastel | 0.45 | 0.03-0.1 | soft little pillows
rosin bag gone hard | 0.3 | 0.01-0.04 | cracked pouch
lost sneaker on the mat stack | 0.4 | 0.02-0.06 | single orphan
jump ropes braided together | 0.5 | 0.03-0.1 | inseparable tangle
frisbees warped in a pile | 0.4 | 0.02-0.08 | sun-cupped discs
kickballs wedged behind a beam | 0.4 | 0.02-0.08 | escape artists
sweatband basket nobody claims | 0.35 | 0.02-0.06 | terry rainbow
tug rope with a taped middle | 0.4 | 0.05-0.15 | flag long gone""",
            "foreground_element": """mat dragged with both fists | 0.6 | 0.15-0.3 | vinyl squealing
cone tower carried like a torch | 0.55 | 0.05-0.2 | chin-high wobble
ball bin steered by its lip | 0.55 | 0.1-0.25 | one-hand pilot
net post walked corner to corner | 0.45 | 0.1-0.3 | end-over-end
hoop stack fanned for the count | 0.45 | 0.05-0.2 | color check
rope coil shoulder-carried | 0.45 | 0.05-0.2 | heavy loop
pinnie bag emptied in a heap | 0.5 | 0.05-0.15 | mesh flood
pump worked with a knee brace | 0.5 | 0.05-0.15 | hissing strokes
door held for the trolley train | 0.5 | 0.05-0.2 | foot as a stop
inventory counted on fingers | 0.45 | 0.03-0.1 | lips moving
sneeze from the chalk drift | 0.4 | 0.03-0.1 | dusty cloud""",
            "time_of_day": INDOOR_UTILITY_TIME,
            "weather": INDOOR_UTILITY_WEATHER,
        },
    },
    # ═════════════════════════════ OUTDOOR ═════════════════════════════
    "outdoor/private_spaces_us": {
        "mall_parking_lot_far_corner": {
            "background": """empty far corner of the lot | 1.0 | 0.5-0.9 | faded stall stripes
light poles spaced down the rows | 0.8 | 0.15-0.4 | twin heads on tapered masts
mall roofline low on the horizon | 0.6 | 0.2-0.45 | long beige band
hedge strip along the boundary | 0.6 | 0.2-0.4 | dusty evergreen wall
cars clustered far off by the doors | 0.6 | 0.2-0.4 | distant glinting cluster
cart corral standing alone | 0.55 | 0.1-0.3 | pipe-rail pen
overflow rows never used | 0.6 | 0.3-0.6 | clean unbroken stripes
retention pond behind a fence | 0.4 | 0.2-0.4 | reedy bank
tree islands on mulch mounds | 0.55 | 0.15-0.35 | staked young trunks
sky taking most of the frame | 0.7 | 0.4-0.7 | wide open ceiling
service road skirting the lot | 0.45 | 0.15-0.35 | kerbed lane""",
            "midground": """single car parked far from all others | 0.6 | 0.15-0.35 | careful owner
stray cart drifted to the kerb | 0.55 | 0.05-0.2 | wheels turned
seagulls working the empty rows | 0.45 | 0.05-0.2 | inland scavengers
puddle stretched across a low row | 0.5 | 0.1-0.3 | sky mirrored
kerbed island with a young maple | 0.5 | 0.1-0.25 | mulch ring
delivery van cutting the diagonal | 0.4 | 0.1-0.25 | unhurried route
skateboarder using the smooth rows | 0.35 | 0.1-0.25 | pushing long lines
snow pile remnant in spring | 0.3 | 0.1-0.25 | gritty grey heap
lamp pole base ringed with concrete | 0.5 | 0.05-0.15 | anchor bolts
line-marking machine parked mid-job | 0.3 | 0.1-0.2 | half-fresh stripe
wind rolling a paper cup | 0.45 | 0.03-0.1 | hollow rattle""",
            "architecture_detail": """asphalt fading to grey | 0.6 | 0.3-0.6 | sun-bleached surface
crack lines filled with tar | 0.55 | 0.1-0.3 | black meanders
kerb stones chipped at the corner | 0.5 | 0.05-0.15 | broken lips
drain grate collecting grit | 0.5 | 0.03-0.1 | silted slots
stall stripes worn to shadows | 0.55 | 0.1-0.3 | ghost grid
wheel stop out of line | 0.45 | 0.05-0.15 | nudged block
weeds claiming the expansion joints | 0.5 | 0.05-0.2 | green seams
mulch washed onto the asphalt | 0.45 | 0.05-0.15 | bark scatter
pole base with old bump scars | 0.45 | 0.03-0.1 | paint transfers
kerb ramp to the walkway | 0.4 | 0.05-0.15 | bumpy domes
oil spots in the popular stalls | 0.45 | 0.05-0.15 | dark medallions""",
            "props": """paper cup rolling in the wind | 0.5 | 0.02-0.06 | hollow scrape
lost sunglasses on the kerb | 0.4 | 0.01-0.05 | one arm bent
bungee cord dropped off a roof rack | 0.35 | 0.01-0.05 | hooked coil
crushed fountain drink at a stall line | 0.4 | 0.02-0.06 | flattened cup
pigeon feather stuck in the tar | 0.35 | 0.01-0.04 | grey quill
balloon deflated against the hedge | 0.35 | 0.02-0.06 | wrinkled skin
hair tie on the asphalt | 0.35 | 0.01-0.03 | dusty loop
cart wheel cover cracked off | 0.35 | 0.01-0.04 | grey disc
mitten dropped by the corral | 0.3 | 0.01-0.04 | salt-stained wool
apple core near the tree island | 0.3 | 0.01-0.04 | browning spiral
zip tie clipped and dropped | 0.3 | 0.01-0.03 | white curl""",
            "foreground_element": """bags heaved into the trunk | 0.65 | 0.1-0.25 | handle dig
cart pushed back the long way | 0.55 | 0.1-0.3 | solo trek
keys blipped from three rows out | 0.55 | 0.05-0.15 | tail lights answering
door opened wide with no neighbors | 0.5 | 0.1-0.25 | luxury of space
coat grabbed against the wind | 0.5 | 0.05-0.2 | flapping hem
phone checked leaning on the car | 0.5 | 0.05-0.2 | one elbow on the roof
kid buckled without the door dance | 0.45 | 0.1-0.25 | full swing room
receiptless walk straight to the car | 0.45 | 0.1-0.25 | freehand stride
puddle looped around mid-row | 0.5 | 0.05-0.2 | wide arc
gull shooed off the hood | 0.35 | 0.05-0.15 | wing flurry
mirror wiped with a sleeve | 0.45 | 0.03-0.1 | dew swipe""",
            "time_of_day": OUTDOOR_QUIET_TIME,
            "weather": OUTDOOR_QUIET_WEATHER,
        },
        "parking_garage_roof_deck": {
            "background": """open top deck under full sky | 1.0 | 0.5-0.9 | pale concrete plain
low parapet wall around the edge | 0.85 | 0.2-0.45 | cast concrete rail
stair core bulkhead standing alone | 0.7 | 0.15-0.35 | painted block hut
ramp mouth surfacing from below | 0.6 | 0.15-0.35 | dark sloped throat
distant rooftops and water towers | 0.6 | 0.25-0.5 | low skyline band
rows almost entirely empty | 0.75 | 0.3-0.6 | bare stall grid
elevator penthouse with a humming vent | 0.45 | 0.1-0.3 | louvered box
cable barriers between levels | 0.45 | 0.1-0.25 | tensioned strands
clouds moving their shadows across | 0.55 | 0.3-0.6 | drifting patches
treetops showing over the parapet | 0.45 | 0.15-0.35 | green fringe
lightning rods on the bulkhead | 0.35 | 0.03-0.1 | thin spikes""",
            "midground": """lone car backed into a corner | 0.55 | 0.15-0.3 | long-stay dust
pigeons pecking the empty deck | 0.45 | 0.05-0.2 | strutting pair
puddles in the deck's low spots | 0.5 | 0.1-0.3 | sky pieces
wind combing grit across the surface | 0.5 | 0.1-0.3 | hissing drift
hooded figure taking the air | 0.35 | 0.1-0.25 | lunch-break escape
maintenance cart by the bulkhead | 0.35 | 0.1-0.2 | coiled hoses
row of solar-light bollards | 0.35 | 0.05-0.2 | dim caps
snow-melt salt bin strapped shut | 0.35 | 0.05-0.15 | yellow tub
antenna cluster on the far corner | 0.4 | 0.05-0.2 | guyed masts
plane crossing high overhead | 0.4 | 0.03-0.1 | thin contrail
runner using the empty rows | 0.3 | 0.1-0.25 | laps in the open""",
            "architecture_detail": """deck coating worn in the lanes | 0.55 | 0.15-0.4 | grey traffic paths
drainage scuppers through the parapet | 0.5 | 0.03-0.1 | weep holes
expansion joints with rubber spines | 0.5 | 0.05-0.2 | ridged strips
parapet top rounded by weather | 0.45 | 0.05-0.2 | smoothed edge
rust bleeding at a rail anchor | 0.45 | 0.02-0.08 | orange tears
bird wire along the bulkhead edge | 0.4 | 0.02-0.08 | thin spikes
hatch to the mechanical space | 0.35 | 0.03-0.1 | padlocked lid
wind moaning at the stair door | 0.45 | 0.05-0.15 | gap whistle
concrete patched in odd rectangles | 0.5 | 0.1-0.25 | shade mismatch
camera dome on a corner mast | 0.35 | 0.02-0.06 | smoked glass eye
lichen freckling the parapet | 0.4 | 0.05-0.15 | slow constellations""",
            "props": """crushed can wedged at a joint | 0.4 | 0.01-0.05 | sun-faded aluminum
cigarette ends by the stair door | 0.45 | 0.02-0.06 | break-time midden
lost scarf caught on the parapet | 0.3 | 0.02-0.06 | flapping tail
grit swept into wind rows | 0.45 | 0.05-0.15 | natural sorting
feather pinned by a pebble | 0.35 | 0.01-0.03 | grey curve
coffee lid without its cup | 0.35 | 0.01-0.04 | rolling disc
bolt and washer shed by a machine | 0.3 | 0.01-0.03 | rusted pair
chalk marks from a forgotten game | 0.3 | 0.03-0.1 | rain-softened lines
zip tie on the cable barrier | 0.3 | 0.01-0.03 | weathered curl
gull posted on the parapet corner | 0.4 | 0.03-0.1 | lookout duty
dead leaves gathered at the bulkhead | 0.45 | 0.05-0.15 | crisp drift""",
            "foreground_element": """hair caught by the rooftop wind | 0.6 | 0.05-0.2 | strands across the face
lean on the parapet for the view | 0.6 | 0.1-0.3 | forearms on concrete
jacket zipped against the gust | 0.55 | 0.05-0.2 | collar up
phone panorama swept slowly | 0.45 | 0.05-0.15 | horizon pass
coffee shielded from the wind | 0.45 | 0.03-0.1 | lid checked
deep breath taken at the edge | 0.5 | 0.1-0.25 | shoulders dropping
sunset watched from the hood | 0.4 | 0.1-0.3 | wind-blown perch
car spotted from above | 0.45 | 0.05-0.2 | pointing it out
stretch after the drive | 0.45 | 0.05-0.2 | arms overhead
grit blinked out of an eye | 0.4 | 0.03-0.1 | knuckle dab
echo tested down the ramp mouth | 0.35 | 0.05-0.15 | quick shout""",
            "time_of_day": OUTDOOR_QUIET_TIME,
            "weather": OUTDOOR_QUIET_WEATHER,
        },
        "side_yard_between_houses": {
            "background": """narrow grass strip between the houses | 1.0 | 0.4-0.8 | shaded thin lawn
neighbor's siding wall close by | 0.8 | 0.3-0.6 | vinyl planks
gate to the front yard | 0.6 | 0.15-0.35 | sagging pickets
AC units side by side | 0.65 | 0.15-0.35 | humming grey boxes
stepping stones sunk in the grass | 0.55 | 0.15-0.35 | tilted pavers
hose reel bolted to the wall | 0.55 | 0.1-0.25 | coiled green
downspouts running the corner | 0.55 | 0.1-0.25 | white elbows
fence panels closing the far end | 0.6 | 0.25-0.5 | greyed boards
window wells with gravel beds | 0.5 | 0.1-0.3 | plastic domes
trash cans staged for the week | 0.55 | 0.15-0.35 | wheeled pair
strip of sky overhead | 0.55 | 0.2-0.45 | ribbon of light""",
            "midground": """ladder stored on wall hooks | 0.5 | 0.15-0.35 | horizontal aluminum
firewood stacked under the eave | 0.5 | 0.15-0.3 | split rows
kiddie pool leaned to dry | 0.4 | 0.15-0.3 | dripping shell
rain barrel under a downspout | 0.45 | 0.1-0.25 | lidded drum
wheelbarrow parked nose down | 0.45 | 0.1-0.25 | crusted tray
bags of mulch half used | 0.45 | 0.1-0.2 | folded tops
dog run wire between posts | 0.35 | 0.1-0.25 | slack line
bicycle under a tarp | 0.4 | 0.1-0.25 | shrouded shape
recycling bins nested | 0.5 | 0.1-0.2 | stacked blue
window screens leaned for cleaning | 0.4 | 0.1-0.25 | mesh row
cat patrolling the fence top | 0.4 | 0.05-0.15 | balanced walk""",
            "architecture_detail": """moss on the shaded foundation | 0.5 | 0.05-0.2 | green skirt
hose bib dripping into a worn dip | 0.45 | 0.02-0.08 | mud cup
meter boxes side by side on the wall | 0.5 | 0.05-0.15 | grey domes
mud line splashed on the siding | 0.45 | 0.05-0.15 | rain spatter
gravel drip strip along the eave | 0.45 | 0.05-0.2 | stone ribbon
gate hinge sprung and mended | 0.4 | 0.02-0.08 | wire fix
cable line looping to the corner | 0.4 | 0.03-0.1 | sagging span
narrow shade that never lifts | 0.5 | 0.2-0.45 | permanent dusk
fence post heaved by frost | 0.4 | 0.05-0.15 | leaning stub
worn path hugging one wall | 0.5 | 0.1-0.3 | bare dirt line
dryer vent flap mid-flutter | 0.4 | 0.02-0.06 | lint whiskers""",
            "props": """watering can by the spigot | 0.45 | 0.02-0.08 | faded plastic
dog toy lost in the strip | 0.4 | 0.01-0.05 | chewed ball
clothes pins clipped to the wire | 0.4 | 0.01-0.05 | weathered wood
trowel stuck in a mulch bag | 0.4 | 0.01-0.05 | rusted blade
flower pots stacked by size | 0.45 | 0.03-0.1 | nested terracotta
sponge dried on the window sill | 0.35 | 0.01-0.04 | stiff yellow
broom against the gate | 0.4 | 0.02-0.08 | splayed bristles
paint can under the eave | 0.35 | 0.02-0.06 | rain-ringed lid
extension cord out a window | 0.35 | 0.02-0.08 | temporary forever
bird nest tucked on the meter box | 0.35 | 0.02-0.06 | grass cup
gutter scoop dropped after the job | 0.3 | 0.01-0.04 | leaf-stained plastic""",
            "foreground_element": """squeeze past the AC units | 0.55 | 0.1-0.25 | sideways shuffle
hose unreeled toward the front | 0.55 | 0.1-0.25 | loops paying out
trash can walked to the kerb | 0.6 | 0.1-0.3 | tilt and roll
gate held for the dog | 0.5 | 0.05-0.2 | nose already through
screen carried flat like a tray | 0.45 | 0.1-0.25 | careful grip
firewood armload stacked high | 0.45 | 0.05-0.2 | chin clamp
stepping stones taken in stride | 0.5 | 0.05-0.2 | practiced hops
barrel lid lifted for a level check | 0.4 | 0.05-0.15 | dark water peek
tarp corner re-tucked | 0.45 | 0.05-0.15 | bungee stretch
wall touched ducking the ladder | 0.45 | 0.05-0.15 | low clearance
cat greeted on the fence line | 0.4 | 0.03-0.12 | offered knuckles""",
            "time_of_day": OUTDOOR_QUIET_TIME,
            "weather": OUTDOOR_QUIET_WEATHER,
        },
        "backyard_behind_the_shed": {
            "background": """back wall of the shed | 1.0 | 0.4-0.7 | weathered boards
fence corner closing the pocket | 0.8 | 0.3-0.6 | grey panel angle
compost heap in its bin | 0.6 | 0.15-0.35 | dark crumbly layers
overgrown strip nobody mows | 0.65 | 0.2-0.45 | knee-high seed heads
firewood rack under a lean-to | 0.55 | 0.2-0.4 | split stack
neighbor's tree hanging over | 0.55 | 0.2-0.45 | borrowed shade
brush pile waiting for a permit | 0.45 | 0.15-0.35 | grey tangle
rain barrel on blocks | 0.45 | 0.1-0.25 | green drum
old swing frame gone to rust | 0.4 | 0.15-0.35 | naked A-frame
bare patch where the pool stood | 0.45 | 0.2-0.4 | yellow ring
vines taking the fence corner | 0.5 | 0.2-0.4 | leafy scramble""",
            "midground": """wheelbarrow tipped against the shed | 0.55 | 0.1-0.25 | rust-freckled tray
stack of spare fence boards | 0.5 | 0.1-0.25 | silvered lumber
garden cart with a flat tire | 0.45 | 0.1-0.25 | slumped corner
chicken wire roll standing | 0.4 | 0.05-0.2 | springy cylinder
bird bath gone green | 0.4 | 0.05-0.15 | algae bowl
burn barrel with a mesh lid | 0.35 | 0.1-0.2 | scorched drum
tomato cages stored in a jumble | 0.45 | 0.1-0.2 | wire cones
dog exploring the long grass | 0.45 | 0.1-0.25 | nose-down patrol
step ladder folded flat on nails | 0.4 | 0.1-0.25 | wall-hung rails
bag of leaves from last fall | 0.45 | 0.1-0.2 | sagging paper sack
rabbit frozen at the fence gap | 0.35 | 0.03-0.1 | ears up""",
            "architecture_detail": """shed skids sinking into the soil | 0.45 | 0.05-0.2 | tilted base
fence boards cupped by the sun | 0.5 | 0.1-0.3 | curled grain
gap under the fence dug by something | 0.45 | 0.03-0.1 | excavated scoop
moss on the shed's north side | 0.5 | 0.1-0.25 | green shingle shade
hinges bleeding rust down the door | 0.45 | 0.03-0.1 | orange streaks
mushrooms at the compost edge | 0.4 | 0.03-0.1 | pale clusters
wasp nest started under the eave | 0.35 | 0.02-0.06 | grey paper knuckle
path worn to the compost | 0.5 | 0.1-0.25 | bare dirt line
window of the shed silted blind | 0.45 | 0.03-0.1 | cobwebbed pane
anthill at the fence post base | 0.4 | 0.02-0.08 | sandy crater
stake line from an old bed | 0.35 | 0.05-0.15 | string long gone""",
            "props": """rusted trowel in the weeds | 0.45 | 0.01-0.05 | lost last season
cracked plant pots in a heap | 0.5 | 0.03-0.1 | terracotta shards
hose end without its nozzle | 0.4 | 0.01-0.05 | dribbling thread
dog bone half buried | 0.4 | 0.01-0.05 | ongoing project
tennis ball gone grey | 0.45 | 0.01-0.04 | fetch veteran
glove stiffened over a post | 0.4 | 0.01-0.05 | scarecrow wave
watering can with a split seam | 0.4 | 0.02-0.06 | weeping side
brick pile from a project | 0.45 | 0.05-0.12 | mossy stack
netting bundled from the berry bed | 0.35 | 0.02-0.08 | tangled mist
frisbee lodged in the vines | 0.35 | 0.01-0.04 | faded disc
snail shells collected on a stump | 0.3 | 0.01-0.04 | spiral row""",
            "foreground_element": """compost turned with a fork | 0.55 | 0.1-0.25 | steaming layers
armload of wood pulled from the rack | 0.55 | 0.1-0.25 | bark crumbs
long grass waded through | 0.55 | 0.1-0.3 | seed heads ticking
shed door shoulder-bumped open | 0.5 | 0.05-0.2 | swollen wood
dog called off a scent | 0.45 | 0.05-0.2 | reluctant return
barrow handles lifted for the run | 0.45 | 0.1-0.25 | balanced load
berry checked along the fence | 0.45 | 0.05-0.15 | ripeness pinch
wasp given a wide berth | 0.45 | 0.05-0.2 | slow sidestep
gloves clapped free of soil | 0.5 | 0.05-0.15 | dust puff
rabbit hole inspected with a frown | 0.4 | 0.05-0.15 | hands on knees
sun found in the one open patch | 0.45 | 0.1-0.25 | face tilted up""",
            "time_of_day": OUTDOOR_QUIET_TIME,
            "weather": OUTDOOR_QUIET_WEATHER,
        },
        "office_courtyard_quiet": {
            "background": """paved courtyard between glass wings | 1.0 | 0.5-0.8 | large format pavers
low concrete planters of grasses | 0.8 | 0.2-0.45 | feathered clumps
single bench facing the small tree | 0.7 | 0.15-0.35 | slatted seat
glass walls reflecting the sky | 0.75 | 0.3-0.6 | mirrored clouds
young trees in metal grates | 0.6 | 0.15-0.35 | caged trunks
gravel margin along the walls | 0.55 | 0.15-0.35 | raked stone band
covered walkway on one side | 0.5 | 0.2-0.4 | column rhythm
sky rectangle framed by rooflines | 0.6 | 0.25-0.5 | clean-cut opening
ivy panel softening a blank wall | 0.45 | 0.2-0.4 | trained green grid
water feature turned off | 0.4 | 0.1-0.3 | still black basin
moss between the shaded pavers | 0.45 | 0.1-0.25 | green joints""",
            "midground": """bench with room for one more | 0.6 | 0.15-0.3 | worn center slat
planter edge used as a seat | 0.55 | 0.1-0.25 | polished concrete lip
sparrows working the paver joints | 0.5 | 0.05-0.2 | hopping pair
bike leaned by the walkway | 0.4 | 0.1-0.25 | quiet corner spot
maintenance hose coiled by a tap | 0.35 | 0.05-0.15 | grey spiral
shade line crossing the court | 0.55 | 0.2-0.45 | moving boundary
leaves gathered at the drain | 0.45 | 0.05-0.15 | brown drift
someone stretching after sitting | 0.4 | 0.1-0.25 | arms overhead
tree shadow cast on the wall | 0.5 | 0.15-0.35 | trembling lace
umbrella base without its pole | 0.3 | 0.05-0.15 | orphaned disc
cat from nowhere crossing calmly | 0.3 | 0.05-0.15 | office mystery""",
            "architecture_detail": """paver joints in a strict grid | 0.55 | 0.15-0.4 | sand-swept lines
drain channel across the low edge | 0.5 | 0.05-0.15 | slotted steel
planter corners chipped by carts | 0.45 | 0.03-0.1 | bruised concrete
mullion shadows laddering the ground | 0.5 | 0.15-0.35 | moving stripes
tap recessed in a wall box | 0.4 | 0.02-0.08 | brass stub
grate rings around the trunks | 0.45 | 0.05-0.15 | radial slots
bench feet bolted to the pavers | 0.4 | 0.02-0.08 | anchor plates
wind eddy corner collecting leaves | 0.45 | 0.05-0.15 | swirl pocket
glass reflections doubling the tree | 0.5 | 0.15-0.35 | ghost twin
lichen starting on the north planter | 0.4 | 0.03-0.1 | pale spots
paver replaced in a wrong shade | 0.4 | 0.03-0.1 | odd one out""",
            "props": """forgotten cardigan on the bench | 0.4 | 0.02-0.08 | folded arm rest
coffee cup on the planter edge | 0.45 | 0.02-0.06 | balanced ring
lunch container lid chasing wind | 0.35 | 0.01-0.05 | skittering disc
earbud case on the bench slat | 0.35 | 0.01-0.04 | pebble-sized
hair clip by the bench leg | 0.3 | 0.01-0.03 | claw teeth
apple core in the gravel | 0.35 | 0.01-0.04 | tidy toss missed
stone lined up on the planter lip | 0.3 | 0.01-0.04 | idle sorting
badge reel dropped in the joint | 0.3 | 0.01-0.04 | coiled cord
pigeon feather on the pavers | 0.35 | 0.01-0.03 | grey curl
water bottle sweating on the bench | 0.4 | 0.02-0.06 | ring forming
sandwich wrapper pinned by a shoe | 0.35 | 0.01-0.05 | wind hostage""",
            "foreground_element": """face tilted to the sun with eyes shut | 0.6 | 0.1-0.25 | stolen minute
lunch eaten from a lap | 0.55 | 0.1-0.25 | container balanced
shoes slipped off under the bench | 0.45 | 0.05-0.15 | socked feet on stone
call taken pacing the pavers | 0.5 | 0.1-0.3 | slow circuits
stretch against the planter | 0.5 | 0.05-0.2 | heel drop
crumbs offered to the sparrows | 0.45 | 0.05-0.15 | patient toss
jacket folded into a cushion | 0.45 | 0.05-0.15 | bench comfort
deep breath before going back in | 0.5 | 0.1-0.25 | shoulders reset
tree leaves touched in passing | 0.4 | 0.03-0.1 | fingertip drag
gravel raked with a shoe edge | 0.35 | 0.03-0.1 | idle lines
cloud reflections watched in the glass | 0.45 | 0.1-0.3 | slow drift""",
            "time_of_day": OUTDOOR_QUIET_TIME,
            "weather": OUTDOOR_QUIET_WEATHER,
        },
        "loading_dock_after_hours": {
            "background": """raised concrete dock face | 1.0 | 0.4-0.7 | bumper-scarred edge
roller doors down for the night | 0.85 | 0.3-0.6 | ribbed steel curtains
empty trailer bay striped by shadow | 0.7 | 0.25-0.5 | oil-stained apron
rubber dock bumpers in pairs | 0.65 | 0.1-0.25 | chunked black blocks
bollards guarding the corners | 0.6 | 0.1-0.25 | scraped steel posts
ramp sloping up to the dock | 0.6 | 0.2-0.4 | broom-rough concrete
back wall of painted block | 0.65 | 0.3-0.6 | utility grey
chain-link gate rolled shut | 0.5 | 0.2-0.4 | padlocked track
dumpster corral swept clean | 0.5 | 0.15-0.3 | fenced pen
lone pole light over the apron | 0.55 | 0.1-0.3 | wide cone of glow
weeds along the fence base | 0.5 | 0.1-0.25 | tough green fringe""",
            "midground": """stack of pallets squared for pickup | 0.65 | 0.15-0.3 | strapped tower
dock plate leaned by the door | 0.5 | 0.1-0.2 | checker steel sheet
wheel chocks on a rope | 0.5 | 0.05-0.15 | yellow wedges
hand truck chained to a rail | 0.45 | 0.05-0.15 | overnight mooring
shrink wrap scraps balled up | 0.45 | 0.03-0.1 | cloudy tumbleweed
trailer dropped without its cab | 0.4 | 0.25-0.45 | landing legs down
puddle mapping the apron slope | 0.5 | 0.1-0.3 | slow mirror
cat hunting along the wall | 0.35 | 0.05-0.15 | shadow work
broom against the bumper | 0.45 | 0.03-0.1 | end-of-shift lean
milk crates stacked by the door | 0.45 | 0.05-0.15 | plastic columns
moths circling the pole light | 0.45 | 0.03-0.1 | dusty orbit""",
            "architecture_detail": """dock edge steel worn silver | 0.55 | 0.05-0.15 | polished lip
bumper bolts bleeding rust | 0.5 | 0.03-0.1 | orange trails
apron joints packed with grit | 0.5 | 0.05-0.2 | dark seams
tire scrub marks arcing in | 0.5 | 0.1-0.25 | black sweeps
downpipe strapped to the block wall | 0.45 | 0.05-0.15 | dented run
seal brushes around the doors | 0.45 | 0.03-0.1 | worn bristle frames
steps up the dock side with a rail | 0.5 | 0.05-0.2 | pipe handhold
paint line marking the trailer path | 0.45 | 0.05-0.2 | faded yellow
block wall scarred at fork height | 0.45 | 0.05-0.15 | gouged band
gate wheels rusted on the track | 0.4 | 0.02-0.08 | seized rollers
night insects ticking at the light | 0.4 | 0.03-0.1 | small collisions""",
            "props": """strapping ribbon curled on the apron | 0.45 | 0.02-0.06 | plastic spiral
lost glove flattened by tires | 0.4 | 0.01-0.05 | pressed leather
coffee cup on the dock edge | 0.4 | 0.01-0.05 | night-shift relic
chalk wedge worn round | 0.35 | 0.01-0.04 | tire marker
bungee hooked to the rail | 0.35 | 0.01-0.04 | idle stretch
zip ties clipped in a scatter | 0.35 | 0.01-0.04 | white commas
pallet splinter pile swept aside | 0.4 | 0.02-0.08 | wood shards
crushed stone tracked off the verge | 0.4 | 0.02-0.08 | grey scatter
moth resting under the light | 0.35 | 0.01-0.03 | folded triangle
rag tied to the gate wire | 0.3 | 0.01-0.04 | faded flag
bottle cap trodden into the seam | 0.3 | 0.01-0.03 | flattened star""",
            "foreground_element": """dock edge sat on with legs hanging | 0.55 | 0.1-0.25 | heels drumming
last pallet nudged square | 0.5 | 0.1-0.2 | boot push
padlock clicked and tugged | 0.5 | 0.03-0.1 | ritual check
ramp walked down backwards with a load | 0.4 | 0.1-0.25 | careful steps
jacket collar up against the night | 0.5 | 0.05-0.2 | cool air
phone light swept under the trailer | 0.4 | 0.05-0.15 | quick check
chock kicked under a wheel | 0.45 | 0.05-0.15 | wedge set
cat offered a bit of sandwich | 0.35 | 0.03-0.1 | shift companion
gate rolled shut with a shoulder | 0.5 | 0.1-0.25 | rattling travel
dust slapped off the thighs | 0.45 | 0.05-0.15 | end of shift
moth waved away from the face | 0.4 | 0.03-0.1 | soft flurry""",
            "time_of_day": OUTDOOR_QUIET_TIME,
            "weather": OUTDOOR_QUIET_WEATHER,
        },
        "school_back_field": {
            "background": """worn grass field behind the school | 1.0 | 0.5-0.9 | patchy green expanse
chain-link boundary fence | 0.8 | 0.2-0.5 | galvanized run
bare goal frames without nets | 0.7 | 0.15-0.35 | white pipe rectangles
treeline closing the far side | 0.65 | 0.25-0.5 | township woods
back wall of the gym block | 0.6 | 0.25-0.5 | tall brick face
gravel track loop around the grass | 0.5 | 0.2-0.4 | grey ribbon
equipment shed at the field corner | 0.5 | 0.1-0.3 | green steel hut
low bleacher of three benches | 0.5 | 0.15-0.3 | silver risers
backstop wedge for the diamond | 0.5 | 0.15-0.35 | tall chain-link sail
storm drain swale crossing the edge | 0.4 | 0.15-0.3 | mowed dip
flag of mown stripes across the grass | 0.5 | 0.25-0.5 | alternating sheen""",
            "midground": """soccer goal dragged off-line | 0.55 | 0.15-0.3 | anchor bags on the frame
line machine's fading stripes | 0.5 | 0.1-0.3 | ghost geometry
sprinkler head throwing its arc | 0.4 | 0.1-0.3 | ticking spray
crows working the mowed rows | 0.45 | 0.05-0.2 | strutting gleaners
long jump pit raked and empty | 0.4 | 0.1-0.2 | sand rectangle
gym class cones left out | 0.45 | 0.05-0.2 | orange scatter
kestrel hovering the fence line | 0.3 | 0.03-0.1 | pinned to the wind
groundsman's mower parked | 0.4 | 0.1-0.25 | grass-caked deck
gate chained at the service track | 0.45 | 0.1-0.2 | sagging leaf
dew still holding in the shade | 0.45 | 0.15-0.35 | silvered patch
lone walker cutting the diagonal | 0.4 | 0.1-0.25 | steady pace""",
            "architecture_detail": """goal mouths worn to bare dirt | 0.55 | 0.1-0.25 | grassless crescents
fence fabric bellied by climbers | 0.5 | 0.05-0.2 | stretched diamond mesh
gate posts concreted at an angle | 0.45 | 0.05-0.15 | tilted pair
sprinkler heads flush in the turf | 0.4 | 0.02-0.08 | brass circles
track edge blurring into grass | 0.45 | 0.1-0.25 | soft boundary
brick wall scarred by ball games | 0.5 | 0.1-0.3 | scuffed band
bleacher feet sunk in the turf | 0.4 | 0.03-0.1 | settled pads
drain grate in the swale | 0.4 | 0.02-0.08 | leaf-caught slots
molehills dotting the far corner | 0.4 | 0.05-0.15 | fresh brown domes
mower stripes bending at the trees | 0.45 | 0.15-0.35 | curved rows
gap worn under the service gate | 0.4 | 0.03-0.1 | scraped hollow""",
            "props": """lost tennis ball in the swale | 0.45 | 0.01-0.04 | grey-green veteran
pinnie snagged on the fence | 0.4 | 0.02-0.06 | mesh flag
water bottle standing alone mid-field | 0.4 | 0.01-0.05 | forgotten sentinel
cleat divots drying in a cluster | 0.45 | 0.03-0.1 | torn crescents
frisbee on the shed roof | 0.35 | 0.01-0.04 | out of reach
hair tie in the track gravel | 0.35 | 0.01-0.03 | dusty loop
practice cone cracked flat | 0.4 | 0.01-0.05 | run-over casualty
sunflower seed shells by the bleacher | 0.4 | 0.02-0.06 | spit constellation
feather stuck upright in the turf | 0.35 | 0.01-0.03 | crow calling card
chewed mouthguard by the bench | 0.3 | 0.01-0.03 | abandoned armor
jacket forgotten on the top bench | 0.4 | 0.02-0.08 | slumped shape""",
            "foreground_element": """lap walked on the gravel loop | 0.6 | 0.1-0.3 | steady crunch
ball kicked long against the fence | 0.5 | 0.1-0.25 | rattling mesh
dew kicked off the toes | 0.45 | 0.05-0.15 | dark shoe tips
bleacher bench climbed for the view | 0.45 | 0.1-0.25 | rocking riser
dog let off for a field sprint | 0.45 | 0.1-0.3 | ecstatic laps
divot pressed back with a heel | 0.4 | 0.03-0.1 | turf repair
goal frame chinned experimentally | 0.35 | 0.05-0.15 | swinging test
stone lofted at the backstop | 0.35 | 0.05-0.15 | ringing hit
grass blade whistled between thumbs | 0.35 | 0.03-0.1 | reedy shriek
track gravel raked with a shoe | 0.4 | 0.03-0.1 | idle grooves
wind read off the treeline | 0.4 | 0.1-0.25 | swaying tops""",
            "time_of_day": OUTDOOR_QUIET_TIME,
            "weather": OUTDOOR_QUIET_WEATHER,
        },
    },
}


def write_file(path: pathlib.Path, text: str, force: bool) -> str:
    if path.exists() and not force:
        return "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "write"


# Kurations-Banlists (Substring) aus tests/unit/test_location_lists_extended.py
INDOOR_BANNED = ["trail", "ridge", "mountain", "forest canopy", "ocean",
                 "beach", "appalachian", "summit", "wilderness", "switchback"]
OUTDOOR_BANNED = ["hardwood floor", "sofa", "office desk", "kitchen counter",
                  "duvet", "shower stall", "bath tub", "indoor pool"]

# Signage-Banlist: alles, was der Encoder als lesbaren Text rendern würde.
# Substring-Match — deshalb tauchen auch "assigned" (sign) oder
# "cartridge" (ridge) nirgends auf.
NO_TEXT_BANNED = [
    "sign", "poster", "letter", "label", "logo", "banner", "menu",
    "flyer", "sticker", "decal", "plaque", "billboard", "graffiti",
    "scoreboard", "chalkboard", "whiteboard", "price", "printed",
    "writing", "written", "brand", "advert", "notice", "bulletin",
]


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
        haystack = f"{name} {parts[3]}".lower()
        for b in banned:
            if b in haystack:
                problems.append(f"banlist '{b}' in {name!r}")
        for b in NO_TEXT_BANNED:
            if b in haystack:
                problems.append(f"signage word '{b}' in {name!r} / {parts[3]!r}")
    if len(names) != len(set(names)):
        dupes = {n for n in names if names.count(n) > 1}
        problems.append(f"duplicates: {dupes}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ok = True
    for category, sets in SETS.items():
        for set_name, elements in sets.items():
            missing = [e for e in ELEMENTS if e not in elements]
            if missing:
                print(f"ERROR {category}/{set_name}: missing {missing}")
                ok = False
                continue
            for element, body in elements.items():
                for p in validate(category, set_name, element, body):
                    print(f"ERROR {category}/{set_name}/{element}: {p}")
                    ok = False
    if not ok:
        return 1

    written = skipped = 0
    for category, sets in SETS.items():
        for set_name, elements in sets.items():
            target = ROOT / "location_lists" / pathlib.Path(category) / set_name
            for element, body in elements.items():
                r = write_file(target / f"{element}.txt",
                               HEADER[element] + body.strip() + "\n", args.force)
                written += r == "write"
                skipped += r == "skip"

    total = sum(len(v) for v in SETS.values())
    print(f"locations: {written} written, {skipped} skipped ({total} sets)")
    if skipped:
        print("Use --force to overwrite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
