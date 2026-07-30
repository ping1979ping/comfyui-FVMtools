"""Generator für private Urlaubsorte: Strand-Varianten und Winter-Spots.

outdoor/beach_variants     Strand in allen Spielarten — sandy/rocky, Meer,
                           See, Fluss, öffentlicher und privater Pool.
                           Busy/Private sind SET-Varianten (eigene Inhalte),
                           Tag/Nacht steckt in den time_of_day-Pools und ist
                           damit per Override erzwingbar
                           ("time_of_day: moonlight silvering the water").
outdoor/vacation_winter_us Private Winterurlaubs-Ecken: Hot-Tub-Deck,
                           zugefrorener See, eingeschneiter Cottage-Hof,
                           leerer Pistenrand nach Betriebsschluss.

Regeln wie bei den Private-Spaces-Sets: Standard-Kuration plus
Signage-Banlist (Substring!) — nichts mit Aufschrift, kein "sign" (auch in
"assigned"), kein "printed". Nacht-Objekte sind zeitneutral formuliert
("fire ring of blackened stones" funktioniert bei Tag und Nacht).

Ausführen:  python scripts/gen_vacation_water_locations.py [--force]
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

# ── Atmosphären-Pools: Tag UND Nacht in einem Pool, Wetter ohne Licht ──

SEA_TIME = """hazy bright morning over the water | 0.8 | - | -
harsh midday sun straight overhead | 0.75 | - | -
late afternoon sun angling low | 0.85 | - | -
golden hour gilding the water | 0.85 | - | -
sunset colors stacked on the horizon | 0.8 | - | -
blue hour with the first stars | 0.6 | - | -
moonlight silvering the water | 0.5 | - | -
deep night under a clear sky | 0.45 | - | -
first light before anyone arrives | 0.6 | - | -
flat overcast glare off the water | 0.65 | - | -"""

SEA_WEATHER = """steady onshore breeze | 0.85 | - | -
salt haze softening the distance | 0.7 | - | -
humid heat pressing down | 0.6 | - | -
gusts kicking up loose sand | 0.55 | - | -
spray drifting off the break | 0.55 | - | -
still sultry air | 0.5 | - | -
thunderheads stacked far out | 0.45 | - | -
cool damp air rolling off the water | 0.55 | - | -
light rain pocking the sand | 0.35 | - | -
flat calm without a ripple of wind | 0.5 | - | -"""

FRESHWATER_TIME = """morning mist lifting off the water | 0.75 | - | -
bright mid morning | 0.8 | - | -
harsh midday sun on the water | 0.7 | - | -
lazy late afternoon | 0.85 | - | -
golden hour across the far shore | 0.85 | - | -
sunset doubling itself in the water | 0.75 | - | -
blue hour with a loon calling | 0.5 | - | -
moonpath laid across the lake | 0.5 | - | -
deep night full of frog song | 0.45 | - | -
overcast stillness | 0.6 | - | -"""

FRESHWATER_WEATHER = """light breeze riffling the surface | 0.8 | - | -
humid summer air | 0.65 | - | -
dead calm mirror water | 0.6 | - | -
gusts dragging cat's paws across | 0.5 | - | -
damp air under the shore trees | 0.55 | - | -
heavy still air before a storm | 0.45 | - | -
drizzle dimpling the shallows | 0.35 | - | -
dragonfly-thick warm air | 0.5 | - | -
cool air pooling near the water | 0.5 | - | -
thin high cloud | 0.6 | - | -"""

POOL_TIME = """bright morning before the rush | 0.7 | - | -
harsh noon glare off the deck | 0.75 | - | -
long hot afternoon | 0.85 | - | -
late sun stretching the shadows | 0.8 | - | -
dusk with the water gone dark | 0.55 | - | -
night with the pool glowing from below | 0.5 | - | -
deep night with the deck empty | 0.4 | - | -
first light on undisturbed water | 0.5 | - | -
overcast flat brightness | 0.6 | - | -
last hour before closing | 0.55 | - | -"""

POOL_WEATHER = """still hot air over the concrete | 0.8 | - | -
faint chlorine tang in the air | 0.75 | - | -
breeze rocking the umbrellas | 0.55 | - | -
humid air thick as a towel | 0.55 | - | -
evening air cooling fast | 0.5 | - | -
heat shimmer over the deck | 0.55 | - | -
distant thunder ending the swim | 0.4 | - | -
light rain dimpling the pool | 0.35 | - | -
dry heat that empties the loungers | 0.5 | - | -
mild air perfect for floating | 0.6 | - | -"""

WINTER_TIME = """low winter sun barely clearing the trees | 0.8 | - | -
bright glare off fresh snow | 0.75 | - | -
flat white overcast midday | 0.8 | - | -
early dusk closing in fast | 0.75 | - | -
alpenglow pink on the high snow | 0.6 | - | -
blue hour over the snowfield | 0.6 | - | -
moonlight bright enough to walk by | 0.5 | - | -
deep night with snow-glow | 0.45 | - | -
first grey light through falling snow | 0.5 | - | -
last light catching the chimney smoke | 0.6 | - | -"""

WINTER_WEATHER = """fat flakes falling straight down | 0.6 | - | -
squeaky-cold still air | 0.6 | - | -
wind-driven flurries stinging | 0.45 | - | -
thaw drip ticking off the eaves | 0.45 | - | -
breath hanging in the air | 0.7 | - | -
clear hard cold | 0.65 | - | -
snow smell on a rising wind | 0.5 | - | -
ice fog hugging the ground | 0.4 | - | -
powder blowing off the roof edge | 0.45 | - | -
mild spell softening the drifts | 0.4 | - | -"""


SETS: dict[str, dict[str, dict[str, str]]] = {
    "outdoor/beach_variants": {
        "sandy_beach_busy": {
            "background": """wide sandy beach packed with umbrellas | 1.0 | 0.5-0.9 | color-dotted crowd field
surf line rolling in sets | 0.85 | 0.3-0.6 | white foam bands
lifeguard chair above the crowd | 0.7 | 0.1-0.3 | tall white frame
dune line fencing the back | 0.6 | 0.2-0.4 | grass-topped ridge line
boardwalk silhouettes far off | 0.5 | 0.15-0.35 | railing and rooftops
swimmers bobbing between the flags | 0.65 | 0.2-0.4 | heads in the swell
horizon stacked with haze | 0.6 | 0.25-0.5 | sea-sky blur
kite wobbling over the beach | 0.45 | 0.05-0.2 | diving diamond
towel patchwork to the waterline | 0.7 | 0.3-0.6 | quilted crowd ground
jetty rocks at the beach end | 0.45 | 0.15-0.35 | dark boulder arm
gull squadrons riding the breeze | 0.55 | 0.1-0.3 | white scatter""",
            "midground": """family camp of chairs and coolers | 0.8 | 0.2-0.4 | claimed territory
sandcastle city under construction | 0.6 | 0.1-0.3 | dribbled towers
umbrella leaning with the wind | 0.65 | 0.1-0.3 | tilted canvas
paddleball game staked out | 0.5 | 0.1-0.3 | back-and-forth pock
cooler wagon dragged past | 0.5 | 0.1-0.25 | fat-wheeled hauler
toddler at the foam edge | 0.55 | 0.05-0.2 | knee-deep bravery
boogie boards planted like fins | 0.5 | 0.1-0.25 | leashed row
seagull raid on an open bag | 0.5 | 0.05-0.2 | wing chaos
sun shelter half collapsed | 0.45 | 0.1-0.25 | flapping panel
walker threading the towel maze | 0.55 | 0.1-0.3 | careful steps
surf fisherman off to one side | 0.35 | 0.1-0.25 | planted rod""",
            "architecture_detail": """tide line of shell and weed | 0.6 | 0.1-0.3 | wrack ribbon
soft sand churned by traffic | 0.6 | 0.2-0.45 | ankle-deep churn
hard wet sand smooth as a floor | 0.55 | 0.2-0.45 | mirror sheen
dune fence slats half buried | 0.5 | 0.05-0.2 | leaning pickets
trash barrel ringed by near-misses | 0.45 | 0.05-0.15 | overflow scatter
access path worn through the dunes | 0.5 | 0.1-0.25 | sandy funnel
flag poles marking the swim zone | 0.5 | 0.03-0.1 | colored pennants
sand fleas popping at the wrack | 0.35 | 0.02-0.08 | tiny leaps
runnels braiding down the slope | 0.45 | 0.1-0.25 | silver threads
foot-burn hop zone of dry sand | 0.45 | 0.15-0.35 | shimmering heat
shell crush line under the step | 0.4 | 0.05-0.2 | crunching mosaic""",
            "props": """flip flops guarding a towel corner | 0.55 | 0.02-0.06 | kicked-off pair
sunscreen bottle capless in the sand | 0.55 | 0.02-0.06 | gritty nozzle
bucket and spade abandoned mid-build | 0.5 | 0.02-0.08 | tide-doomed works
frisbee stuck upright | 0.45 | 0.01-0.05 | quivering disc
mesh bag of shells started | 0.45 | 0.02-0.06 | curated haul
paperback tented on a towel | 0.5 | 0.02-0.06 | wind-flipped pages
goggles half buried | 0.4 | 0.01-0.04 | one lens up
juice pouch squeezed flat | 0.4 | 0.01-0.04 | straw bent
sandy phone in a zip bag | 0.4 | 0.01-0.04 | precaution taken
inflatable ring drifting loose | 0.45 | 0.03-0.1 | escaping toy
lost shovel at the waterline | 0.4 | 0.01-0.04 | red plastic blade""",
            "foreground_element": """towel snapped flat against the wind | 0.6 | 0.1-0.25 | double-handed flourish
sunscreen worked into a shoulder | 0.6 | 0.05-0.2 | white streaks
hot-sand sprint to the water | 0.55 | 0.1-0.3 | high-kneed dash
wave braced with a shriek | 0.55 | 0.1-0.3 | arms up
sand brushed off a sandwich | 0.5 | 0.05-0.15 | losing battle
umbrella screwed deeper | 0.5 | 0.05-0.2 | two-hand twist
kid buried to the waist | 0.5 | 0.1-0.25 | patted mound
hair wrung over one shoulder | 0.55 | 0.05-0.15 | salt twist
cooler lid raided for a drink | 0.55 | 0.05-0.15 | ice dig
ball chased into the shallows | 0.5 | 0.1-0.25 | splash pursuit
squint out to the swimmers | 0.5 | 0.05-0.2 | hand visor""",
            "time_of_day": SEA_TIME,
            "weather": SEA_WEATHER,
        },
        "sandy_beach_private": {
            "background": """empty sand stretching both ways | 1.0 | 0.5-0.9 | unmarked expanse
dunes rolling back in grass tufts | 0.8 | 0.25-0.5 | seagrass mounds
surf running unwatched | 0.8 | 0.3-0.6 | steady white lines
single set of footprints along the wet sand | 0.6 | 0.1-0.3 | dotted line to nowhere
driftwood log half buried | 0.55 | 0.1-0.3 | silvered trunk
headland closing the far view | 0.5 | 0.2-0.4 | hazy bluff
sky doing most of the talking | 0.7 | 0.4-0.7 | big weather ceiling
sandpipers working the foam edge | 0.55 | 0.05-0.2 | clockwork sprints
old fire ring of blackened stones | 0.45 | 0.05-0.2 | charcoaled circle
tide pools left in the low ripples | 0.45 | 0.1-0.3 | warm shallow mirrors
beach rose thicket at the dune toe | 0.4 | 0.1-0.3 | pink-dotted green""",
            "midground": """one blanket anchored with shoes | 0.6 | 0.1-0.25 | solitary claim
gull dozing on one leg | 0.45 | 0.03-0.1 | folded lookout
crab scuttling between holes | 0.4 | 0.02-0.08 | sideways dash
kelp rope beached in a coil | 0.45 | 0.05-0.2 | rubbery tangle
distant walker with a dog | 0.45 | 0.05-0.2 | the only company
wind rippling the dry sand | 0.5 | 0.15-0.35 | traveling shivers
tern hovering then folding into a dive | 0.4 | 0.03-0.1 | white dart
beached horseshoe crab shell | 0.35 | 0.02-0.08 | brown helmet
sand devil spinning briefly | 0.3 | 0.05-0.15 | grit whirl
low dune shadow creeping out | 0.45 | 0.15-0.35 | evening reach
single kayak pulled above the line | 0.35 | 0.1-0.2 | resting hull""",
            "architecture_detail": """unbroken tide line down the beach | 0.6 | 0.1-0.3 | continuous wrack ribbon
wind-ripple corduroy in the dry sand | 0.55 | 0.15-0.35 | fine ridges
dune scarp cut by the last storm | 0.45 | 0.1-0.25 | bitten edge
ghost crab holes with fan piles | 0.45 | 0.03-0.1 | excavated sprays
shell drifts sorted by the swash | 0.5 | 0.05-0.2 | graded windrows
lone fence post from a lost boundary | 0.35 | 0.02-0.08 | leaning grey stub
freshwater seep darkening the slope | 0.35 | 0.05-0.15 | glistening fan
foam globs shivering on the wet sand | 0.4 | 0.03-0.1 | quaking meringue
sand so dry it squeaks | 0.4 | 0.1-0.3 | singing steps
swash line redrawn by each wave | 0.5 | 0.1-0.3 | fresh scallops
storm-tossed buoy stranded high | 0.3 | 0.02-0.08 | barnacled float""",
            "props": """one pair of shoes above the tide line | 0.5 | 0.02-0.06 | socks tucked in
water bottle standing in the sand | 0.45 | 0.01-0.05 | half buried base
found stick for drawing in sand | 0.4 | 0.01-0.04 | wave-smoothed wand
sea glass pieces pocketed slowly | 0.4 | 0.01-0.04 | frosted gems
towel rolled as a pillow | 0.45 | 0.02-0.08 | head dent
hat weighted with a stone | 0.4 | 0.01-0.05 | wind insurance
binoculars resting on the blanket | 0.35 | 0.01-0.05 | strap coiled
apple core tossed to the gulls | 0.35 | 0.01-0.03 | instant regret
whole sand dollar found unbroken | 0.35 | 0.01-0.03 | careful trophy
thermos wedged upright | 0.4 | 0.01-0.05 | steam curl
folded chair never unfolded | 0.35 | 0.03-0.1 | optimism on standby""",
            "foreground_element": """bare feet at the foam edge | 0.65 | 0.05-0.2 | cold nibble
name traced big in the wet sand | 0.45 | 0.05-0.2 | stick calligraphy
stone skipped across a lull | 0.45 | 0.05-0.15 | counting hops
shell held up against the light | 0.5 | 0.03-0.1 | translucent check
hair surrendered to the wind | 0.55 | 0.05-0.15 | streaming strands
sweater sleeves pulled over hands | 0.5 | 0.05-0.15 | evening chill
long exhale facing the horizon | 0.55 | 0.1-0.3 | shoulders dropping
dog stick thrown into the surf | 0.4 | 0.1-0.25 | ecstatic retrieve
jeans rolled two turns too few | 0.45 | 0.05-0.15 | soaked hems
sand poured from palm to palm | 0.4 | 0.03-0.1 | idle hourglass
horizon watched without a phone | 0.5 | 0.1-0.3 | pocketed world""",
            "time_of_day": SEA_TIME,
            "weather": SEA_WEATHER,
        },
        "rocky_shore_cove": {
            "background": """boulder-armored cove | 1.0 | 0.5-0.9 | grey rounded giants
cliff walls cupping the water | 0.8 | 0.3-0.6 | layered rock faces
swell surging between rocks | 0.75 | 0.25-0.5 | white collars forming
pocket of coarse pebble beach | 0.6 | 0.2-0.4 | rattling shingle
sea stack standing offshore | 0.45 | 0.1-0.3 | wave-cut pillar
spray bursting on the outer rocks | 0.55 | 0.15-0.35 | white explosions
cormorants drying spread-winged | 0.45 | 0.05-0.15 | heraldic poses
deep green water in the shelter | 0.6 | 0.25-0.5 | glassy jade
cave mouth breathing with the swell | 0.35 | 0.1-0.25 | dark exhale
pines leaning over the cliff lip | 0.45 | 0.15-0.35 | wind-combed crowns
horizon cut by the cove arms | 0.5 | 0.2-0.4 | framed slice of sea""",
            "midground": """flat-topped boulder claimed for sitting | 0.6 | 0.1-0.3 | warm granite throne
rock pools stepped down the shore | 0.55 | 0.15-0.35 | mirror terraces
seal head periscoping the entrance | 0.35 | 0.03-0.1 | sleek curious dome
kelp forest swaying under the surface | 0.45 | 0.15-0.35 | amber shadows
gull dropping a shell on the rocks | 0.4 | 0.03-0.1 | smash technique
swimmer's slow circuit of the cove | 0.35 | 0.1-0.25 | quiet crawl
driftwood jammed in a crevice | 0.45 | 0.05-0.15 | wedged silver limb
crab pot float washed into a corner | 0.35 | 0.02-0.08 | barnacled sphere
narrow scramble path down the cliff | 0.45 | 0.1-0.25 | polished handholds
mussel beds blackening the low rocks | 0.5 | 0.1-0.3 | blue-black crust
anemones open in the still pools | 0.45 | 0.05-0.15 | green blooms""",
            "architecture_detail": """barnacle line marking high water | 0.55 | 0.1-0.3 | white crust band
wave-polished stone underfoot | 0.55 | 0.15-0.35 | slick rounded backs
quartz vein lightning through granite | 0.45 | 0.03-0.12 | white zigzag
rock warm on top cold beneath | 0.4 | 0.1-0.25 | two-temperature seat
surge channel funneling each set | 0.45 | 0.1-0.25 | rhythmic rush
tafoni pockets in the soft layers | 0.35 | 0.05-0.15 | honeycomb weathering
tide leaving the pools one by one | 0.45 | 0.15-0.35 | slow retreat
salt crust in the splash-zone dips | 0.4 | 0.03-0.1 | white rime
periwinkles dotting every crack | 0.45 | 0.05-0.15 | small dark studs
echo of the swell under a ledge | 0.4 | 0.1-0.25 | hollow booming
lichen zoning the upper rocks | 0.45 | 0.1-0.25 | orange to grey bands""",
            "props": """shoes wedged safe in a dry crack | 0.5 | 0.02-0.06 | above the spray
towel spread on the flat boulder | 0.5 | 0.03-0.1 | corner-weighted
found buoy fragment faded pink | 0.35 | 0.01-0.05 | wave-worn foam
snorkel mask drying on a rock | 0.4 | 0.01-0.05 | drips in the sun
crab claw abandoned by a gull | 0.35 | 0.01-0.03 | orange pincer
sketchbook wedged under a stone | 0.3 | 0.01-0.05 | wind-safe pages
rope end frayed into a mop | 0.35 | 0.01-0.05 | ship's castoff
sea urchin test balanced whole | 0.3 | 0.01-0.03 | fragile lantern
water shoes drip-drying toe-up | 0.4 | 0.01-0.05 | rubber pair
lunch tin defended from gulls | 0.4 | 0.02-0.06 | latched box
walking stick parked in a crevice | 0.35 | 0.01-0.05 | trusted third leg""",
            "foreground_element": """scramble steadied on three points | 0.6 | 0.1-0.25 | crab-wise descent
pool crouched over like a window | 0.55 | 0.1-0.25 | hands on knees
anemone tickled to a close | 0.45 | 0.03-0.1 | gentle fingertip
cold-water gasp mid-wade | 0.5 | 0.05-0.2 | involuntary octave
mussel shell pried for a look | 0.4 | 0.03-0.1 | blue interior
spray dodged with a laugh | 0.5 | 0.1-0.25 | half-turn flinch
boulder-top lunch with a view | 0.5 | 0.1-0.3 | sandwich and horizon
seal watched dead-still | 0.4 | 0.05-0.2 | mutual curiosity
echo tested into the cave mouth | 0.35 | 0.05-0.15 | quick shout
warm rock lain back on | 0.5 | 0.1-0.3 | eyes closed
pebble pyramid balanced patiently | 0.4 | 0.03-0.1 | fifth stone wobbling""",
            "time_of_day": SEA_TIME,
            "weather": SEA_WEATHER,
        },
        "rocky_point_tidepools": {
            "background": """rock shelf running out to the point | 1.0 | 0.5-0.9 | table of wet stone
pools scattered like dropped mirrors | 0.85 | 0.3-0.6 | sky pieces everywhere
families bent over the water windows | 0.6 | 0.2-0.4 | pointing clusters
surf working the shelf edge | 0.7 | 0.2-0.45 | white surge line
headland lighthouse-less and bare | 0.45 | 0.15-0.35 | wind-scoured hump
gulls loitering for dropped finds | 0.5 | 0.1-0.25 | opportunist line
weed-slick channels between pools | 0.55 | 0.15-0.35 | green glass runs
low tide stretching the playground | 0.6 | 0.3-0.6 | exposed acreage
kids hopping pool to pool | 0.55 | 0.15-0.35 | shrieking parkour
parking pull-off above the rocks | 0.4 | 0.1-0.25 | guardrail glint
mainland cliffs curving away | 0.5 | 0.2-0.4 | receding walls""",
            "midground": """bucket brigade of young collectors | 0.55 | 0.1-0.3 | intent expedition
starfish prised nowhere near loose | 0.45 | 0.03-0.1 | five-point suction
hermit crab race across a pool floor | 0.45 | 0.03-0.1 | borrowed-shell hustle
dad ankle-deep pointing things out | 0.5 | 0.1-0.25 | guided tour
slip caught by a quick arm | 0.45 | 0.05-0.15 | seaweed lesson
sculpin darting under a ledge | 0.4 | 0.02-0.08 | camouflage failure
photographer flat on the rock | 0.4 | 0.05-0.15 | macro devotion
tide chart consulted on a phone | 0.4 | 0.02-0.08 | return-time math
picnic wedged in a dry hollow | 0.4 | 0.05-0.2 | wind-sheltered spread
dog forbidden from every pool | 0.4 | 0.05-0.15 | straining leash
teenager narrating a find | 0.4 | 0.05-0.15 | crowd of two""",
            "architecture_detail": """pools zoned by depth and life | 0.55 | 0.15-0.35 | layered neighborhoods
coralline pink crusting the rims | 0.45 | 0.05-0.15 | rosy stone lace
surge slot too wide to hop | 0.45 | 0.05-0.15 | respected gap
mussel mat quilting the mid-shelf | 0.5 | 0.15-0.3 | blue-black quilt
sea lettuce flags in the drains | 0.45 | 0.05-0.15 | bright green streamers
barnacles closing as the water falls | 0.4 | 0.05-0.15 | tiny trapdoors
non-slip friction of dry rock | 0.45 | 0.1-0.25 | grippy patches
treacherous sheen of the wet slopes | 0.5 | 0.1-0.25 | tested carefully
sand pockets floored in old shell | 0.4 | 0.05-0.15 | crushed white beds
tide creeping back up the channels | 0.5 | 0.15-0.35 | quiet reclaim
limpet scars ringing empty seats | 0.35 | 0.03-0.1 | home ovals""",
            "props": """clear bucket of temporary residents | 0.5 | 0.02-0.08 | catch and release
net on a bamboo handle | 0.45 | 0.01-0.05 | optimistic gear
sandals in a dry-spot pile | 0.5 | 0.02-0.06 | communal parking
magnifying jar fogged with use | 0.35 | 0.01-0.04 | breath and salt
field guide splayed on a rock | 0.35 | 0.01-0.05 | wind-fluttered pages
juice boxes lined on a ledge | 0.4 | 0.01-0.05 | expedition rations
crab molt displayed on a palm-sized stone | 0.35 | 0.01-0.03 | perfect ghost
wet socks abandoned mid-shelf | 0.4 | 0.01-0.04 | strategy change
phone in a chest pocket at risk | 0.4 | 0.01-0.03 | leaning danger
band-aid tin for barnacle knees | 0.35 | 0.01-0.04 | expedition medic
found lure with a rusted hook | 0.3 | 0.01-0.03 | handled carefully""",
            "foreground_element": """pool leaned over till hair touches water | 0.6 | 0.1-0.25 | close encounter
starfish arm touched one finger | 0.55 | 0.03-0.1 | gentle contact
hermit crab walked across a palm | 0.5 | 0.03-0.1 | ticklish march
anemone poke bet lost | 0.45 | 0.03-0.1 | quick close
slippery ledge crossed hand in hand | 0.5 | 0.1-0.25 | chain crossing
find shouted across the shelf | 0.5 | 0.05-0.2 | come-see wave
sneaker soaked in a misjudged hop | 0.5 | 0.05-0.15 | squelching pride
bucket tipped back at day's end | 0.45 | 0.05-0.15 | residents returned
knees dimpled by the rock | 0.45 | 0.05-0.15 | red honeycomb
camera lowered nearly to the surface | 0.4 | 0.05-0.15 | held breath
tide race back to shore giggled through | 0.45 | 0.1-0.25 | splashing retreat""",
            "time_of_day": SEA_TIME,
            "weather": SEA_WEATHER,
        },
        "lake_beach_public": {
            "background": """roped swim area dotted with heads | 1.0 | 0.4-0.8 | float-line rectangle
coarse sand beach on the lake | 0.85 | 0.4-0.7 | trucked-in strip
far shore of solid trees | 0.7 | 0.25-0.5 | green wall doubled in water
swim raft crowded at the corner | 0.6 | 0.1-0.3 | tilting takeoffs
lifeguard chair with an umbrella | 0.55 | 0.1-0.25 | shaded perch
grass bank of blankets behind the sand | 0.6 | 0.25-0.5 | patchwork slope
boat launch at the beach end | 0.45 | 0.15-0.3 | concrete tongue
buoy line ticking in the chop | 0.55 | 0.1-0.25 | white beads
picnic pavilion under the oaks | 0.5 | 0.2-0.4 | green roof shade
canoes racked by the launch | 0.45 | 0.1-0.25 | nested hulls
paddleboarders wobbling out wide | 0.45 | 0.1-0.25 | standing silhouettes""",
            "midground": """kids cannonballing off the raft | 0.6 | 0.1-0.3 | tucked splashes
toddler zone of ankle water | 0.55 | 0.1-0.3 | bucket brigade
floats drifting past the rope | 0.5 | 0.1-0.25 | escaping donuts
sandy towels shaken like flags | 0.5 | 0.05-0.2 | grit clouds
snapping turtle rumor clearing a corner | 0.35 | 0.05-0.15 | wide berth
grill smoke drifting from the pavilion | 0.5 | 0.1-0.3 | charcoal ribbons
minnows schooling in the shallows | 0.45 | 0.05-0.15 | silver flickers
dad throwing kids in rotation | 0.5 | 0.1-0.25 | launch service
goose delegation marching the sand | 0.45 | 0.1-0.2 | honking landlords
swim lesson clinging to the rope | 0.4 | 0.1-0.25 | kickboard row
dropped ice cream tragedy in progress | 0.4 | 0.03-0.1 | seagull-less mourning""",
            "architecture_detail": """sand ending in a mowed-grass seam | 0.55 | 0.1-0.3 | maintained edge
gravel path around to the launch | 0.5 | 0.1-0.25 | crunching loop
lake water tea-colored in the shallows | 0.55 | 0.15-0.35 | tannin amber
weed line where the sand runs out | 0.5 | 0.1-0.25 | soft green boundary
rope anchors rusting at the posts | 0.4 | 0.02-0.08 | orange stains
raft ladder polished by feet | 0.45 | 0.02-0.08 | silver rungs
foot-rinse spigot by the grass | 0.45 | 0.03-0.1 | dripping ritual
pavilion posts carved by decades | 0.4 | 0.03-0.1 | initialed history
lake bottom mapped by toes | 0.45 | 0.1-0.25 | sand then silt then weeds
buoy rope beard of algae | 0.4 | 0.03-0.1 | green fringe
shade line of the big oaks | 0.5 | 0.2-0.4 | cool boundary""",
            "props": """diving toys sunk on purpose | 0.5 | 0.02-0.08 | bright targets
inflatable flamingo overstaying | 0.45 | 0.03-0.1 | pink landmark
zinc-striped noses in every direction | 0.4 | 0.02-0.06 | war paint
mesh bag of lake toys | 0.5 | 0.02-0.08 | dripping arsenal
towel clothesline between two oaks | 0.4 | 0.03-0.1 | sagging gallery
floaties deflating on the grass | 0.45 | 0.02-0.08 | wrinkling arms
sandy sandwich eaten anyway | 0.45 | 0.01-0.05 | crunch accepted
goggles on a forehead all day | 0.45 | 0.01-0.04 | red-ringed badge
minnow trap checked hourly | 0.35 | 0.01-0.05 | wire promise
lost swim ring rolling down the beach | 0.4 | 0.02-0.08 | wind-powered escape
grape bunch passed down a towel row | 0.4 | 0.01-0.05 | family conveyor""",
            "foreground_element": """running start off the raft | 0.55 | 0.1-0.25 | board-rattling launch
rope ducked under to the deep side | 0.5 | 0.05-0.2 | forbidden crossing
weeds high-stepped through | 0.5 | 0.05-0.2 | squeamish tiptoe
towel cocoon on the grass bank | 0.5 | 0.1-0.25 | shivering burrito
foot rinse hop at the spigot | 0.5 | 0.03-0.1 | one-legged dance
float claimed with a belly flop | 0.5 | 0.05-0.2 | territorial splash
minnow cupped for a second | 0.45 | 0.03-0.1 | quick release
lake gulp mid-laugh regretted | 0.4 | 0.03-0.1 | sputtering recovery
goose stared down over a sandwich | 0.4 | 0.05-0.15 | tense standoff
raft king-of-the-hill round | 0.45 | 0.1-0.25 | shoving contest
sun-warmed shallows walked slowly | 0.5 | 0.1-0.25 | bath-water stretch""",
            "time_of_day": FRESHWATER_TIME,
            "weather": FRESHWATER_WEATHER,
        },
        "lake_cove_private": {
            "background": """still cove ringed by hemlocks | 1.0 | 0.5-0.9 | dark green bowl
weathered dock reaching out | 0.8 | 0.2-0.45 | grey planks on posts
water black-green and glassy | 0.75 | 0.3-0.6 | depth without color
far point hiding the main lake | 0.55 | 0.2-0.4 | wooded arm
rowboat tied nose-in | 0.55 | 0.1-0.3 | patient tether
boulder shore instead of beach | 0.55 | 0.2-0.4 | mossy shoulders
cabin roof through the trees | 0.45 | 0.1-0.3 | cedar shakes
heron statue-still on a deadhead | 0.4 | 0.03-0.1 | grey sentinel
lily pads cornering the shallows | 0.5 | 0.15-0.35 | green tile floor
mist pocket lingering late | 0.4 | 0.15-0.35 | slow to burn off
loon surfacing without a ripple | 0.35 | 0.03-0.1 | black-and-white check""",
            "midground": """ladder off the dock end | 0.55 | 0.05-0.15 | silver rungs into dark
towels over the dock rail | 0.5 | 0.05-0.15 | drying flags
dragonflies stitching the air | 0.5 | 0.05-0.2 | blue needles
kayak hauled half out | 0.45 | 0.1-0.25 | dripping stern
fish rising in soft rings | 0.5 | 0.05-0.2 | widening circles
rope swing over the deep corner | 0.4 | 0.05-0.2 | knotted invitation
turtle log with a full house | 0.4 | 0.05-0.15 | shell lineup
water striders dimpling the film | 0.4 | 0.03-0.1 | skating pinpricks
canoe crossing the cove mouth | 0.35 | 0.1-0.2 | silent paddler
beaver wake cutting the calm | 0.3 | 0.05-0.15 | V of intent
firefly meadow at the shore back | 0.35 | 0.1-0.3 | evening sparks""",
            "architecture_detail": """dock boards gapped and springy | 0.55 | 0.1-0.3 | flexing walk
post rings worn by decades of rope | 0.4 | 0.02-0.08 | polished grooves
water so clear the rocks layer down | 0.5 | 0.2-0.4 | drowned staircase
pine needles rafting the corners | 0.45 | 0.05-0.15 | drifting mats
sap beads on the dock rail | 0.35 | 0.02-0.06 | amber pearls
drop-off visible as a color line | 0.5 | 0.1-0.3 | green to black
frog plop punctuation | 0.45 | 0.02-0.08 | shoreline exits
moss carpeting the north rocks | 0.45 | 0.1-0.25 | deep pile
old anchor chain into the weeds | 0.3 | 0.02-0.08 | rusted disappearance
cattail stand guarding the inlet | 0.45 | 0.1-0.3 | brown pokers
morning spider lines silvering the rail | 0.4 | 0.03-0.1 | dew-strung harps""",
            "props": """fishing rod propped on the rail | 0.45 | 0.02-0.08 | line slack
tackle box open on the boards | 0.4 | 0.02-0.08 | lure trays fanned
enamel mug on the dock post | 0.45 | 0.01-0.05 | steam in the cool
life jackets in a faded stack | 0.45 | 0.03-0.1 | sun-bleached pile
paddle laid along the dock edge | 0.45 | 0.02-0.08 | drip line
worn deck of cards on a towel | 0.35 | 0.01-0.05 | evening plans
citronella bucket half burned | 0.4 | 0.01-0.05 | waxy crater
lantern waiting for dusk | 0.4 | 0.01-0.05 | glass and wick
frayed rope coiled on a cleat | 0.4 | 0.01-0.05 | working loops
minnow bucket lidded in the shade | 0.35 | 0.01-0.05 | live cargo
paperback swollen from dock life | 0.4 | 0.01-0.04 | humidity casualty""",
            "foreground_element": """toes tested off the ladder | 0.6 | 0.05-0.2 | temperature verdict
quiet slide in without a splash | 0.5 | 0.1-0.25 | respectful entry
back float staring at the sky | 0.55 | 0.1-0.3 | ears underwater
dock boards warm under the back | 0.55 | 0.1-0.3 | sun-stored heat
line cast toward the lily edge | 0.45 | 0.05-0.2 | soft plop
loon call answered badly | 0.35 | 0.03-0.1 | laughing failure
rope swing arc ending in a shriek | 0.4 | 0.1-0.25 | release point
mug warmed both hands | 0.45 | 0.03-0.1 | morning ritual
canoe steadied for a shaky boarding | 0.4 | 0.1-0.25 | white-knuckle step
dragonfly landed on a knee | 0.4 | 0.02-0.08 | held breath
evening watched from the dock end | 0.55 | 0.1-0.3 | feet in the water""",
            "time_of_day": FRESHWATER_TIME,
            "weather": FRESHWATER_WEATHER,
        },
        "river_bend_beach": {
            "background": """sand-and-gravel bar on the inside bend | 1.0 | 0.5-0.9 | river-sorted crescent
green water sliding past | 0.85 | 0.3-0.6 | glassy current lanes
cutbank forest wall opposite | 0.7 | 0.25-0.5 | rooty overhang
riffle entering at the bar head | 0.6 | 0.15-0.35 | broken sparkle
swimming hole under the cutbank | 0.6 | 0.2-0.4 | slow deep green
sycamores leaning over the run | 0.55 | 0.2-0.4 | mottled trunks
canoe beached bow-up | 0.45 | 0.1-0.25 | aluminum grin
bluff rising downstream | 0.45 | 0.15-0.35 | layered stone
gravel shallows shimmering | 0.6 | 0.2-0.4 | knee-deep glitter
island splitting the flow above | 0.4 | 0.15-0.3 | willow tuft
swallows strafing the surface | 0.45 | 0.05-0.15 | insect patrol""",
            "midground": """camp chairs half in the water | 0.55 | 0.1-0.25 | cooling-seat trick
cooler anchored in the shallows | 0.5 | 0.05-0.2 | current-chilled
tube flotilla roped together | 0.45 | 0.1-0.3 | lazy convoy
kids damming a side trickle | 0.5 | 0.1-0.25 | civil engineering
rope swing over the deep hole | 0.45 | 0.05-0.2 | tested knot
fly caster working the riffle tail | 0.4 | 0.1-0.2 | rhythmic loops
dog swimming a wide circle | 0.45 | 0.05-0.2 | nose-up paddle
smallmouth flashing off a rock | 0.35 | 0.02-0.08 | bronze turn
driftwood fort mid-construction | 0.4 | 0.1-0.25 | stacked silver
crawdad hunt in the gravel | 0.45 | 0.05-0.2 | flipped-stone patrol
heron lifting off downstream | 0.4 | 0.05-0.15 | slow-motion exit""",
            "architecture_detail": """gravel graded fine to cobble | 0.55 | 0.15-0.35 | sorted bands
wet sand rim following the waterline | 0.5 | 0.1-0.3 | dark margin
current seams braiding off the point | 0.5 | 0.1-0.3 | glassy sutures
flood-line trash of leaves and sticks | 0.45 | 0.05-0.2 | high-water necklace
undercut roots combed by the flow | 0.45 | 0.1-0.25 | woody curtain
sun-warmed shallows over dark stone | 0.5 | 0.15-0.35 | bath pockets
clam trails wandering the silt | 0.35 | 0.03-0.1 | dotted wanderings
midstream boulder with a wake | 0.45 | 0.05-0.15 | standing ripple
sand cliff kids keep collapsing | 0.4 | 0.05-0.15 | engineered erosion
water-smoothed glass cobbles | 0.4 | 0.05-0.15 | jelly-bean colors
mayfly husks on every stone | 0.35 | 0.03-0.1 | paper ghosts""",
            "props": """river shoes drying on a log | 0.5 | 0.02-0.06 | drip row
skipping-stone stockpile curated | 0.45 | 0.01-0.05 | flat elite
snorkel mask for the deep hole | 0.4 | 0.01-0.05 | fogged lens
towels draped on willow limbs | 0.45 | 0.03-0.1 | branch laundry
crawdad in a clear cup | 0.4 | 0.01-0.04 | temporary celebrity
sunscreen standing in a shoe | 0.4 | 0.01-0.04 | sand-free trick
watermelon cooling in the shallows | 0.4 | 0.02-0.08 | green buoy
found antler bleached pale | 0.3 | 0.01-0.04 | river treasure
rope coil for the swing repair | 0.35 | 0.01-0.05 | fresh knots
sandwich bag of trail mix | 0.4 | 0.01-0.04 | shared rations
paddle jammed upright as a marker | 0.35 | 0.01-0.05 | camp flag""",
            "foreground_element": """flat stone sidearmed down the run | 0.55 | 0.05-0.2 | four-skip pride
current walked against thigh-deep | 0.55 | 0.1-0.25 | leaning wade
crawdad held up pinching air | 0.5 | 0.03-0.1 | careful grip
tube spun slowly mid-drift | 0.45 | 0.1-0.25 | lazy rotation
cold hole gasped into | 0.5 | 0.05-0.2 | spring-fed shock
gravel walked barefoot gingerly | 0.5 | 0.05-0.2 | wincing progress
watermelon slice dripped over the water | 0.45 | 0.03-0.1 | seed-spitting range
rope swing queue self-organized | 0.4 | 0.1-0.25 | turn debates
minnow school stood in until nibbled | 0.45 | 0.05-0.15 | ticklish audit
wet dog dodged too late | 0.45 | 0.05-0.2 | spray radius
riffle listened to with eyes shut | 0.45 | 0.1-0.25 | white noise bath""",
            "time_of_day": FRESHWATER_TIME,
            "weather": FRESHWATER_WEATHER,
        },
        "resort_pool_busy": {
            "background": """free-form pool packed shoulder to shoulder | 1.0 | 0.4-0.8 | churned turquoise
lounger ranks three deep | 0.85 | 0.3-0.6 | towel-claimed rows
swim-up bar under a palm roof | 0.55 | 0.15-0.35 | stools in the water
palm cluster shading a corner | 0.6 | 0.2-0.4 | rustling umbrella crowns
water slide corkscrewing in | 0.5 | 0.15-0.35 | squealing delivery
hotel wings framing the deck | 0.6 | 0.3-0.55 | balcony grids
towel hut with a queue | 0.5 | 0.1-0.25 | rolled white stacks
kiddie splash zone fountaining | 0.55 | 0.15-0.35 | mushroom rain
umbrella field cranked open | 0.6 | 0.25-0.5 | canvas archipelago
hot tub corner at capacity | 0.45 | 0.1-0.25 | simmering circle
pool deck shimmering with heat | 0.55 | 0.25-0.5 | wet-print mosaic""",
            "midground": """cannonball contest at the deep end | 0.55 | 0.1-0.3 | judged splashes
float traffic jam mid-pool | 0.55 | 0.15-0.35 | vinyl gridlock
server weaving with a loaded tray | 0.5 | 0.1-0.25 | balanced circus
kids chain-diving for rings | 0.5 | 0.1-0.25 | bubbling relay
sunscreen assembly line on a lounger | 0.5 | 0.1-0.25 | family production
marco polo drifting out of bounds | 0.45 | 0.1-0.3 | eyes-shut wandering
towel flicked at a sibling | 0.45 | 0.05-0.15 | snap war
lifeguard rotating stations | 0.45 | 0.05-0.15 | whistle handoff
belly flop absorbed with honor | 0.45 | 0.05-0.15 | red-chested bow
pool noodle sword duel | 0.45 | 0.05-0.2 | foam fencing
napper defended by an eye mask | 0.4 | 0.05-0.15 | oblivious island""",
            "architecture_detail": """deck tile too hot for bare feet | 0.55 | 0.15-0.35 | hop-inducing
infinity edge sheeting over | 0.45 | 0.1-0.25 | glass lip
underwater bench along one wall | 0.45 | 0.1-0.25 | submerged shelf
depth tiles worn smooth | 0.4 | 0.03-0.1 | numberless markers
gutter slots sipping the chop | 0.45 | 0.05-0.15 | rhythmic slurp
umbrella sockets in the deck | 0.4 | 0.02-0.08 | brass rings
outdoor shower on a teak square | 0.45 | 0.05-0.15 | rinse ritual
planter walls doubling as seats | 0.45 | 0.1-0.25 | warm ledges
towel-corner knots on every lounger | 0.5 | 0.05-0.2 | claim flags
spa spillway warming the corner | 0.4 | 0.05-0.15 | steady sheet
palm shadow moving across the water | 0.5 | 0.15-0.35 | swaying lattice""",
            "props": """frozen drink sweating on a ledge | 0.55 | 0.02-0.06 | umbrella garnish
goggles collection by a family camp | 0.5 | 0.02-0.06 | strap pile
inflatable unicorn on patrol | 0.5 | 0.03-0.12 | absurd majesty
paperback fattening with humidity | 0.45 | 0.01-0.05 | deck-chair novel
waterproof speaker doing too much | 0.4 | 0.01-0.05 | contested playlist
dive rings glinting on the bottom | 0.5 | 0.02-0.08 | scattered targets
kids' crocs in a bright heap | 0.5 | 0.02-0.06 | molded pile
aloe bottle standing by | 0.4 | 0.01-0.04 | evening forecast
room key card in a sandal | 0.45 | 0.01-0.03 | classic hiding spot
snack basket down to crumbs | 0.45 | 0.01-0.05 | fry graveyard
wet footmarks evaporating in rows | 0.5 | 0.05-0.2 | vanishing trail""",
            "foreground_element": """lounger claimed at dawn defended at noon | 0.55 | 0.1-0.25 | towel diplomacy
slide exit wiped out with style | 0.5 | 0.05-0.2 | sideways arrival
drink handed down into the pool | 0.5 | 0.05-0.15 | careful transfer
sunscreen reapplied under protest | 0.5 | 0.05-0.2 | wriggling target
float boarded like a walrus | 0.5 | 0.05-0.2 | third attempt
hot deck crossed on tiptoe | 0.55 | 0.05-0.2 | yelping sprint
hair slicked back at the wall | 0.5 | 0.03-0.15 | breath caught
kid launched from joined hands | 0.5 | 0.05-0.2 | catapult service
shade negotiated by inches | 0.45 | 0.05-0.15 | umbrella politics
pool bar order mimed over splashing | 0.45 | 0.05-0.15 | held-up fingers
goggle marks compared like medals | 0.45 | 0.03-0.1 | red-ring pride""",
            "time_of_day": POOL_TIME,
            "weather": POOL_WEATHER,
        },
        "backyard_pool_private": {
            "background": """rectangular pool behind the house | 1.0 | 0.4-0.8 | still blue rectangle
privacy fence wrapping the yard | 0.8 | 0.3-0.55 | tall board wall
concrete deck with a few loungers | 0.7 | 0.25-0.5 | sun-bleached pair
diving board with a worn tip | 0.5 | 0.1-0.25 | sandpaper nose
string lights crossing the patio | 0.5 | 0.1-0.3 | idle bulbs by day
pool shed tucked in the corner | 0.5 | 0.1-0.25 | filter hum inside
back porch steps down to the deck | 0.55 | 0.15-0.35 | screen door above
maple throwing afternoon shade | 0.55 | 0.2-0.4 | leaf-shadow lace
neighbor rooflines beyond the fence | 0.5 | 0.15-0.35 | shingled horizon
water so still it doubles the sky | 0.6 | 0.25-0.5 | inverted clouds
towel line strung post to post | 0.45 | 0.1-0.25 | drying stripes""",
            "midground": """skimmer net leaned on the fence | 0.55 | 0.05-0.2 | long-handled sentinel
float drifting its own slow lap | 0.55 | 0.1-0.25 | unhurried orbit
hose topping off the water level | 0.4 | 0.05-0.15 | quiet trickle
robot cleaner tracing the floor | 0.4 | 0.05-0.15 | humming crawler
dog stationed at the wet edge | 0.45 | 0.05-0.15 | ball-drop hint
lounger angled into the last sun | 0.5 | 0.1-0.25 | tracked afternoon
bug dish floating its catch | 0.35 | 0.02-0.08 | overnight harvest
sprinkler ticking in the side yard | 0.35 | 0.05-0.15 | lawn duty
firefly shift starting at the fence | 0.35 | 0.05-0.2 | evening lights
towel cape drying over a chairback | 0.45 | 0.05-0.15 | superhero off duty
grill parked within splash range | 0.45 | 0.1-0.25 | dinner ambitions""",
            "architecture_detail": """coping stones warm underhand | 0.5 | 0.05-0.2 | rounded lip
deck crack sealed and resealed | 0.45 | 0.05-0.15 | tar seams
skimmer door flapping softly | 0.45 | 0.02-0.06 | plastic heartbeat
underwater light lens in the wall | 0.45 | 0.02-0.08 | glass eye
ladder rails hot in the sun | 0.45 | 0.02-0.08 | scorching chrome
water line tile band | 0.5 | 0.05-0.15 | blue mosaic stripe
filter return dimpling the surface | 0.45 | 0.03-0.1 | steady pulse
wet deck darkening in splash arcs | 0.5 | 0.1-0.25 | evaporating maps
gate latch up high out of kid reach | 0.45 | 0.02-0.06 | safety height
diving board bolts counter-sunk | 0.35 | 0.01-0.05 | flush steel
evening steam lifting off the water | 0.4 | 0.1-0.3 | day's heat leaving""",
            "props": """goggles on the pool edge | 0.5 | 0.01-0.05 | drip-drying
sunscreen and a rolled towel on a chair | 0.55 | 0.02-0.08 | ready kit
pool test strips tube by the shed | 0.35 | 0.01-0.04 | ritual chemistry
diving rings stacked on a step | 0.5 | 0.02-0.06 | rainbow quoits
noodle collection in a barrel | 0.5 | 0.03-0.1 | faded bouquet
radio on the porch rail | 0.4 | 0.01-0.05 | ballgame murmur
citronella torches ringing the deck | 0.4 | 0.03-0.1 | unlit til dusk
popsicle sticks on a napkin | 0.4 | 0.01-0.04 | afternoon evidence
dog ball waterlogged and prized | 0.45 | 0.01-0.04 | soggy treasure
book folded over a chair arm | 0.45 | 0.01-0.05 | pool-side pause
lemonade sweating on the steps | 0.45 | 0.01-0.05 | condensation ring""",
            "foreground_element": """first toe-in sending rings out | 0.6 | 0.05-0.2 | silence breaker
slow breaststroke keeping the hair dry | 0.5 | 0.1-0.25 | careful glide
float lain on like furniture | 0.55 | 0.1-0.25 | drifting recliner
cannonball audience of one dog | 0.5 | 0.05-0.2 | judged entry
skimmer run around the edge | 0.5 | 0.1-0.25 | leaf patrol
night swim glow admired from the water | 0.4 | 0.1-0.3 | lit from below
towel wrapped watching the sunset | 0.5 | 0.1-0.25 | dripping quietly
handstand contest scored loosely | 0.45 | 0.05-0.2 | legs at angles
edge sat with legs stirring circles | 0.55 | 0.05-0.2 | idle propellers
sun-warmed deck lain on to dry | 0.5 | 0.1-0.3 | starfish pose
lightning bugs watched from the shallows | 0.4 | 0.1-0.25 | chin at water level""",
            "time_of_day": POOL_TIME,
            "weather": POOL_WEATHER,
        },
        "hotel_pool_after_dark": {
            "background": """empty pool glowing from below | 1.0 | 0.4-0.8 | electric turquoise slab
deck chairs squared off for the night | 0.8 | 0.25-0.5 | aligned ranks
hotel windows checkering the dark | 0.7 | 0.25-0.5 | lit and unlit grid
palm silhouettes gone black | 0.55 | 0.15-0.35 | paper cutouts
steam curling off the warm water | 0.6 | 0.15-0.35 | slow ghosts
hot tub simmering in its corner | 0.5 | 0.1-0.25 | ringed jacuzzi glow
towel hut shuttered | 0.45 | 0.1-0.25 | closed hatch
underwater lights laddering the lanes | 0.6 | 0.15-0.35 | glowing rungs
city or stars beyond the fence line | 0.5 | 0.2-0.4 | distant glitter
umbrella spokes folded like roosting birds | 0.5 | 0.1-0.3 | furled canvas
night air holding the chlorine close | 0.5 | 0.2-0.4 | heavy scent blanket""",
            "midground": """lone swimmer stitching quiet laps | 0.5 | 0.1-0.25 | rippling seam
couple whispering in the hot tub | 0.4 | 0.1-0.2 | steam-wrapped voices
abandoned float turning in the current | 0.5 | 0.05-0.2 | slow clockwork
moths orbiting the deck lamps | 0.45 | 0.03-0.1 | dusty satellites
pool cleaner hose breathing on the bottom | 0.4 | 0.05-0.15 | coiled worker
wet footmarks leading to the gate | 0.45 | 0.05-0.15 | recent departure
security round passing unhurried | 0.35 | 0.05-0.15 | keyring jingle
night breeze scalloping the surface | 0.5 | 0.1-0.3 | broken reflections
towel forgotten on a lounger | 0.45 | 0.03-0.1 | pale rectangle
ice machine glow from the alcove | 0.4 | 0.05-0.15 | blue-white spill
bat cutting through the lamp light | 0.35 | 0.02-0.08 | flickering pass""",
            "architecture_detail": """light cones wavering on the pool floor | 0.55 | 0.15-0.35 | dancing nets
deck tile cool at last underfoot | 0.5 | 0.1-0.3 | day's heat gone
gutter slurp louder in the quiet | 0.45 | 0.03-0.1 | rhythmic sips
handrail beaded with condensation | 0.4 | 0.02-0.08 | cool chrome sweat
depth shadows stacked in layers | 0.45 | 0.1-0.3 | dark stairsteps
gate spring loud as a gunshot | 0.4 | 0.02-0.08 | echoing clang
reflection of the building laid on the water | 0.5 | 0.2-0.4 | trembling twin
steam fog softening every lamp | 0.5 | 0.1-0.3 | halo effect
lane rope shadows waving on the floor | 0.4 | 0.05-0.2 | sine curves
puddles holding their own small lights | 0.45 | 0.05-0.15 | scattered coins
hot tub timer dial ticking down | 0.35 | 0.01-0.05 | mechanical patience""",
            "props": """single sandal under a lounger | 0.4 | 0.01-0.04 | orphan of the day
folded towel stack left out | 0.45 | 0.02-0.08 | white monolith
wine glasses by the hot tub rim | 0.35 | 0.01-0.05 | stems in steam
goggles hanging off a chair arm | 0.4 | 0.01-0.04 | swinging slightly
room key on a damp towel | 0.4 | 0.01-0.03 | trust or forgetfulness
float parked against the steps | 0.45 | 0.03-0.1 | nudged by the current
drink can sweating on the deck | 0.4 | 0.01-0.04 | ring forming
phone face-down on a lounger | 0.35 | 0.01-0.03 | do not disturb
pool thermometer bobbing on its cord | 0.4 | 0.01-0.04 | patient float
ice bucket abandoned mid-melt | 0.35 | 0.01-0.05 | clinking quietly
moth floating spread-winged | 0.35 | 0.01-0.03 | night casualty""",
            "foreground_element": """quiet slip into the glowing water | 0.6 | 0.1-0.25 | barely a ripple
back float under the night sky | 0.55 | 0.1-0.3 | ears full of hum
steam parted with a slow arm | 0.5 | 0.05-0.2 | ghost-swimming
whisper carrying across the water | 0.45 | 0.05-0.15 | conspirators' pool
underwater glide lit blue-green | 0.5 | 0.1-0.25 | glowing passage
hot tub eased into by degrees | 0.45 | 0.05-0.2 | hissing breath
wet hair pushed back under the lamps | 0.5 | 0.03-0.15 | slicked shine
towel hugged against the night air | 0.5 | 0.05-0.2 | shivering pause
pool edge sat on with a drink | 0.5 | 0.05-0.2 | legs in the glow
midnight lap counted to nobody | 0.45 | 0.1-0.25 | private race
gate closed softly behind | 0.45 | 0.05-0.15 | considerate exit""",
            "time_of_day": POOL_TIME,
            "weather": POOL_WEATHER,
        },
    },
    "outdoor/vacation_winter_us": {
        "cabin_hot_tub_deck": {
            "background": """snow-loaded deck around the tub | 1.0 | 0.4-0.8 | white-capped rails
hot tub steaming hard | 0.9 | 0.2-0.4 | rolling vapor column
cabin wall of stacked logs | 0.75 | 0.3-0.55 | snow-chinked seams
pines sagging under snow loads | 0.7 | 0.25-0.5 | white-shouldered ranks
valley lights far below | 0.45 | 0.15-0.35 | scattered warm dots
icicle fringe along the eave | 0.6 | 0.1-0.3 | glass teeth row
firewood stack under a white cap | 0.55 | 0.15-0.3 | dusted cordwood
snowfield rolling off into the trees | 0.6 | 0.3-0.6 | unbroken blue-white
chimney smoke standing straight up | 0.5 | 0.1-0.3 | windless column
deck lantern haloed in vapor | 0.5 | 0.05-0.2 | soft glow ball
stars sharp above the steam | 0.45 | 0.2-0.5 | hard bright points""",
            "midground": """robes hung on hooks by the door | 0.6 | 0.1-0.25 | waiting warmth
snow shoveled into a path | 0.6 | 0.15-0.3 | walled walkway
boots steaming on the mat | 0.5 | 0.05-0.15 | melting crust
cover shell propped against the rail | 0.5 | 0.1-0.25 | frosted clamshell
snow sliding off a branch in a whump | 0.45 | 0.05-0.2 | powder burst
deer prints stitching the yard | 0.45 | 0.05-0.2 | dotted lines
sled parked nose-in a drift | 0.4 | 0.05-0.15 | day's veteran
thermometer needle deep in the blue | 0.4 | 0.01-0.05 | honest cold
snowman supervising from the yard | 0.4 | 0.05-0.15 | stick-armed guard
lantern path lights half buried | 0.45 | 0.05-0.15 | glowing mounds
owl calling from the dark treeline | 0.35 | 0.03-0.1 | question marks""",
            "architecture_detail": """deck boards squeaking with frost | 0.5 | 0.1-0.3 | bound crystals
tub rim ice where splashes froze | 0.5 | 0.03-0.1 | glazed ring
rail caps carrying perfect snow bars | 0.5 | 0.05-0.2 | knife-edge loaves
steam-frost feathering the window | 0.5 | 0.05-0.15 | fern patterns
jets churning the surface hard | 0.55 | 0.1-0.25 | rolling boil
path salt biting through the ice | 0.4 | 0.03-0.1 | gritty patches
icicle drips re-freezing mid-fall | 0.4 | 0.02-0.08 | growing spears
snow squeak underfoot in the cold | 0.5 | 0.05-0.2 | styrofoam song
tub light shifting through its colors | 0.4 | 0.05-0.2 | slow rainbow
frozen towel forgotten on the rail | 0.4 | 0.02-0.08 | stiff as a board
breath and steam merging in clouds | 0.5 | 0.1-0.3 | double vapor""",
            "props": """cocoa mugs on the tub edge | 0.5 | 0.01-0.06 | marshmallow rafts
wool hats worn in the water | 0.5 | 0.02-0.06 | absurd but correct
towels folded inside the door | 0.45 | 0.02-0.08 | strategic staging
lantern with a candle stub | 0.4 | 0.01-0.05 | soft flicker
snowball stockpile within reach | 0.4 | 0.01-0.05 | ambush ready
slippers waiting under the eave | 0.45 | 0.01-0.05 | dry promise
book abandoned for the view | 0.35 | 0.01-0.04 | face-down tent
headlamp on the rail post | 0.35 | 0.01-0.04 | path insurance
cider thermos wedged in snow | 0.4 | 0.01-0.05 | natural cooler backwards
icicle harvested as a trophy | 0.35 | 0.01-0.04 | melting sword
robe belt trailing in the snow | 0.4 | 0.01-0.05 | dropped sash""",
            "foreground_element": """steam-blind shuffle to the tub | 0.6 | 0.1-0.25 | robe clutched
gasp at the air-to-water swap | 0.6 | 0.05-0.2 | 40-degree leap
hair frosting white at the tips | 0.5 | 0.03-0.12 | instant aging
snowball lobbed from the water | 0.45 | 0.05-0.15 | artillery advantage
shoulders sunk to the chin | 0.55 | 0.1-0.25 | thermostat found
star search through the steam gaps | 0.5 | 0.1-0.3 | patient tilt
snow angel dared and regretted | 0.35 | 0.1-0.25 | shrieking return
toe out to test the deck cold | 0.45 | 0.03-0.1 | instant retreat
cocoa steam sipped through the vapor | 0.5 | 0.03-0.1 | double warmth
wet hand hissing on the cold rail | 0.4 | 0.02-0.08 | steam print
quiet listened to between jets | 0.5 | 0.1-0.3 | winter hush""",
            "time_of_day": WINTER_TIME,
            "weather": WINTER_WEATHER,
        },
        "frozen_lake_shore": {
            "background": """white lake plain to the far shore | 1.0 | 0.5-0.9 | wind-packed snow over ice
dark treeline ringing the shore | 0.75 | 0.25-0.5 | spruce silhouette wall
ice heaved into shore ridges | 0.6 | 0.15-0.35 | tilted blue plates
snow dunes sculpted across the ice | 0.55 | 0.2-0.45 | wind-carved waves
far shacks of ice fishermen | 0.4 | 0.1-0.25 | scattered dots
cattails locked in at the inlet | 0.5 | 0.15-0.3 | frozen pokers
cloud shadows crossing the white | 0.5 | 0.25-0.5 | traveling grey
pressure crack wandering out | 0.45 | 0.1-0.3 | dark lightning line
boarded summer dock iced in | 0.5 | 0.15-0.3 | locked-in pilings
island pines wearing snow coats | 0.45 | 0.15-0.3 | offshore hummock
low sun dragging long blue shadows | 0.55 | 0.25-0.5 | stretched forms""",
            "midground": """cleared skating patch swept dark | 0.5 | 0.15-0.35 | grey window in the white
shovel standing in a snow pile | 0.45 | 0.05-0.15 | upright marker
hockey net dragged onto the ice | 0.4 | 0.1-0.2 | sagging twine
fox tracks looping the shoreline | 0.45 | 0.05-0.2 | purposeful stitching
ice angler hunched on a bucket | 0.4 | 0.05-0.15 | patient statue
snow squall marching across the far end | 0.35 | 0.2-0.4 | grey curtain
kids testing the shore ice loudness | 0.45 | 0.1-0.25 | stomping chorus
dog delighted by the slick | 0.45 | 0.05-0.2 | splayed scramble
bench iced over facing the lake | 0.45 | 0.05-0.15 | glazed slats
old bubbler hole skinned with new ice | 0.35 | 0.02-0.08 | thin black window
sled train heading out from shore | 0.35 | 0.1-0.25 | rope caravan""",
            "architecture_detail": """booming crack rolling under the ice | 0.5 | 0.1-0.3 | whale-song thunder
clear black ice windows between snow | 0.5 | 0.1-0.3 | bubbles locked mid-rise
shore stones capped in ice helmets | 0.45 | 0.05-0.2 | glazed cobbles
frost flowers blooming on new ice | 0.4 | 0.03-0.12 | feather gardens
wind-polished slick streaks | 0.45 | 0.1-0.3 | mirror lanes
slush layer hiding under fresh snow | 0.4 | 0.05-0.2 | grey surprise
ice ridge sheared and refrozen | 0.4 | 0.05-0.2 | broken stacked panes
reed stems cased in glass | 0.4 | 0.03-0.12 | crystal straws
snow squeak pitch rising with the cold | 0.45 | 0.05-0.2 | temperature audible
dock ladder locked in mid-air | 0.35 | 0.02-0.08 | frozen reach
bubbles stair-stepped down the black | 0.4 | 0.05-0.15 | frozen ascent""",
            "props": """skates knotted over a shoulder | 0.5 | 0.02-0.06 | blade pair
thermos planted in the snow | 0.45 | 0.01-0.05 | steam flag
hockey stick bridge over a crack | 0.4 | 0.01-0.05 | cautious tool
hand auger leaned on a bucket | 0.35 | 0.01-0.05 | cork-screwed steel
tip-up flag waiting mid-lake | 0.35 | 0.01-0.04 | spring-loaded promise
mittens clipped to a jacket | 0.45 | 0.01-0.04 | idiot-string wisdom
puck lost in a snowbank | 0.4 | 0.01-0.03 | spring discovery
sled rope frozen stiff | 0.4 | 0.01-0.04 | wire curve
hand warmers shaken furiously | 0.45 | 0.01-0.03 | pocket heat
broken shovel handle marking thin ice | 0.35 | 0.01-0.05 | improvised warning
orange peel bright on the white | 0.35 | 0.01-0.03 | snack evidence""",
            "foreground_element": """first glide wobbled then trusted | 0.55 | 0.1-0.25 | ankle negotiation
ice stared through hands-cupped | 0.5 | 0.05-0.15 | window to the dark
crack boom flinched at then laughed off | 0.5 | 0.05-0.2 | nervous humor
snow shoved clear with a boot sweep | 0.5 | 0.05-0.2 | widening the rink
stone slid whirring for distance | 0.45 | 0.05-0.2 | curling minimalism
breath fog thick as speech bubbles | 0.5 | 0.03-0.12 | visible words
mitten wiped across a runny nose | 0.45 | 0.02-0.08 | winter honesty
skate laces yanked tight with teeth | 0.4 | 0.03-0.1 | frozen-finger workaround
penguin shuffle across a slick patch | 0.5 | 0.05-0.2 | dignity traded
sunset colors read off the ice | 0.45 | 0.1-0.3 | pink mirror
hot drink passed glove to glove | 0.45 | 0.03-0.1 | shared thaw""",
            "time_of_day": WINTER_TIME,
            "weather": WINTER_WEATHER,
        },
        "snowed_in_cottage_yard": {
            "background": """cottage buried to the sills | 1.0 | 0.4-0.8 | white-drifted hunker
path dug shoulder-deep to the door | 0.75 | 0.2-0.4 | canyon walk
woodshed with its drifted lean-to | 0.6 | 0.2-0.4 | white-capped shelter
fence posts down to their caps | 0.6 | 0.1-0.3 | dotted line of knobs
orchard trees iced to filigree | 0.5 | 0.2-0.4 | glass branches
smoke rising from the buried chimney | 0.6 | 0.1-0.3 | grey ribbon
clothesline sagging with snow rope | 0.45 | 0.1-0.25 | white cable
drift curling off the roof edge | 0.55 | 0.15-0.35 | frozen wave
birdfeeder mobbed in the quiet | 0.5 | 0.05-0.2 | flickering visitors
lane unplowed to the main road | 0.5 | 0.2-0.4 | untracked ribbon
mailbox drowned to its flag | 0.45 | 0.05-0.15 | red periscope""",
            "midground": """shovel parked mid-job | 0.6 | 0.05-0.2 | leaning worker
snow fort under construction | 0.45 | 0.1-0.25 | block architecture
toboggan tipped by the steps | 0.5 | 0.05-0.2 | red curl
rabbit tracks under the feeder | 0.45 | 0.05-0.15 | hop punctuation
snow blower cowled in its drift | 0.4 | 0.1-0.2 | defeated machine
icicles farmed along the shed eave | 0.5 | 0.05-0.2 | growing teeth
cat refusing the doorstep threshold | 0.4 | 0.02-0.08 | one-paw verdict
firewood ferried load by load | 0.5 | 0.1-0.25 | sled shuttle
chickadees swapping at the feeder | 0.45 | 0.03-0.1 | polite rotation
drift cornice cracking off the roof | 0.4 | 0.1-0.25 | slow-motion calve
snow-cave entrance glowing faint blue | 0.35 | 0.05-0.15 | kid-sized burrow""",
            "architecture_detail": """windowpanes ferned with frost | 0.55 | 0.05-0.2 | crystal gardens
door swollen and shoulder-opened | 0.45 | 0.03-0.1 | winter fit
steps packed to a slick ramp | 0.5 | 0.05-0.15 | treacherous grade
drift strata telling the storm history | 0.45 | 0.1-0.3 | layered chapters
snow load creaking the shed roof | 0.4 | 0.05-0.2 | weight complaint
buried rain barrel domed white | 0.4 | 0.03-0.1 | igloo cameo
wind scallops around every corner | 0.45 | 0.1-0.25 | carved eddies
blue shadow pooling in the path | 0.5 | 0.1-0.3 | cold light
glitter drift when the sun breaks | 0.45 | 0.1-0.3 | diamond air
eave icicle drips counting seconds | 0.45 | 0.03-0.1 | slow metronome
porch light cap of perfect snow | 0.4 | 0.01-0.05 | white beret""",
            "props": """snowshoes crossed by the door | 0.45 | 0.02-0.08 | webbed pair
sled rope in a stiff coil | 0.45 | 0.01-0.05 | frozen spiral
bird seed scoop in the bin | 0.4 | 0.01-0.04 | buried to the handle
carrot budgeted for the snowman | 0.4 | 0.01-0.03 | nose in waiting
mittens drying on the stove-side line | 0.45 | 0.02-0.06 | steaming row
thermos parked on the porch rail | 0.4 | 0.01-0.05 | steam curl
snow brush worn to a nub | 0.4 | 0.01-0.04 | car veteran
kindling basket topped with birch | 0.45 | 0.02-0.08 | paper-bark curls
salt bucket with a frozen scoop | 0.4 | 0.01-0.05 | chipped free
lantern for the woodshed run | 0.4 | 0.01-0.05 | swinging glow
storm chocolate rationed slowly | 0.35 | 0.01-0.03 | foil corner""",
            "foreground_element": """shovel bite and heave rhythm | 0.6 | 0.1-0.25 | steady excavation
drift fallen into backwards laughing | 0.5 | 0.1-0.25 | powder crater
firewood armload chin-steadied | 0.55 | 0.05-0.2 | bark shedding
icicle snapped off for inspection | 0.5 | 0.03-0.1 | crystal sword
snow squint into the bright yard | 0.5 | 0.05-0.2 | hand visor
boot snow stomped off on the step | 0.55 | 0.03-0.12 | ritual thunder
feeder refilled with numb fingers | 0.45 | 0.03-0.1 | seed spill
path widened one scoop a day | 0.45 | 0.1-0.25 | siege patience
snowball packed and reconsidered | 0.45 | 0.03-0.1 | truce held
frost window scraped for a look out | 0.5 | 0.05-0.15 | fingernail porthole
deep-snow wade thigh-high | 0.5 | 0.1-0.3 | slow-motion march""",
            "time_of_day": WINTER_TIME,
            "weather": WINTER_WEATHER,
        },
        "empty_ski_slope_edge": {
            "background": """groomed slope empty in the last light | 1.0 | 0.5-0.9 | corduroy lanes
lift chairs hanging dead still | 0.75 | 0.15-0.35 | swaying stopped
treeline walls channeling the run | 0.7 | 0.25-0.5 | dark spruce banks
mountain shoulder above the top station | 0.55 | 0.2-0.4 | wind-scoured white
snow fence rows catching drift | 0.5 | 0.1-0.3 | slatted ranks
lodge lights warm at the base | 0.5 | 0.15-0.3 | amber windows
snow gun hibernating under ice | 0.45 | 0.1-0.25 | frozen cannon
half-buried hay bales at the pylon | 0.45 | 0.05-0.2 | padding mounds
groomer working a far slope | 0.4 | 0.1-0.25 | crawling lights
untouched powder stash off the edge | 0.5 | 0.2-0.4 | tempting margin
first stars over the ridgeline | 0.45 | 0.2-0.4 | early sparks""",
            "midground": """last skier's tracks setting up | 0.55 | 0.15-0.35 | frozen curves
sled path worn beside the run | 0.45 | 0.1-0.25 | walkers' groove
snow cat road switchbacking wide | 0.4 | 0.15-0.3 | packed lane
powder pillows on the boundary pines | 0.5 | 0.15-0.3 | loaded branches
lift terminal humming to itself | 0.4 | 0.1-0.2 | idle machinery
ptarmigan tracks lacing the fence line | 0.35 | 0.03-0.1 | bird stitching
wind lifting spindrift off the crest | 0.5 | 0.1-0.3 | smoking ridge line
snowbank seats carved by waiting | 0.4 | 0.05-0.15 | slumped couches
last chair rocking as it stops | 0.4 | 0.05-0.15 | pendulum settling
maintenance ski-doo parked | 0.35 | 0.05-0.15 | key-in trust
moon rising over the far shoulder | 0.4 | 0.1-0.3 | cold lamp""",
            "architecture_detail": """corduroy grooves crisp underfoot | 0.55 | 0.15-0.4 | fresh-pressed lines
ice glaze where the sun hit at noon | 0.5 | 0.1-0.3 | refrozen sheen
pylon pads scarred by seasons | 0.4 | 0.03-0.1 | patched cushions
cable sag between towers | 0.45 | 0.05-0.2 | catenary curves
drift knife-edging the fence tops | 0.45 | 0.05-0.2 | sculpted crests
boot pack trail hardening to concrete | 0.45 | 0.05-0.2 | stomped stair
snow squeak sharpening as it cools | 0.45 | 0.05-0.2 | cold audible
gun whales lined down the pitch | 0.4 | 0.1-0.3 | man-made moguls
shadow line racing up the slope | 0.5 | 0.2-0.4 | sunset tide
tower lights blinking on in series | 0.4 | 0.05-0.15 | one-by-one wake
tracks refrozen into rails | 0.45 | 0.1-0.25 | set grooves""",
            "props": """skis planted X in the snow | 0.5 | 0.02-0.08 | crossed marker
poles leaned on the fence | 0.45 | 0.01-0.05 | strap tangle
goggles pushed up dead of use | 0.45 | 0.01-0.04 | mirrored brow
lone glove atop a fence post | 0.4 | 0.01-0.04 | lost-and-found beacon
wax scraper in a jacket pocket | 0.35 | 0.01-0.03 | plastic edge
thermos shared at the overlook | 0.45 | 0.01-0.05 | passed warmth
boot buckles finally released | 0.45 | 0.01-0.04 | day's-end relief
trail snack bar frozen solid | 0.4 | 0.01-0.03 | jaw workout
hand warmers at half power | 0.4 | 0.01-0.03 | fading coals
helmet hung on a pole grip | 0.4 | 0.01-0.05 | resting shell
snowball packed on a dare | 0.35 | 0.01-0.04 | pointless ammo""",
            "foreground_element": """last run stance taken and held | 0.55 | 0.1-0.3 | savoring the empty
boots clomped up the boot pack | 0.5 | 0.1-0.25 | frankenstein steps
goggle fog wiped with a thumb | 0.5 | 0.03-0.1 | smeared clarity
spindrift taken full in the face | 0.45 | 0.05-0.2 | laughing whiteout
skis shouldered for the walk down | 0.5 | 0.1-0.25 | A-frame carry
quiet listened to after the lifts die | 0.5 | 0.1-0.3 | mountain silence
untouched margin poached guiltily | 0.4 | 0.1-0.3 | powder theft
lodge lights aimed for gratefully | 0.5 | 0.1-0.25 | homing descent
cold toes wiggled for a verdict | 0.45 | 0.02-0.08 | numb count
sunset colors watched from the fence | 0.5 | 0.1-0.3 | leaning audience
track carved slow and perfect | 0.45 | 0.1-0.3 | flawless arc""",
            "time_of_day": WINTER_TIME,
            "weather": WINTER_WEATHER,
        },
    },
}


def write_file(path: pathlib.Path, text: str, force: bool) -> str:
    if path.exists() and not force:
        return "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "write"


OUTDOOR_BANNED = ["hardwood floor", "sofa", "office desk", "kitchen counter",
                  "duvet", "shower stall", "bath tub", "indoor pool"]

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
        for b in OUTDOOR_BANNED:
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
