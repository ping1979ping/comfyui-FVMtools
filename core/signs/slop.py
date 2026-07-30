"""Heuristics that detect implausible or gibberish rendered text ("slop").

Diffusion models happily paint letter-shaped noise. Three cheap, offline
signals catch most of it: the tokens are not words, the letter pairs do not
occur in latin script, or SAM3 sees a text region where OCR reads nothing at
all (the pseudo-glyph case). All functions here are pure and do no I/O.
"""

# ──── Embedded lexicon ────

_WORDS_EN = """
OPEN CLOSED ENTER EXIT ENTRANCE PUSH PULL STOP GO SALE PRICE FREE NEW HOT COLD
FRESH DAILY SPECIAL MENU COFFEE TEA BEER WINE WATER MILK BREAD CAKE PIZZA
BURGER SANDWICH SALAD SOUP SHOP STORE MARKET CAFE BAR HOTEL MOTEL BANK POST
OFFICE SCHOOL LIBRARY MUSEUM PARK STREET ROAD AVENUE LANE SQUARE STATION
AIRPORT PARKING GARAGE TAXI BUS TRAIN METRO SUBWAY PLATFORM TICKET GATE PLEASE
THANK THANKS WELCOME HELLO GOODBYE YES NO MAYBE HELP INFO INFORMATION WARNING
DANGER CAUTION NOTICE PRIVATE PUBLIC STAFF ONLY RESERVED VACANT OCCUPIED
TOILET RESTROOM LADIES GENTLEMEN MEN WOMEN CHILDREN FAMILY FIRE EMERGENCY
POLICE HOSPITAL DOCTOR PHARMACY CLINIC DENTIST THE AND FOR YOU ARE NOT BUT ALL
ANY CAN HAS HAD HIS HER OUR THEIR WITH FROM THIS THAT THESE THOSE HAVE WILL
WOULD SHOULD COULD ABOUT AFTER AGAIN AGAINST BEFORE BEING BELOW BETWEEN BOTH
DURING EACH FEW MORE MOST OTHER SOME SUCH THAN THEN THERE THROUGH UNDER UNTIL
VERY WHAT WHEN WHERE WHICH WHILE WHO WHY HOW ONE TWO THREE FOUR FIVE SIX SEVEN
EIGHT NINE TEN ELEVEN TWELVE TWENTY THIRTY FIFTY HUNDRED THOUSAND MILLION
MONDAY TUESDAY WEDNESDAY THURSDAY FRIDAY SATURDAY SUNDAY JANUARY FEBRUARY
MARCH APRIL MAY JUNE JULY AUGUST SEPTEMBER OCTOBER NOVEMBER DECEMBER NORTH
SOUTH EAST WEST LEFT RIGHT UP DOWN FRONT BACK TOP BOTTOM CENTER CENTRE RED
GREEN BLUE BLACK WHITE YELLOW ORANGE PURPLE PINK BROWN GREY GRAY SILVER GOLD
BOOK BOOKS PAPER NEWS TIMES WEEKLY JOURNAL MAGAZINE PRESS EDITION VOLUME
CHAPTER PAGE LIMITED COMPANY GROUP BROTHERS SONS ASSOCIATES PARTNERS TRADING
SERVICE SERVICES SUPPLY SUPPLIES WORKS FACTORY STUDIO GALLERY SALON BARBER
LAUNDRY BAKERY BUTCHER GROCERY DELI DINER BISTRO PUB TAVERN LOUNGE CLUB
THEATER THEATRE CINEMA SIZE SMALL MEDIUM LARGE EXTRA SUPER MEGA MINI LOVE LIFE
TIME WORLD PEACE DREAM STAR SUN MOON SKY SEA OCEAN RIVER MOUNTAIN FOREST CITY
TOWN VILLAGE HOME HOUSE ROOM DOOR WINDOW KEY MADE QUALITY ORIGINAL GENUINE
NATURAL ORGANIC PREMIUM CLASSIC MODERN VINTAGE RETRO BEST GOOD GREAT TOP FIRST
LAST NEXT BUY SELL SAVE ONLINE DELIVERY PICKUP TAKEAWAY CASH CARD CREDIT DEBIT
CHANGE RECEIPT TOTAL SUBTOTAL DISCOUNT OFFER DEAL NAME ADDRESS PHONE EMAIL WEB
SITE FLAMMABLE POISON TOXIC HIGH VOLTAGE SPEED LIMIT SLOW YIELD ROUTE HIGHWAY
BRIDGE TUNNEL HARBOUR HARBOR BEACH SUMMER WINTER SPRING AUTUMN NIGHT MORNING
EVENING TODAY TONIGHT HAPPY LUCKY ROYAL GRAND CENTRAL UNION NATIONAL GLOBAL
INTERNATIONAL EXPRESS RAPID DIRECT LOCAL SUPERMARKET PHARMACY GARDEN FLOWER
FLOWERS MUSIC SOUND RADIO VIDEO PHOTO PHOTOGRAPHY DESIGN ART ARTS CRAFT
SPORTS FITNESS GYM YOGA DANCE SCHOOLS ACADEMY COLLEGE UNIVERSITY INSTITUTE
CHURCH TEMPLE CASTLE PALACE TOWER PLAZA MALL CENTRE COURT HALL FLOOR LEVEL
BASEMENT ROOF WAIT READY CLOSE FINISH START BEGIN
"""

_WORDS_DE = """
DER DIE DAS UND ODER ABER NICHT EIN EINE EINEN MIT VON ZUM ZUR AUF AUS BEI NACH
VOR UEBER UNTER DURCH FUER OHNE ICH DU ER SIE ES WIR IHR SIND IST WAR HABEN HAT
WIRD WERDEN KANN MUSS SOLL BAHNHOF HAUPTBAHNHOF FLUGHAFEN AUSGANG EINGANG
AUSFAHRT EINFAHRT NOTAUSGANG HALTESTELLE PARKPLATZ PARKHAUS STRASSE WEG PLATZ
GASSE ALLEE RING MARKT MARKTPLATZ RATHAUS KIRCHE SCHULE BIBLIOTHEK APOTHEKE
KRANKENHAUS ARZT ZAHNARZT POLIZEI FEUERWEHR BAECKEREI METZGEREI FLEISCHEREI
KONDITOREI BUCHHANDLUNG BLUMEN FRISEUR GASTHAUS GASTHOF RESTAURANT KNEIPE
BRAUEREI BROT KUCHEN TORTE KAFFEE BIER WEIN WASSER MILCH ZUCKER SALZ MEHL
BUTTER WURST FLEISCH FISCH OBST OFFEN GESCHLOSSEN RUHETAG FEIERTAG PREIS PREISE
ANGEBOT SONDERANGEBOT NEU FRISCH TAEGLICH ACHTUNG VORSICHT GEFAHR VERBOTEN
PRIVAT EINTRITT FREI KEIN AUSKUNFT NOTRUF RAUCHEN HERREN DAMEN TOILETTE KINDER
FAMILIE MONTAG DIENSTAG MITTWOCH DONNERSTAG FREITAG SAMSTAG SONNTAG EINS ZWEI
DREI VIER SECHS SIEBEN ACHT NEUN ZEHN NORD SUED OST WEST LINKS RECHTS OBEN
UNTEN ROT BLAU SCHWARZ WEISS GELB BRAUN GRAU HAUS ZIMMER FENSTER STADT DORF
LAND WELT ZEIT LEBEN LIEBE TRAUM STERN SONNE MOND HIMMEL MEER FLUSS BERG WALD
GUT BESSER BESTE GROSS KLEIN LANG KURZ ALT JUNG SCHNELL LANGSAM DANKE BITTE
HALLO WILLKOMMEN GMBH VERLAG ZEITUNG NACHRICHTEN WERKSTATT BAUSTELLE FAHRRAD
AUTOHAUS TANKSTELLE SUPERMARKT KAUFHAUS SCHREIBWAREN SPIELWAREN MOEBEL GARTEN
BLUMENLADEN THEATER KINO MUSEUM GALERIE HOTEL PENSION HERBERGE CAMPING STRAND
SEE INSEL BRUECKE TUNNEL BAHNSTEIG GLEIS ABFAHRT ANKUNFT FAHRPLAN FAHRKARTE
SCHALTER KASSE EINKAUF VERKAUF MIETE WOHNUNG BUERO PRAXIS KANZLEI SCHULHOF
SPIELPLATZ FRIEDHOF BAHNHOFSTRASSE HAUPTSTRASSE KIRCHPLATZ MUENSTER DOM
"""

WORD_LIST = frozenset(
    token for token in (_WORDS_EN + " " + _WORDS_DE).upper().split() if token
)

# ──── Embedded bigram table (weights 1..10, 10 = very common) ────

MAX_BIGRAM_WEIGHT = 10.0

BIGRAM_FREQ = {
    "AB": 5, "AC": 6, "AD": 5, "AE": 3, "AF": 3, "AG": 4, "AH": 3, "AI": 5,
    "AL": 8, "AM": 6, "AN": 10, "AP": 4, "AR": 8, "AS": 7, "AT": 9, "AU": 5,
    "AV": 3, "AW": 2, "AY": 4,
    "BA": 6, "BE": 7, "BI": 4, "BL": 4, "BO": 5, "BR": 5, "BU": 4, "BY": 3,
    "CA": 6, "CE": 6, "CH": 9, "CI": 4, "CK": 6, "CL": 4, "CO": 8, "CR": 4,
    "CT": 5, "CU": 3,
    "DA": 5, "DE": 9, "DI": 6, "DO": 5, "DR": 3, "DU": 3,
    "EA": 7, "EB": 2, "EC": 5, "ED": 8, "EE": 5, "EF": 3, "EG": 3, "EH": 3,
    "EI": 8, "EK": 2, "EL": 7, "EM": 5, "EN": 10, "EO": 2, "EP": 3, "ER": 10,
    "ES": 8, "ET": 6, "EU": 3, "EV": 3, "EW": 2, "EX": 3, "EY": 3,
    "FA": 4, "FE": 5, "FF": 4, "FI": 5, "FL": 4, "FO": 6, "FR": 5, "FT": 4,
    "FU": 3,
    "GA": 4, "GE": 8, "GH": 4, "GI": 3, "GL": 3, "GO": 4, "GR": 5, "GU": 2,
    "HA": 8, "HE": 10, "HI": 6, "HL": 2, "HN": 3, "HO": 7, "HR": 3, "HT": 3,
    "HU": 3,
    "IA": 3, "IC": 7, "ID": 4, "IE": 7, "IF": 3, "IG": 4, "II": 1, "IK": 2,
    "IL": 6, "IM": 5, "IN": 10, "IO": 6, "IP": 3, "IR": 5, "IS": 8, "IT": 8,
    "IV": 3, "IZ": 2,
    "JA": 2, "JE": 2, "JO": 2, "JU": 2,
    "KA": 3, "KE": 4, "KI": 3, "KL": 2, "KO": 3, "KR": 2, "KT": 2, "KU": 2,
    "LA": 6, "LB": 2, "LD": 6, "LE": 8, "LF": 2, "LI": 6, "LK": 2, "LL": 7,
    "LM": 2, "LO": 6, "LP": 2, "LS": 3, "LT": 4, "LU": 3, "LY": 4,
    "MA": 7, "ME": 7, "MI": 5, "MM": 3, "MO": 6, "MP": 3, "MU": 3, "MY": 2,
    "NA": 5, "NC": 5, "ND": 9, "NE": 8, "NF": 2, "NG": 8, "NH": 2, "NI": 5,
    "NK": 3, "NN": 4, "NO": 6, "NS": 6, "NT": 8, "NU": 2, "NZ": 2,
    "OB": 3, "OC": 4, "OD": 4, "OF": 6, "OG": 3, "OH": 3, "OI": 2, "OK": 2,
    "OL": 5, "OM": 6, "ON": 9, "OO": 4, "OP": 6, "OR": 9, "OS": 5, "OT": 4,
    "OU": 8, "OV": 3, "OW": 5, "OX": 1,
    "PA": 5, "PE": 7, "PF": 2, "PH": 4, "PI": 5, "PL": 5, "PO": 5, "PP": 4,
    "PR": 6, "PS": 2, "PT": 3, "PU": 3,
    "QU": 4,
    "RA": 7, "RB": 2, "RC": 3, "RD": 5, "RE": 10, "RG": 3, "RI": 7, "RK": 3,
    "RL": 3, "RM": 4, "RN": 4, "RO": 8, "RR": 2, "RS": 5, "RT": 5, "RU": 4,
    "RY": 3,
    "SA": 6, "SC": 6, "SE": 8, "SH": 8, "SI": 6, "SK": 2, "SL": 3, "SM": 3,
    "SN": 2, "SO": 6, "SP": 5, "SS": 5, "ST": 10, "SU": 5, "SW": 2,
    "TA": 7, "TC": 2, "TE": 9, "TH": 10, "TI": 8, "TL": 2, "TN": 1, "TO": 8,
    "TR": 6, "TS": 4, "TT": 4, "TU": 4, "TW": 2, "TY": 4, "TZ": 3,
    "UB": 2, "UC": 3, "UD": 3, "UE": 3, "UF": 3, "UG": 3, "UH": 2, "UL": 5,
    "UM": 4, "UN": 8, "UP": 3, "UR": 6, "US": 6, "UT": 5, "UZ": 1,
    "VA": 3, "VE": 7, "VI": 4, "VO": 4,
    "WA": 5, "WE": 6, "WH": 4, "WI": 5, "WN": 3, "WO": 4, "WR": 2,
    "XI": 1, "XP": 2, "XT": 2,
    "YA": 1, "YE": 3, "YO": 3, "YS": 2, "YT": 1,
    "ZA": 2, "ZE": 4, "ZI": 3, "ZU": 3, "ZW": 2,
}

# ──── Scoring configuration ────

DEFAULT_SLOP_WEIGHTS = {
    "ocr_conf": 0.20,
    "dictionary": 0.45,
    "bigram": 0.15,
    "repeat": 0.10,
    "empty_but_detected": 0.10,
    "vlm": 0.10,
}

SIGNAL_KEYS = tuple(DEFAULT_SLOP_WEIGHTS)

SLOP_THRESHOLD = 0.5
EMPTY_DETECTED_FLOOR = 0.7
UNKNOWN_SCORE = 0.5
MIN_TOKEN_LENGTH = 2
MIN_REPEAT_RUN = 3

# Umlauts are folded so the bigram table stays plain ASCII (str.upper()
# already turns eszett into "SS").
_FOLD_MAP = {
    "Ä": "AE", "Ö": "OE", "Ü": "UE",
    "Á": "A", "À": "A", "Â": "A", "É": "E", "È": "E", "Ê": "E",
    "Í": "I", "Ì": "I", "Î": "I", "Ó": "O", "Ò": "O", "Ô": "O",
    "Ú": "U", "Ù": "U", "Û": "U", "Ç": "C", "Ñ": "N",
}


def _fold(text):
    """Uppercase and fold accented latin characters to plain ASCII."""
    out = []
    for ch in (text or "").upper():
        out.append(_FOLD_MAP.get(ch, ch))
    return "".join(out)


def _tokens(text):
    """Split into alphanumeric tokens that are long enough to judge."""
    result = []
    for raw in _fold(text).split():
        cleaned = "".join(ch for ch in raw if ch.isalnum())
        if len(cleaned) < MIN_TOKEN_LENGTH:
            continue
        if not any(ch.isalpha() for ch in cleaned):
            continue
        result.append(cleaned)
    return result


def _clamp01(value):
    """Clamp a number into 0..1, mapping anything unusable to 0.0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))


def dictionary_ratio(text):
    """Fraction of eligible tokens found in WORD_LIST (0..1)."""
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in WORD_LIST)
    return hits / float(len(tokens))


def bigram_plausibility(text):
    """Mean normalised bigram weight over all letter pairs (0..1)."""
    weights = []
    for token in _tokens(text):
        letters = "".join(ch for ch in token if ch.isalpha())
        for i in range(len(letters) - 1):
            pair = letters[i:i + 2]
            weights.append(BIGRAM_FREQ.get(pair, 0))
    if not weights:
        return 0.0
    return _clamp01(sum(weights) / (len(weights) * MAX_BIGRAM_WEIGHT))


def repeated_glyph_ratio(text):
    """
    Share of characters caught in a stutter pattern (0..1).

    Counts runs of three or more identical characters plus immediately
    duplicated pairs such as ``ABAB``, which both betray tiled glyph noise.
    """
    chars = "".join(ch for ch in _fold(text) if not ch.isspace())
    if not chars:
        return 0.0

    covered = set()

    i = 0
    while i < len(chars):
        j = i
        while j + 1 < len(chars) and chars[j + 1] == chars[i]:
            j += 1
        if (j - i + 1) >= MIN_REPEAT_RUN:
            covered.update(range(i, j + 1))
        i = j + 1

    for k in range(len(chars) - 3):
        if chars[k] == chars[k + 1]:
            continue
        if chars[k:k + 2] == chars[k + 2:k + 4]:
            covered.update(range(k, k + 4))

    return len(covered) / float(len(chars))


def _effective_confidence(ocr_conf, char_confs):
    """Blend the region confidence with per-glyph confidences if available."""
    region = _clamp01(ocr_conf)
    values = []
    for value in (char_confs or []):
        if value is None:
            continue
        values.append(_clamp01(value))
    if not values:
        return region
    mean = sum(values) / float(len(values))
    char_conf = 0.5 * mean + 0.5 * min(values)
    return min(region, char_conf) if region > 0.0 else char_conf


def _resolve_weights(weights):
    """Merge user overrides over the defaults, ignoring unknown keys."""
    merged = dict(DEFAULT_SLOP_WEIGHTS)
    if isinstance(weights, dict):
        for key, value in weights.items():
            if key not in merged:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number < 0.0:
                continue
            merged[key] = number
    return merged


def _unknown_result():
    """Result shape used when there is nothing at all to judge."""
    return {
        "score": UNKNOWN_SCORE,
        "verdict": "unknown",
        "signals": {key: None for key in SIGNAL_KEYS},
    }


def score_slop(ocr_text="", ocr_conf=0.0, char_confs=None,
               text_region_detected=True, vlm_legible=None, weights=None):
    """
    Rate how likely a rendered text region is gibberish.

    Args:
        ocr_text: Text the OCR backend read back from the region.
        ocr_conf: Region level OCR confidence (0..1).
        char_confs: Optional per-character confidences (0..1).
        text_region_detected: True when SAM3 grounded a text region here.
        vlm_legible: Optional vision LLM legibility vote (bool or 0..1).
            None means "not asked" — the signal is dropped and the remaining
            weights are renormalised.
        weights: Optional overrides for DEFAULT_SLOP_WEIGHTS.

    Returns:
        Dict with "score" (0..1, 1 = definitely slop), "verdict"
        ("slop" | "clean" | "unknown") and the raw per-signal values.
    """
    text = ocr_text if isinstance(ocr_text, str) else ("" if ocr_text is None else str(ocr_text))
    has_text = bool(text.strip())
    detected = bool(text_region_detected)

    if not has_text and not detected and vlm_legible is None:
        return _unknown_result()

    signals = {key: None for key in SIGNAL_KEYS}
    signals["ocr_conf"] = 1.0 - _effective_confidence(ocr_conf, char_confs)

    if has_text:
        signals["dictionary"] = 1.0 - dictionary_ratio(text)
        signals["bigram"] = 1.0 - bigram_plausibility(text)
        signals["repeat"] = repeated_glyph_ratio(text)
        signals["empty_but_detected"] = 0.0
    else:
        signals["empty_but_detected"] = 1.0 if detected else 0.0

    if vlm_legible is not None:
        if isinstance(vlm_legible, bool):
            signals["vlm"] = 0.0 if vlm_legible else 1.0
        else:
            signals["vlm"] = 1.0 - _clamp01(vlm_legible)

    resolved = _resolve_weights(weights)
    total = 0.0
    accumulated = 0.0
    for key, value in signals.items():
        if value is None:
            continue
        weight = resolved.get(key, 0.0)
        total += weight
        accumulated += weight * _clamp01(value)

    score = accumulated / total if total > 0.0 else UNKNOWN_SCORE
    if signals["empty_but_detected"] == 1.0:
        score = max(score, EMPTY_DETECTED_FLOOR)
    score = _clamp01(score)

    verdict = "slop" if score >= SLOP_THRESHOLD else "clean"
    return {"score": score, "verdict": verdict, "signals": signals}
