"""Erweitere den bestehenden Song-Prompt um die Formvorgaben eines Covers.

Der Music-Production-Toolkit-Zweig baut schon einen System- und einen
User-Prompt (MiniMaxStructuredPromptV20) und laesst LM Studio daraus vier
Sektionen schreiben: [Style], [Lyrics], [Title], [Image_Prompt]. Dieses
Ausgabeformat parst MiniMaxParseExternalLLMOutputV16 weiter — es darf also
NICHT angetastet werden.

Dieser Node haengt sich dazwischen: Er nimmt beide Prompts, fuegt die gemessene
Form des Referenztexts (FVM_LyricsProsodyAnalyze) und den gewuenschten Flavor
hinzu und gibt beide Prompts erweitert zurueck. Ein LLM-Aufruf, ein Ausgabe-
format, nur mehr Vorgaben.

Drei Flavor-Arten, weil der Wunsch unterschiedlich konkret sein kann:
  thema    — ein Thema ("Abschied am Bahnhof"), das Modell erfindet die Bilder
  begriffe — Begriffe/Reizwoerter, die vorkommen sollen
  umdichten— ein vorhandener Text, der auf Form und Reimschema gebracht wird

Bewusst im System-Prompt verankert: Der neue Text darf keine Formulierungen des
Originals uebernehmen. Die Form wird nachgebaut, nicht der Text kopiert — das
ist die Grenze zwischen Cover und Textklau, und ein LLM haelt sie nur ein, wenn
man es ihm sagt.
"""

FLAVOR_ARTEN = ["thema", "begriffe", "umdichten"]

STRENGE_TOLERANZ = {"streng": 0, "nah": 1, "frei": 2}

KOPFZEILE = "COVER FORM TARGET (measured from the reference recording)"

REGELN = """LYRIC FORM RULES (these override generic lyric advice above)
- Keep the section order and the number of lines per section exactly as listed.
- Match the syllable count of every line within +/-{toleranz} syllables.
- Reproduce the rhyme scheme per section (equal letters must rhyme, different letters must not).
- Write singable phrasing: stresses on the strong beats, no crammed consonant clusters.
- Write an ORIGINAL text. Do not reuse lines, phrases or distinctive wording from the reference
  lyrics; the form is the target, the words are yours. Do not translate the original either.
- Keep the language of the new lyrics: {sprache}.
- Keep the four-section output contract ([Style], [Lyrics], [Title], [Image_Prompt]) unchanged.
- Count syllables as they are sung. If a line does not fit, rewrite it rather than padding with
  filler words such as oh, yeah or na."""

FLAVOR_TEXTE = {
    "thema": "NEW LYRIC CONTENT — theme to write about:\n{text}",
    "begriffe": (
        "NEW LYRIC CONTENT — work these terms/images in naturally "
        "(not as a list, and not all in one line):\n{text}"
    ),
    "umdichten": (
        "NEW LYRIC CONTENT — rewrite the following text so it fits the measured form "
        "above. Keep its meaning and imagery, change wording and line breaks as needed:\n{text}"
    ),
}


def baue_bloecke(
    prosody_brief: str,
    flavor_art: str,
    flavor_text: str,
    sprache: str,
    strenge: str,
    referenztext: str = "",
    referenz_mitgeben: bool = False,
) -> str:
    """Der Block, der an den User-Prompt angehaengt wird."""
    toleranz = STRENGE_TOLERANZ.get(strenge, 1)
    teile = [
        KOPFZEILE,
        (prosody_brief or "").strip(),
        "",
        REGELN.format(toleranz=toleranz, sprache=sprache or "same as the reference"),
    ]
    vorlage = FLAVOR_TEXTE.get(flavor_art)
    if vorlage and (flavor_text or "").strip():
        teile += ["", vorlage.format(text=flavor_text.strip())]
    if referenz_mitgeben and (referenztext or "").strip():
        teile += [
            "",
            "REFERENCE LYRICS (for form and stress positions only — never copy wording):",
            referenztext.strip(),
        ]
    return "\n".join(teile)


def system_zusatz(strenge: str) -> str:
    """Kurzer Zusatz fuer den System-Prompt, damit die Regeln Gewicht haben."""
    toleranz = STRENGE_TOLERANZ.get(strenge, 1)
    return (
        "\nCOVER MODE\n"
        "The user supplies a measured form target (sections, lines, syllables per line, rhyme "
        "scheme) taken from an existing recording. Treat it as a hard constraint for [Lyrics]: "
        f"same section order, same line count, syllables within +/-{toleranz}, same rhyme scheme. "
        "Write original words — never reuse phrases from the reference lyrics and never translate "
        "them. Style, Title and Image_Prompt keep their normal task, and the four-section output "
        "contract stays exactly as specified above."
    )


class FVM_CoverLyricsPromptBuilder:
    """System- und User-Prompt um Cover-Formvorgaben erweitern."""

    CATEGORY = "FVMtools/music"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": ("STRING", {"forceInput": True}),
                "system_prompt": ("STRING", {"forceInput": True}),
                "prosody_brief": ("STRING", {"forceInput": True}),
                "flavor_art": (FLAVOR_ARTEN, {"default": "thema"}),
                "flavor_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Thema, Begriffe oder der umzudichtende Text.",
                    },
                ),
                "strenge": (
                    list(STRENGE_TOLERANZ),
                    {
                        "default": "nah",
                        "tooltip": "Silben-Toleranz pro Zeile: streng 0, nah 1, frei 2.",
                    },
                ),
                "referenz_mitgeben": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Originaltext in den Prompt legen. Genauere Betonungen, "
                        "aber das Modell neigt dann zum Abschreiben.",
                    },
                ),
            },
            "optional": {
                "language": ("STRING", {"forceInput": True}),
                "reference_lyrics": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("user_prompt", "system_prompt", "info")
    FUNCTION = "execute"

    def execute(
        self,
        user_prompt,
        system_prompt,
        prosody_brief,
        flavor_art,
        flavor_text,
        strenge,
        referenz_mitgeben,
        language="",
        reference_lyrics="",
    ):
        block = baue_bloecke(
            prosody_brief,
            flavor_art,
            flavor_text,
            language,
            strenge,
            reference_lyrics,
            referenz_mitgeben,
        )
        neuer_user = f"{(user_prompt or '').rstrip()}\n\n{block}\n"
        neuer_system = f"{(system_prompt or '').rstrip()}\n{system_zusatz(strenge)}\n"
        info = (
            f"Flavor: {flavor_art} ({len((flavor_text or '').split())} Woerter) | "
            f"Strenge: {strenge} (+/-{STRENGE_TOLERANZ.get(strenge, 1)} Silben) | "
            f"Referenztext im Prompt: {'ja' if referenz_mitgeben else 'nein'} | "
            f"User-Prompt: {len(neuer_user)} Zeichen, System-Prompt: {len(neuer_system)} Zeichen"
        )
        return (neuer_user, neuer_system, info)
