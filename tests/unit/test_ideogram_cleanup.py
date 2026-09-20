"""The rules of the cleanup loop, which are the ones that can destroy a picture.

Every other step here can only fail to improve the image. This one actively
repaints areas, so its two judgements have to be exactly right:

* **What counts as slop.** One planned region carries one line of text. The line
  that belongs is the biggest one in it — the name is always set larger than the
  decoration the model added around it. Get this backwards and the loop erases
  the word and keeps the decoration.
* **What counts as a line at all.** Run the detector at 3x and it reports
  speckles of two by two pixels as text, dozens of them inside one lit shop
  window. Emptying those would repaint a window full of picture frames for
  nothing — that is how a tidying loop starts destroying the picture.
"""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "live", "ideogram"),
)
cleanup = pytest.importorskip("cleanup")


class TestBigEnough:
    def test_a_detector_speckle_is_not_a_line(self):
        assert not cleanup.big_enough((1035, 349, 1037, 351))

    def test_a_real_small_line_is(self):
        assert cleanup.big_enough((100, 200, 190, 212))

    def test_thin_but_long_still_counts(self):
        """Small print under a name is exactly this shape and is the thing the
        loop exists to find — the filter must not throw it away."""
        assert cleanup.big_enough((100, 200, 160, 206))

    def test_short_but_tall_does_not(self):
        assert not cleanup.big_enough((100, 200, 108, 240))


class TestStrayRule:
    """`stray_lines` needs a detector, so the rule is tested through its parts."""

    def test_share_inside_is_computed_on_the_line_not_the_region(self):
        line = (100, 100, 120, 110)
        region = (0, 0, 1000, 1000)
        assert cleanup._iou_inside(line, region) == pytest.approx(1.0)
        # And the other way round it is nearly nothing — which is why the check
        # is directional. Using the symmetric IoU here would file every line as
        # "not really inside" and the loop would erase the words themselves.
        assert cleanup._iou_inside(region, line) < 0.01

    def test_a_line_straddling_a_border_is_not_inside(self):
        assert cleanup._iou_inside(
            (90, 100, 110, 110), (100, 100, 200, 200)
        ) == pytest.approx(0.5)
        assert 0.5 < cleanup.INSIDE

    def test_disjoint(self):
        assert cleanup._iou_inside((0, 0, 10, 10), (500, 500, 600, 600)) == 0.0


class TestSmallRatio:
    """The rule that decides whether a second line is decoration or half a word.

    The first version kept the tallest line and emptied every other one. The
    numbers approved of it — the detector count fell and the loop declared
    convergence — while the picture got worse: a partly-occluded `TA` of TABAK
    was erased and replaced with a yellow smudge. On the noticeboard, where every
    sheet carries a two-line word, it would have emptied one half of each.
    """

    def test_a_second_line_of_the_same_size_is_kept(self):
        """`KAFFEE` over `KASSE` — same word, two lines, both must survive."""
        keep_h, other_h = 40, 38
        assert other_h >= cleanup.SMALL_RATIO * keep_h

    def test_a_small_decoration_line_is_removed(self):
        """The `GERLALCL` under a variety name is a third the height."""
        keep_h, other_h = 40, 12
        assert other_h < cleanup.SMALL_RATIO * keep_h

    def test_the_threshold_leaves_room_for_a_smaller_second_line(self):
        """A two-line setting is often not perfectly even — `KUEHL` over
        `SCHRANK` sets the longer word smaller. The cut must sit below that."""
        assert 0.5 <= cleanup.SMALL_RATIO <= 0.75

    def test_height_not_area_decides(self):
        """A long thin line of small print out-AREAS a short tall word.

        `PERLE` at 90x40 is 3600 px; a line of small print under it at 200x10 is
        2000 — but widen the label and the small print wins on area while losing
        on height. Ranked by area the loop picks the decoration as the main text
        and then empties the name.
        """
        wort = (100, 100, 190, 140)  # 90 breit, 40 hoch  -> 3600 px
        kleindruck = (100, 150, 500, 162)  # 400 breit, 12 hoch -> 4800 px
        assert (kleindruck[2] - kleindruck[0]) * (kleindruck[3] - kleindruck[1]) > (
            wort[2] - wort[0]
        ) * (wort[3] - wort[1])
        assert cleanup.pick_main([wort, kleindruck]) == 0
        assert cleanup.pick_main([kleindruck, wort]) == 1

    def test_identical_boxes_do_not_confuse_the_choice(self):
        """Two detections can carry the same coordinates; picking by identity
        would then spare the wrong one — the word instead of the decoration."""
        gleich = (10, 10, 90, 34)  # 80x24, klar zeilenfoermig
        assert cleanup.pick_main([gleich, gleich, (10, 60, 90, 68)]) in (0, 1)

    def test_a_crest_is_not_the_main_line(self):
        """The failure that survived three attempted fixes: on a wine label the
        ornamental crest is TALLER than the name (29 px against 14). Ranked by
        height alone it was kept as the main line and the NAME was emptied —
        `PHOENIX` came back as `PXLGES`, then `RELGES`, then `PELGES`, and none
        of the three fixes touched the cause."""
        wappen = (340, 470, 371, 500)  # 31x29, Verhaeltnis 1,07:1
        name = (335, 507, 396, 521)  # 61x14, Verhaeltnis 4,4:1
        klein = (336, 522, 392, 528)  # 56x6
        assert not cleanup.line_like(wappen)
        assert cleanup.line_like(name) and cleanup.line_like(klein)
        assert cleanup.pick_main([wappen, name, klein]) == 1

    def test_a_mark_is_never_emptied_either(self):
        """A crest is not text, so it is neither the main line nor slop."""
        assert not cleanup.line_like((340, 470, 371, 500))


class TestGrowBox:
    def test_it_grows(self):
        got = cleanup.grow_box((100, 100, 200, 120), (768, 1280))
        assert got[0] < 100 and got[1] < 100 and got[2] > 200 and got[3] > 120

    def test_it_grows_by_the_line_height_not_a_constant(self):
        """A tall line needs a wider collar than a short one; a fixed pad either
        under-covers big lettering or swamps small print."""
        small = cleanup.grow_box((100, 100, 200, 108), (768, 1280))
        big = cleanup.grow_box((100, 100, 200, 160), (768, 1280))
        assert (big[2] - 200) > (small[2] - 200)

    def test_it_never_leaves_the_frame(self):
        got = cleanup.grow_box((0, 0, 40, 20), (768, 1280))
        assert got[0] >= 0 and got[1] >= 0
        got = cleanup.grow_box((1240, 740, 1280, 768), (768, 1280))
        assert got[2] <= 1280 and got[3] <= 768

    def test_it_never_grows_into_a_line_being_kept(self):
        """The failure this guard exists for: small print sits two pixels under
        the name it decorates. Grown by its own height it reaches into the name,
        the blank element repaints that too, and a cleanup run turned `PHOENIX`
        into `PXLGES` and `HELIOS` into `IFGLE` while its counter reported
        success."""
        name = (100, 100, 200, 124)
        klein = (100, 126, 210, 134)
        ohne = cleanup.grow_box(klein, (768, 1280))
        mit = cleanup.grow_box(klein, (768, 1280), keep=[name])
        assert ohne[1] < name[3], "Gegenprobe: ungeschont muss es hineingreifen"
        assert mit[1] >= name[3]

    def test_it_clips_from_above_as_well(self):
        name = (100, 100, 200, 124)
        oben = (100, 88, 210, 96)
        assert cleanup.grow_box(oben, (768, 1280), keep=[name])[3] <= name[1]

    def test_a_line_beside_the_kept_one_is_not_clipped(self):
        """Only a vertical overlap matters — a find to the SIDE of the name may
        grow freely, and clipping it would leave its ink uncovered."""
        name = (100, 100, 200, 124)
        daneben = (400, 110, 500, 118)
        frei = cleanup.grow_box(daneben, (768, 1280))
        assert cleanup.grow_box(daneben, (768, 1280), keep=[name]) == frei

    def test_there_is_always_some_collar(self):
        """Even a one-pixel-high find gets a real margin: ink that pokes out of
        the mask survives bit-identically, which is the coverage law."""
        got = cleanup.grow_box((100, 100, 200, 101), (768, 1280))
        assert (100 - got[0]) >= 3


class TestScaleSweep:
    def test_more_than_one_scale_is_used(self):
        """A single-scale pass reported 10 lines where three scales found 16.
        The small print never reaches the threshold at native size, so a
        one-scale loop reports a clean picture and leaves the decoration."""
        assert len(cleanup.DB_SCALES) >= 2
        assert max(s for s, _ in cleanup.DB_SCALES) >= 2.0

    def test_the_thresholds_are_not_loosened_into_noise(self):
        """3x at box_thresh 0.4 produced dozens of two-pixel speckles."""
        assert all(t >= 0.45 for _, t in cleanup.DB_SCALES)
