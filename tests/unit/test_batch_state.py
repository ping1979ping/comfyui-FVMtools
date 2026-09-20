"""Progress bookkeeping — the part that has to survive files moving away."""

import json
import os

import pytest

from core.batch_state import (
    STATE_FILENAME, clear_done, get_done, list_images, next_file, progress,
    save_state, set_done,
)


@pytest.fixture
def folder(tmp_path):
    """A directory with five images and two files that are not images."""
    for name in ("b.jpg", "a.png", "c.jpeg", "d.webp", "e.PNG"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "sidecar.json").write_text("{}")
    return str(tmp_path)


class TestListing:
    def test_only_images(self, folder):
        assert list_images(folder) == ["a.png", "b.jpg", "c.jpeg", "d.webp", "e.PNG"]

    def test_case_insensitive_extension(self, folder):
        assert "e.PNG" in list_images(folder)

    def test_missing_directory_is_empty(self, tmp_path):
        assert list_images(str(tmp_path / "nope")) == []

    def test_state_file_is_not_an_image(self, folder):
        set_done(folder, {"a.png"})
        assert STATE_FILENAME not in list_images(folder)

    def test_sort_by_modified(self, folder):
        os.utime(os.path.join(folder, "e.PNG"), (1, 1))
        assert list_images(folder, sort_by="modified")[0] == "e.PNG"

    def test_subdirs_walked_only_when_asked(self, folder):
        os.makedirs(os.path.join(folder, "keep"))
        (open(os.path.join(folder, "keep", "moved.jpg"), "wb")).write(b"x")
        assert "keep/moved.jpg" not in list_images(folder)
        assert "keep/moved.jpg" in list_images(folder, include_subdirs=True)

    def test_dot_folders_always_skipped(self, folder):
        os.makedirs(os.path.join(folder, ".cache"))
        (open(os.path.join(folder, ".cache", "hidden.jpg"), "wb")).write(b"x")
        assert not any("hidden" in name
                       for name in list_images(folder, include_subdirs=True))


class TestState:
    def test_roundtrip(self, folder):
        set_done(folder, {"a.png", "b.jpg"})
        assert get_done(folder) == {"a.png", "b.jpg"}

    def test_trackers_are_independent(self, folder):
        set_done(folder, {"a.png"}, tracker="one")
        set_done(folder, {"b.jpg"}, tracker="two")
        assert get_done(folder, "one") == {"a.png"}
        assert get_done(folder, "two") == {"b.jpg"}

    def test_clear_leaves_other_trackers(self, folder):
        set_done(folder, {"a.png"}, tracker="one")
        set_done(folder, {"b.jpg"}, tracker="two")
        clear_done(folder, "one")
        assert get_done(folder, "one") == set()
        assert get_done(folder, "two") == {"b.jpg"}

    def test_clearing_last_tracker_removes_the_file(self, folder):
        set_done(folder, {"a.png"})
        clear_done(folder)
        assert not os.path.exists(os.path.join(folder, STATE_FILENAME))

    def test_corrupt_state_reads_as_empty(self, folder):
        with open(os.path.join(folder, STATE_FILENAME), "w") as handle:
            handle.write("{not json")
        assert get_done(folder) == set()

    def test_write_is_atomic_no_temp_left_behind(self, folder):
        save_state(folder, {"default": {"done": ["a.png"]}})
        assert not [n for n in os.listdir(folder) if n.endswith(".tmp")]

    def test_state_file_is_valid_json(self, folder):
        set_done(folder, {"a.png"}, extra={"last": "a.png"})
        with open(os.path.join(folder, STATE_FILENAME), encoding="utf-8") as handle:
            data = json.load(handle)
        assert data["default"]["done"] == ["a.png"]
        assert data["default"]["last"] == "a.png"


class TestNextFile:
    def test_hands_out_in_order(self, folder):
        done = set()
        seen = []
        for _ in range(5):
            name, done, _wrapped = next_file(folder, done)
            seen.append(name)
            done = done | {name}
        assert seen == ["a.png", "b.jpg", "c.jpeg", "d.webp", "e.PNG"]

    def test_returns_none_when_exhausted(self, folder):
        name, _done, _wrapped = next_file(folder, set(list_images(folder)))
        assert name is None

    def test_loop_restarts_and_reports_it(self, folder):
        name, done, wrapped = next_file(folder, set(list_images(folder)), loop=True)
        assert name == "a.png"
        assert wrapped is True
        assert done == set()

    def test_loop_on_empty_folder_still_stops(self, tmp_path):
        name, _done, wrapped = next_file(str(tmp_path), set(), loop=True)
        assert name is None and wrapped is False

    def test_survives_files_moving_away(self, folder):
        """The move-mode case: handing out must not skip when the listing shrinks.

        A counter-based loader breaks here — after the first file leaves, index 1
        points at the third picture.
        """
        done, seen = set(), []
        for _ in range(5):
            name, done, _wrapped = next_file(folder, done)
            if name is None:
                break
            seen.append(name)
            done = done | {name}
            os.remove(os.path.join(folder, name))      # the Save node moved it
        assert seen == ["a.png", "b.jpg", "c.jpeg", "d.webp", "e.PNG"]


class TestProgress:
    def test_counts_what_was_handed_out(self, folder):
        assert progress(folder, set())[0:2] == (0, 5)
        assert progress(folder, {"a.png", "b.jpg"})[0:2] == (2, 5)

    def test_total_holds_when_files_move_away(self, folder):
        os.remove(os.path.join(folder, "a.png"))
        processed, total, remaining = progress(folder, {"a.png"})
        assert total == 5, "a moved file still belongs to the batch"
        assert remaining == 4
        assert processed == 1

    def test_finished_reports_full(self, folder):
        position, total, remaining = progress(folder, set(list_images(folder)))
        assert (position, total, remaining) == (5, 5, 0)
