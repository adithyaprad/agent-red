"""Locating the data files the repository carries, from wherever the package was installed."""

from __future__ import annotations

import pytest

from agentred.paths import ROOT_ENV_VAR, candidate_roots, repo_path


class TestCandidateRoots:
    def test_the_package_tree_and_the_working_directory_are_both_tried(self):
        roots = candidate_roots({})
        assert len(roots) == 2
        assert (roots[0] / "src" / "agentred" / "paths.py").exists()

    def test_an_override_is_tried_first(self, tmp_path):
        roots = candidate_roots({ROOT_ENV_VAR: str(tmp_path)})
        assert roots[0] == tmp_path

    def test_a_blank_override_is_not_a_location(self, tmp_path):
        assert candidate_roots({ROOT_ENV_VAR: "   "}) == candidate_roots({})


class TestRepoPath:
    def test_the_shop_and_the_corpus_are_found_from_the_package_tree(self):
        assert (repo_path("data", "store") / "orders.json").is_file()
        assert list(repo_path("data", "techniques").glob("*.yaml"))

    def test_an_override_wins_over_the_package_tree(self, tmp_path):
        (tmp_path / "data" / "store").mkdir(parents=True)
        found = repo_path("data", "store", env={ROOT_ENV_VAR: str(tmp_path)})
        assert found == tmp_path / "data" / "store"

    def test_a_location_that_holds_nothing_falls_through_to_one_that_does(self, tmp_path):
        assert repo_path("data", "store", env={ROOT_ENV_VAR: str(tmp_path)}).is_dir()

    def test_nothing_anywhere_names_every_place_it_looked(self, tmp_path):
        with pytest.raises(FileNotFoundError) as raised:
            repo_path("data", "not_a_directory", env={ROOT_ENV_VAR: str(tmp_path)})
        message = str(raised.value)
        assert str(tmp_path) in message
        assert ROOT_ENV_VAR in message

    def test_it_refuses_rather_than_returning_a_path_that_does_not_exist(self, tmp_path):
        """A returned path is one that resolves, so no caller has to check.

        The alternative is a tool server that starts against a shop it could not find, in
        which every declared rule reports as never in play and the run reads as a clean sheet.
        """
        with pytest.raises(FileNotFoundError):
            repo_path("nothing_here", env={ROOT_ENV_VAR: str(tmp_path)})
