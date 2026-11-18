"""gh-pages has one writer per file: a rig's publish commits only its own paths.

If a rig also committed the top-level pages it happened to build locally, it could
push a Scores page built from CSVs master doesn't have, and the CI rebuild would
then fight it. These tests use real git against a local bare remote."""
import importlib
import subprocess

publish = importlib.import_module("publish")


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def _setup(tmp_path, monkeypatch):
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(k, v)
    origin = tmp_path / "origin.git"
    wt = tmp_path / "wt"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(tmp_path, "init", "-q", "-b", publish.BRANCH, str(wt))
    (wt / "index.html").write_text("built by CI")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "ci pages")
    _git(wt, "remote", "add", "origin", str(origin))
    _git(wt, "push", "-q", "origin", publish.BRANCH)

    def sync():
        _git(wt, "fetch", "-q", "origin", publish.BRANCH)
        _git(wt, "reset", "-q", "--hard", f"origin/{publish.BRANCH}")

    monkeypatch.setattr(publish, "WORKTREE", wt)
    monkeypatch.setattr(publish, "_sync_worktree", sync)
    monkeypatch.setattr(publish, "_branch_exists_remote", lambda b: True)
    return origin, wt


def _remote_file(origin, path):
    return _git(origin, "show", f"{publish.BRANCH}:{path}")


def test_rig_publish_commits_only_its_run_dir(tmp_path, monkeypatch):
    origin, wt = _setup(tmp_path, monkeypatch)

    def stage():
        (wt / "index.html").write_text("built by a rig from local CSVs")  # must NOT ship
        (wt / "linux-default").mkdir(exist_ok=True)
        (wt / "linux-default" / "results.csv").write_text("rows")

    assert publish._commit_and_push(["linux-default"], "Publish bench run linux-default", stage)
    assert _remote_file(origin, "linux-default/results.csv") == "rows"
    assert _remote_file(origin, "index.html") == "built by CI"


def test_rejected_push_resyncs_and_keeps_the_other_writers_files(tmp_path, monkeypatch):
    origin, wt = _setup(tmp_path, monkeypatch)
    other = tmp_path / "other"  # CI pushing new landing pages mid-publish
    _git(tmp_path, "clone", "-q", "-b", publish.BRANCH, str(origin), str(other))
    pushed = {"done": False}

    def stage():
        if not pushed["done"]:  # land CI's push after this attempt synced
            (other / "index.html").write_text("newer CI build")
            _git(other, "commit", "-q", "-am", "ci rebuild")
            _git(other, "push", "-q", "origin", publish.BRANCH)
            pushed["done"] = True
        (wt / "windows-default").mkdir(exist_ok=True)
        (wt / "windows-default" / "results.csv").write_text("rows")

    assert publish._commit_and_push(["windows-default"], "Publish bench run windows-default", stage)
    assert _remote_file(origin, "windows-default/results.csv") == "rows"
    assert _remote_file(origin, "index.html") == "newer CI build"


def test_no_changes_commits_nothing(tmp_path, monkeypatch):
    origin, wt = _setup(tmp_path, monkeypatch)
    before = _git(origin, "rev-parse", publish.BRANCH)
    assert publish._commit_and_push(["linux-default"], "noop", lambda: None) is False
    assert _git(origin, "rev-parse", publish.BRANCH) == before
