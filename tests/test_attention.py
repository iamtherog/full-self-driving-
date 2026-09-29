"""Tests for the attention monitor: the maths, the statistics, and the safety boundary."""

import json
import math
import random
import re

import pytest

from autodrive import Config, scenarios
from autodrive.attention import AttentionModel, AttentionMonitor, KneserNey, TokenSpec
from autodrive.attention import corpus as corpus_module
from autodrive.attention.corpus import Recorder, nominal_scenario
from autodrive.attention.evaluate import score
from autodrive.attention.events import MODE_SEVERITY, events
from autodrive.attention.kneser_ney import BOS, EOS, UNK, estimate_discounts
from autodrive.attention.model import ArtifactError
from autodrive.attention.monitor import PREFIX, SEVERITY, TEMPLATES
from autodrive.attention.sprt import SPRT
from autodrive.attention.training import Plan, surprisals, threshold_at, train
from autodrive.observe import Snapshot
from autodrive.sim import run


@pytest.fixture(scope="module")
def shipped() -> AttentionModel:
    return AttentionModel.load()


def snap(t, mode="engaged", aeb=False, alert=None, behavior="cruise", accel=0.0,
         lateral=0.0, ttc=math.inf, age=0.0):
    return Snapshot(t=t, mode=mode, aeb=aeb, alert=alert, behavior=behavior, speed=20.0,
                    accel=accel, lane_lateral=lateral, ttc=ttc, perception_age=age)


# --- Kneser-Ney ---------------------------------------------------------------

def test_probabilities_match_a_hand_calculation():
    # Corpus "<s> a b </s>", order 2. Every count is 1, so counts-of-counts are
    # too sparse to estimate and all discounts take the fallback (0.5, 0.75, 0.9).
    # |V| = {a, b, </s>, <unk>} = 4.
    #   Unigram continuation counts a, b, </s> = 1 each, total 3, gamma = 3 * 0.5:
    #     P1(a)     = (1 - 0.5 + 1.5 * 1/4) / 3 = 0.2916...
    #     P1(<unk>) = (0       + 1.5 * 1/4) / 3 = 0.125
    #   Bigram context "a" has one successor, b, count 1, gamma = 0.5:
    #     P2(b | a) = (1 - 0.5 + 0.5 * P1(b)) / 1 = 0.6458...
    #     P2(a | a) = (0       + 0.5 * P1(a)) / 1 = 0.1458...
    lm = KneserNey.fit([["a", "b"]], order=2)
    p1 = (0.5 + 1.5 / 4) / 3
    assert lm.prob(["<unk>-context"], "a") == pytest.approx(p1)      # unseen context -> unigram
    assert lm.prob(["zzz"], "never-seen") == pytest.approx(0.125)
    assert lm.prob(["a"], "b") == pytest.approx(0.5 + 0.5 * p1)
    assert lm.prob(["a"], "a") == pytest.approx(0.5 * p1)


def test_discounts_follow_chen_goodman_equation_26():
    counts = [1] * 10 + [2] * 5 + [3] * 3 + [4] * 2
    d = estimate_discounts(counts)
    y = 10 / (10 + 2 * 5)
    assert d.d1 == pytest.approx(1 - 2 * y * 5 / 10)
    assert d.d2 == pytest.approx(2 - 3 * y * 3 / 5)
    assert d.d3 == pytest.approx(3 - 4 * y * 2 / 3)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_every_context_yields_a_strictly_positive_distribution_summing_to_one(order):
    rng = random.Random(order)
    corpus = [[rng.choice("abcde") for _ in range(rng.randint(1, 30))] for _ in range(40)]
    lm = KneserNey.fit(corpus, order=order)
    contexts = [list(g[:-1]) for g in lm.counts]                         # every seen context
    contexts += [[rng.choice("abcdeXY") for _ in range(order)] for _ in range(50)]   # unseen, OOV
    contexts += [[], [BOS]]
    for context in contexts:
        probs = [lm.prob(context, w) for w in lm.vocabulary]
        assert math.fsum(probs) == pytest.approx(1.0, abs=1e-12)
        assert min(probs) > 0.0


def test_continuation_counts_not_frequency_drive_the_lower_order():
    # "b" is frequent but only ever follows "a"; "c" is rare but follows many
    # tokens. After an unseen context, Kneser-Ney must prefer "c": that is the
    # whole point of continuation counts over raw frequency.
    corpus = [["a", "b"]] * 50 + [[x, "c"] for x in "defgh"]
    lm = KneserNey.fit(corpus, order=2)
    assert lm.prob(["never-seen"], "c") > lm.prob(["never-seen"], "b")


def test_unknown_tokens_are_surprising_but_never_impossible():
    lm = KneserNey.fit([["a", "b", "a", "b"]], order=3)
    assert math.isfinite(lm.surprisal(["a", "b"], "totally-new"))
    assert lm.surprisal(["a", "b"], "totally-new") > lm.surprisal(["a", "b"], "a")


def test_serialisation_round_trip_is_exact():
    rng = random.Random(7)
    lm = KneserNey.fit([[rng.choice("wxyz") for _ in range(50)] for _ in range(10)], order=3)
    copy = KneserNey.from_dict(json.loads(json.dumps(lm.to_dict())))
    for context in (["w", "x"], ["z"], [], ["q", "q"]):
        for w in lm.vocabulary:
            assert copy.prob(context, w) == lm.prob(context, w)


def test_derived_statistics_do_not_depend_on_count_order():
    rng = random.Random(3)
    lm = KneserNey.fit([[rng.choice("pqrst") for _ in range(80)] for _ in range(20)], order=4)
    items = list(lm.counts.items())
    rng.shuffle(items)
    shuffled = KneserNey(4, dict(items))
    for context in (["p", "q", "r"], ["t"], []):
        for w in lm.vocabulary:
            assert shuffled.prob(context, w) == lm.prob(context, w)     # bit for bit, not approx


@pytest.mark.parametrize("bad", [BOS, EOS, UNK])
def test_reserved_symbols_are_rejected_in_training_data(bad):
    with pytest.raises(ValueError):
        KneserNey.fit([["a", bad]], order=2)


# --- Sequential test -----------------------------------------------------------

def test_sprt_stops_exactly_when_the_likelihood_ratio_crosses_a_boundary():
    test = SPRT()   # p0 = 0.5, p1 = 0.8, alpha = beta = 0.01
    wins = math.ceil(test.upper / math.log(0.8 / 0.5))
    losses = math.ceil(test.lower / math.log(0.2 / 0.5))
    assert test.run([True] * 100).trials == wins == 10
    assert test.run([False] * 100).trials == losses == 6
    assert test.run([True, False] * 3).verdict == "undecided"
    assert test.run([None] * 50 + [True] * 10).trials == 10    # ties carry no evidence


@pytest.mark.parametrize(("p_true", "verdict", "bound"), [(0.5, "accept", 0.01 / 0.99),
                                                           (0.8, "reject", 0.01 / 0.99)])
def test_sprt_error_rates_respect_walds_bounds(p_true, verdict, bound):
    rng = random.Random(1945)
    runs = 4000
    wrong = sum(SPRT().run(rng.random() < p_true for _ in range(1000)).verdict == verdict
                for _ in range(runs))
    # Three standard errors of Monte Carlo slack on top of Wald's bound.
    assert wrong / runs <= bound + 3 * math.sqrt(bound * (1 - bound) / runs)


# --- Events --------------------------------------------------------------------

def test_windows_are_aligned_and_a_trailing_partial_window_is_dropped():
    stream = [snap(k * 0.01) for k in range(175)]           # 1.75 s at 100 Hz
    out = list(events(stream, TokenSpec()))
    assert [(e.t_start, e.t_end) for e in out] == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)]


def test_a_single_step_of_trouble_marks_its_whole_window():
    stream = [snap(k * 0.01) for k in range(100)]
    stream[20] = snap(0.20, mode="fault", aeb=True, accel=-6.5, ttc=1.1, lateral=0.5, age=0.3)
    first = next(iter(events(stream, TokenSpec())))
    f = first.features
    banded = (f.mode, f.aeb, f.stale, f.accel_band, f.ttc_band, f.lateral_band)
    assert banded == ("fault", True, True, 0, 0, 2)
    assert first.token == "fault+aeb|cruise|a0|t0|l2|stale"


def test_token_spec_matches_the_safety_limits_it_mirrors():
    spec, safety = TokenSpec(), Config().safety
    assert spec.ttc_edges[0] == safety.aeb_ttc
    assert spec.stale_after == safety.sensor_timeout
    assert set(MODE_SEVERITY) == {"off", "engaged", "lat_override", "fault"}


def test_any_change_to_the_token_spec_changes_its_digest():
    base = TokenSpec().digest()
    assert TokenSpec(window=0.25).digest() != base
    assert TokenSpec(ttc_edges=(1.5, 3.0, 6.0)).digest() != base


# --- Training pieces -----------------------------------------------------------

def test_threshold_respects_the_flag_budget_even_with_ties():
    values = [1.0] * 900 + [5.0] * 90 + [9.0] * 10
    threshold = threshold_at(values, 0.99)                      # budget: 10 of 1000
    assert sum(v >= threshold for v in values) <= 10
    assert threshold_at(values, 0.999) > 9.0                    # budget of 1 cannot be met by a tie of 10


def test_plan_rejects_invalid_settings():
    with pytest.raises(ValueError):
        Plan(fractions=(0.5, 0.5, 0.0))
    with pytest.raises(ValueError):
        Plan(orders=(3, 2))


# --- The safety boundary --------------------------------------------------------

@pytest.mark.parametrize("name", list(scenarios.ALL))
def test_the_monitor_cannot_change_how_the_car_drives(name, shipped):
    plain = run(scenarios.ALL[name]())
    watched = run(scenarios.ALL[name](), observer=AttentionMonitor(shipped))
    for key, values in plain.log.items():
        if key != "advisory":
            assert watched.log[key] == values, key


def test_a_crashing_observer_is_detached_and_the_drive_continues_unchanged():
    class Broken:
        def observe(self, snapshot):
            if snapshot.t > 5.0:
                raise RuntimeError("boom")

    plain = run(scenarios.hard_brake())
    broken = run(scenarios.hard_brake(), observer=Broken())
    assert broken.observer_error == "RuntimeError: boom"
    assert broken.log["speed"] == plain.log["speed"]
    assert broken.passed


def test_online_scores_equal_offline_scores(shipped):
    # The same drive, scored by the live monitor and by the training code, must
    # agree exactly. Any difference would be train/serve skew.
    recorder, monitor = Recorder(), AttentionMonitor(shipped)
    run(nominal_scenario(12345), observer=recorder)
    for s in recorder.snapshots:
        monitor.observe(s)
    tokens = [e.token for e in events(recorder.snapshots, shipped.spec)]
    assert [bits for _, bits in monitor.scores] == surprisals(shipped.lm, tokens)


def test_every_message_comes_from_a_fixed_template(shipped):
    patterns = [re.escape(PREFIX + t + ".").replace(r"\{ttc:\.1f\}", r"[\d.]+")
                .replace(r"\{accel:\.1f\}", r"-?[\d.]+").replace(r"\{lateral:\.2f\}", r"[\d.]+")
                for t in TEMPLATES.values()]
    for name in scenarios.ALL:
        for flag in score(name, shipped).flags:
            assert any(re.fullmatch(p, flag.message) for p in patterns), flag.message


def test_nothing_is_shown_over_a_safety_alert_or_while_the_driver_drives(shipped):
    for name in scenarios.ALL:
        row = score(name, shipped)
        log = row.result.log
        for flag in row.delivered:
            i = round(flag.t / 0.01)
            assert log["alert"][i] is None and log["mode"][i] == "engaged", (name, flag)


def test_hold_off_limits_repeats_but_never_blocks_an_escalation(shipped):
    eager = AttentionModel(lm=shipped.lm, spec=shipped.spec, threshold_bits=-math.inf, holdoff=3.0)
    monitor = AttentionMonitor(eager)
    shown = []
    for k in range(150):                        # benign windows end at 0.5 s and 1.0 s; both flagged
        advisory = monitor.observe(snap(k * 0.01))
        if advisory:
            shown.append(advisory)
    assert [a.t for a in shown] == [0.5]
    assert monitor.flags[-1].withheld_because == "hold-off"
    for k in range(150, 210):                   # closing fast; its window ends at 2.0 s, inside the hold-off
        advisory = monitor.observe(snap(k * 0.01, behavior="follow", accel=-1.0, ttc=2.0))
        if advisory:
            shown.append(advisory)
    assert len(shown) == 2 and "closing on the vehicle ahead" in shown[1].message
    assert shown[1].t == pytest.approx(2.0)
    assert SEVERITY["closing"] > SEVERITY["unusual"]


# --- The artifact ---------------------------------------------------------------

def test_a_tampered_artifact_is_refused(shipped, tmp_path):
    path = shipped.save(tmp_path / "model.json")
    blob = json.loads(path.read_text())
    blob["threshold_bits"] += 1.0
    path.write_text(json.dumps(blob))
    with pytest.raises(ArtifactError, match="SHA-256"):
        AttentionModel.load(path)


def test_an_artifact_for_a_different_token_spec_is_refused():
    with pytest.raises(ArtifactError, match="token spec"):
        AttentionModel.load(spec=TokenSpec(window=0.25))


def test_training_is_deterministic_for_any_worker_count():
    serial = corpus_module.build(range(8), TokenSpec(), workers=1)
    parallel = corpus_module.build(range(8), TokenSpec(), workers=3)
    assert serial.drives == parallel.drives
    assert train(Plan(drives=15)).digest() == train(Plan(drives=15)).digest()


@pytest.mark.slow
def test_the_shipped_artifact_reproduces_from_its_provenance(shipped):
    plan = shipped.provenance["plan"]
    rebuilt = train(Plan(drives=plan["drives"], first_seed=plan["first_seed"]))
    assert rebuilt.digest() == shipped.digest()
