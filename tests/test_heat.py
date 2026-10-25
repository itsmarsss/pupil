import math

from gaze.control.heat import GazeCloud, ambiguous, neighbour, scores, sigma_from_accuracy
from gaze.control.snapping import Element


def tabs():
    return [Element("AXRadioButton", f"tab{i}", 200 * i, 40, 200, 34, parent_role="AXTabGroup") for i in range(4)]


def test_mass_is_exact_and_sums_to_one_over_the_plane():
    c = GazeCloud([(100.0, 100.0, 1.0)], sigma=50)
    assert abs(c.mass(-1e5, -1e5, 2e5, 2e5) - 1) < 1e-9
    assert abs(c.mass(100, -1e5, 1e5, 2e5) - 0.5) < 1e-9  # half the plane
    one_sigma = math.erf(1 / math.sqrt(2))
    assert abs(c.mass(50, -1e5, 100, 2e5) - one_sigma) < 1e-9  # +-1 sigma in x


def test_a_point_between_two_tabs_is_too_close_to_call_but_language_settles_it():
    els = tabs()
    cloud = GazeCloud([(400.0, 57.0, 1.0)], sigma=sigma_from_accuracy(150))  # right on the tab1 | tab2 border
    shares = scores(els, cloud)
    assert ambiguous(shares)
    named = scores(els, cloud, fit=[0.05, 0.05, 0.9, 0.05])  # "close the tab2 tab": Jev knows which
    assert not ambiguous(named) and max(range(4), key=named.__getitem__) == 2


def test_a_tight_cloud_inside_one_tab_is_not_ambiguous():
    shares = scores(tabs(), GazeCloud([(300.0, 57.0, 1.0)] * 10, sigma=25))
    assert not ambiguous(shares) and max(range(4), key=shares.__getitem__) == 1


def test_low_confidence_samples_spread_wider():
    sure, unsure = GazeCloud([(0.0, 0.0, 1.0)], sigma=40), GazeCloud([(0.0, 0.0, 0.25)], sigma=40)
    assert unsure.mass(-40, -40, 80, 80) < sure.mass(-40, -40, 80, 80)


def test_attention_weights_favour_the_final_fixation_and_round_trip():
    els = [Element("AXButton", "first", 0, 0, 100, 100), Element("AXButton", "second", 300, 0, 100, 100)]
    points = [(50.0, 50.0, 1.0)] * 4 + [(350.0, 50.0, 1.0)] * 4
    baseline = scores(els, GazeCloud(points, sigma=25))
    cloud = GazeCloud(points, sigma=25, weights=[1.0] * 4 + [2.0] * 4)
    weighted = scores(els, cloud)
    assert abs(baseline[0] - baseline[1]) < 1e-6
    assert weighted[1] > weighted[0]
    summary = cloud.summary()
    restored = GazeCloud([tuple(p) for p in summary["points"]], summary["sigma"], summary["weights"])
    assert scores(els, restored) == weighted


def test_neighbours_stay_in_the_row_and_kind():
    els = tabs() + [Element("AXButton", "Back", 210, 80, 30, 30)]
    t1 = els[1]
    assert neighbour(t1, els, "left").label == "tab0" and neighbour(t1, els, "right").label == "tab2"
    assert neighbour(els[0], els, "left") is None and neighbour(t1, els, "down") is None  # Back isn't a tab


def test_closing_anything_in_one_popup_is_not_a_choice():
    from gaze.control.gate import _element_route

    field = Element("AXTextField", "Email", 500, 400, 300, 30, parent="Sign in")
    close = Element("AXButton", "Close", 700, 300, 20, 20, parent="Sign in", subrole="AXCloseButton")
    cloud = GazeCloud([(650.0, 410.0, 1.0)], sigma=90)
    choices = {"element_action": type("A", (), {"choice": "close"})()}
    route, _, extras = _element_route([(0.6, field), (0.5, close)], (650, 410), choices, "", cloud)
    assert route == "close"
    choices = {"element_action": type("A", (), {"choice": "activate"})()}
    tab_a = Element("AXRadioButton", "a", 0, 0, 200, 30, parent_role="AXTabGroup")
    tab_b = Element("AXRadioButton", "b", 200, 0, 200, 30, parent_role="AXTabGroup")
    route, _, extras = _element_route([(0.6, tab_a), (0.6, tab_b)], (200, 15), choices, "", GazeCloud([(200.0, 15.0, 1.0)], 90))
    assert route == "choose" and {el.label for _, el in extras["options"]} == {"a", "b"}
