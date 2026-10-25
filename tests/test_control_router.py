from gaze.control.router import correct, route

LABELS = ["Search repositories", "Pull requests", "Issues"]


def test_snaps_misheard_commands():
    assert correct("closed time.") == "close tab"
    assert correct("go bag") == "go back"
    assert correct("never mine") == "never mind"
    assert correct("scroll dawn a lot") == "scroll down a lot"


def test_snaps_to_labels_near_gaze():
    assert correct("click search repose", LABELS) == "click Search repositories"
    assert correct("open issue", LABELS) == "open Issues"


def test_leaves_exact_phrases_chatter_and_requests_alone():
    for text in ["close tab.", "click", "Oh.", "Yeah.", "It's only...", "Did they do seconds?", "postcard",
                 "write an email to my professor"]:
        assert correct(text, LABELS) is None, text


def test_corrected_phrase_routes():
    assert route(correct("never mine"), wake=False).kind == "cancel"
    assert route(correct("go bag"), wake=False).kind == "shortcut"  # back, via the app's menu


def test_short_action_references_route_without_the_agent():
    assert route("again").args == {"mode": "again"}
    assert route("do that again").kind == "reference"
    assert route("do the same here").args == {"mode": "same_here"}
    assert route("close the one I just clicked").args == {"mode": "close_last"}
