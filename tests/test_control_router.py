from gaze.control.router import route


def test_short_action_references_route_without_the_agent():
    assert route("again").args == {"mode": "again"}
    assert route("do that again").kind == "reference"
    assert route("do the same here").args == {"mode": "same_here"}
    assert route("close the one I just clicked").args == {"mode": "close_last"}
