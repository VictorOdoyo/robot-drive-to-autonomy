from pathlib import Path

from robotlab.cases import load_case, plan_route


def test_custom_example_has_a_feasible_route():
    case = load_case(Path(__file__).resolve().parents[1] / "examples" / "custom_case.json")
    route, _, _, cost = plan_route(case)
    assert len(route) >= 2 and cost > 0
    assert case.events[0].kind == "grade"
