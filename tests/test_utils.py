from interaction_complexity.utils import dynamic_n_step, scenario_id_from_path, stable_split


def test_scenario_id_from_path():
    assert scenario_id_from_path("data/foo/CHN_SIND-1_10_LEFT_T-25.xml") == "CHN_SIND-1_10_LEFT_T-25"


def test_dynamic_full_horizon():
    assert dynamic_n_step("CHN_SIND-1_10_LEFT_T-25", 0) == 15
    assert dynamic_n_step("CHN_SIND-1_10_LEFT_T-25", 5) == 5


def test_stable_split_is_stable():
    sid = "CHN_SIND-1_10_LEFT_T-25"
    assert stable_split(sid) == stable_split(sid)
