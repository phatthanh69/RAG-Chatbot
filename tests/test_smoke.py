from tests._entry import EXPECTED_URL_PREFIXES


def test_app_factory_boots(app):
    assert app is not None
    assert app.name


def test_expected_blueprints_registered(app):
    rules = {r.rule for r in app.url_map.iter_rules()}
    # Every API area must expose at least one rule under its prefix.
    for prefix in EXPECTED_URL_PREFIXES:
        assert any(rule.startswith(prefix) for rule in rules), (
            f"No route registered under {prefix}. Rules: {sorted(rules)}"
        )


def test_root_and_api_index_routes_exist(app):
    rules = {r.rule for r in app.url_map.iter_rules()}
    assert "/" in rules
    assert "/api" in rules
