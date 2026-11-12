"""Static security-contract checks for inference API route declarations."""

import ast
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "bindings"
    / "python"
    / "SneppX_ALG"
    / "interface_bindings"
    / "inference_server.py"
)


def _route_definitions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    routes = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not isinstance(decorator.func, ast.Attribute):
                continue
            if decorator.func.attr not in {"get", "post", "put", "patch", "delete"}:
                continue
            if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                continue
            routes[(decorator.func.attr.upper(), decorator.args[0].value)] = node
    return routes


def _has_require_auth_dependency(route):
    for decorator in route.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg != "dependencies" or not isinstance(keyword.value, ast.List):
                continue
            for dependency in keyword.value.elts:
                if (
                    isinstance(dependency, ast.Call)
                    and isinstance(dependency.func, ast.Name)
                    and dependency.func.id == "Depends"
                    and dependency.args
                    and isinstance(dependency.args[0], ast.Name)
                    and dependency.args[0].id == "require_auth"
                ):
                    return True
    return False


def test_serving_control_routes_require_authentication():
    routes = _route_definitions()
    protected = (
        ("GET", "/metrics"),
        ("POST", "/v1/deploy"),
        ("POST", "/v1/traffic"),
        ("POST", "/v1/generate/continuous-batch"),
        ("POST", "/v1/models/quantize"),
    )
    for key in protected:
        assert key in routes, f"Missing expected route: {key}"
        assert _has_require_auth_dependency(routes[key]), (
            f"{key} must enforce require_auth"
        )


def test_admin_credentials_are_header_parameters():
    routes = _route_definitions()
    admin_routes = (
        ("GET", "/v1/admin/keys"),
        ("POST", "/v1/admin/keys"),
        ("GET", "/v1/admin/keys/{key_id}"),
        ("DELETE", "/v1/admin/keys/{key_id}"),
        ("PATCH", "/v1/admin/keys/{key_id}"),
        ("GET", "/v1/admin/keys/{key_id}/usage"),
    )
    for key in admin_routes:
        route = routes[key]
        authorization = next(
            (arg for arg in route.args.args if arg.arg == "authorization"),
            None,
        )
        assert authorization is not None, f"{key} is missing Authorization"
        position = route.args.args.index(authorization)
        default_index = position - (
            len(route.args.args) - len(route.args.defaults)
        )
        assert default_index >= 0, f"{key} Authorization must have a default"
        default = route.args.defaults[default_index]
        assert (
            isinstance(default, ast.Call)
            and isinstance(default.func, ast.Name)
            and default.func.id == "Header"
        ), f"{key} must read Authorization from an HTTP header, not a query string"


def test_auth_dependency_enforces_rate_limiting():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    require_auth = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "require_auth"
    )
    calls_rate_limiter = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "check_rate_limit"
        for node in ast.walk(require_auth)
    )
    assert calls_rate_limiter, "require_auth must enforce the configured rate limiter"
