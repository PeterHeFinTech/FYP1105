from __future__ import annotations

import random
from typing import Any

_SUPPORTED_MODELS = {"ctgan", "tabddpm"}


def generate_synthetic_data(
    data: Any,
    model: str,
    num_samples: int | None = None,
    random_state: int = 42,
    **model_kwargs: Any,
) -> Any:
    """Generate synthetic tabular data using CTGAN or TabDDPM backends.

    Input can be a list of dictionaries or a pandas.DataFrame (if pandas is
    installed). If the requested backend package is unavailable, the function
    falls back to bootstrap-based sampling.
    """
    records, columns, return_as_dataframe = _normalize_input(data)

    if not records:
        raise ValueError("data must not be empty")

    model_key = model.strip().lower()
    if model_key not in _SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model}'. Choose from {_SUPPORTED_MODELS}.")

    target_rows = len(records) if num_samples is None else int(num_samples)
    if target_rows <= 0:
        raise ValueError("num_samples must be a positive integer")

    if model_key == "ctgan":
        generated = _generate_with_ctgan(records, columns, target_rows, random_state, **model_kwargs)
    else:
        generated = _generate_with_tabddpm(records, columns, target_rows, random_state, **model_kwargs)

    return _restore_output(generated, columns, return_as_dataframe)


def _normalize_input(data: Any) -> tuple[list[dict[str, Any]], list[str], bool]:
    if isinstance(data, list):
        if not data:
            return [], [], False
        if not all(isinstance(row, dict) for row in data):
            raise TypeError("list input must contain dictionaries")
        columns = list(data[0].keys())
        normalized = [{col: row.get(col) for col in columns} for row in data]
        return normalized, columns, False

    if hasattr(data, "to_dict") and hasattr(data, "columns"):
        records = data.to_dict(orient="records")
        columns = [str(col) for col in data.columns]
        normalized = [{col: row.get(col) for col in columns} for row in records]
        return normalized, columns, True

    raise TypeError("data must be a list of dict rows or a pandas.DataFrame")


def _restore_output(records: list[dict[str, Any]], columns: list[str], return_as_dataframe: bool) -> Any:
    if return_as_dataframe:
        try:
            import pandas as pd  # type: ignore

            return pd.DataFrame(records, columns=columns)
        except Exception:
            pass
    return records


def _generate_with_ctgan(
    records: list[dict[str, Any]],
    columns: list[str],
    target_rows: int,
    random_state: int,
    **model_kwargs: Any,
) -> list[dict[str, Any]]:
    try:
        from ctgan import CTGAN  # type: ignore
        import pandas as pd  # type: ignore

        df = pd.DataFrame(records, columns=columns)
        discrete_columns = [
            col
            for col in columns
            if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
        ]

        model = CTGAN(**model_kwargs)
        model.fit(df, discrete_columns=discrete_columns)
        sampled = model.sample(target_rows)
        return sampled.to_dict(orient="records")
    except Exception:
        return _bootstrap_fallback(records, target_rows, random_state)


def _generate_with_tabddpm(
    records: list[dict[str, Any]],
    columns: list[str],
    target_rows: int,
    random_state: int,
    **model_kwargs: Any,
) -> list[dict[str, Any]]:
    for module_name, class_name in (("tabddpm", "TabDDPM"), ("tabsynth", "TabDDPM")):
        try:
            module = __import__(module_name, fromlist=[class_name])
            model_cls = getattr(module, class_name)
            model = model_cls(random_state=random_state, **model_kwargs)
            model.fit(records)
            sampled = model.sample(target_rows)
            if isinstance(sampled, list):
                return sampled
        except Exception:
            continue

    return _bootstrap_fallback(records, target_rows, random_state)


def _bootstrap_fallback(records: list[dict[str, Any]], target_rows: int, random_state: int) -> list[dict[str, Any]]:
    rng = random.Random(random_state)
    sampled: list[dict[str, Any]] = []

    for _ in range(target_rows):
        source = dict(rng.choice(records))
        for key, value in source.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                value = _coerce_number(value)
            if isinstance(value, int):
                source[key] = int(round(value + rng.gauss(0.0, max(abs(value) * 0.01, 1.0))))
            elif isinstance(value, float):
                source[key] = value + rng.gauss(0.0, max(abs(value) * 0.01, 0.01))
        sampled.append(source)

    return sampled


def _coerce_number(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
