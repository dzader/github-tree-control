#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

EPS = 1e-12


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def canonical_output(value: object) -> str:
    name = normalize_name(value)
    aliases = {
        "current": ("current", "currenta", "amperage", "amps", "i"),
        "power": ("power", "powerw", "watt", "watts", "p"),
        "voltage": ("voltage", "voltagev", "volt", "volts", "v"),
        "stress": ("stress", "engineeringstress", "truestress"),
        "force": ("force", "load", "newtons"),
        "cl": ("cl", "liftcoefficient", "coefficientoflift"),
        "cd": ("cd", "dragcoefficient", "coefficientofdrag"),
        "cm": ("cm", "momentcoefficient", "coefficientofmoment"),
    }
    for canonical, names in aliases.items():
        if name in names or any(len(alias) >= 4 and alias in name for alias in names):
            return canonical
    return name or "output"


def header_role(value: object, protocol: Mapping[str, object]) -> str | None:
    name = normalize_name(value)
    expectations = protocol.get("parser_expectations", {})
    role_synonyms = {
        "independent": expectations.get("independent_variable_synonyms", []),
        "output": expectations.get("output_synonyms", []),
        "context": expectations.get("context_synonyms", []),
    }
    for role, synonyms in role_synonyms.items():
        normalized = [normalize_name(item) for item in synonyms]
        if any(token and (name == token or (len(token) >= 3 and token in name)) for token in normalized):
            return role
    return None


def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    converted = {}
    for column in frame.columns:
        converted[str(column)] = pd.to_numeric(frame[column], errors="coerce")
    return pd.DataFrame(converted)


def read_tables(path: Path) -> Iterable[Tuple[str, pd.DataFrame]]:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        for sheet, frame in pd.read_excel(path, sheet_name=None).items():
            yield f"{path.name}::{sheet}", frame
        return
    if suffix in (".csv", ".tsv", ".txt", ".dat"):
        attempts = [
            {"sep": None, "engine": "python"},
            {"sep": "\t"},
            {"sep": r"\s+", "engine": "python"},
            {"sep": ";"},
            {"sep": ","},
        ]
        for kwargs in attempts:
            try:
                frame = pd.read_csv(path, **kwargs)
                if len(frame.columns) >= 2 and len(frame) >= 4:
                    yield path.name, frame
                    return
            except Exception:
                continue
        return
    if suffix == ".parquet":
        yield path.name, pd.read_parquet(path)
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(errors="replace"))
        yield path.name, pd.json_normalize(payload if isinstance(payload, list) else [payload])
        return
    if suffix == ".mat":
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        vectors: Dict[str, np.ndarray] = {}
        for key, value in payload.items():
            if key.startswith("__"):
                continue
            array = np.asarray(value)
            if array.ndim == 1 and array.size >= 4 and np.issubdtype(array.dtype, np.number):
                vectors[key] = array.astype(float)
        lengths = [len(value) for value in vectors.values()]
        if lengths:
            length = max(set(lengths), key=lengths.count)
            selected = {key: value for key, value in vectors.items() if len(value) == length}
            if len(selected) >= 2:
                yield path.name, pd.DataFrame(selected)
        return
    if suffix in (".h5", ".hdf5"):
        vectors: Dict[str, np.ndarray] = {}
        with h5py.File(path) as handle:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    array = np.asarray(obj)
                    if array.ndim == 1 and array.size >= 4 and np.issubdtype(array.dtype, np.number):
                        vectors[name] = array.astype(float)
            handle.visititems(visit)
        lengths = [len(value) for value in vectors.values()]
        if lengths:
            length = max(set(lengths), key=lengths.count)
            selected = {key: value for key, value in vectors.items() if len(value) == length}
            if len(selected) >= 2:
                yield path.name, pd.DataFrame(selected)


@dataclass
class Curve:
    curve_id: str
    family_key: str
    split: str
    x: np.ndarray
    outputs: Dict[str, np.ndarray]
    context: np.ndarray
    source: str


def dedupe_curve(x: np.ndarray, outputs: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    order = np.argsort(x, kind="mergesort")
    x = np.asarray(x, float)[order]
    sorted_outputs = {name: np.asarray(value, float)[order] for name, value in outputs.items()}
    rounded = np.round(x, 12)
    unique = np.unique(rounded)
    new_outputs = {name: [] for name in sorted_outputs}
    for value in unique:
        mask = rounded == value
        for name, output in sorted_outputs.items():
            new_outputs[name].append(float(np.mean(output[mask])))
    return unique.astype(float), {name: np.asarray(values, float) for name, values in new_outputs.items()}


def extract_curves(protocol: Mapping[str, object], extraction_manifest: Sequence[Mapping[str, object]], data_root: Path) -> Tuple[List[Curve], List[Dict[str, object]]]:
    curves: List[Curve] = []
    failures: List[Dict[str, object]] = []
    for item in extraction_manifest:
        path = data_root / str(item["local_relative_path"])
        family_key = str(item["family_key"])
        split = str(item["split"])
        found = 0
        try:
            tables = list(read_tables(path))
        except Exception as exc:
            failures.append({"path": str(path), "reason": f"table_read_exception:{type(exc).__name__}"})
            continue
        for table_name, raw in tables:
            if len(raw) < 8 or len(raw.columns) < 2:
                continue
            numeric = numeric_frame(raw)
            roles = {str(column): header_role(column, protocol) for column in raw.columns}
            independent_candidates = [column for column, role in roles.items() if role == "independent" and numeric[column].notna().sum() >= 10]
            if not independent_candidates:
                numeric_counts = sorted(((int(numeric[column].notna().sum()), column) for column in numeric.columns), reverse=True)
                independent_candidates = [column for count, column in numeric_counts if count >= 10][:1]
            if not independent_candidates:
                continue
            x_column = independent_candidates[0]
            output_columns = [column for column, role in roles.items() if role == "output" and column != x_column and numeric[column].notna().sum() >= 10]
            if not output_columns:
                candidates = []
                for column in numeric.columns:
                    if column == x_column or roles.get(column) == "context":
                        continue
                    series = numeric[column]
                    finite = series[np.isfinite(series)]
                    if len(finite) >= 10 and float(np.ptp(finite)) > 1e-10:
                        candidates.append((float(np.std(finite)), column))
                output_columns = [column for _, column in sorted(candidates, reverse=True)[:2]]
            if not output_columns:
                continue
            output_columns = output_columns[:2]
            context_columns = []
            for column, role in roles.items():
                if role != "context" or column == x_column or column in output_columns:
                    continue
                unique = raw[column].dropna().nunique()
                if 1 < unique <= 40:
                    context_columns.append(column)
            groups: Iterable[Tuple[object, pd.DataFrame]]
            if context_columns:
                groups = raw.groupby(context_columns, dropna=False, sort=True)
            else:
                groups = [("all", raw)]
            for group_key, group in groups:
                group_numeric = numeric_frame(group)
                mask = np.isfinite(group_numeric[x_column].to_numpy(float))
                for column in output_columns:
                    mask &= np.isfinite(group_numeric[column].to_numpy(float))
                if int(np.sum(mask)) < 12:
                    continue
                x = group_numeric.loc[mask, x_column].to_numpy(float)
                outputs = {
                    canonical_output(column): group_numeric.loc[mask, column].to_numpy(float)
                    for column in output_columns
                }
                x, outputs = dedupe_curve(x, outputs)
                if len(x) < 12 or float(np.ptp(x)) <= 1e-12:
                    continue
                valid_outputs = {
                    name: value
                    for name, value in outputs.items()
                    if float(np.ptp(value)) > max(1e-10, 1e-6 * max(float(np.max(np.abs(value))), 1.0))
                }
                if not valid_outputs:
                    continue
                context_values: List[float] = []
                for column in context_columns[:4]:
                    series = pd.to_numeric(group[column], errors="coerce")
                    finite = series[np.isfinite(series)]
                    context_values.append(float(np.median(finite)) if len(finite) else 0.0)
                while len(context_values) < 4:
                    context_values.append(0.0)
                key_text = json.dumps(group_key, sort_keys=True, default=str)
                curve_id = hashlib.sha256(f"{family_key}\0{item['source_path']}\0{table_name}\0{key_text}".encode()).hexdigest()[:20]
                curves.append(Curve(
                    curve_id=curve_id,
                    family_key=family_key,
                    split=split,
                    x=x,
                    outputs=valid_outputs,
                    context=np.asarray(context_values, float),
                    source=f"{item['source_path']}::{table_name}::{key_text}",
                ))
                found += 1
        if found == 0:
            failures.append({"path": str(path), "reason": "no_valid_curve_detected"})
    return curves, failures


def normalize_axis(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if not upper > lower:
        return np.zeros_like(values)
    return 2.0 * (values - lower) / (upper - lower) - 1.0


def basis(z: np.ndarray, lower: float, upper: float, degree: int = 8) -> np.ndarray:
    return np.polynomial.chebyshev.chebvander(normalize_axis(z, lower, upper), degree)


def ridge_solve(matrix: np.ndarray, target: np.ndarray, penalty: float, weights: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(matrix, float)
    y = np.asarray(target, float)
    if weights is not None:
        scale = np.sqrt(np.asarray(weights, float))
        X = X * scale[:, None]
        y = y * scale
    regularizer = np.eye(X.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + regularizer, X.T @ y)


def fit_affine(shape: np.ndarray, target: np.ndarray, penalty: float = 2e-5) -> Tuple[float, float]:
    matrix = np.column_stack([np.ones(len(shape)), shape])
    coefficients = ridge_solve(matrix, target, penalty)
    return float(coefficients[0]), float(coefficients[1])


def robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    center = float(np.median(values))
    scale = max(float(np.std(values)), float(np.percentile(values, 90) - np.percentile(values, 10)) / 2.563, 1e-8)
    return (values - center) / scale


def orthogonalize_mode(base_coefficients: np.ndarray, mode_coefficients: np.ndarray, z_samples: np.ndarray, lower: float, upper: float) -> np.ndarray:
    matrix = basis(z_samples, lower, upper)
    base_values = matrix @ base_coefficients
    mode_values = matrix @ mode_coefficients
    projection = np.linalg.lstsq(np.column_stack([np.ones(len(base_values)), base_values]), mode_values, rcond=None)[0]
    constant = np.zeros_like(base_coefficients)
    constant[0] = 1.0
    adjusted = mode_coefficients - projection[0] * constant - projection[1] * base_coefficients
    scale = max(float(np.std(matrix @ adjusted)), 1e-8)
    return adjusted / scale


@dataclass
class Atlas:
    x_lower: float
    x_upper: float
    output_coefficients: Dict[str, np.ndarray]
    output_floors: Dict[str, float]
    context_center: np.ndarray
    context_scale: np.ndarray
    c_prior: np.ndarray
    c_lower: float = -1.7
    c_upper: float = 1.7
    c_penalty: float = 0.025

    def values(self, curve: Curve, output: str, c_value: float) -> np.ndarray:
        matrix = basis(curve.x, self.x_lower, self.x_upper)
        coefficients = self.output_coefficients[output]
        return matrix @ coefficients[:, 0] + c_value * (matrix @ coefficients[:, 1])

    def prior(self, curve: Curve) -> float:
        standardized = (curve.context - self.context_center) / self.context_scale
        features = np.r_[1.0, standardized]
        return float(np.clip(features @ self.c_prior, self.c_lower, self.c_upper))


def shared_outputs(curves: Sequence[Curve], minimum_curves: int = 3) -> List[str]:
    counts: Dict[str, int] = {}
    for curve in curves:
        for output in curve.outputs:
            counts[output] = counts.get(output, 0) + 1
    return sorted(output for output, count in counts.items() if count >= minimum_curves)


def initialize_atlas(curves: Sequence[Curve], outputs: Sequence[str]) -> Atlas:
    all_x = np.concatenate([curve.x for curve in curves])
    lower = float(np.quantile(all_x, 0.005))
    upper = float(np.quantile(all_x, 0.995))
    span = max(upper - lower, 1e-8)
    lower -= 0.05 * span
    upper += 0.05 * span
    context = np.vstack([curve.context for curve in curves])
    context_center = np.median(context, axis=0)
    context_scale = np.std(context, axis=0)
    context_scale[context_scale < 1e-8] = 1.0
    output_coefficients: Dict[str, np.ndarray] = {}
    floors: Dict[str, float] = {}
    z_samples = np.linspace(lower, upper, 400)
    for output in outputs:
        blocks = []
        targets = []
        weights = []
        ranges = []
        for curve in curves:
            if output not in curve.outputs:
                continue
            blocks.append(basis(curve.x, lower, upper))
            targets.append(robust_normalize(curve.outputs[output]))
            weights.append(np.full(len(curve.x), 1.0 / len(curve.x)))
            ranges.append(float(np.ptp(curve.outputs[output])))
        matrix = np.vstack(blocks)
        target = np.concatenate(targets)
        weight = np.concatenate(weights)
        base = ridge_solve(matrix, target, 0.006, weight)
        epsilon = max((upper - lower) * 1e-4, 1e-8)
        derivative = (basis(z_samples + epsilon, lower, upper) @ base - basis(z_samples - epsilon, lower, upper) @ base) / (2 * epsilon)
        tangent = ridge_solve(basis(z_samples, lower, upper), derivative, 0.002)
        tangent = orthogonalize_mode(base, tangent, z_samples, lower, upper)
        output_coefficients[output] = np.column_stack([base, tangent])
        floors[output] = max(float(np.median(ranges)) * 0.15, 1e-8)
    return Atlas(
        x_lower=lower,
        x_upper=upper,
        output_coefficients=output_coefficients,
        output_floors=floors,
        context_center=context_center,
        context_scale=context_scale,
        c_prior=np.zeros(1 + context.shape[1]),
    )


def curve_score(atlas: Atlas, curve: Curve, c_value: float, indices: np.ndarray | None = None) -> float:
    if indices is None:
        indices = np.arange(len(curve.x))
    errors = []
    for output in atlas.output_coefficients:
        if output not in curve.outputs:
            continue
        shape = atlas.values(curve, output, c_value)
        a, b = fit_affine(shape[indices], curve.outputs[output][indices])
        denominator = max(float(np.ptp(curve.outputs[output][indices])), atlas.output_floors[output])
        errors.append(float(np.sqrt(np.mean((a + b * shape[indices] - curve.outputs[output][indices]) ** 2)) / denominator))
    if not errors:
        return float("inf")
    return float(np.exp(np.mean(np.log(np.maximum(errors, EPS)))))


def estimate_c_full(atlas: Atlas, curve: Curve) -> float:
    grid = np.linspace(atlas.c_lower, atlas.c_upper, 101)
    scores = np.asarray([curve_score(atlas, curve, float(value)) for value in grid])
    return float(grid[int(np.argmin(scores))])


def update_coefficients(atlas: Atlas, curves: Sequence[Curve], c_values: Mapping[str, float]) -> None:
    z_samples = np.linspace(atlas.x_lower, atlas.x_upper, 400)
    for output in list(atlas.output_coefficients):
        blocks = []
        targets = []
        weights = []
        for curve in curves:
            if output not in curve.outputs:
                continue
            matrix = basis(curve.x, atlas.x_lower, atlas.x_upper)
            c_value = float(c_values[curve.curve_id])
            blocks.append(np.hstack([matrix, c_value * matrix]))
            targets.append(robust_normalize(curve.outputs[output]))
            weights.append(np.full(len(curve.x), 1.0 / len(curve.x)))
        coefficients = ridge_solve(np.vstack(blocks), np.concatenate(targets), 0.01, np.concatenate(weights))
        width = coefficients.shape[0] // 2
        base = coefficients[:width]
        tangent = orthogonalize_mode(base, coefficients[width:], z_samples, atlas.x_lower, atlas.x_upper)
        atlas.output_coefficients[output] = np.column_stack([base, tangent])


def fit_atlas(curves: Sequence[Curve]) -> Atlas:
    outputs = shared_outputs(curves)
    if not outputs:
        raise RuntimeError("no output appears in at least three development curves")
    atlas = initialize_atlas(curves, outputs)
    c_values = {curve.curve_id: 0.0 for curve in curves}
    for _ in range(5):
        c_values = {curve.curve_id: estimate_c_full(atlas, curve) for curve in curves}
        update_coefficients(atlas, curves, c_values)
    context = np.vstack([curve.context for curve in curves])
    standardized = (context - atlas.context_center) / atlas.context_scale
    features = np.column_stack([np.ones(len(curves)), standardized])
    targets = np.asarray([c_values[curve.curve_id] for curve in curves])
    atlas.c_prior = ridge_solve(features, targets, 0.4)
    return atlas


def select_probes(curve: Curve, count: int = 5) -> np.ndarray:
    normalized = normalize_axis(curve.x, float(np.min(curve.x)), float(np.max(curve.x)))[:, None]
    unique_indices = np.unique(np.round(normalized, 10), axis=0, return_index=True)[1]
    candidates = np.sort(unique_indices)
    if len(candidates) < count:
        candidates = np.arange(len(curve.x))
    selected = [int(candidates[np.argmin(np.square(normalized[candidates, 0]))])]
    while len(selected) < count:
        remaining = np.asarray([index for index in candidates if int(index) not in selected], int)
        if len(remaining) == 0:
            remaining = np.asarray([index for index in range(len(curve.x)) if index not in selected], int)
        distances = np.min(np.square(normalized[remaining] - normalized[np.asarray(selected)].T), axis=1)
        selected.append(int(remaining[np.argmax(distances)]))
    return np.asarray(sorted(selected), int)


def cubic_predict(curve: Curve, probes: np.ndarray, outputs: Sequence[str]) -> Dict[str, np.ndarray]:
    normalized = normalize_axis(curve.x, float(np.min(curve.x)), float(np.max(curve.x)))
    matrix = np.column_stack([np.ones(len(normalized)), normalized, normalized ** 2, normalized ** 3])
    predictions = {}
    for output in outputs:
        if output not in curve.outputs:
            continue
        coefficients = ridge_solve(matrix[probes], curve.outputs[output][probes], 0.002)
        predictions[output] = matrix @ coefficients
    return predictions


def atlas_predict(atlas: Atlas, curve: Curve, probes: np.ndarray) -> Tuple[Dict[str, np.ndarray], float]:
    prior = atlas.prior(curve)
    grid = np.unique(np.clip(np.r_[np.linspace(prior - 1.1, prior + 1.1, 89), prior, 0.0], atlas.c_lower, atlas.c_upper))
    best = (float("inf"), prior, {})
    for c_value in grid:
        parameters = {}
        normalized_errors = []
        for output in atlas.output_coefficients:
            if output not in curve.outputs:
                continue
            shape = atlas.values(curve, output, float(c_value))
            a, b = fit_affine(shape[probes], curve.outputs[output][probes], 2e-4)
            parameters[output] = (a, b)
            normalized_errors.append(float(np.sqrt(np.mean((a + b * shape[probes] - curve.outputs[output][probes]) ** 2)) / atlas.output_floors[output]))
        if not normalized_errors:
            continue
        score = float(np.exp(np.mean(np.log(np.maximum(normalized_errors, EPS))))) + atlas.c_penalty * ((c_value - prior) / 0.9) ** 2
        if score < best[0]:
            best = (score, float(c_value), parameters)
    _, c_value, parameters = best
    predictions = {}
    for output, (a, b) in parameters.items():
        predictions[output] = a + b * atlas.values(curve, output, c_value)
    return predictions, c_value


def choose_blend(atlas: Atlas, curve: Curve, probes: np.ndarray) -> float:
    rows = []
    for held in probes:
        subset = probes[probes != held]
        atlas_prediction, _ = atlas_predict(atlas, curve, subset)
        fixed_prediction = cubic_predict(curve, subset, list(atlas.output_coefficients))
        rows.append((int(held), atlas_prediction, fixed_prediction))
    best = (float("inf"), 0.0)
    for alpha in np.linspace(0.0, 1.0, 11):
        errors = []
        for held, atlas_prediction, fixed_prediction in rows:
            output_errors = []
            for output in atlas_prediction:
                value = alpha * atlas_prediction[output][held] + (1.0 - alpha) * fixed_prediction[output][held]
                output_errors.append(abs(value - curve.outputs[output][held]) / atlas.output_floors[output])
            if output_errors:
                errors.append(float(np.exp(np.mean(np.log(np.maximum(output_errors, EPS))))))
        score = float(np.mean(errors)) + 0.008 * alpha if errors else float("inf")
        if score < best[0]:
            best = (score, float(alpha))
    return best[1]


def safe_atlas_predict(atlas: Atlas, curve: Curve, probes: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, float]:
    atlas_prediction, c_value = atlas_predict(atlas, curve, probes)
    fixed_prediction = cubic_predict(curve, probes, list(atlas.output_coefficients))
    alpha = choose_blend(atlas, curve, probes)
    predictions = {
        output: alpha * atlas_prediction[output] + (1.0 - alpha) * fixed_prediction[output]
        for output in atlas_prediction
    }
    return predictions, c_value, alpha


def context_features(curve: Curve, x: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.tile(curve.context, (len(x), 1))])


@dataclass
class Controls:
    models: Dict[str, Dict[str, object]]


def fit_controls(curves: Sequence[Curve], outputs: Sequence[str]) -> Controls:
    models: Dict[str, Dict[str, object]] = {"direct_poly": {}, "extra_trees": {}, "hist_gb": {}, "mlp": {}}
    for output in outputs:
        relevant = [curve for curve in curves if output in curve.outputs]
        features = np.vstack([context_features(curve, curve.x) for curve in relevant])
        targets = np.concatenate([curve.outputs[output] for curve in relevant])
        polynomial = make_pipeline(PolynomialFeatures(3, include_bias=True), Ridge(alpha=0.03))
        polynomial.fit(features, targets)
        models["direct_poly"][output] = polynomial
        trees = ExtraTreesRegressor(n_estimators=220, min_samples_leaf=4, max_features=0.9, random_state=2101, n_jobs=-1)
        trees.fit(features, targets)
        models["extra_trees"][output] = trees
        gradient = HistGradientBoostingRegressor(max_iter=240, learning_rate=0.05, max_leaf_nodes=18, l2_regularization=0.03, random_state=2101)
        gradient.fit(features, targets)
        models["hist_gb"][output] = gradient
        neural = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(40, 24), activation="tanh", alpha=0.02, max_iter=500, early_stopping=True, random_state=2101))
        neural.fit(features, targets)
        models["mlp"][output] = neural
    return Controls(models=models)


def calibrated_control_predict(controls: Controls, method: str, curve: Curve, probes: np.ndarray) -> Dict[str, np.ndarray]:
    features = context_features(curve, curve.x)
    predictions = {}
    for output, model in controls.models[method].items():
        if output not in curve.outputs:
            continue
        raw = np.asarray(model.predict(features), float)
        a, b = fit_affine(raw[probes], curve.outputs[output][probes], 4e-4)
        predictions[output] = a + b * raw
    return predictions


def prediction_error(atlas: Atlas, curve: Curve, prediction: Mapping[str, np.ndarray], indices: np.ndarray) -> Tuple[float, Dict[str, float]]:
    errors = {}
    for output, values in prediction.items():
        if output not in curve.outputs:
            continue
        denominator = max(float(np.ptp(curve.outputs[output][indices])), atlas.output_floors[output])
        errors[output] = float(np.sqrt(np.mean((values[indices] - curve.outputs[output][indices]) ** 2)) / denominator)
    if not errors:
        return float("inf"), errors
    return float(np.exp(np.mean(np.log(np.maximum(list(errors.values()), EPS))))), errors


def aggregate(records: Sequence[Mapping[str, object]], method: str) -> Dict[str, float]:
    values = np.asarray([float(record[f"{method}_joint"]) for record in records])
    fixed = np.asarray([float(record["fixed_joint"]) for record in records])
    ratio = values / np.maximum(fixed, EPS)
    return {
        "median_joint_nrmse": float(np.median(values)),
        "geometric_mean_joint_nrmse": float(np.exp(np.mean(np.log(np.maximum(values, EPS))))),
        "geometric_mean_ratio_vs_fixed": float(np.exp(np.mean(np.log(np.maximum(ratio, EPS))))),
        "median_ratio_vs_fixed": float(np.median(ratio)),
        "win_fraction_vs_fixed": float(np.mean(ratio < 1.0)),
        "p90_ratio_vs_fixed": float(np.quantile(ratio, 0.9)),
        "worst_ratio_vs_fixed": float(np.max(ratio)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--extraction-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    extraction_manifest = json.loads(Path(args.extraction_manifest).read_text())
    curves, parse_failures = extract_curves(protocol, extraction_manifest, Path(args.data_root))
    development = [curve for curve in curves if curve.split == "development"]
    validation = [curve for curve in curves if curve.split == "validation"]
    confirmation_curves = [curve for curve in curves if curve.split == "confirmation"]
    if confirmation_curves:
        raise RuntimeError("confirmation curve appeared in development extraction")
    if len(development) < 3 or len(validation) < 2:
        raise RuntimeError(f"insufficient parsed curves: development={len(development)} validation={len(validation)}")
    atlas = fit_atlas(development)
    controls = fit_controls(development, list(atlas.output_coefficients))
    records: List[Dict[str, object]] = []
    for curve in validation:
        supported = [output for output in atlas.output_coefficients if output in curve.outputs]
        if not supported:
            parse_failures.append({"curve_id": curve.curve_id, "reason": "no_supported_validation_output"})
            continue
        probes = select_probes(curve, 5)
        probe_set = set(probes.tolist())
        scored = np.asarray([index for index in range(len(curve.x)) if index not in probe_set], int)
        if len(scored) < 7:
            parse_failures.append({"curve_id": curve.curve_id, "reason": "too_few_scored_rows"})
            continue
        methods: Dict[str, Dict[str, np.ndarray]] = {}
        methods["fixed"] = cubic_predict(curve, probes, supported)
        raw_atlas, c_value = atlas_predict(atlas, curve, probes)
        safe_atlas, _, atlas_weight = safe_atlas_predict(atlas, curve, probes)
        methods["raw_atlas"] = raw_atlas
        methods["safe_atlas"] = safe_atlas
        for method in controls.models:
            methods[method] = calibrated_control_predict(controls, method, curve, probes)
        row: Dict[str, object] = {
            "curve_id": curve.curve_id,
            "family_key": curve.family_key,
            "source": curve.source,
            "points": len(curve.x),
            "outputs": ";".join(supported),
            "atlas_c": c_value,
            "atlas_weight": atlas_weight,
            "probe_indices": ";".join(str(index) for index in probes.tolist()),
        }
        for method, prediction in methods.items():
            joint, errors = prediction_error(atlas, curve, prediction, scored)
            row[f"{method}_joint"] = joint
            for output, error in errors.items():
                row[f"{method}_{output}"] = error
        records.append(row)
    if len(records) < 2:
        raise RuntimeError(f"fewer than two valid validation curves: {len(records)}")
    methods = ["fixed", "safe_atlas", "raw_atlas", "direct_poly", "extra_trees", "hist_gb", "mlp"]
    summaries = {method: aggregate(records, method) for method in methods}
    candidate = summaries["safe_atlas"]
    best_learned = min(summaries[method]["geometric_mean_joint_nrmse"] for method in ("direct_poly", "extra_trees", "hist_gb", "mlp"))
    gate = {
        "pass": bool(
            candidate["geometric_mean_ratio_vs_fixed"] <= 0.90
            and candidate["win_fraction_vs_fixed"] >= 0.65
            and candidate["worst_ratio_vs_fixed"] <= 1.75
            and candidate["geometric_mean_joint_nrmse"] <= 0.90 * best_learned
        ),
        "requirements": {
            "geometric_mean_ratio_vs_fixed_lte": 0.90,
            "win_fraction_vs_fixed_gte": 0.65,
            "worst_ratio_vs_fixed_lte": 1.75,
            "ratio_vs_best_learned_control_lte": 0.90,
        },
        "best_learned_control_geometric_mean_joint_nrmse": best_learned,
        "atlas_ratio_vs_best_learned_control": candidate["geometric_mean_joint_nrmse"] / best_learned,
    }
    summary = {
        "protocol_id": protocol["protocol_id"],
        "domain": protocol["domain"],
        "confirmation_values_accessed": False,
        "parsed_curve_count": len(curves),
        "development_curve_count": len(development),
        "validation_curve_count": len(validation),
        "valid_validation_curve_count": len(records),
        "parse_failures": parse_failures,
        "outputs": list(atlas.output_coefficients),
        "methods": summaries,
        "atlas_diagnostics": {
            "median_c": float(np.median([record["atlas_c"] for record in records])),
            "median_atlas_weight": float(np.median([record["atlas_weight"] for record in records])),
            "nonzero_atlas_weight_fraction": float(np.mean([record["atlas_weight"] > 0 for record in records])),
        },
        "development_gate": gate,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "development_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (output_dir / "per_curve.csv").open("w", newline="") as handle:
        fieldnames = sorted({key for record in records for key in record})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    model = {
        "model_id": "causal-loom-v021-cross-domain-projective-response-atlas",
        "domain": protocol["domain"],
        "x_lower": atlas.x_lower,
        "x_upper": atlas.x_upper,
        "output_coefficients": {name: values.tolist() for name, values in atlas.output_coefficients.items()},
        "output_floors": atlas.output_floors,
        "context_center": atlas.context_center.tolist(),
        "context_scale": atlas.context_scale.tolist(),
        "c_prior": atlas.c_prior.tolist(),
        "c_lower": atlas.c_lower,
        "c_upper": atlas.c_upper,
        "c_penalty": atlas.c_penalty,
        "probe_count": 5,
        "basis_degree": 8,
    }
    (output_dir / "development_model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print("V021_CROSS_DOMAIN_DEVELOPMENT=" + json.dumps({
        "domain": summary["domain"],
        "development_gate": gate,
        "curve_counts": {
            "parsed": len(curves),
            "development": len(development),
            "validation": len(validation),
            "valid_validation": len(records),
        },
        "safe_atlas": summaries["safe_atlas"],
        "fixed": summaries["fixed"],
        "atlas_diagnostics": summary["atlas_diagnostics"],
    }, sort_keys=True))
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
