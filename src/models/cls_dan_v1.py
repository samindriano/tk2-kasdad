"""Small deterministic NumPy implementation for the frozen CLS_DAN_V1 pilot.

This module has no data access or fitting side effects at import time.  The
official runner supplies only a frozen unigram vocabulary, mean-count CSR
matrices, saved C2_B structured features, and prefix labels.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit


EMBEDDING_WIDTH = 32
DAN_HIDDEN_WIDTHS = (64, 32)
DROPOUT_RATE = 0.10
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPSILON = 1e-8
L2_COEFFICIENT = 1e-4
RANDOM_SEED = 42
ARCHITECTURES = ("DAN", "LINEAR")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def unigram_terms(vectorizer: Any) -> tuple[list[str], dict[str, int]]:
    """Return the frozen no-space vocabulary in original vectorizer order."""
    vocabulary = getattr(vectorizer, "vocabulary_", None)
    require(isinstance(vocabulary, Mapping) and len(vocabulary) > 0,
            "A fitted frozen vectorizer vocabulary is required")
    entries = sorted(((str(term), int(index)) for term, index in vocabulary.items()
                      if " " not in str(term)), key=lambda item: item[1])
    require(entries and len({term for term, _ in entries}) == len(entries),
            "Frozen unigram vocabulary is empty or duplicated")
    indices = [index for _, index in entries]
    require(len(indices) == len(set(indices)), "Frozen unigram indices are duplicated")
    terms = [term for term, _ in entries]
    return terms, {term: index for index, term in enumerate(terms)}


def build_mean_unigram_counts(
    documents: Sequence[str], vectorizer: Any,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build row-normalized unigram counts with multiplicity and OOV auditing.

    Normalization is by the number of in-vocabulary token occurrences, so
    multiplying this matrix by E returns the mean embedding. OOV occurrences
    do not enter the mean, and an all-OOV/empty document yields a zero row.
    """
    terms, token_to_column = unigram_terms(vectorizer)
    preprocess = vectorizer.build_preprocessor()
    tokenize = vectorizer.build_tokenizer()
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    all_token_counts = np.zeros(len(documents), dtype=np.int64)
    known_token_counts = np.zeros(len(documents), dtype=np.int64)

    for row, document in enumerate(documents):
        require(isinstance(document, str), "Documents must be normalized strings")
        tokens = tokenize(preprocess(document))
        all_token_counts[row] = len(tokens)
        row_counts: dict[int, int] = {}
        for token in tokens:
            column = token_to_column.get(token)
            if column is not None:
                row_counts[column] = row_counts.get(column, 0) + 1
        known = int(sum(row_counts.values()))
        known_token_counts[row] = known
        if known:
            divisor = float(known)
            for column in sorted(row_counts):
                indices.append(column)
                values.append(row_counts[column] / divisor)
        indptr.append(len(indices))

    matrix = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float64),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(documents), len(terms)), dtype=np.float64,
    )
    matrix.sort_indices()
    coverage = np.divide(
        known_token_counts, all_token_counts,
        out=np.zeros(len(documents), dtype=np.float64), where=all_token_counts > 0,
    )
    require(np.isfinite(matrix.data).all() and np.isfinite(coverage).all()
            and np.all((coverage >= 0.0) & (coverage <= 1.0)),
            "Mean-count matrix or token coverage is invalid")
    return matrix, {
        "all_token_counts": all_token_counts,
        "known_token_counts": known_token_counts,
        "coverage": coverage,
        "unigram_terms": terms,
        "vocabulary_size": len(terms),
        "all_oov_rows": int(np.count_nonzero((all_token_counts > 0) & (known_token_counts == 0))),
        "empty_token_rows": int(np.count_nonzero(all_token_counts == 0)),
    }


def _as_csr(counts: Any) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    require(matrix.ndim == 2 and np.isfinite(matrix.data).all()
            and np.all(matrix.data >= 0.0), "Mean-count matrix must be finite and nonnegative")
    return matrix


def _validate_inputs(counts: Any, structured: Any, labels: Any | None = None) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray | None]:
    mean_counts = _as_csr(counts)
    features = np.asarray(structured, dtype=np.float64)
    require(features.ndim == 2 and features.shape[0] == mean_counts.shape[0]
            and np.isfinite(features).all(), "Structured features must be a finite aligned matrix")
    target = None if labels is None else np.asarray(labels, dtype=np.float64).reshape(-1)
    if target is not None:
        require(len(target) == mean_counts.shape[0] and np.isfinite(target).all()
                and np.isin(target, (0.0, 1.0)).all(), "Labels must be aligned binary 0/1")
    return mean_counts, features, target


def initialize_parameters(vocabulary_size: int, structured_width: int, architecture: str,
                          rng: np.random.Generator) -> dict[str, np.ndarray]:
    require(architecture in ARCHITECTURES, f"Unknown frozen architecture: {architecture}")
    require(vocabulary_size > 0 and structured_width >= 0,
            "Vocabulary and structured widths must be valid")
    input_width = EMBEDDING_WIDTH + structured_width
    params: dict[str, np.ndarray] = {
        "E": rng.normal(0.0, 0.05, size=(vocabulary_size, EMBEDDING_WIDTH)).astype(np.float64),
    }
    if architecture == "DAN":
        widths = (input_width, *DAN_HIDDEN_WIDTHS, 1)
        for layer, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:]), start=1):
            params[f"W{layer}"] = rng.normal(
                0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)
            ).astype(np.float64)
            params[f"b{layer}"] = np.zeros(fan_out, dtype=np.float64)
    else:
        params["Wout"] = rng.normal(
            0.0, np.sqrt(2.0 / input_width), size=(input_width, 1)
        ).astype(np.float64)
        params["bout"] = np.zeros(1, dtype=np.float64)
    return params


def _forward(
    params: Mapping[str, np.ndarray], counts: sparse.csr_matrix,
    structured: np.ndarray, architecture: str,
    dropout_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pooled = np.asarray(counts @ params["E"], dtype=np.float64)
    x = np.concatenate((pooled, structured), axis=1)
    cache: dict[str, np.ndarray] = {"pooled": pooled, "x": x}
    if architecture == "LINEAR":
        logits = (x @ params["Wout"] + params["bout"]).reshape(-1)
        return logits, cache
    require(architecture == "DAN", f"Unknown frozen architecture: {architecture}")
    z1 = x @ params["W1"] + params["b1"]
    h1 = np.maximum(z1, 0.0)
    h1d = h1 if dropout_masks is None else h1 * dropout_masks[0]
    z2 = h1d @ params["W2"] + params["b2"]
    h2 = np.maximum(z2, 0.0)
    h2d = h2 if dropout_masks is None else h2 * dropout_masks[1]
    logits = (h2d @ params["W3"] + params["b3"]).reshape(-1)
    cache.update({"z1": z1, "h1d": h1d, "z2": z2, "h2d": h2d})
    return logits, cache


def _regularized_parameters(params: Mapping[str, np.ndarray]) -> list[str]:
    return [name for name in params if name == "E" or name.startswith("W")]


def loss_and_gradients(
    params: Mapping[str, np.ndarray], counts: Any, structured: Any,
    labels: Any, architecture: str,
    dropout_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[float, dict[str, np.ndarray]]:
    """Mean BCE-with-logits plus the frozen nonbias L2 penalty and gradients."""
    mean_counts, features, target = _validate_inputs(counts, structured, labels)
    assert target is not None
    if dropout_masks is not None:
        require(architecture == "DAN" and len(dropout_masks) == 2,
                "Dropout masks apply only to the DAN hidden layers")
        require(dropout_masks[0].shape == (len(target), DAN_HIDDEN_WIDTHS[0])
                and dropout_masks[1].shape == (len(target), DAN_HIDDEN_WIDTHS[1]),
                "Dropout mask shapes do not align with the frozen hidden layers")
    logits, cache = _forward(params, mean_counts, features, architecture, dropout_masks)
    data_loss = float(np.mean(np.logaddexp(0.0, logits) - target * logits))
    penalty_names = _regularized_parameters(params)
    penalty = 0.5 * L2_COEFFICIENT * sum(float(np.sum(params[name] ** 2)) for name in penalty_names)
    probability = expit(logits)
    dlogits = ((probability - target) / len(target)).reshape(-1, 1)
    gradients: dict[str, np.ndarray] = {}

    if architecture == "LINEAR":
        gradients["Wout"] = cache["x"].T @ dlogits + L2_COEFFICIENT * params["Wout"]
        gradients["bout"] = np.sum(dlogits, axis=0)
        dinput = dlogits @ params["Wout"].T
    else:
        m1, m2 = dropout_masks if dropout_masks is not None else (None, None)
        gradients["W3"] = cache["h2d"].T @ dlogits + L2_COEFFICIENT * params["W3"]
        gradients["b3"] = np.sum(dlogits, axis=0)
        dh2 = dlogits @ params["W3"].T
        if m2 is not None:
            dh2 *= m2
        dz2 = dh2 * (cache["z2"] > 0.0)
        gradients["W2"] = cache["h1d"].T @ dz2 + L2_COEFFICIENT * params["W2"]
        gradients["b2"] = np.sum(dz2, axis=0)
        dh1 = dz2 @ params["W2"].T
        if m1 is not None:
            dh1 *= m1
        dz1 = dh1 * (cache["z1"] > 0.0)
        gradients["W1"] = cache["x"].T @ dz1 + L2_COEFFICIENT * params["W1"]
        gradients["b1"] = np.sum(dz1, axis=0)
        dinput = dz1 @ params["W1"].T

    dpooled = np.asarray(dinput[:, :EMBEDDING_WIDTH], dtype=np.float64)
    gradients["E"] = np.asarray(mean_counts.T @ dpooled, dtype=np.float64) + L2_COEFFICIENT * params["E"]
    total_loss = data_loss + penalty
    require(np.isfinite(total_loss) and all(np.isfinite(value).all() for value in gradients.values()),
            "Loss or gradient is non-finite")
    return total_loss, gradients


def _mean_loss(params: Mapping[str, np.ndarray], counts: sparse.csr_matrix,
               structured: np.ndarray, labels: np.ndarray, architecture: str,
               chunk_size: int = 1024) -> dict[str, float]:
    loss_sum = 0.0
    n = len(labels)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        logits, _ = _forward(params, counts[start:stop], structured[start:stop], architecture)
        y = labels[start:stop]
        loss_sum += float(np.sum(np.logaddexp(0.0, logits) - y * logits))
    data_loss = loss_sum / n
    penalty = 0.5 * L2_COEFFICIENT * sum(
        float(np.sum(params[name] ** 2)) for name in _regularized_parameters(params)
    )
    return {"data_bce": data_loss, "regularization": penalty,
            "total_objective": data_loss + penalty}


def adam_update(
    params: Mapping[str, np.ndarray], gradients: Mapping[str, np.ndarray],
    first_moment: Mapping[str, np.ndarray], second_moment: Mapping[str, np.ndarray],
    step: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One ordinary Adam step with the frozen constants, for training and QA."""
    require(step >= 0 and set(params) == set(gradients) == set(first_moment) == set(second_moment),
            "Adam state keys or step are invalid")
    t = step + 1
    new_params: dict[str, np.ndarray] = {}
    new_m: dict[str, np.ndarray] = {}
    new_v: dict[str, np.ndarray] = {}
    for name in params:
        gradient = np.asarray(gradients[name], dtype=np.float64)
        m = BETA1 * first_moment[name] + (1.0 - BETA1) * gradient
        v = BETA2 * second_moment[name] + (1.0 - BETA2) * gradient * gradient
        m_hat = m / (1.0 - BETA1 ** t)
        v_hat = v / (1.0 - BETA2 ** t)
        updated = params[name] - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)
        require(np.isfinite(updated).all() and np.isfinite(m).all() and np.isfinite(v).all(),
                f"Adam produced non-finite values for {name}")
        new_params[name], new_m[name], new_v[name] = updated, m, v
    return new_params, new_m, new_v


def fit_fixed_recipe(counts: Any, structured: Any, labels: Any,
                     architecture: str) -> dict[str, Any]:
    """Train one complete frozen DAN or LINEAR predictor; never early-stops."""
    mean_counts, features, target = _validate_inputs(counts, structured, labels)
    assert target is not None
    require(len(target) > 0 and architecture in ARCHITECTURES,
            "Official fit requires at least one row and a frozen architecture")
    rng = np.random.default_rng(RANDOM_SEED)
    params = initialize_parameters(mean_counts.shape[1], features.shape[1], architecture, rng)
    initial_loss = _mean_loss(params, mean_counts, features, target, architecture)
    first_moment = {name: np.zeros_like(value) for name, value in params.items()}
    second_moment = {name: np.zeros_like(value) for name, value in params.items()}
    step = 0
    epoch_batch_objectives: list[float] = []

    for _epoch in range(EPOCHS):
        order = rng.permutation(len(target))
        batch_objectives: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            masks = None
            if architecture == "DAN":
                masks = (
                    (rng.random((len(rows), DAN_HIDDEN_WIDTHS[0])) >= DROPOUT_RATE).astype(np.float64)
                    / (1.0 - DROPOUT_RATE),
                    (rng.random((len(rows), DAN_HIDDEN_WIDTHS[1])) >= DROPOUT_RATE).astype(np.float64)
                    / (1.0 - DROPOUT_RATE),
                )
            objective, gradients = loss_and_gradients(
                params, mean_counts[rows], features[rows], target[rows], architecture, masks
            )
            params, first_moment, second_moment = adam_update(
                params, gradients, first_moment, second_moment, step
            )
            step += 1
            batch_objectives.append(objective)
        epoch_batch_objectives.append(float(np.mean(batch_objectives)))

    final_loss = _mean_loss(params, mean_counts, features, target, architecture)
    require(step == EPOCHS * int(np.ceil(len(target) / BATCH_SIZE)),
            "Fit did not complete the exact frozen epoch/update count")
    require(all(np.isfinite(value).all() for value in params.values()),
            "Final model state contains non-finite parameters")
    return {
        "architecture": architecture,
        "params": params,
        "adam_first_moment": first_moment,
        "adam_second_moment": second_moment,
        "adam_steps": step,
        "training": {
            "epochs_completed": EPOCHS,
            "batch_size": BATCH_SIZE,
            "updates": step,
            "initial_no_dropout_loss": initial_loss,
            "final_no_dropout_loss": final_loss,
            "mean_minibatch_objective_by_epoch": epoch_batch_objectives,
            "converged_claim": False,
        },
    }


def predict_logits(model: Mapping[str, Any], counts: Any, structured: Any,
                   chunk_size: int = 1024) -> np.ndarray:
    mean_counts, features, _ = _validate_inputs(counts, structured)
    params = model["params"]
    architecture = str(model["architecture"])
    require(architecture in ARCHITECTURES and mean_counts.shape[1] == params["E"].shape[0]
            and features.shape[1] + EMBEDDING_WIDTH
            == (params["Wout"].shape[0] if architecture == "LINEAR" else params["W1"].shape[0]),
            "Prediction inputs do not align with the saved model architecture")
    outputs = np.empty(mean_counts.shape[0], dtype=np.float64)
    for start in range(0, len(outputs), chunk_size):
        stop = min(len(outputs), start + chunk_size)
        outputs[start:stop], _ = _forward(
            params, mean_counts[start:stop], features[start:stop], architecture
        )
    require(np.isfinite(outputs).all(), "Model logits contain non-finite values")
    return outputs


def predict_probabilities(model: Mapping[str, Any], counts: Any, structured: Any,
                          chunk_size: int = 1024) -> np.ndarray:
    probabilities = expit(predict_logits(model, counts, structured, chunk_size))
    require(np.isfinite(probabilities).all() and np.all((probabilities >= 0.0) & (probabilities <= 1.0)),
            "Model probabilities are invalid")
    return probabilities


def model_arrays(model: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return a pickle-free, deterministic NPZ payload including Adam buffers."""
    architecture = str(model["architecture"])
    metadata = {k: v for k, v in model.items()
                if k not in {"params", "adam_first_moment", "adam_second_moment"}}
    payload: dict[str, np.ndarray] = {
        "architecture": np.asarray(architecture),
        "adam_steps": np.asarray(int(model["adam_steps"]), dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    }
    for prefix, key in (("param", "params"), ("adam_m", "adam_first_moment"),
                        ("adam_v", "adam_second_moment")):
        for name, value in model[key].items():
            payload[f"{prefix}_{name}"] = np.asarray(value, dtype=np.float64)
    return payload


def model_from_arrays(payload: Mapping[str, Any]) -> dict[str, Any]:
    architecture = str(np.asarray(payload["architecture"]).item())
    require(architecture in ARCHITECTURES, "Saved model architecture is invalid")
    metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    names = [name[6:] for name in payload if name.startswith("param_")]
    require(names and set(names) == {name[7:] for name in payload if name.startswith("adam_m_")}
            == {name[7:] for name in payload if name.startswith("adam_v_")},
            "Saved model parameter/Adam buffer schemas differ")
    params = {name: np.asarray(payload[f"param_{name}"], dtype=np.float64).copy() for name in names}
    first = {name: np.asarray(payload[f"adam_m_{name}"], dtype=np.float64).copy() for name in names}
    second = {name: np.asarray(payload[f"adam_v_{name}"], dtype=np.float64).copy() for name in names}
    require(all(params[k].shape == first[k].shape == second[k].shape for k in names),
            "Saved model Adam buffers have invalid shapes")
    for prefix, arrays in (("parameters", params), ("Adam first moments", first),
                           ("Adam second moments", second)):
        require(all(np.isfinite(value).all() for value in arrays.values()),
                f"Saved model {prefix} contain non-finite values")
    return {
        **metadata,
        "architecture": architecture,
        "params": params,
        "adam_first_moment": first,
        "adam_second_moment": second,
        "adam_steps": int(np.asarray(payload["adam_steps"]).item()),
    }
