"""Decoding and metrics matching the approved HAMOS evaluation protocol."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


CATEGORY_NAMES = ["AMENITY", "BRANDING", "EXPERIENCE", "FACILITY", "LOYALTY", "SERVICE"]
SENTIMENT_NAMES = ["negative", "neutral", "positive"]


def decode_outputs(outputs, objectness_threshold, flag_threshold=0.5):
    result = []
    obj = outputs["objectness"].detach().cpu().sigmoid().numpy()
    cat = outputs["category"].detach().cpu().argmax(-1).numpy()
    sent = outputs["sentiment"].detach().cpu().argmax(-1).numpy()
    ai = outputs["aspect_implicit"].detach().cpu().sigmoid().numpy()
    oi = outputs["opinion_implicit"].detach().cpu().sigmoid().numpy()
    for b in range(len(obj)):
        items = []
        for j in range(obj.shape[1]):
            if obj[b, j] >= objectness_threshold:
                items.append({"category": int(cat[b, j]), "sentiment": int(sent[b, j]),
                              "aspect_implicit": int(ai[b, j] >= flag_threshold),
                              "opinion_implicit": int(oi[b, j] >= flag_threshold),
                              "slot": j})
        result.append(items)
    return result


def _assign(predictions, gold, keys, exact_first=False):
    """Maximum-cardinality assignment; optionally prioritize exact full tuples."""
    if not predictions or not gold:
        return [], [], list(range(len(predictions))), list(range(len(gold)))
    n = min(len(predictions), len(gold))
    matrix = np.zeros((len(predictions), len(gold)), dtype=np.float64)
    max_hamming = len(keys)
    exact_bonus = max_hamming * n + 1.0
    for i, pred in enumerate(predictions):
        for j, target in enumerate(gold):
            mismatches = sum(int(pred[k] != target[k]) for k in keys)
            matrix[i, j] = mismatches
            if exact_first and mismatches:
                matrix[i, j] += exact_bonus
    pi, gi = linear_sum_assignment(matrix)
    pi, gi = pi.tolist(), gi.tolist()
    up = sorted(set(range(len(predictions))) - set(pi))
    ug = sorted(set(range(len(gold))) - set(gi))
    return pi, gi, up, ug


def _class_scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def _multiclass_stats(predictions, gold, labels, names=None):
    counts = {label: [0, 0, 0] for label in labels}  # TP, FP, FN
    for pred, target in zip(predictions, gold):
        if pred is None:
            counts[target][2] += 1
        elif target is None:
            counts[pred][1] += 1
        elif pred == target:
            counts[target][0] += 1
        else:
            counts[pred][1] += 1
            counts[target][2] += 1
    names = names or {label: str(label) for label in labels}
    per_class = {names[k]: _class_scores(*v) for k, v in counts.items()}
    micro = _class_scores(sum(v[0] for v in counts.values()),
                          sum(v[1] for v in counts.values()), sum(v[2] for v in counts.values()))
    micro["macro_f1"] = float(np.mean([v["f1"] for v in per_class.values()])) if labels else 0.0
    micro["per_class"] = per_class
    micro["class_order"] = [names[k] for k in labels]
    micro["gold_support_by_class"] = {names[k]: int(counts[k][0] + counts[k][2]) for k in labels}
    micro["predicted_support_by_class"] = {names[k]: int(counts[k][0] + counts[k][1]) for k in labels}
    return micro


def _primary_counts(pred, gold):
    gold_imp = [g for g in gold if g["aspect_implicit"]]
    pred_imp = [p for p in pred if p["aspect_implicit"]]
    pi, gi, _, _ = _assign(pred_imp, gold_imp, ("category", "sentiment"), exact_first=True)
    tp = sum(pred_imp[a]["category"] == gold_imp[b]["category"] and
             pred_imp[a]["sentiment"] == gold_imp[b]["sentiment"] for a, b in zip(pi, gi))
    return tp, len(pred_imp) - tp, len(gold_imp) - tp


def _implicit_component_scores(predictions, gold_sets, key, labels, names):
    """Component F1 over implicit-gated multisets, matching each component independently."""
    pred_values, gold_values = [], []
    for pred, gold in zip(predictions, gold_sets):
        pred_imp = [p for p in pred if p["aspect_implicit"]]
        gold_imp = [g for g in gold if g["aspect_implicit"]]
        pi, gi, up, ug = _assign(pred_imp, gold_imp, (key,), exact_first=True)
        for p_idx, g_idx in zip(pi, gi):
            pred_values.append(pred_imp[p_idx][key])
            gold_values.append(gold_imp[g_idx][key])
        for p_idx in up:
            pred_values.append(pred_imp[p_idx][key])
            gold_values.append(None)
        for g_idx in ug:
            pred_values.append(None)
            gold_values.append(gold_imp[g_idx][key])
    scores = _multiclass_stats(pred_values, gold_values, labels, names)
    scores["assignment"] = f"independent exact-first one-to-one matching on {key}"
    scores["prediction_gate"] = "aspect_implicit=true"
    return scores


def _confusion_payload(matrix, class_names):
    return {"rows_gold": list(class_names),
            "columns_prediction": list(class_names) + ["MISS"],
            "values": matrix.tolist(),
            "unmatched_predicted_items": "ignored by the gold-implicit-conditioned diagnostic"}


def _macro_support_status(metric, minimum_class_support):
    supports = list(metric["gold_support_by_class"].values())
    min_support = min(supports, default=0)
    if min_support < minimum_class_support:
        metric["macro_f1_status"] = (
            f"withheld: every fixed class requires at least {minimum_class_support} gold examples; "
            f"minimum observed support is {min_support}")
        metric["macro_f1"] = None
    else:
        metric["macro_f1_status"] = (
            f"reported: every fixed class has at least {minimum_class_support} gold examples")


def evaluate_structured(predictions, gold_sets, *, category_names=None, sentiment_names=None):
    """Score a list of decoded prediction sets against integer-encoded gold sets."""
    if len(predictions) != len(gold_sets):
        raise ValueError("predictions and gold_sets must have equal length")
    category_names = list(category_names or CATEGORY_NAMES)
    sentiment_names = list(sentiment_names or SENTIMENT_NAMES)
    if len(category_names) != 6 or len(sentiment_names) != 3:
        raise ValueError("expected names for six categories and three sentiments")
    full_keys = ("category", "sentiment", "aspect_implicit", "opinion_implicit")
    total_tp = total_fp = total_fn = exact_segments = cardinality_abs = cardinality_signed = 0
    field_pred, field_gold = defaultdict(list), defaultdict(list)
    primary_tp = primary_fp = primary_fn = 0
    conditional_cat_pred, conditional_cat_gold = [], []
    conditional_sent_pred, conditional_sent_gold = [], []
    conditional_exact_pair = conditional_gold_count = 0
    diagnostic_cat_matrix = np.zeros((6, 7), dtype=np.int64)
    diagnostic_sent_matrix = np.zeros((3, 4), dtype=np.int64)
    aspect_fp = aspect_fn = opinion_fp = opinion_fn = 0

    for pred, gold in zip(predictions, gold_sets):
        pi, gi, up, ug = _assign(pred, gold, full_keys, exact_first=True)
        exact_here = 0
        for a, b in zip(pi, gi):
            p, g = pred[a], gold[b]
            is_exact = all(p[k] == g[k] for k in full_keys)
            exact_here += int(is_exact)
            for key in full_keys:
                field_pred[key].append(p[key]); field_gold[key].append(g[key])
            if g["aspect_implicit"] and p["aspect_implicit"] != g["aspect_implicit"]:
                aspect_fn += 1
            if not g["aspect_implicit"] and p["aspect_implicit"]:
                aspect_fp += 1
            if g["opinion_implicit"] and not p["opinion_implicit"]:
                opinion_fn += 1
            if not g["opinion_implicit"] and p["opinion_implicit"]:
                opinion_fp += 1
        for idx in up:
            p = pred[idx]
            for key in full_keys:
                field_pred[key].append(p[key]); field_gold[key].append(None)
            if p["aspect_implicit"]: aspect_fp += 1
            if p["opinion_implicit"]: opinion_fp += 1
        for idx in ug:
            g = gold[idx]
            for key in full_keys:
                field_pred[key].append(None); field_gold[key].append(g[key])
            if g["aspect_implicit"]: aspect_fn += 1
            if g["opinion_implicit"]: opinion_fn += 1
        tps = exact_here
        total_tp += tps
        total_fp += len(pred) - tps
        total_fn += len(gold) - tps
        exact_segments += int(tps == len(pred) == len(gold))
        cardinality_abs += abs(len(pred) - len(gold))
        cardinality_signed += len(pred) - len(gold)

        imp_tp, imp_fp, imp_fn = _primary_counts(pred, gold)
        primary_tp += imp_tp
        primary_fp += imp_fp
        primary_fn += imp_fn

        # Independent gold-implicit-conditioned assignment (no flag keys/costs).
        gold_imp = [g for g in gold if g["aspect_implicit"]]
        cpi, cgi, _, cug = _assign(pred, gold_imp, ("category", "sentiment"), exact_first=True)
        for a, b in zip(cpi, cgi):
            p, g = pred[a], gold_imp[b]
            conditional_cat_pred.append(p["category"]); conditional_cat_gold.append(g["category"])
            conditional_sent_pred.append(p["sentiment"]); conditional_sent_gold.append(g["sentiment"])
            diagnostic_cat_matrix[g["category"], p["category"]] += 1
            diagnostic_sent_matrix[g["sentiment"], p["sentiment"]] += 1
            conditional_exact_pair += int(p["category"] == g["category"] and p["sentiment"] == g["sentiment"])
        for b in cug:
            g = gold_imp[b]
            conditional_cat_pred.append(None); conditional_cat_gold.append(g["category"])
            conditional_sent_pred.append(None); conditional_sent_gold.append(g["sentiment"])
            diagnostic_cat_matrix[g["category"], 6] += 1
            diagnostic_sent_matrix[g["sentiment"], 3] += 1
        conditional_gold_count += len(gold_imp)

    set_scores = _class_scores(total_tp, total_fp, total_fn)
    # Field-wise classification scores from the deterministic full-set assignment.
    label_maps = {"category": range(6), "sentiment": range(3),
                  "aspect_implicit": range(2), "opinion_implicit": range(2)}
    names_by_field = {"category": category_names, "sentiment": sentiment_names,
                      "aspect_implicit": ["explicit", "implicit"],
                      "opinion_implicit": ["explicit", "implicit"]}
    component_scores = {}
    for key, labels in label_maps.items():
        pvals, gvals = field_pred[key], field_gold[key]
        counts = {label: [0, 0, 0] for label in labels}
        for p, g in zip(pvals, gvals):
            if p is None:
                counts[g][2] += 1
            elif g is None:
                counts[p][1] += 1
            elif p == g:
                counts[g][0] += 1
            else:
                counts[p][1] += 1; counts[g][2] += 1
        per_class = {names_by_field[key][k]: _class_scores(*v) for k, v in counts.items()}
        component_scores[key] = {"macro_f1": float(np.mean([x["f1"] for x in per_class.values()])),
                                 "per_class": per_class,
                                 "class_order": list(names_by_field[key]),
                                 "gold_support_by_class": {names_by_field[key][k]: int(v[0] + v[2])
                                                            for k, v in counts.items()},
                                 "predicted_support_by_class": {names_by_field[key][k]: int(v[0] + v[1])
                                                                 for k, v in counts.items()}}
    cond_cat = _multiclass_stats(conditional_cat_pred, conditional_cat_gold, range(6), category_names)
    cond_sent = _multiclass_stats(conditional_sent_pred, conditional_sent_gold, range(3), sentiment_names)
    end_to_end_cat = _implicit_component_scores(predictions, gold_sets, "category", range(6), category_names)
    end_to_end_sent = _implicit_component_scores(predictions, gold_sets, "sentiment", range(3), sentiment_names)
    primary = _class_scores(primary_tp, primary_fp, primary_fn)
    return {
        "primary_implicit_category_sentiment": primary,
        "gold_implicit_conditioned": {"category": cond_cat, "sentiment": cond_sent,
            "exact_category_sentiment_rate": conditional_exact_pair / conditional_gold_count if conditional_gold_count else 0.0,
            "gold_implicit_count": conditional_gold_count,
            "confusion_matrices": {
                "category": _confusion_payload(diagnostic_cat_matrix, category_names),
                "sentiment": _confusion_payload(diagnostic_sent_matrix, sentiment_names)}},
        "end_to_end_implicit_components": {"category": end_to_end_cat, "sentiment": end_to_end_sent},
        "complete_set": {**set_scores, "exact_set_accuracy": exact_segments / len(gold_sets) if gold_sets else 0.0},
        "components": component_scores,
        "aspect_implicit_positive": component_scores["aspect_implicit"]["per_class"]["implicit"],
        "opinion_implicit_positive": component_scores["opinion_implicit"]["per_class"]["implicit"],
        "cardinality": {"mae": cardinality_abs / len(gold_sets) if gold_sets else 0.0,
                        "signed_bias": cardinality_signed / len(gold_sets) if gold_sets else 0.0},
        "segments": len(gold_sets),
    }


def select_objectness_threshold(outputs, gold_sets, grid):
    """Choose global threshold by dev primary F1; ties go to highest threshold."""
    results = []
    for threshold in grid:
        predictions = decode_outputs(outputs, float(threshold))
        counts = np.sum(np.asarray([_primary_counts(p, g) for p, g in zip(predictions, gold_sets)], dtype=np.int64), axis=0)
        score = _class_scores(counts[0], counts[1], counts[2])["f1"]
        results.append((score, float(threshold), counts.tolist()))
    best_score = max(x[0] for x in results)
    tied = [x for x in results if np.isclose(x[0], best_score, rtol=0, atol=1e-12)]
    score, threshold, counts = max(tied, key=lambda x: x[1])
    return {"threshold": threshold, "development_primary_f1": score,
            "development_primary_counts": {"tp": counts[0], "fp": counts[1], "fn": counts[2]},
            "grid": [float(x) for x in grid], "tie_break": "highest threshold"}


def paired_hotel_bootstrap(text_predictions, multimodal_predictions, gold_sets, hotel_ids,
                           *, replicates=2000, seed=1729):
    """Paired hotel-cluster bootstrap for multimodal-minus-text primary F1."""
    if not (len(text_predictions) == len(multimodal_predictions) == len(gold_sets) == len(hotel_ids)):
        raise ValueError("bootstrap inputs must have equal lengths")
    hotels = sorted(set(hotel_ids))
    rng = np.random.default_rng(seed)
    # Metric components are segment-additive; calculate assignments once, then
    # resample cluster-level sufficient statistics rather than rescoring millions
    # of repeated segments across bootstrap replicates.
    grouped = {"text": defaultdict(lambda: np.zeros(3, dtype=np.int64)),
               "multimodal": defaultdict(lambda: np.zeros(3, dtype=np.int64))}
    for tpred, mpred, gold, hotel in zip(text_predictions, multimodal_predictions, gold_sets, hotel_ids):
        grouped["text"][hotel] += np.asarray(_primary_counts(tpred, gold), dtype=np.int64)
        grouped["multimodal"][hotel] += np.asarray(_primary_counts(mpred, gold), dtype=np.int64)
    deltas, text_scores, multimodal_scores = [], [], []
    multimodal_wins = ties = text_wins = 0
    for _ in range(replicates):
        sampled = rng.choice(hotels, size=len(hotels), replace=True)
        tc = sum((grouped["text"][h] for h in sampled), start=np.zeros(3, dtype=np.int64))
        mc = sum((grouped["multimodal"][h] for h in sampled), start=np.zeros(3, dtype=np.int64))
        t = _class_scores(tc[0], tc[1], tc[2])["f1"]
        m = _class_scores(mc[0], mc[1], mc[2])["f1"]
        text_scores.append(t); multimodal_scores.append(m); deltas.append(m - t)
        if m > t:
            multimodal_wins += 1
        elif m < t:
            text_wins += 1
        else:
            ties += 1
    def interval(values):
        return [float(x) for x in np.quantile(values, [0.025, 0.975])]
    return {"replicates": replicates, "seed": seed,
            "text_primary_f1_ci95": interval(text_scores),
            "multimodal_primary_f1_ci95": interval(multimodal_scores),
            "difference_ci95": interval(deltas),
            "mean_difference": float(np.mean(deltas)),
            "multimodal_better_replicates": multimodal_wins,
            "ties": ties,
            "text_only_better_replicates": text_wins,
            "proportion_multimodal_better": multimodal_wins / replicates if replicates else 0.0}


def evaluate_by_language(predictions, gold_sets, languages, *,
                         category_names=None, sentiment_names=None, min_class_support=5):
    """Evaluate language-filtered segment subsets with explicit support and macro gating."""
    lengths = {len(predictions), len(gold_sets), len(languages)}
    if len(lengths) != 1:
        raise ValueError("per-language inputs must have equal lengths")
    category_names = list(category_names or CATEGORY_NAMES)
    sentiment_names = list(sentiment_names or SENTIMENT_NAMES)
    output = {}
    for language in sorted(set(languages)):
        indices = [i for i, value in enumerate(languages) if value == language]
        language_predictions = [predictions[i] for i in indices]
        language_gold = [gold_sets[i] for i in indices]
        metrics = evaluate_structured(language_predictions, language_gold,
                                      category_names=category_names,
                                      sentiment_names=sentiment_names)
        gold_implicit = [g for gold in language_gold for g in gold if g["aspect_implicit"]]
        primary = metrics["primary_implicit_category_sentiment"]
        for metric in metrics["end_to_end_implicit_components"].values():
            _macro_support_status(metric, min_class_support)
        for metric in metrics["gold_implicit_conditioned"]["category"], metrics["gold_implicit_conditioned"]["sentiment"]:
            _macro_support_status(metric, min_class_support)
        for metric in metrics["components"].values():
            _macro_support_status(metric, min_class_support)
        output[language] = {
            "support": {"eligible_segments": len(indices),
                        "gold_quad_count": sum(len(gold) for gold in language_gold),
                        "gold_implicit_quad_count": len(gold_implicit),
                        "primary_predicted_implicit_count": primary["tp"] + primary["fp"],
                        "primary_gold_implicit_count": primary["tp"] + primary["fn"],
                        "gold_implicit_category_support": {
                            name: sum(g["category"] == idx for g in gold_implicit)
                            for idx, name in enumerate(category_names)},
                        "gold_implicit_sentiment_support": {
                            name: sum(g["sentiment"] == idx for g in gold_implicit)
                            for idx, name in enumerate(sentiment_names)},
                        "implicitness_macro_min_class_support": min_class_support},
            "metrics": metrics}
    return output
