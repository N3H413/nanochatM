"""
Rare-word recall eval (Track A).

Measures whether the model assigns high probability / high rank to the TRUE
next token specifically at positions where that token is "rare" (per your
engram's own token_blocked definition), vs "common" positions -- and lets you
compare that gap across model variants (baseline / different block_fraction).

Usage sketch (fill in your own tokenizer + text):

    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()

    val_texts = [...]  # list of raw strings, held-out from training if possible

    probes_rare, probes_common = build_probes(
        val_texts, tokenizer, model.engram.token_blocked,
        max_context=model.config.sequence_len, max_probes_per_bucket=300,
    )

    results = evaluate_probes(model, probes_rare, device)
    print(summarize(results, "RARE"))
    results_c = evaluate_probes(model, probes_common, device)
    print(summarize(results_c, "COMMON"))
"""
import math
import random
import torch


def build_probes(texts, tokenizer, token_blocked, max_context=256,
                  max_probes_per_bucket=300, max_repeats_per_token=3, min_context=8, seed=0):
    """Scan `texts`, find every position where the next token is "rare" (kept,
    i.e. token_blocked[token] is False) or "common" (token_blocked[token] is
    True), and build (context_ids, target_id) probes for both buckets.

    max_repeats_per_token caps how many probes come from the same target token
    id, so the rare bucket isn't dominated by whichever rare word happens to
    repeat the most in your sample text -- you want breadth across many
    distinct rare tokens, not depth on one or two.
    """
    rng = random.Random(seed)
    rare_probes = []
    common_probes = []
    rare_counts = {}
    common_counts = {}

    all_texts = list(texts)
    rng.shuffle(all_texts)

    for text in all_texts:
        ids = tokenizer.encode(text)
        if len(ids) <= min_context + 1:
            continue
        for i in range(min_context, len(ids)):
            target = ids[i]
            is_blocked = bool(token_blocked[target].item())
            context = ids[max(0, i - max_context):i]

            if not is_blocked:  # rare / kept
                if rare_counts.get(target, 0) >= max_repeats_per_token:
                    continue
                if len(rare_probes) >= max_probes_per_bucket:
                    continue
                rare_probes.append((context, target))
                rare_counts[target] = rare_counts.get(target, 0) + 1
            else:  # common / blocked
                if common_counts.get(target, 0) >= max_repeats_per_token:
                    continue
                if len(common_probes) >= max_probes_per_bucket:
                    continue
                common_probes.append((context, target))
                common_counts[target] = common_counts.get(target, 0) + 1

        if len(rare_probes) >= max_probes_per_bucket and len(common_probes) >= max_probes_per_bucket:
            break

    print(f"Built {len(rare_probes)} rare-word probes covering {len(rare_counts)} distinct tokens")
    print(f"Built {len(common_probes)} common-word probes covering {len(common_counts)} distinct tokens")
    if len(rare_probes) < max_probes_per_bucket:
        print(f"  (note: only found {len(rare_probes)}/{max_probes_per_bucket} rare probes -- "
              f"pass more/longer texts, or lower max_probes_per_bucket)")
    return rare_probes, common_probes


@torch.no_grad()
def evaluate_probes(model, probes, device, top_k=(1, 5, 20)):
    """For each (context, target), run the model and record where the TRUE
    target token landed in the model's own ranking of next-token predictions.
    """
    model.eval()
    ranks = []
    target_probs = []
    top1_preds = []  # (target, predicted) for qualitative inspection

    for context, target in probes:
        idx = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(idx)  # (1, T, vocab) -- targets=None path
        last_logits = logits[0, -1]  # (vocab,)
        probs = torch.softmax(last_logits.float(), dim=-1)

        target_prob = probs[target].item()
        # rank: how many tokens have strictly higher probability than the target (1 = best possible)
        rank = int((probs > probs[target]).sum().item()) + 1
        pred = int(torch.argmax(probs).item())

        ranks.append(rank)
        target_probs.append(max(target_prob, 1e-12))  # avoid log(0)
        top1_preds.append((target, pred))

    return {"ranks": ranks, "target_probs": target_probs, "top1_preds": top1_preds, "top_k": top_k}


def summarize(results, label, tokenizer=None):
    ranks = results["ranks"]
    target_probs = results["target_probs"]
    n = len(ranks)
    if n == 0:
        return f"[{label}] no probes evaluated"

    lines = [f"--- {label} (n={n}) ---"]
    for k in results["top_k"]:
        acc_at_k = sum(1 for r in ranks if r <= k) / n
        lines.append(f"  top-{k} accuracy: {100*acc_at_k:.1f}%")

    mean_rank = sum(ranks) / n
    median_rank = sorted(ranks)[n // 2]
    mean_neg_log2_prob = sum(-math.log2(p) for p in target_probs) / n  # "bits" to encode the true token

    lines.append(f"  mean rank of true token: {mean_rank:.1f}   median rank: {median_rank}")
    lines.append(f"  mean bits to encode true token (-log2 P): {mean_neg_log2_prob:.3f}  (lower = more confident/correct)")

    # A handful of wrong predictions for qualitative eyeballing, decoded if a tokenizer is given
    wrong = [(t, p) for (t, p), r in zip(results["top1_preds"], ranks) if r > 1]
    if wrong and tokenizer is not None:
        lines.append("  sample misses (true -> predicted):")
        for t, p in wrong[:8]:
            try:
                t_str = tokenizer.decode([t])
                p_str = tokenizer.decode([p])
            except Exception:
                t_str, p_str = str(t), str(p)
            lines.append(f"    {t_str!r} -> {p_str!r}")

    return "\n".join(lines)
