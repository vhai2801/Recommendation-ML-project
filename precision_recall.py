import numpy as np

def precision_recall_at_k_1(test_df, global_mean, k, threshold=3.5):
    user_groups = test_df.groupby('userId')

    precisions = []
    recalls = []

    for uid, group in user_groups:
        top_k = group.head(k)
        all_user = group

        n_rel_and_rec = sum(
            true_r >= threshold
            for true_r in top_k['rating']
            if global_mean >= threshold
        )

        n_rec = k if global_mean >= threshold else 0
        n_rel = sum(all_user['rating'] >= threshold)

        if n_rec > 0:
            precisions.append(n_rel_and_rec / n_rec)
        if n_rel > 0:
            recalls.append(n_rel_and_rec / n_rel)

    return np.mean(precisions) if precisions else 0, np.mean(recalls) if recalls else 0


def precision_recall_at_k_2(predictions, k, threshold=3.5):
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel = sum(true_r >= threshold for (_, true_r) in user_ratings)
        n_rec_k = sum(est >= threshold for (est, _)    in top_k)
        n_rel_and_rec_k = sum(est >= threshold and true_r >= threshold for (est, true_r) in top_k)

        if n_rec_k > 0:
            precisions.append(n_rel_and_rec_k / n_rec_k)
        if n_rel > 0:
            recalls.append(n_rel_and_rec_k / n_rel)

    return np.mean(precisions) if precisions else 0, np.mean(recalls) if recalls else 0


def precision_recall_at_k_3(test_df, predictions, k, threshold=3.5):
    df = test_df.copy()
    df['pred'] = predictions

    precisions = []
    recalls = []

    for _, group in df.groupby('userId'):
        group_sorted = group.sort_values('pred', ascending=False)
        top_k = group_sorted.head(k)

        n_rel = sum(group['rating'] >= threshold)
        n_rec_k = sum(top_k['pred']   >= threshold)
        n_rel_and_rec_k = sum((top_k['pred'] >= threshold) & (top_k['rating'] >= threshold))

        if n_rec_k > 0:
            precisions.append(n_rel_and_rec_k / n_rec_k)
        if n_rel > 0:
            recalls.append(n_rel_and_rec_k / n_rel)

    return np.mean(precisions) if precisions else 0, np.mean(recalls) if recalls else 0
