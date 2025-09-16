import pandas as pd

def filter_min_interactions(ratings: pd.DataFrame, min_user=5, min_item=20):
    r = ratings.copy()
    # Filter users
    vc_u = r['userId'].value_counts()
    keep_users = vc_u[vc_u >= min_user].index
    r = r[r['userId'].isin(keep_users)]
    # Filter items
    vc_i = r['movieId'].value_counts()
    keep_items = vc_i[vc_i >= min_item].index
    r = r[r['movieId'].isin(keep_items)]
    return r.reset_index(drop=True)

def last_item_test_split(ratings: pd.DataFrame):
    # For each user, take their last interaction (by timestamp if available) as test, rest train
    r = ratings.copy()
    if 'timestamp' in r.columns:
        r = r.sort_values(['userId','timestamp'])
    else:
        r = r.sort_values(['userId'])
    test_idx = r.groupby('userId').tail(1).index
    test = r.loc[test_idx]
    train = r.drop(index=test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)
