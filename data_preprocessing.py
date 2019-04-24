import numpy as np
import pandas as pd

from polara import RecommenderData
from polara import get_movielens_data as get_ml_data
from polara import get_bookcrossing_data as get_bx_data
from polara import get_amazon_data as get_az_data
from polara.lib.similarity import stack_features, cosine_similarity
from polara.recommender.coldstart.data import ColdStartSimilarityDataModel
from polara.recommender.hybrid.data import IdentityDiagonalMixin, SideRelationsMixin


class SimilarityDataModel(SideRelationsMixin, RecommenderData): pass


def get_movielens_data(path, fixes_path=None, meta_path=None, implicit=False, filter_data=None, filter_no_meta=False):
    ratings = get_ml_data(path)
    meta_info = None
    
    if implicit:
        ratings = ratings.query('rating>=4').assign(rating=1)
    
    if fixes_path is not None:
        id_fix = pd.read_csv(fixes_path)
        fix_col = id_fix.columns[0]
        ratings.movieid.replace(id_fix.set_index(fix_col).movieid, inplace=True)
        ratings = ratings.drop_duplicates()

    if meta_path is not None:
        meta_info = (pd.read_csv(meta_path, sep=';', na_filter=False).set_index('movieid')
                     .reindex(index=ratings['movieid'].unique(), fill_value=[]))
        meta_cols = meta_info.columns
        meta_info.loc[:, meta_cols] = (meta_info.loc[:, meta_cols]
                                       .applymap(lambda x: x.split(',') if x else []))
        
        if filter_data:
            for field, values in filter_data.items():
                meta_info.loc[:, field] = meta_info[field].apply(lambda x: [v for v in x if v not in values])
    
        if filter_no_meta:
            not_empty = meta_info.applymap(len).sum(axis=1) > 0
            if not not_empty.all():
                meta_info = meta_info.loc[not_empty]
                ratings = ratings.query(f'movieid in @meta_info.index')
    return ratings, meta_info


def get_bookcrossing_data(path, get_books_meta=False, implicit=False, pcore=5, filter_data=None, filter_no_meta=False):
    data = get_bx_data(path, get_ratings=True, get_books=get_books_meta)
    ratings = data
    meta_info = None
    
    if get_books_meta:
        ratings, books_meta = data
    
    if implicit:
        ratings = ratings.query('rating==0').assign(rating=1)
    else:
        ratings = ratings.query('rating>0').assign(rating=lambda x: x['rating'] / 2)
    
    too_active_users = ratings['userid'].value_counts().pipe(lambda x: x[x > 1000].index)
    ratings = ratings.query(f'userid not in @too_active_users')

    if get_books_meta:
        meta_info = (books_meta.set_index('isbn')
                     .reindex(index=ratings['isbn'].unique(), fill_value=np.NaN)
                     .applymap(lambda x: [x] if x==x else []))
        
        if filter_data:
            for field, values in filter_data.items():
                meta_info.loc[:, field] = meta_info[field].apply(lambda x: [v for v in x if v not in values])
                
        if filter_no_meta:
            not_empty = meta_info.applymap(len).sum(axis=1) > 0
            if not not_empty.all():
                meta_info = meta_info.loc[not_empty]
                ratings = ratings.query(f'isbn in @meta_info.index')
    
    while pcore: # do only if pcore is specified
        valid_users = ratings['userid'].value_counts() >= pcore
        valid_items = ratings['isbn'].value_counts() >= pcore
        clean_data = (valid_items.all() and valid_users.all())
        if clean_data:
            break
        else:
            ratings = ratings.query('userid in @valid_users.loc[@valid_users].index and '
                                    'isbn in @valid_items.loc[@valid_items].index')
    return ratings, meta_info


def get_amazon_data(path, meta_path=None, implicit=False, pcore=5, filter_data=None, filter_no_meta=False, flat_categories=False):
    ratings = get_az_data(path)
    
    if implicit:
        ratings = ratings.query('rating>=4').assign(rating=1)
        
    meta_info = None
    if meta_path:
        item_meta = get_az_data(meta_path=meta_path)
        meta_info = (item_meta.loc[:, ['asin', 'categories', 'brand']]
                     .set_index('asin')
                     .reindex(index=ratings['asin'].unique(), fill_value=[])
                     .assign(brand=lambda x: x['brand'].apply(lambda b: [b] if b==b else []))
                     .assign(categories=lambda x: x['categories'].apply(lambda c: [f for v in c for f in v]))
                    )
                
        if filter_data:
            for field, values in filter_data.items():
                meta_info.loc[:, field] = meta_info[field].apply(lambda x: [v for v in x if v not in values])
        
        if flat_categories:
            top_level_cat = { # according to the distributed data
                'Books',
                'Electronics',
                'Movies and TV',
                'CDs and Vinyl',
                'Clothing, Shoes and Jewelry',
                'Home and Kitchen',
                'Kindle Store',
                'Sports and Outdoors',
                'Cell Phones and Accessories',
                'Health and Personal Care',
                'Toys and Games',
                'Video Games',
                'Tools and Home Improvement',
                'Beauty',
                'Apps for Android',
                'Office Products',
                'Pet Supplies',
                'Automotive',
                'Grocery and Gourmet Food',
                'Patio, Lawn and Garden',
                'Baby',
                'Digital Music',
                'Musical Instruments',
                'Amazon Instant Video',
            }
            top_level_cat = top_level_cat.union({item.replace('and', '&') for item in top_level_cat})
            # take only the most specific category;
            # some items have multiple category hierarchies - take the last category from each hierarchy
            # such categories are located 1 step before a new top level category
            cats = meta_info['categories']
            end_cats = cats.apply(lambda s: s[-1:] if s[-1] not in top_level_cat else [])
            meta_info.loc[:, 'categories'] = (cats.apply(lambda x: [x[i-1] for i, c in enumerate(x)
                                                                    if (c in top_level_cat) and (i>0)])
                                                  .combine(end_cats, lambda x, y: x+y) # handle last hierarchy
                                             )

        if filter_no_meta:
            not_empty = meta_info.applymap(len).sum(axis=1) > 0
            if not not_empty.all():
                meta_info = meta_info.loc[not_empty]
                ratings = ratings.query(f'asin in @meta_info.index')
                
    while pcore: # do only if pcore is specified
        valid_users = ratings['userid'].value_counts() >= pcore
        valid_items = ratings['asin'].value_counts() >= pcore
        clean_data = (valid_items.all() and valid_users.all())
        if clean_data:
            break
        else:
            valid_users = valid_users.loc[valid_users].index
            valid_items = valid_items.loc[valid_items].index
            ratings = ratings.query('userid in @valid_users and '
                                    'asin in @valid_items')
            meta_info = meta_info.query('asin in @valid_items')
    return ratings, meta_info


def get_yahoo_music_data(path, meta_path=None, implicit=False, pcore=5, filter_data=None, filter_no_meta=False):
    ratings = pd.read_csv(path)

    meta_info = None
    if meta_path:
        item_meta = pd.read_csv(meta_path, index_col=0)
        meta_info = item_meta.applymap(lambda x: [x] if x==x else [])

        if filter_data:
            for field, values in filter_data.items():
                meta_info.loc[:, field] = meta_info[field].apply(lambda x: [v for v in x if v not in values])

        if filter_no_meta:
            not_empty = meta_info.applymap(len).sum(axis=1) > 0
            if not not_empty.all():
                meta_info = meta_info.loc[not_empty]
                ratings = ratings.query(f'songid in @meta_info.index')

    while pcore: # do only if pcore is specified
        valid_users = ratings['userid'].value_counts() >= pcore
        valid_items = ratings['songid'].value_counts() >= pcore
        clean_data = (valid_items.all() and valid_users.all())
        if clean_data:
            break
        else:
            valid_users = valid_users.loc[valid_users].index
            valid_items = valid_items.loc[valid_items].index
            ratings = ratings.query('userid in @valid_users and '
                                    'songid in @valid_items')
            meta_info = meta_info.query('songid in @valid_items')
    return ratings, meta_info


def get_similarity_data(meta_info, metric='common', assume_binary=True, fill_diagonal=True):
    feat_mat, lbls = stack_features(meta_info, normalize=False)
    
    if metric == 'common':
        item_similarity = feat_mat.dot(feat_mat.T)
        item_similarity = item_similarity / item_similarity.data.max()
        item_similarity.setdiag(1.0)
    
    if (metric == 'cosine') or (metric == 'salton'):
        item_similarity = cosine_similarity(feat_mat,
                                            assume_binary=assume_binary,
                                            fill_diagonal=fill_diagonal)
        
    if item_similarity.format == 'csr':
        item_similarity = item_similarity.T # ensure CSC format (matrix is symmetric)

    userid = 'userid'
    itemid = meta_info.index.name
    similarities = {userid: None, itemid: item_similarity}
    indices = {userid: None, itemid: meta_info.index}
    labels = {userid: None, itemid: lbls}
    return similarities, indices, labels


def prepare_data_model(data_label, raw_data, similarities, sim_indices, item_meta, seed=0, feedback=None):
    userid = 'userid'
    itemid = item_meta[data_label].index.name
    data_model = SimilarityDataModel(similarities[data_label],
                                     sim_indices[data_label],
                                     raw_data[data_label],
                                     userid, itemid,
                                     feedback=feedback,
                                     seed=seed)
    data_model.test_fold = 1
    data_model.holdout_size = 1
    data_model.random_holdout = True
    data_model.warm_start = False
    data_model.verbose = False
    data_model.prepare()
    return data_model


def prepare_cold_start_data_model(data_label, raw_data, similarities, sim_indices, item_meta, seed=0, feedback=None):
    userid = 'userid'
    itemid = item_meta[data_label].index.name
    data_model = ColdStartSimilarityDataModel(similarities[data_label],
                                              sim_indices[data_label],
                                              raw_data[data_label],
                                              userid, itemid,
                                              feedback=feedback,
                                              seed=seed)
    data_model.test_fold = 1
    data_model.holdout_size = 1
    data_model.random_holdout = True
    data_model.verbose = False
    data_model.prepare()
    return data_model
