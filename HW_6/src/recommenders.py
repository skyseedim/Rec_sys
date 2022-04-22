import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True, filter_it=-9):
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        self.filter_it = filter_it


        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):
        # your_code
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        return user_item_matrix.astype(float)

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4) # K - кол-во билжайших соседей
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    # your_code
    def get_rec(self, x):
        """Находим 1 наиболее близкий товар к заданному"""

        recs = self.model.similar_items(self.itemid_to_id[x], N=3)
        top_rec = recs[1][0]  # потому что на позии recs[0][0] - находится id самого x
        rec = self.id_to_itemid[top_rec]
        # заглушка, можно заменять на "менее похожий товар" в случае если попался товар не из ТОП-5000
        if rec == self.filter_it:
            # rec = x
            rec = self.id_to_itemid[recs[2][0]]
        return rec

    # your_code
    def get_recommendation(self, user, N=5, similar_users=False):
        """Рекомендуем товары"""

        # your_code
        sparse_user_item = csr_matrix(self.user_item_matrix).tocsr()

        model = self.model
        if similar_users:
            model = self.own_recommender

        res = [self.id_to_itemid[rec[0]] for rec in
               model.recommend(userid=self.userid_to_id[user],
                               user_items=sparse_user_item,  # на вход user-item matrix
                               N=N,
                               filter_already_liked_items=False,
                               filter_items=[self.itemid_to_id[self.filter_it]],
                               recalculate_user=True)]

        #assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_items_recommendation(self, data, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        popularity = data[data.user_id == user].groupby(['item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity = popularity[popularity['item_id'] != self.filter_it]
        popularity = popularity.head(N)

        goods = popularity.item_id.to_list()
        res = list(map(lambda x: self.get_rec(x), goods))

        #assert len(res) == N, 'Количество рекомендаций != {}'.format(user)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

    # your_code
        sim_user = self.model.similar_users(self.userid_to_id[user], N=2)[1][0]
        res = self.get_recommendation(sim_user, N=5, similar_users=True)

        #assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res