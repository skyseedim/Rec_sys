
def prefilter_items(data):
    popularity = data.groupby('item_id')['user_id'].nunique() / data['user_id'].nunique()
    popularity = popularity.reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    # Уберем самые популярные товары (их и так купят)
    # top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    # data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев

    # Уберем не интересные для рекоммендаций категории (department)

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.

    # Уберем слишком дорогие товары

    # ...

    return data


def postfilter_items(user_id, recommednations):
    pass


def top_items(data, top_num=5000, fictive=-9):
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(top_num).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = fictive

    return data
