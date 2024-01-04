## the following functions derive features from the file historical_data.csv and return train_df.csv

import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta


## update types
## create target variable
def addFeatures_durations(historical_data: pd.DataFrame) -> pd.DataFrame:
    ## Update types and create the target variable
    historical_data['created_at'] = pd.to_datetime(historical_data['created_at'])
    historical_data['actual_delivery_time'] = pd.to_datetime(historical_data['actual_delivery_time'])
    ## calculate delivery duration
    historical_data['actual_total_delivery_duration'] = (
                historical_data['actual_delivery_time'] - historical_data['created_at']).dt.total_seconds()
    ## estimated time spent outside the store/not on order preparation
    historical_data['est_time_non-prep'] = historical_data['estimated_order_place_duration'] + historical_data[
        'estimated_store_to_consumer_driving_duration']

    return historical_data


def addFeatures_store_prep_stats(historical_data: pd.DataFrame) -> pd.DataFrame:
    ## returns a separate dataframe with the min, max, median and mean and standard deviation of the est_prep_time_per_item for each store_id

    historical_data = addFeatures_durations(historical_data=historical_data)
    ## calculate the amount of time (in seconds) the order spends in the store, then divide by the total number of items in the order
    historical_data['est_time_prep_per_item'] = (historical_data['actual_total_delivery_duration'] - historical_data[
        'est_time_non-prep']) / historical_data['total_items']
    historical_data = historical_data[historical_data['est_time_prep_per_item'] > 0]
    store_prep_stats_df = historical_data[['store_id', 'est_time_prep_per_item']].groupby(
        'store_id').est_time_prep_per_item.aggregate([np.mean, np.std])
    ## calculate z-score
    store_prep_facts_df = pd.merge(
        left=store_prep_stats_df.reset_index(),
        right=historical_data[['store_id', 'est_time_prep_per_item']],
        how='inner',
        on='store_id'
    )
    store_prep_facts_df['z_score'] = (store_prep_facts_df['est_time_prep_per_item'] - store_prep_facts_df['mean']) / \
                                     store_prep_facts_df['std']
    ## drop outliers
    store_prep_stats_df = store_prep_facts_df.loc[
        (store_prep_facts_df['z_score'] <= 3) & (store_prep_facts_df['z_score'] >= -3)]
    store_prep_stats_df = store_prep_stats_df.drop(columns=['mean', 'std'])
    store_prep_stats_df = store_prep_stats_df.groupby('store_id').est_time_prep_per_item.aggregate(
        ['min', 'max', 'median', np.mean, np.std])
    store_prep_stats_df = store_prep_stats_df.add_prefix('store_est_time_prep_per_item_')

    return store_prep_stats_df


def addFeatures_store_prep_day_created_stats(
        historical_data: pd.DataFrame) -> pd.DataFrame:  # this function would be done in practice by SQL or Spark
    ## returns a separate dataframe with the min, max, median and mean and standard deviation of the est_prep_time_per_item for each store_id by the day of week

    historical_data = addFeatures_durations(historical_data=historical_data)
    ## calculate the amount of time (in seconds) the order spends in the store, then divide by the total number of items in the order
    historical_data['est_time_prep_per_item'] = (historical_data['actual_total_delivery_duration'] - historical_data[
        'est_time_non-prep']) / historical_data['total_items']
    historical_data = historical_data[historical_data['est_time_prep_per_item'] > 0]
    historical_data['created_day_of_week'] = historical_data['created_at'].dt.day_of_week
    store_prep_stats_df = historical_data[['store_id', 'created_day_of_week', 'est_time_prep_per_item']].groupby(
        ['store_id', 'created_day_of_week']).est_time_prep_per_item.aggregate(['min', 'max', 'median', np.mean, np.std])

    # store_prep_stats_df = store_prep_stats_df.groupby(['store_id','created_day_of_week']).est_time_prep_per_item.aggregate(['min','max','median',np.mean,np.std])
    store_prep_stats_df = store_prep_stats_df.add_prefix('store_day_of_week_est_time_prep_per_item_')

    return store_prep_stats_df


def addFeatures_category_prep_stats(historical_data: pd.DataFrame) -> pd.DataFrame:
    ## returns a separate dataframe with the min,max, median, mean and standard deviation for each category
    try:
        historical_data = addFeatures_durations(historical_data=historical_data)
        ## calculate the amount of time (in seconds) the order spends in the store, then divide by the total number of items in the order
        historical_data['est_time_prep_per_item'] = (historical_data['actual_total_delivery_duration'] -
                                                     historical_data['est_time_non-prep']) / historical_data[
                                                        'total_items']
        historical_data = historical_data[historical_data['est_time_prep_per_item'] > 0]
        category_prep_stats_df = historical_data[['clean_store_primary_category', 'est_time_prep_per_item']].groupby(
            'clean_store_primary_category').est_time_prep_per_item.aggregate([np.mean, np.std])
        ## calculate z-score
        category_prep_facts_df = pd.merge(
            left=category_prep_stats_df.reset_index(),
            right=historical_data[['created_at', 'clean_store_primary_category', 'est_time_prep_per_item']],
            how='inner',
            on='clean_store_primary_category'
        )
        category_prep_facts_df['z_score'] = (category_prep_facts_df['est_time_prep_per_item'] - category_prep_facts_df[
            'mean']) / category_prep_facts_df['std']
        ## drop outliers
        category_prep_stats_df = category_prep_facts_df.loc[
            (category_prep_facts_df['z_score'] <= 3) & (category_prep_facts_df['z_score'] >= -3)]
        category_prep_stats_df = category_prep_stats_df.drop(columns=['mean', 'std', 'created_at'])
        category_prep_stats_df = category_prep_stats_df.groupby(
            'clean_store_primary_category').est_time_prep_per_item.aggregate(['min', 'max', 'median', np.mean, np.std])
        category_prep_stats_df = category_prep_stats_df.add_prefix('category_est_time_prep_per_item_')
    except Exception as ex:
        raise ex
    return category_prep_stats_df


def addFeatures_time_of_day(historical_data: pd.DataFrame) -> pd.DataFrame:
    ## Time-of-day features
    try:
        historical_data['created_hour_of_day'] = historical_data['created_at'].dt.hour
        historical_data['created_day_of_week'] = historical_data['created_at'].dt.day_of_week
    except Exception as ex:
        raise ex
    return historical_data


def addFeatures_ratios(historical_data: pd.DataFrame) -> pd.DataFrame:
    ## ratio features
    try:
        # historical_data['total_onshift_dashers'].mask(historical_data['total_busy_dashers'] > historical_data['total_onshift_dashers'],historical_data['total_busy_dashers'],inplace=True)
        historical_data['available_dashers'] = historical_data['total_onshift_dashers'] - historical_data[
            'total_busy_dashers']
        historical_data['orders_without_dashers'] = historical_data['total_outstanding_orders'] - historical_data[
            'total_onshift_dashers']
        historical_data['available_to_outstanding'] = historical_data['available_dashers'] / historical_data[
            'total_outstanding_orders']
        historical_data['available_to_outstanding'].replace(to_replace=[np.inf, -np.inf],
                                                            value=historical_data['available_dashers'][1], inplace=True)
        historical_data['busy_to_onshift'] = historical_data['total_onshift_dashers'] / historical_data[
            'total_busy_dashers']
        historical_data['busy_to_onshift'].replace(to_replace=[np.inf, -np.inf],
                                                   value=historical_data['total_onshift_dashers'][1], inplace=True)
        historical_data['busy_to_outstanding'] = historical_data['total_busy_dashers'] / historical_data[
            'total_outstanding_orders']
        historical_data['busy_to_outstanding'].replace(to_replace=[np.inf, -np.inf, np.nan], value=0, inplace=True)
        historical_data['onshift_to_outstanding'] = historical_data['total_onshift_dashers'] / historical_data[
            'total_outstanding_orders']
        historical_data['onshift_to_outstanding'].replace([np.inf, -np.inf], 0, inplace=True)
    except Exception as ex:
        raise ex
    return historical_data


def addFeatures_relative_abundances(historical_data: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    ## outstanding order and Dasher relative abundances per market
    # Independent sample tests conducted below indicate that the average order by hour populations differ between markets, so we also group by market in addition to hour
    assert 'market_id', 'created_hour_of_day' in train_df.columns;
    'Matching market_id and/or created_hour_of_day columns not found in train_df'
    try:
        market_hour_abd = historical_data.groupby(['market_id', 'created_hour_of_day'])[
            'total_outstanding_orders', 'total_onshift_dashers', 'total_busy_dashers'].aggregate(np.mean).add_prefix(
            'market_hour_mean_')
        # market_hour_std_dev = historical_data.groupby(['market_id','created_hour_of_day'])['total_outstanding_orders','total_onshift_dashers','total_busy_dashers'].aggregate(np.std).add_prefix('market_hour_std_dev_')
        hour_abd = historical_data.groupby(['created_hour_of_day'])[
            'total_outstanding_orders', 'total_onshift_dashers', 'total_busy_dashers'].aggregate(np.mean).add_prefix(
            'hour_mean_')
        # hour_std_dev = historical_data.groupby(['created_hour_of_day'])['total_outstanding_orders','total_onshift_dashers','total_busy_dashers'].aggregate(np.std).add_prefix('hour_std_dev_')

        train_df = pd.merge(left=train_df, right=market_hour_abd, on=['market_id', 'created_hour_of_day'])
        # train_df = pd.merge(left=train_df,right=market_hour_std_dev,on=['market_id','created_hour_of_day'])
        train_df = pd.merge(left=train_df, right=hour_abd, on=['created_hour_of_day'])
        # train_df = pd.merge(left=train_df,right=hour_std_dev,on=['created_hour_of_day'])
        # market hour abundances
        train_df['market_hour_onshift_outs_avg'] = train_df['market_hour_mean_total_onshift_dashers'] / train_df[
            'market_hour_mean_total_outstanding_orders']
        train_df['market_hour_onshift_to_outstanding_abd'] = train_df['onshift_to_outstanding'] / train_df[
            'market_hour_onshift_outs_avg']
        train_df['market_hour_busy_outs_avg'] = train_df['market_hour_mean_total_busy_dashers'] / train_df[
            'market_hour_mean_total_outstanding_orders']
        train_df['market_hour_busy_to_outstanding_abd'] = train_df['busy_to_outstanding'] / train_df[
            'market_hour_busy_outs_avg']
        train_df['market_hour_outs_order_abd'] = train_df['total_outstanding_orders'] / train_df[
            'market_hour_mean_total_outstanding_orders']
        train_df['market_hour_busy_dasher_abd'] = train_df['total_busy_dashers'] / train_df[
            'market_hour_mean_total_busy_dashers']
        # hour abundances
        train_df['hour_onshift_outs_avg'] = train_df['hour_mean_total_onshift_dashers'] / train_df[
            'hour_mean_total_outstanding_orders']
        train_df['hour_onshift_to_outstanding_abd'] = train_df['onshift_to_outstanding'] / train_df[
            'hour_onshift_outs_avg']
        train_df['hour_busy_outs_avg'] = train_df['hour_mean_total_busy_dashers'] / train_df[
            'hour_mean_total_outstanding_orders']
        train_df['hour_busy_to_outstanding_abd'] = train_df['busy_to_outstanding'] / train_df['hour_busy_outs_avg']
        train_df['hour_outs_order_abd'] = train_df['total_outstanding_orders'] / train_df[
            'hour_mean_total_outstanding_orders']
        train_df['hour_busy_dasher_abd'] = train_df['total_busy_dashers'] / train_df['hour_mean_total_busy_dashers']
    except Exception as ex:
        raise ex
    return train_df


def addFeatures_market_dayOfWeek_ratios(historical_data: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    ## ratios and value aggregations for each market by the day of the week
    assert 'market_id', 'created_day_of_week' in historical_data.columns;
    'Matching market_id and/or created_day_of_week columns not found in historical_data'
    assert 'market_id', 'created_day_of_week' in train_df.columns;
    'Matching market_id and/or created_day_of_week columns not found in train_df'
    try:
        market_day_abd = historical_data.groupby(['market_id', 'created_day_of_week'])[
            'total_outstanding_orders', 'total_onshift_dashers', 'total_busy_dashers'].aggregate(np.mean).add_prefix(
            'market_day_mean_')
        day_abd = historical_data.groupby(['created_day_of_week'])[
            'total_outstanding_orders', 'total_onshift_dashers', 'total_busy_dashers'].aggregate(np.mean).add_prefix(
            'created_day_mean_')

        train_df = pd.merge(left=train_df, right=market_day_abd, on=['market_id', 'created_day_of_week'])
        train_df = pd.merge(left=train_df, right=day_abd, on=['created_day_of_week'])
    except Exception as ex:
        raise ex
    return train_df


def addFeatures_dummies(historical_data: pd.DataFrame, dummy_column: str) -> pd.DataFrame:
    ## add dummies for categories
    try:
        dumm = pd.get_dummies(historical_data[dummy_column], prefix=str(dummy_column + '_'), dtype=float)
        # concat dummies
        historical_data = pd.concat([historical_data, dumm], axis=1)
        historical_data = historical_data.drop(columns=[dummy_column])
    except Exception as ex:
        raise ex
    return historical_data


def cleanFeatures_remove_outliers(historical_data: pd.DataFrame) -> pd.DataFrame:
    ## remove confounding outliers
    try:
        historical_data = historical_data.loc[
            historical_data['actual_total_delivery_duration'] > historical_data['est_time_non-prep']]
        historical_data = historical_data[
            historical_data['actual_total_delivery_duration'] < 86400]  # the equivalent of an entire day
    except Exception as ex:
        raise ex
    return historical_data


## Run all functions

historical_data = pd.read_csv(r'historical_data.csv')

store_id_unique = historical_data["store_id"].unique().tolist()
store_id_and_category = {store_id: historical_data[historical_data.store_id == store_id].store_primary_category.mode()
                         for store_id in store_id_unique}


def fill(store_id):
    try:
        return store_id_and_category[store_id].values[0]
    except:
        return np.nan


# fill null values
historical_data["clean_store_primary_category"] = historical_data.store_id.apply(fill)

## build store, category prep time statistics DataFrames
store_prep_stats_df = addFeatures_store_prep_stats(historical_data=historical_data)
store_prep_by_day_df = addFeatures_store_prep_day_created_stats(historical_data=historical_data)
## build the training dataset using the functions above
train_df = addFeatures_durations(historical_data=historical_data)
train_df = addFeatures_ratios(historical_data=train_df)
train_df = addFeatures_time_of_day(historical_data=train_df)
train_df = addFeatures_relative_abundances(historical_data=train_df, train_df=train_df)
train_df = addFeatures_market_dayOfWeek_ratios(historical_data=train_df, train_df=train_df)
## merge store, category prep stats DataFrames with train_df
train_df = pd.merge(
    left=train_df,
    right=store_prep_stats_df.reset_index(),
    how='inner',
    on='store_id'
)
train_df = pd.merge(left=train_df, right=store_prep_by_day_df.reset_index(), how='inner',
                    on=['store_id', 'created_day_of_week'])
## the category_prep_stats features do not make signficant contributions to models, so the table is dropped
# train_df = pd.merge(
#    left=train_df,
#    right=category_prep_stats_df.reset_index(),
#    how='inner',
#    on='clean_store_primary_category'
# )
## remove confounding outliers
train_df = cleanFeatures_remove_outliers(historical_data=train_df)
## add total prep time estimations
train_df['store_est_median_total_prep_time'] = train_df['total_items'] * train_df['store_est_time_prep_per_item_median']
train_df['store_est_mean_total_prep_time'] = train_df['total_items'] * train_df['store_est_time_prep_per_item_mean']
## by day
train_df['store_day_median_total_prep_time'] = train_df['total_items'] * train_df[
    'store_day_of_week_est_time_prep_per_item_median']
train_df['store_day_mean_total_prep_time'] = train_df['total_items'] * train_df[
    'store_day_of_week_est_time_prep_per_item_mean']
## add dummy columns for clean_store_primary_category, market_id, order_protocol
train_df = addFeatures_dummies(historical_data=train_df, dummy_column='clean_store_primary_category')
train_df = addFeatures_dummies(historical_data=train_df, dummy_column='market_id')
train_df = addFeatures_dummies(historical_data=train_df, dummy_column='order_protocol')
train_df = addFeatures_dummies(historical_data=train_df, dummy_column='created_hour_of_day')
train_df = addFeatures_dummies(historical_data=train_df, dummy_column='created_day_of_week')
## drop the native 'store_primary_category' column and all remaining non-feature columns, then do a headcheck
train_df.drop(
    columns=['store_primary_category', 'created_at', 'actual_delivery_time', 'store_id', 'est_time_prep_per_item'],
    inplace=True)
## drop NaN's
train_df.dropna(inplace=True)
train_df = train_df.astype("float32")

train_df['latest_update'] = dt.today()

train_df.to_csv(path_or_buf=r"/Users/advit/Documents/Capstone Project/Project/doordash_delivery_est/train_df.csv")