import pandas as pd

lu_matrix_path = r"D:\stdl\lu_matrix.csv"
lc_matrix_path = r"D:\stdl\lc_matrix.csv"
lc_predictions = r"C:\tools\adele\test_area4\LC\predictions_lc.csv"
lu_predictions = r"C:\tools\adele\test_area4\LU\predictions_lu.csv"
change_prediction = r"C:\tools\adele\test_area4\change\predictions_changes.csv"
base = r"C:\tools\adele\test_area4\base.csv"
PROBABILITY_THRESHOLD = 0.2

df_lu_weights = pd.read_csv(lu_matrix_path, sep=";")
df_lc_weights = pd.read_csv(lc_matrix_path, sep=";")
df_base = pd.read_csv(base)
df_lc = pd.read_csv(lc_predictions)
df_lc.rename({'prediction': 'prediction_lc'}, axis=1, inplace=True)
df_lu = pd.read_csv(lu_predictions)
df_lu.rename({'prediction': 'prediction_lu'}, axis=1, inplace=True)
df_change = pd.read_csv(change_prediction)
df_change.rename({'1': 'proba_change'}, axis=1, inplace=True)
small_classes = {
    'lc': [13, 14, 15, 16, 17, 33, 34, 35, 42, 43, 46, 63, 64],
    'lu': [101, 102, 104, 106, 108, 121, 123, 124, 125, 141, 142, 143, 144,
           145, 146, 147, 164, 162, 163, 164, 165, 166, 201, 202, 302, 303,
           304, 403, 422, 423, 424]
}

df_data = df_base.merge(df_lc[['RELI', 'prediction_lc']], left_on='RELI', right_on='RELI')
df_data = df_data.merge(df_lu[['RELI', 'prediction_lu']], left_on='RELI', right_on='RELI')
df_data = df_data.merge(df_change[['RELI', 'proba_change']], left_on='RELI', right_on='RELI')
df_data['has_changed'] = (df_data['LC3_27'] != df_data['LC4_27']) | (df_data['LU3_46'] != df_data['LU4_46'])

df_lc_pred_change = df_data[df_data['prediction_lc'] != df_data['LC3_27']]
df_lu_pred_change = df_data[df_data['prediction_lu'] != df_data['LU3_46']]

filter1 = df_data['RELI'].isin(df_lc_pred_change['RELI'].values)
filter2 = df_data['RELI'].isin(df_lu_pred_change['RELI'].values)
filter3 = df_data['proba_change'] < PROBABILITY_THRESHOLD
df_data['no_change_predicted'] = ~filter1 & ~filter2 & filter3
filter3 = df_data['LC3_27'].isin(small_classes['lc'])
filter4 = df_data['LU3_46'].isin(small_classes['lu'])
df_data['small_class'] = filter3 | filter4

# small classes are completely filtered
df_data = df_data[~df_data['small_class']]

df_actual_change = df_data[df_data['has_changed']]
change_weights = dict()
for cindex, row in df_actual_change.iterrows():
    w_lc = df_lc_weights[df_lc_weights['destination'] == row['LC4_27']][str(row['LC3_27'])].to_numpy()[0]
    w_lu = df_lu_weights[df_lu_weights['destination'] == row['LU4_46']][str(row['LU3_46'])].to_numpy()[0]
    if (row['LC3_27'] in (14, 35)) and (row['LC4_27'] in (14, 35)) \
            and (row['LU4_46'] == 203) and (row['LU3_46'] == 203):
        w_lc = 3
    if (row['LC3_27'] == 16) and (row['LC4_27'] in (41, 42, 43, 44, 45, 46, 47)) and \
            (row['LU3_46'] < 200) and row['LU4_46'] < 200:
        w_lc = 2
    if (row['LC4_27'] == 16) and (row['LC3_27'] in (41, 42, 43, 44, 45, 46, 47)) and \
            (row['LU3_46'] < 200) and row['LU4_46'] < 200:
        w_lc = 2
    if (row['LC3_27'] == 17) and (row['LC4_27'] in (21, 31, 32)) and \
            (row['LU3_46'] < 200) and row['LU4_46'] < 200:
        w_lc = 2
    if (row['LC4_27'] == 17) and (row['LC3_27'] in (21, 31, 32)) and \
            (row['LU3_46'] < 200) and row['LU4_46'] < 200:
        w_lc = 2
    if (row['LC3_27'] in (31, 32)) and (row['LC4_27'] in (31, 32)) \
            and (row['LU4_46'] == 421) and (row['LU3_46'] == 421):
        w_lc = 3
    weight = w_lc ** 3 + w_lu ** 3
    change_weights[row['RELI']] = weight

df_w = pd.DataFrame.from_dict(change_weights, orient='index', columns=['weight'])
df_data = df_data.merge(df_w, how='left', left_on='RELI', right_index=True)

miss_changes = df_data[df_data['no_change_predicted']]['has_changed'].sum()
print('missed changes: {:.0f}'.format(miss_changes))

# ratio of missed changed over total changes
miss_changed_ratio = miss_changes / df_data['has_changed'].sum()
print('missed changes w.r.t total changes: {:.3f}'.format(miss_changed_ratio))

miss_weighted_changes = df_data[(df_data['no_change_predicted']) & (df_data['has_changed'])]['weight'].sum()
print('missed weighted change: {:.0f}'.format(miss_weighted_changes))

miss_weighted_changed_ratio = miss_weighted_changes / df_data['weight'].sum()
print('missed weighted changes w.r.t total weighted changes: {:.3f}'.format(miss_weighted_changed_ratio))

automatized_points = df_data['no_change_predicted'].sum()
print('automatized points: {:.0f}'.format(automatized_points))

automatized_capacity = automatized_points / (~df_data['has_changed']).sum()
print('automatized capacity: {:.3f}'.format(automatized_capacity))

metric = automatized_capacity * (0.1 - miss_changed_ratio) / 0.1
print('raw metric: {:.3f}'.format(metric))

metric = automatized_capacity * (0.1 - miss_weighted_changed_ratio) / 0.1
print('weighted metric: {:.3f}'.format(metric))
