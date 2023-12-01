import pandas as pd

origin = pd.read_csv('origin.csv')
not_null = (origin.loc[:, 'Rating'].isnull() == False)
origin.loc[not_null, 'Rating'] = (origin.loc[not_null, 'Rating'].values >= 5).astype(int)
origin.to_csv('old_raw.csv', index=False)
origin = origin[not_null]
origin.to_csv('raw.csv', index=False)
origin.dropna().to_csv("no_mv.csv", index=False)