from pytrends.request import TrendReq
import pandas as pd
import datetime as dt

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["Wendys", "Burger King", "McDonalds", "Arbys"]


pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
trends = pytrends.interest_over_time()
days = pd.to_datetime(trends.index).to_series()#.apply(lambda x: dt.datetime.strftime(x, '%m/%d/%Y'))
trends["week"] = days.dt.week
trends["year"] = days.dt.year
comps = ["wendys", "burgerking", "mcdonalds", "arbys"]
for i in range(len(comps)):
    filename = comps[i] + ".csv"
    origFile = pd.read_csv(filename)
    compTrends = trends.ix[:,[kw_list[i], "week", "year"]]
    newFile = pd.merge(origFile, compTrends,  how='inner', left_on=['week','year'], right_on = ['week','year'])
    f_out = comps[i] + "1.csv"
    newFile.to_csv(f_out)
