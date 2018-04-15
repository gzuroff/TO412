from pytrends.request import TrendReq
import pandas as pd
import datetime as dt

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["Wendys", "Burger King", "McDonalds", "Arbys"]
comps = ["wendys", "burgerking", "mcdonalds", "arbys"]
for j in range(len(kw_list)):
    pytrends.build_payload([kw_list[j]], cat=0, timeframe='today 5-y', geo='', gprop='')
    trends = pytrends.interest_over_time()
    days = pd.to_datetime(trends.index).to_series()
    trends["week"] = days.dt.week
    trends["year"] = days.dt.year
    filename = comps[j] + ".csv"
    origFile = pd.read_csv(filename)
    compTrends = trends.ix[:,[kw_list[j], "week", "year"]]
    newFile = pd.merge(origFile, compTrends,  how='inner', left_on=['week','year'], right_on = ['week','year'])
    f_out = comps[j] + "1.csv"
    newFile.to_csv(f_out)
