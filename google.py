from pytrends.request import TrendReq
import pandas as pd
import datetime as dt

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["Wendys", "Arbys", "McDonalds", "Burger King"]


pytrends.build_payload(kw_list, cat=0, timeframe='today 2-y', geo='', gprop='')
trends = pytrends.interest_over_time()
days = pd.to_datetime(trends.index).to_series()#.apply(lambda x: dt.datetime.strftime(x, '%m/%d/%Y'))
trends["week"] = days.dt.week
trends["year"] = days.dt.year
print(trends)