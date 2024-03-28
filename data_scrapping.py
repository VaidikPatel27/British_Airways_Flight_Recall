import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

reviews = []
headers = []
names = []
dates = []
recommend = []
ratings = []
traveller_type = []
seat_type = []
route = []
for i in range(1,100):
  print(f'Initializing page:{i} ---->')
  url = f'https://www.airlinequality.com/airline-reviews/british-airways/page/{i}/?sortby=post_date%3ADesc&pagesize=100'
  req = requests.get(url)
  soup = BeautifulSoup(req.text,'lxml')

  reviews_all = soup.find_all('div',{'class','text_content'})
  for rev in range(0,len(reviews_all)):
    reviews.append(str(reviews_all[rev].text.strip()))

  headers_all = soup.find_all('h2' , {'class','text_header'})
  for header in range(0,len(headers_all)):
    headers.append(str(headers_all[header].string).replace('"','').strip())

  names_all = soup.find_all('h3',{'class','text_sub_header'})
  for name in range(0,len(names_all)):
    names.append(names_all[name].span()[0].string)

  dates_all = soup.find_all('h3',{'class','text_sub_header'})
  for date in range(0,len(dates_all)):
    dates.append(dates_all[date].time.string)

  rating_all = soup.find_all('div',{'class','rating-10'})
  for rate in range(1,len(rating_all)):
    ratings.append(rating_all[rate].text.strip()[0])

  recommend_all = soup.find_all('div',{'class','review-stats'})
  for recom in range(0,len(recommend_all)):
    if 'no' in str(recommend_all[recom]):
      recommend.append('no')
    elif 'yes' in str(recommend_all[recom]):
      recommend.append('yes')
    else:
      recommend.append(np.NaN)

  travel = soup.find_all('div',{'class','review-stats'})
  for idx_travel in range(len(travel)):
    flag = 0
    for node_travel in travel[idx_travel].find_all('tr'):
      if 'type_of_traveller' in str(node_travel):
        traveller_type.append(node_travel.find('td',{'class','review-value'}).string)
        flag = 1
    if flag ==0:
      traveller_type.append(np.nan)

  seat_find = soup.find_all('div',{'class','review-stats'})
  for idx_cabin in range(len(seat_find)):
    flag1 = 0
    for node_cabin in seat_find[idx_cabin].find_all('tr'):
      if 'cabin_flown' in str(node_cabin):
        seat_type.append(node_cabin.find('td',{'class','review-value'}).string)
        flag1 = 1
    if flag1 == 0:
      seat_type.append(np.nan)

  route_find = soup.find_all('div',{'class','review-stats'})
  for idx_route in range(len(route_find)):
    flag2 = 0
    for node_route in route_find[idx_route].find_all('tr'):
      if 'route' in str(node_route):
        route.append(node_route.find('td',{'class','review-value'}).string)
        flag2 = 1
    if flag2 == 0:
      route.append(np.nan)
      
  print('')

df = pd.DataFrame({
    'review_title':headers,
    'review':reviews,
    'customer_name':names,
    'date':dates,
    'traveller_type': traveller_type,
    'seat_type': seat_type,
    'route': route,
    'rating': ratings,
    'recommended':recommend,
})

df['rating'] = df['rating'].replace('n',np.nan)

df['date'] = pd.to_datetime(df['date'])


path = 'path where you want to store this file'
df.to_csv('path/sentiment.csv',index=False)


