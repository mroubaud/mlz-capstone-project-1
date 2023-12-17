import requests

url = 'http://localhost:9696/predict'

data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/b/be/Domestic_horse_at_Suryachaur_and_the_mountains_in_the_back1.jpg'}

result = requests.post(url, json=data).json()
print(result)