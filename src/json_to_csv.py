import csv
import json

with open('resources/reviews.json', 'r', encoding='utf8', errors='ignore') as json_file:
  data = json.load(json_file)

f = csv.writer(open("reviews.csv", "w", newline='', encoding='utf8', errors='ignore'))
f.writerow(["userName","date","score","scoreText","text"])

for person in data["data"]:
  f.writerow([person["userName"],
  person["date"],
  person["score"],
  person["scoreText"],
  person["text"]])

