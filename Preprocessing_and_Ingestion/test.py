import json


with open("source_data.json", 'r') as file:
    data = json.load(file)



items = data.get("courses")
print("Number of courses: {}".format(len(items)))