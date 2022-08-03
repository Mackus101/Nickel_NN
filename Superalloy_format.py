import pandas as pd
import json

search_terms = ['Yield Strength',
                'Ultimate Tensile Strength',
                'Elongation',
                # 'Heat treatment 1 Temperature',
                # 'Heat treatment 1 Time',
                # 'Heat treatment 2 Temperature',
                # 'Heat treatment 2 Time',
                # 'Heat treatment 3 Temperature',
                # 'Heat treatment 3 Time',
                # 'Heat treatment 4 Temperature',
                # 'Heat treatment 4 Time',
                # 'Heat treatment 5 Temperature',
                # 'Heat treatment 5 Time',
                'Area under heat treatment curve',
                'Strengthening Precipitate Phase',
                'Powder processed',
                'Pressure treated']


def load_matmine_data(filepath):

    with open(filepath, 'r') as f:
        data = json.loads(f.read())

    return format_superalloys(data)


def format_entry(entry):
    elements = pd.json_normalize(entry['composition']).set_index('element').T
    name = pd.DataFrame({"names": [entry["names"][0]]})
    properties = pd.json_normalize(entry["properties"]).set_index("name")
    properties = properties[:][properties.index.isin(search_terms)]
    properties = properties["scalars"].apply(lambda x: x[0]["value"])

    formed_entry = pd.concat([name, pd.DataFrame(properties).T.reset_index(
        drop=True), elements.reset_index(drop=True)], axis='columns')

    return formed_entry


def format_superalloys(data):
    formatted_data = pd.DataFrame()
    for entry in data:
        formatted_data = pd.concat(
            [formatted_data, format_entry(entry)], ignore_index=True)

    return formatted_data


if __name__ == "__main__":
    data = load_matmine_data("ni_superalloys_3.json")
    print(data)
