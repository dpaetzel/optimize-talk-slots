import pandas as pd
import math


def preprocess_excel(path):
    """
    This function reads the excel list and returns the needed data from it
    :param path: Path to the excel list
    :return: Dict(
        "id_to_name_dict": Dict(Int -> String),
        "name_to_id_dict": Dict(String -> Int),
        "workshop_list": List(String),
        "speaker_dict": Dict(int -> String),
        "prio_matrix": Matrix( number_of_participants x number_of_workshops)
        )
    """

    with open(path, 'rb') as excel:
        data = pd.read_excel(excel)

    # ID to Name
    names = data['Name']

    id_to_name = {}
    name_to_id = {}
    for id, name in names.iteritems():
        id_to_name[id] = name
        name_to_id[name] = id

    # Workshop ID
    workshop_data_raw = set(data.iloc[:, 5].to_list())
    workshop_data_raw.remove('Ich bin kein Speaker / Workshop Owner.')
    # This removes the nan element from the list
    workshop_data_list = list(filter(lambda x: x == x, workshop_data_raw))

    # manually add workshop for external speaker:
    workshop_data_list.append("W2 Building the City of the Future")

    speakers = {}
    no_workshops = len(workshop_data_list)
    priorities_of_humans = []
    for _, entry in data.iterrows():
        # If person is speaker assign workshop to them
        speaks_at = entry[5]
        if speaks_at != 'Ich bin kein Speaker / Workshop Owner.':
            try:
                math.isnan(speaks_at)
            except:
                speakers[name_to_id[entry[4]]] = entry[5]
        # Create prio list for the participants
        prios = [100 for _ in range(no_workshops)]
        for i in range(6, 11):
            # Catch error if user has not filled out all prios
            try:
                workshop_name = entry[i]
                workshop_id = workshop_data_list.index(workshop_name)
                prios[workshop_id] = i - 5 - 1
            except ValueError:
                pass
        priorities_of_humans.append(prios)

    return {
        "id_to_name_dict": id_to_name,
        "name_to_id_dict": name_to_id,
        "workshop_list": workshop_data_list,
        "speaker_dict": speakers,
        "prio_matrix": priorities_of_humans
    }


if __name__ == '__main__':
    res = preprocess_excel('Workshop Preference Voting Retreat 2022.xlsx')
