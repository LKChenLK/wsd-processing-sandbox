import csv
from os import PathLike


def load_keys(input_file: PathLike) -> None:
    data = []
    with open(input_file, "r", encoding="'iso-8859-1'") as f:
        reader = csv.reader(f, delimiter="\t")
        # Skip the header
        next(reader)        
        prev_target_id = None
        tmp_list = []
        for el in reader:
            if not prev_target_id:
                prev_target_id = el[0]
                continue
            if prev_target_id != el[0]:
                prev_target_id = el[0]
                data.append(tmp_list)
                tmp_list = []
            tmp_list.append(el)
        return data