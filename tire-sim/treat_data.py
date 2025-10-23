from collections import defaultdict
import csv
from datetime import datetime
import os
import yaml

data_file_prefix = "B2356run"

def split_data(data_folder_path, selected_runs, output_folder_path, note="none"):
    selected_data_names = [data_file_prefix + str(rn) + ".dat" for rn in selected_runs]
    selected_data_paths = [os.path.join(data_folder_path, data_name) for data_name in selected_data_names]

    if not os.path.isdir(data_folder_path):
        print("Data folder does not exist.")
        return 1
    elif len(os.listdir(data_folder_path)) == 0:
        print("Data folder is empty.")
        return 2
    
    dir_data_names = os.listdir(data_folder_path)
    if any(data_name not in dir_data_names for data_name in selected_data_names):
        print("Data folder missing selected file.")
        return 3
    
    # Keep track of all data columns that exist somewhere in the dataset
    all_columns = set()

    metadata = {
        "note": note,
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "runs": selected_runs, 
        "tires": set(), 
        "units": defaultdict(set)
    }

    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    for i in range(len(selected_data_names)):
        data_number = selected_runs[i]
        data_name = selected_data_names[i]
        data_path = selected_data_paths[i]

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()

        # Parse metadata
        metadata_line = lines[0].split(";")

        tire_header = metadata_line[4]
        tire_name = tire_header.split("=")[1]
        rim_header = metadata_line[5]
        rim_width = rim_header.split("=")[1]

        metadata["tires"].add(tire_name + ", RW" + rim_width)
        
        # Parse column labels
        header_line = lines[1].split("\t")
        all_columns.update(header_line)

        units_line = lines[2].split('\t')

        column_units = dict(zip(header_line, units_line))
        for column in header_line:
            metadata["units"][column].add(column_units[column])

        data_rows = [row.split('\t') for row in lines[3:]]

        output_csv_path = os.path.join(output_folder_path, f"run{data_number}.csv")

        # --- Write cleaned CSV (header + numeric data only) ---
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(header_line)
            writer.writerows(data_rows)

    yaml_path = os.path.join(output_folder_path, "metadata.yaml")

    metadata["tires"] = list(metadata["tires"])
    metadata["units"] = dict(metadata["units"])
    for column in metadata["units"]:
        metadata["units"][column] = list(metadata["units"][column])[0] if len(metadata["units"][column]) == 1 else list(metadata["units"][column])

    with open(os.path.join(output_folder_path, "info.yaml"), 'w', encoding='utf-8') as yf:
        yaml.dump(metadata, yf, sort_keys=False, allow_unicode=True)

    print(f"YAML written to {yaml_path}")
    print(f"Clean CSV written to {output_csv_path}")

if __name__ == "__main__":
    split_data(
        "RunData_Cornering_ASCII_SI_Round9", 
        [2, 4, 5, 6, 7, 8, 9], 
        "./SN5_R9_Lateral", 
        note="Round 9 TTC Cornering Data with Hoosier 16x7.5"
    )

    split_data(
        "RunData_DriveBrake_ASCII_SI_Round9", 
        [68, 69, 70, 71, 72, 73], 
        "./SN5_R9_Longitudinal", 
        note="Round 9 TTC DriveBrake Data with Hoosier 18x6"
    )