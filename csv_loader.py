import os
import csv
import configargparse
import numpy as np

def load_and_average_data():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--folder', default='./eb5/', help='Folder containing CSV files')
    parser.add_argument('--prefixes', nargs='+', required=True, help='List of CSV file prefixes, for example: --prefix flashmaskv1 old_flashmaskv3 new_flashmaskv3')
    args = parser.parse_args()

    fw_times = {}
    bw_times = {}
    category_counts = {}
    
    for prefix in args.prefixes:
        csv_files = [f for f in os.listdir(args.folder) 
                    if f.startswith(prefix) and f.endswith('.csv')]
        
        if not csv_files:
            print(f"Warning: No CSV files found with prefix '{prefix}' in folder '{args.folder}'")
            continue
        
        for csv_file in csv_files:
            file_path = os.path.join(args.folder, csv_file)
            
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                
                for row in reader:
                    row = row[0].split('\t')
                    if len(row) < 4:
                        continue
                    
                    operation = row[0].strip()
                    try:
                        fw_time = float(row[1])
                        bw_time = float(row[2])
                    except ValueError:
                        continue
                    
                    if operation not in fw_times:
                        fw_times[operation] = {}
                        bw_times[operation] = {}
                        category_counts[operation] = {}
                    
                    if prefix not in fw_times[operation]:
                        fw_times[operation][prefix] = 0.0
                        bw_times[operation][prefix] = 0.0
                        category_counts[operation][prefix] = 0
                    
                    fw_times[operation][prefix] += fw_time
                    bw_times[operation][prefix] += bw_time
                    category_counts[operation][prefix] += 1
    
    for operation in fw_times:
        for prefix in fw_times[operation]:
            count = category_counts[operation][prefix]
            if count > 0:
                fw_times[operation][prefix] /= count
                bw_times[operation][prefix] /= count
    
    categories = list(fw_times.keys())
    methods = args.prefixes
    
    fw_data = {}
    bw_data = {}
    
    for category in categories:
        fw_data[category] = [fw_times[category].get(prefix, 0.0) for prefix in methods]
        bw_data[category] = [bw_times[category].get(prefix, 0.0) for prefix in methods]
    
    return categories, methods, fw_data, bw_data

if __name__ == "__main__":
    categories, methods, fw_times, bw_times = load_and_average_data()
    # test
    print("Categories:", categories)
    print("Methods:", methods)
    print("FW Times:", fw_times)
    print("BW Times:", bw_times)