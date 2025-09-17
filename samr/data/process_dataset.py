import os
import csv

from samr.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH

datasets = {
    'train': os.path.join(RAW_DATA_PATH, 'train'),
    'test': os.path.join(RAW_DATA_PATH, 'test')
}

output_file = os.path.join(PROCESSED_DATA_PATH, "dataset.csv")
data = []

for dataset_name, dataset_path in datasets.items():

    for label_name in ('neg', 'pos'):
        label_dir = os.path.join(dataset_path, label_name)
        
        if not os.path.exists(label_dir):
            continue
        
        for filename in os.listdir(label_dir):
            if not filename.endswith('.txt'):
                continue
            
            file_path = os.path.join(label_dir, filename)
            id, rating = filename.replace('.txt', '').split('_')
            with open(file_path, 'r', encoding='utf-8') as f:
                review = f.read().replace('\n', ' ').strip()
                
            data.append([rating, review])

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['rating', 'review'])
    writer.writerows(data)

print(f"Created {output_file} with {len(data)} entries.")  
