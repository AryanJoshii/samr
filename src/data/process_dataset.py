import os
import csv

raw_dataset_path = '../../data/raw'
processed_dataset_path = '../../data/processed'
script_dir = os.path.dirname(__file__)
raw_abs_path = os.path.abspath(os.path.join(script_dir, raw_dataset_path))
processed_abs_path = os.path.abspath(os.path.join(script_dir, processed_dataset_path))

datasets = {
    'train': os.path.join(raw_abs_path, 'train'),
    'test': os.path.join(raw_abs_path, 'test')
}

for dataset_name, dataset_path in datasets.items():
    output_file = os.path.join(processed_abs_path, f"{dataset_name}.csv")
    data = []

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
                
            data.append([id, rating, review])

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label', 'review'])
        writer.writerows(data)

    print(f"Created {output_file} with {len(data)} entries.")  
