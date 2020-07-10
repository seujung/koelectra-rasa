import random

def split_train_val(file_name, val_ratio=0.2):
    train_file_name = file_name.replace('.md', '_train.md')
    val_file_name = file_name.replace('.md', '_val.md')

    markdown_lines = open(file_name, encoding="utf-8").readlines()
    data_list = dict()
    for l in markdown_lines:
        if "##" in l:
            if 'intent' in l:
                data_list[l] = []
                current_key = l
        else:
            data_list[current_key].append(l.replace('\n', ''))
    
    train_dataset = dict()
    val_dataset = dict()

    for key, value in data_list.items():
        train_dataset[key] = []
        val_dataset[key] = []
        for v in value:
            if random.random() <= val_ratio:
                val_dataset[key].append(v)
            else:
                train_dataset[key].append(v)
    
    train_md = """"""
    for key, value in train_dataset.items():
        train_md += key
        for v in value:
            train_md += v
            train_md += "\n"
        train_md += "\n"
    
    val_md = """"""
    for key, value in val_dataset.items():
        val_md += key
        for v in value:
            val_md += v
            val_md += "\n"
        val_md += "\n"   

    text_file = open(train_file_name, "w")
    _ = text_file.write(train_md)
    text_file.close()

    text_file = open(val_file_name, "w")
    _ = text_file.write(val_md)
    text_file.close()

    return train_file_name, val_file_name