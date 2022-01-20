import os
import yaml

raw_dataset_dir = '/media/neofelis/Jingyu-SSD/carla_dataset'
dataset_dir = './carla_data'
sequence_dir = os.path.join(dataset_dir, 'sequences')
sequence_ref_dict = {}
is_cylindrical = False
sequence_ref_dict['is_cylindrical'] = is_cylindrical

train_val_test_split = []


os.system(f'ls {raw_dataset_dir}')

# make directory

os.system(f'mkdir {dataset_dir}')
os.system(f'mkdir {sequence_dir}')

if is_cylindrical:
    coordinate = 'cylindrical'
else:
    coordinate = 'cartesian'

# os.system(os.path.join(raw_dataset_dir, 'Train'))
sequences = sorted(os.listdir(os.path.join(raw_dataset_dir, 'Train')))
for i in range(len(sequences)):
    sequence_ref_dict[i] = sequences[i]
    
    os.system(f'sudo ln -s {raw_dataset_dir}/Train/{sequences[i]}/{coordinate} {os.path.join(sequence_dir, str(i).zfill(2))}')
    # generate soft links

train_val_test_split.append(len(sequences))
len_train = len(sequences)

sequences = sorted(os.listdir(os.path.join(raw_dataset_dir, 'Val')))
for i in range(len(sequences)):
    sequence_ref_dict[i+len_train] = sequences[i]
    
    os.system(f'sudo ln -s {raw_dataset_dir}/Val/{sequences[i]}/{coordinate} {os.path.join(sequence_dir, str(i+len_train).zfill(2))}')
    # generate soft links

len_train_val = len_train + len(sequences)
train_val_test_split.append(len(sequences))


sequences = sorted(os.listdir(os.path.join(raw_dataset_dir, 'Test')))
for i in range(len(sequences)):
    sequence_ref_dict[i+len_train_val] = sequences[i]
    
    os.system(f'sudo ln -s {raw_dataset_dir}/Test/{sequences[i]}/{coordinate} {os.path.join(sequence_dir, str(i+len_train_val).zfill(2))}')
    # generate soft links
train_val_test_split.append(len(sequences))

sequence_ref_dict['split'] = train_val_test_split

print(sequence_ref_dict)

with open(os.path.join(dataset_dir, 'carla_seq.yaml'), 'w') as outfile:
    yaml.dump(sequence_ref_dict, outfile)