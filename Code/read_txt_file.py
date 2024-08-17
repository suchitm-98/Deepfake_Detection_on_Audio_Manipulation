import pandas as pd

meta_file= open('metadata_test_PA.txt','rt')

meta_dict ={}
for line in meta_file:
    res = line.split()
    #print(res[0])
    #print(res[1])
    #print(res)
    meta_dict.update({res[1]:res[4]})    
    #print(line)

#print(d)
del res, line 

df = pd.DataFrame( list(meta_dict.items()) , columns = ['key','label'] )
df.columns
meta_file.close()
df.to_csv("Metadata_test_PA.csv")

metadata = pd.read_csv("Metadata_test_PA.csv")
metadata = metadata.drop(columns = 'Unnamed: 0')

real_df = metadata[metadata['label'] == 'bonafide']
real_df = real_df.reset_index()
real_final = real_df.iloc[ : , 1]

fake_df = metadata[metadata['label'] == 'spoof']
fake_df = fake_df.reset_index()
fake_final = fake_df.iloc[ : , 1 ]

del metadata,fake_df, real_df

