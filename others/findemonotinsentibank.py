import os
"""find out emotions not in sentibank"""

file_path=r'E:\bi_concepts1553'

files = os.listdir(file_path)

print('====files====')
testemo=['happy', 'jolly', 'joyful', 'depressed', 'furious', 'frightened', 'dirty', 'disgusting',
'messy', 
 'sad', 'angry', 'pitiful', 'amazing', 'excited', 'irritated', 'annoyed', 'scared', 'daunting']

emoset=set()
for file in files:
    splits=file.split('_')
    emo=splits[0]
    if emo=='sad':
        print(file)
    emoset.add(emo)

zero_shot_emo=[]
"""find out emotions that do not exists in Sentibank"""    
print('====zero-shot emotions not in sentibank====')
for emo in testemo:
    if emo not in emoset:  
        zero_shot_emo.append(emo)
print(zero_shot_emo)    
train_emo=testemo
for emo in zero_shot_emo:
    train_emo.remove(emo)
    
print("====train emotions===")
print(train_emo)
