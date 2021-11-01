"""
Finds emotions not in input_path
"""

import shutil
import os

file_path=r'E:\bi_concepts1553'

files = os.listdir(file_path)

print('====files====')
testemo=['happy', 'jolly', 'joyful', 'depressed', 'furious', 'frightened', 'dirty', 'disgusting', 
 'sad', 'angry', 'pitiful', 'amazing', 'excited', 'irritated', 'annoyed', 
 'scared', 'daunting', 'untidy', 'marvelous']

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
"""move files that belong to training set to other directory """
original = r'E:\bi_concepts1553'
target = r'./traindata'

if not os.path.exists(target):
    os.mkdir(target)
files=os.listdir(original)
print('====train emo====')
print(train_emo)

pathList=[]

"""copy files"""
# for file in files:
    # #todo:
    # splits=file.split('_')
    # if splits[0] in train_emo:
        # fns=os.listdir(os.path.join(original,file))
        # for fn in fns:
            # splits=fn.split('.')
            # if splits[-1]=='jpg':
                # if not os.path.exists(os.path.join(target,file)):
                    # os.mkdir(os.path.join(target,file))
                # shutil.copyfile(os.path.join(original, file, fn), os.path.join(target,file, fn))


    # if splits[0] in train_emo:

    
# for file in files:
    # splits=file.split('.')
    # emo=splits[0].split('_')[0]
    # if emo == 'contentment':
        # shutil.copyfile(os.path.join(original, file), os.path.join(target,file))
        
        
        
        
        
        
        
