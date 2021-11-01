"""
remove images in test data(input_path) from train data(target_path)
"""
import os

input_path='./testdata'
target_path='./smalltraindata'

def getFiles(root):
    files=os.listdir(root)
    pathList=[]
    for file in files:
        full_path=os.path.join(root, file)
        if not os.path.isdir(full_path):
            pathList.append(full_path)
        else:
            pathList.extend(getFiles(full_path))
    return pathList
    

pathList=getFiles(input_path)


for path in pathList:
    splits=path.split('\\')
    remove_target_path=os.path.join(target_path, splits[-2], splits[-1])
    if os.path.exists(remove_target_path):
        os.remove(remove_target_path)
        print(path, 'removed')