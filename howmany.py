"""
shows how many files
"""

import os

def getFileList(root):
    files=os.listdir(root)
    pathList=[]
    for file in files:
        path=os.path.join(root, file)
        if os.path.isdir(path):
            pathList.extend(getFileList(path))
        else:
            pathList.append(path)
    return pathList

file_path='./smalltraindata'

files=getFileList(file_path)
print('len:',len(files))