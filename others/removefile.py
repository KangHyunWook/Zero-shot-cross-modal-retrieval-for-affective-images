"""
remove bad image files
"""
import os


"""remove bad images"""
fw=open('delete.txt', 'w')
print('====bad files====')
for path in imagePathList:
    fr=open(path, 'rb')
    check_chars = fr.read()[-2:]
    if check_chars != b'\xff\xd9':
        fw.write(path+'\n')
        print(path, 'removed')

fw.close()

path='./delete.txt'

print('hi')
with open(path,'r') as f:
    for line in f:
        line=line.strip()
        os.remove(line)
        print(line,'removed')
