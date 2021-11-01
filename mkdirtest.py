import os

cnt=0
target_emotion='irritated face'
save_path=os.path.join('./testresult',target_emotion,str(cnt)+'.jpg')
    # if not os.path.exists(save_path):
os.mkdir(os.path.join('testresult',target_emotion))
