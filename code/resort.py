#针对calling_images中的图片重新编号
import os

def resort(path):

    files = os.listdir(r'./'+path)
    files.sort(key=lambda x: int(x[:-4]))

    count = 1

    for filename in files:
        new_name = filename.replace(filename, str(count)+".jpg")
        os.rename(os.path.join(path, filename), os.path.join(path, new_name))

        count += 1

if __name__=="__main__":
    path = 'calling_images'
    resort(path)