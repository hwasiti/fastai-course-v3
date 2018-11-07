import matplotlib.pyplot as plt

def plots(ims, figsize=(12,6), columns=3, titles=None, fontsize=16): # ims = np.stack()
    rows = len(ims)/columns +1
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, columns, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=fontsize)
        plt.imshow(ims[i])
        
# An example how to use it:
# from my_python_tricks import *
# import matplotlib.image as mpimg

# files = os.listdir(path/'tests/')
# rows = int(len(files)/4)+1
# imgs = np.array([mpimg.imread(path/'tests'/x) for x in files])

# plots(imgs, figsize=(15,15),  columns=3, titles=files)