
'''
TELEGRAM Notifications
======================
https://forums.fast.ai/t/training-metrics-as-notifications-on-mobile-using-callbacks/17330/4?u=hwasiti

Derived from Matt Potma's basic telegram bot usage script.

Utilizes this API Library: https://github.com/python-telegram-bot/python-telegram-bot To install: pip install python-telegram-bot --upgrade

To generate an Access Token, you have to talk to BotFather: https://t.me/botfather and follow a few simple steps (described here: https://core.telegram.org/bots#6-botfather ).

For a simple example: https://github.com/python-telegram-bot/python-telegram-bot/wiki/Introduction-to-the-API

'TOKEN' should be replaced by the API token you received from @BotFather:

In ~/.telegram on the machine running the job, put

{"api_key": "442766545:", "chat_id": ""}

For me it was: {"api_key": "630257503:AAF9Q4mrnALdX6nBjR6ImgdG1uwDlnIGvtg", "chat_id": "-318454638"}

https://stackoverflow.com/a/50661601/1970830
Here's how you get an API key: https://core.telegram.org/api/obtaining_api_id Here's how you get your chat ID: (See my Diigo highlights) https://stackoverflow.com/a/50661601/1970830

https://my.telegram.org/apps

Here is an example that can be used with fastai to send Telegram messages with every epoch results

@dataclass
class NotificationCallback(Callback):
        
    def on_train_begin(self, metrics_names: StrList, **kwargs: Any) -> None:
        notify_me("Epoch: train_loss , valid_loss , error_rate")

    def on_epoch_end(self,  epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        super().on_epoch_end(**kwargs)
        val_loss, accuracy = last_metrics[0], last_metrics[1]
        message = str(epoch) + ": " + f"{smooth_loss.item():.4f}" + " , " + f"{val_loss:.4f}" + " , " + f"{accuracy:.4f}"
        notify_me(message)
        return False # Haider: if return true it will stop training at this point
        
learn = create_cnn(data, models.resnet34, metrics=error_rate)
notif_cb = NotificationCallback()
learn.fit_one_cycle(4, callbacks=[notif_cb])

'''

import telegram
import json
import sys
import os
import matplotlib.pyplot as plt

def notify_me(message="Job's done!"):
    if 'win' in sys.platform:
        filename = 'C:' + os.environ['HOMEPATH'] + '\\.telegram'
    else:
        filename = os.environ['HOME'] + '/.telegram'

    with open(filename) as f:
        json_blob = f.read()
        credentials = json.loads(json_blob)

    # Initialize bot
    bot = telegram.Bot(token=credentials['api_key'])

    # Send message
    bot.send_message(chat_id=credentials['chat_id'], text=message)

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



# https://github.com/kasparlund/fastaiNotebooks/blob/master/show_transforms/show_transforms.ipynb
def plotImageMosaic( ims, masks=None, titles=None, figSize=(12,12), imCM=None, mskCM=None ):
    fig = plt.figure(figsize=figSize)
    
    nb    = len(ims)
    nrows = int(np.sqrt(nb))
    ncols = int(math.ceil(nb/nrows))

    gs = gridspec.GridSpec(nrows, ncols=ncols, height_ratios=None if masks is None else (np.zeros(nrows)+ 2), 
                           wspace=0.01, hspace=0.01 if titles is None else 0.2)
    for i in range(nb):
        inner = gridspec.GridSpecFromSubplotSpec(1 if masks is None else 2, 1,subplot_spec=gs[i], 
                                                 wspace=0.01, hspace=0.01)
        
        ax = plt.subplot(inner[0])
        ax.axis('off')
        ax.imshow(ims[i],cmap=imCM)
        if titles is not None: ax.set_title(titles[i])
        
        if masks is not None:
            ax = plt.subplot(inner[1])
            ax.axis('off')
            ax.imshow(masks[i],cmap=mskCM)
            
            
def brkpt(): 
    pass

