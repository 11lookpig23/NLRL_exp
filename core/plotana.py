from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
def plottrain(res,repeat,total_step):
    gap = int(total_step/repeat)
    res = res[:len(res)-(len(res)%repeat)]
    out = res.reshape(gap,repeat)
    dicfreq = {-1:[],0:[],1:[]}
    for k in out:
        freq = Counter(k)
        dicfreq[-1].append(freq[-1]/repeat)
        dicfreq[1].append(freq[1]/repeat)
        dicfreq[0].append(freq[0]/repeat)
    plt.plot(range(gap),dicfreq[1],color = 'r')
    plt.plot(range(gap),dicfreq[-1],color = 'blue')
    plt.plot(range(gap),dicfreq[0],color = 'yellow')
    plt.savefig("trainres.png")
    plt.show()