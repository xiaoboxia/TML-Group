import numpy as np
import matplotlib.pyplot as plt
import torch


plt.plot(x,f_True_mean_list,'m--',linewidth=1.0,label='Forward-T')
plt.fill_between(x,f_True_mean_list_up,f_True_mean_list_under,color='m',alpha=0.1)

plt.plot(x,f_Anchor_mean_list,'g--',linewidth=1.0,label='Forward-A')
plt.fill_between(x,f_Anchor_mean_list_up,f_Anchor_mean_list_under,color='g',alpha=0.1)


plt.plot(x,f_correctiona_mean_list,'b--',linewidth=1.0,label='Forward-A-R')
plt.fill_between(x,f_correctiona_mean_list_up,f_correctiona_mean_list_under,color='b',alpha=0.1)

plt.plot(x,f_noAnchor_mean_list,'r--',linewidth=1.0,label='Forward-N/A')
plt.fill_between(x,f_noAnchor_mean_list_up,f_noAnchor_mean_list_under,color='r',alpha=0.1)


plt.plot(x,f_correction_mean_list,'k--',linewidth=1.0,label='Forward-N/A-R')
plt.fill_between(x,f_correction_mean_list_up,f_correction_mean_list_under,color='gray',alpha=0.1)



plt.plot(x,re_True_mean_list,'m',linewidth=1.0,label='Reweight-T')
plt.fill_between(x,re_True_mean_list_up,re_True_mean_list_under,color='m',alpha=0.1)

plt.plot(x,re_Anchor_mean_list,'g',linewidth=1.0,label='Reweight-A')
plt.fill_between(x,re_Anchor_mean_list_up,re_Anchor_mean_list_under,color='g',alpha=0.1)

plt.plot(x,re_correctiona_mean_list,'b',linewidth=1.0,label='Reweight-A-R')
plt.fill_between(x,re_correctiona_mean_list_up,re_correctiona_mean_list_under,color='b',alpha=0.1)

plt.plot(x,re_noAnchor_mean_list,'r',linewidth=1.0,label='Reweight-N/A')
plt.fill_between(x,re_noAnchor_mean_list_up,re_noAnchor_mean_list_under,color='r',alpha=0.1)
plt.plot(x,re_correction_mean_list,'k',linewidth=1.0,label='Reweight-N/A-R')
plt.fill_between(x,re_correction_mean_list_up,re_correction_mean_list_under,color='gray',alpha=0.1)


