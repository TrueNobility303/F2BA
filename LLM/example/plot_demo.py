import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('F2SA.csv')

F2SA_ST = df.iloc[:, 4].tolist() # single-timescale
F2SA_TT = df.iloc[:, 8].to_list() # two-timescale

x_F2SA_ST = list(range(len(F2SA_ST)))  # Original x values
x_F2SA_ST = [x * 10 for x in x_F2SA_ST]  # Expand by 10x, which corresponds the the number of inner loops

x_F2SA_TT = list(range(len(F2SA_TT)))  # Similarly
x_F2SA_TT = [x * 10 for x in x_F2SA_TT]  

# Enable LaTeX rendering
# plt.rc('text', usetex=True)
plt.rc('font', size=24)

plt.figure()
plt.plot(x_F2SA_ST, F2SA_ST, ':b', label=r'${\rm lr}_y={\rm lr}_x$', linewidth=3)
plt.plot(x_F2SA_TT, F2SA_TT, '-k', label=r'${\rm lr}_y \ll {\rm lr}_x$', linewidth=3)
plt.legend()
plt.grid()
plt.xlabel('#Total Iterations')
plt.ylabel(r'$w_{\rm noisy}$')
plt.tight_layout()
plt.show()
plt.savefig('GPT2_demo.png')
plt.savefig('GPT2_demo.pdf', format='pdf')
