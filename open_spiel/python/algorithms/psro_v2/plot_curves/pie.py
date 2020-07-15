import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = ["0.364 two devs",
          "0.636 one dev",
          ]

# data = [float(x.split()[0]) for x in recipe]
# ingredients = [x.split()[-1] for x in recipe]

data = [0.636, 0.364]
ingredients = ["One dev", "Two devs"]


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct)


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data), textprops=dict(color="w"), colors=['C1', 'C2'])

# wedges, texts, autotexts = ax.pie(data,  textprops=dict(color="w"))


ax.legend(wedges, ingredients,
          title="Iterations",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

# ax.set_title("Matplotlib bakery: A pie")

plt.show()