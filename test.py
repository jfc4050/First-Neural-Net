import numpy as np

perc_tr = 0.9
a = np.random.randn(8, 3)

m_tr = int(perc_tr * a.shape[0])

order = np.random.permutation(a.shape[0])

order_tr = order[0:m_tr]
order_cv = order[m_tr]

print(order)
print(order_tr)
print(order_cv)

a_tr = np.take(a, order_tr, axis=0)
a_cv = np.take(a, order_cv, axis=0)
print(a)
print(a_tr)
print(a_cv)