# SVM

- Hyperplane: $w_0 + w_1 X_1 + w_2 X_2 + \ldots + w_n X_n = 0$
- For a point $(x_1', x_2')$
  - If $w_0 + w_1 x_1' + w_2 x_2' + \ldots + w_n x_n' > 0$, point above line
  - If $w_0 + w_1 x_1' + w_2 x_2' + \ldots + w_n x_n' < 0$, point below line
- If correctly separate classes, then
  - For training point $i$ with $y_i = 1$, $w_0 + w_1 x_{i1} + w_2 x_{i2} + \ldots + w_n x_{in} > 0$
  - For training point $i$ with $y_i = -1$, $w_0 + w_1 x_{i1} + w_2 x_{i2} + \ldots + w_n x_{in} < 0$
  - Combined into one inequality:
    - $y_i(w_0 + w_1 x_{i1} + w_2 x_{i2} + \ldots + w_n x_{in}) > 0$
- Margin: minimum distance from any training point to the hyperplane