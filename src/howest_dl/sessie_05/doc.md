# Windowing

- Input

```raw
time:  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15
   x: x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15
   y: y0 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15
   
padding P = [0,...,0]
```

- Resulting feature set with window = 3:
```raw
--- Feature ---, -->  --- Target ---,
[ [ P,  P, X0],         [ [y0],
  [ P, X0, X1],           [y1],
  [X0, X1, X2],           [y2],
  [X1, X2, X3],           [y3],
  ...,                    ...,
  [Xn-2, Xn-1, Xn],       [yn],
]                       ] 
```
