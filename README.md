# Solving-a-coupled-ODE-using-Multivariate-Newton-Raphson-Method.
Solving a coupled ODE using Multivariate Newton Raphson Method.


![](https://github.com/DrAliSeif/Solving-a-coupled-ODE-using-Multivariate-Newton-Raphson-Method./blob/main/plot.png)

$\ \dfrac{dy_{1}}{dt}= -0.04y_{1} + 10^4y_{2}y_{3} \$

$\ \dfrac{dy_{2}}{dt} = 0.04y_{1} - 10^4y_{2}y_{3} - 3 * 10^7y_{2}^2 \$

$\ \dfrac{dy_{3}}{dt} = 3 * 10^7y_{2}^2 \$



$\ y_{1}^{n} - y_{1}^{n-1}  - dt \cdot (-0.04y_1 + 10^4y_2y_3) = f_{1}^{n}(y_1, y_2, y_3) \$

$\ y_{2}^{n} - y_{2}^{n-1} - dt \cdot (0.04y_1 - 10^4y_2y_3 - 3 \cdot 10^7y_2^2) = f_{2}^{n}(y_1, y_2, y_3) \$

$\ y_{3}^{n} - y_{3}^{n-1} - dt \cdot (3 \cdot 10^7y_2^2) = f_{3}^{n}(y_1, y_2, y_3) \$


$\ \Rightarrow y_{1, k+1}^{n} = y_{1, k}^{n} - J_{f_1} f_1y_{1, k}^{n} \$

$\ \Rightarrow y_{2, k+1}^{n} = y_{2, k}^{n} - J_{f_2} f_2y_{2, k}^{n} \$

$\ \Rightarrow y_{3, k+1}^{n} = y_{2, k}^{n} - J_{f_3} f_3y_{3, k}^{n} \$
