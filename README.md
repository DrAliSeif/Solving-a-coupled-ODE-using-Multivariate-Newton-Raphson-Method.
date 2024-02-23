# Solving-a-coupled-ODE-using-Multivariate-Newton-Raphson-Method.
Solving a coupled ODE using Multivariate Newton Raphson Method.


![](https://github.com/DrAliSeif/Solving-a-coupled-ODE-using-Multivariate-Newton-Raphson-Method./blob/main/plot.png)

$\ \frac{dy_{1}}{dt}= -0.04y_{1} + 10^4y_{2}y_{3} \$

$\ \frac{dy_{2}}{dt} = 0.04y_{1} - 10^4y_{2}y_{3} - 3 * 10^7y_{2}^2 \$

$\ \frac{dy_{3}}{dt} = 3 * 10^7y_{2}^2 \$



$\ y_{1}^{n} - y_{1}^{n-1}  - dt \cdot (-0.04y_1 + 10^4y_2y_3) = f_{1}^{n}(y_1, y_2, y_3) \$

$\ y_{2}^{n} - y_{2}^{n-1} - dt \cdot (0.04y_1 - 10^4y_2y_3 - 3 \cdot 10^7y_2^2) = f_{2}^{n}(y_1, y_2, y_3) \$

$\ y_{3}^{n} - y_{3}^{n-1} - dt \cdot (3 \cdot 10^7y_2^2) = f_{3}^{n}(y_1, y_2, y_3) \$

