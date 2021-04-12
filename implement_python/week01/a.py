def CostFunction(X, y, theta):
    # 目的関数
    J = 0
    m = 8
    for i, x in enumerate(X):
        J += ((theta[0] + x*theta[1]) - y[i])*((theta[0] + x*theta[1]) - y[i])
    J = J/(2*m)

    return J


def GradientDescent(X, y, theta, alpha):
    # 最急降下法/勾配降下法
    # シータの更新
    temp0 = theta[0] - alpha*2*CostFunction(X, y, theta)
    temp1 = theta[1] - alpha*2*CostFunction(X, y, theta)
    theta[0] = temp0
    theta[1] = temp1

    return theta


# 線形回帰

X = [12, 24, 35, 49, 60, 70, 89, 100]
y = [67, 127, 187, 247, 307, 367, 427, 487]
theta = [0, 0]
alpha = 0.5

for i in range(20):
    theta = GradientDescent(X, y, theta, alpha)

print(theta)
