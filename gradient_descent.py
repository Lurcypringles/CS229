```python

import csv

def preprocessing():
    cleaned_data = []
    with open("/Users/lucysheng/Downloads/house-prices.csv") as file:
        for line in file:
            if line.startswith("Home"):
                continue
            else:
                home, price, sqft, bedrooms, bathrooms, offers, brick, neighborhood = line.rstrip().split(",")
                cleaned_data.append((home, price, sqft, bedrooms, bathrooms, offers))
    return cleaned_data

def get_price(list):
    price = []
    for i in list:
        price.append(float(i[1]))
    return price

def get_sqft(list):
    sqft = []
    for i in list:
        sqft.append(float(i[2]))
    return sqft

def get_bedrooms(list):
    bedrooms = []
    for i in list:
        bedrooms.append(float(i[3]))
    return bedrooms

def get_bathrooms(list):
    bathrooms = []
    for i in list:
        bathrooms.append(float(i[4]))
    return bathrooms

def gradient_descent(data, learning_rate=0.0000001, epochs=1000):
    theta_0 = 0.0
    theta_1 = 0.0
    theta_2 = 0.0
    theta_3 = 0.0

    n = len(data)

    X1 = get_sqft(data)
    X2 = get_bedrooms(data)
    X3 = get_bathrooms(data)
    Y = get_price(data)

    for _ in range(epochs):
        m_gradient = 0
        b_gradient = 0
        theta_2_gradient = 0
        theta_3_gradient = 0

        for i in range(n):
            x1 = X1[i]
            x2 = X2[i]
            x3 = X3[i]
            y = Y[i]

            prediction = theta_0 + theta_1 * x1 + theta_2 * x2 + theta_3 * x3
            error = y - prediction

            b_gradient += -(2/n) * error
            m_gradient += -(2/n) * x1 * error
            theta_2_gradient += -(2/n) * x2 * error
            theta_3_gradient += -(2/n) * x3 * error
        
        theta_0 = theta_0 - learning_rate * b_gradient
        theta_1 = theta_1 - learning_rate * m_gradient
        theta_2 = theta_2 - learning_rate * theta_2_gradient
        theta_3 = theta_3 - learning_rate * theta_3_gradient

    return theta_0, theta_1, theta_2, theta_3

def stochastic_gradient_descent(data, learning_rate=0.0000001, epochs=1000):
    theta_0 = 0.0
    theta_1 = 0.0
    theta_2 = 0.0
    theta_3 = 0.0

    n = len(data)

    X1 = get_sqft(data)
    X2 = get_bedrooms(data)
    X3 = get_bathrooms(data)
    Y = get_price(data)

    for _ in range(epochs):
        for i in range(n):
            x1 = X1[i]
            x2 = X2[i]
            x3 = X3[i]
            y = Y[i]

            prediction = theta_0 + theta_1 * x1 + theta_2 * x2 + theta_3 * x3
            error = y - prediction

            b_gradient = -2 * error
            m_gradient = -2 * x1 * error
            theta_2_gradient = -2 * x2 * error
            theta_3_gradient = -2 * x3 * error
        
            theta_0 = theta_0 - learning_rate * b_gradient
            theta_1 = theta_1 - learning_rate * m_gradient
            theta_2 = theta_2 - learning_rate * theta_2_gradient
            theta_3 = theta_3 - learning_rate * theta_3_gradient

    return theta_0, theta_1, theta_2, theta_3

def prediction(sqft, bedroom, bathrooms):
    theta_0, theta_1, theta_2, theta_3 = gradient_descent(preprocessing())
    return theta_0 + theta_1 * sqft + theta_2 * bedroom + theta_3 * bathrooms

data = preprocessing()
theta_0, theta_1, theta_2, theta_3 = gradient_descent(data)
a, b, c, d = stochastic_gradient_descent(data)
print(f"Gradient Descent: {theta_0}, {theta_1}, {theta_2}, {theta_3}")
print(f"Stochastic Gradient Descent: {a}, {b}, {c}, {d}")

```
