class Model: 
    def __init__(self): 
        self.f = ["x**2 + y + sin(x) + cos(y)", "x*y + exp(y)"]
        self.g = ["exp(a) + log(b) + a*b - a**2", "a**3 - b**3"]
        self.state = ["x", "y"]
        self.algeb = ["a", "b"]