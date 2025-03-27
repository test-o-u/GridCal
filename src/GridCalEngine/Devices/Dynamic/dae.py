class DAE: 
    def __init__(self):

        self.nx = 0
        self.ny = 0 
        self.x = None
        self.y = {}
        self.f = None
        self.g = None 

        # Jacobian sparse matrices
        self.dfx = None
        self.dfy = None
        self.dgx = None
        self.dgy = None
