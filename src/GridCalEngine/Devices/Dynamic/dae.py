from scipy.sparse import coo_matrix
from collections import defaultdict

class DAE:
    def __init__(self):

        self.nx = 0
        self.ny = 0

        self.x = None
        self.y = {}
        self.f = None
        self.g = None

        # Dictionaries to accumulate Jacobian values
        self.dfx = {}
        self.dfy = {}
        self.dgx = {}
        self.dgy = {}

        # Sets to store sparsity pattern
        self.sparsity_fx = set()
        self.sparsity_fy = set()
        self.sparsity_gx = set()
        self.sparsity_gy = set()

        # Dictionary with all the parameters
        self.params_dict = defaultdict(dict)

        # Dictionary with all the residuals for updating jacobian
        self.residuals_dict = defaultdict(dict)

    def add_to_jacobian(self, jac_dict, sparsity_set, row, col, value):
        """
        Accumulate values and track sparsity pattern.
        """
        if (row, col) in jac_dict:
            jac_dict[(row, col)] += value
        else:
            jac_dict[(row, col)] = value  # First assignment
            sparsity_set.add((row, col))  # Store pattern

    def build_sparse_matrix(self, jac_dict, sparsity_set, shape):
        """
        Convert accumulated values into a sparse matrix using the precomputed pattern.
        """
        rows, cols = zip(*sparsity_set) if sparsity_set else ([], [])
        values = [jac_dict.get((r, c), 0) for r, c in sparsity_set]
        return coo_matrix((values, (rows, cols)), shape=shape)

    def finalize_jacobians(self):
        """
        Builds all Jacobian matrices from stored triplets and sparsity patterns.
        """
        # self.dfx = self.build_sparse_matrix(self.dfx, self.sparsity_fx, (self.nx, self.nx))
        # self.dfy = self.build_sparse_matrix(self.dfy, self.sparsity_fy, (self.nx, self.ny))
        # self.dgx = self.build_sparse_matrix(self.dgx, self.sparsity_gx, (self.ny, self.nx))
        self.dgy = self.build_sparse_matrix(self.dgy, self.sparsity_gy, (self.ny, self.ny))


