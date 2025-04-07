import numpy as np
from scipy.sparse import coo_matrix
from collections import defaultdict


class DAE:
    """
    DAE class to store numerical paramter, state and algebraic variable, jacobian and residual values.
    """
    def __init__(self):

        self.nx = 0
        self.ny = 0

        self.x = None
        self.y = None
        self.f = None
        self.g = None

        # Dictionaries to accumulate Jacobian values
        self.dfx = {}
        self.dfy = {}
        self.dgx = {}
        self.dgy = {}

        # Sets to store sparsity pattern
        self.sparsity_fx = list()
        self.sparsity_fy = list()
        self.sparsity_gx = list()
        self.sparsity_gy = list()

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
            sparsity_set.append((row, col))  # Store pattern

    def build_sparse_matrix(self, jac_dict, sparsity_set, shape, jac_type):
        """
        Convert accumulated values into a sparse matrix using the precomputed pattern.
        """

        rows, cols = zip(*sparsity_set) if sparsity_set else ([], [])
        if jac_type == 'dfx':
            values = [jac_dict.get((r, c), 0) for r, c in sparsity_set]
        if jac_type == 'dfy':
            values = [jac_dict.get((r, c + self.nx), 0) for r, c in sparsity_set]
        if jac_type == 'dgx':
            values = [jac_dict.get((r + self.nx, c), 0) for r, c in sparsity_set]
        if jac_type == 'dgy':
            values = [jac_dict.get((r + self.nx, c + self.nx), 0) for r, c in sparsity_set]

        return coo_matrix((values, (rows, cols)), shape=shape)

    def finalize_jacobians(self):
        """
        Builds all Jacobian matrices from stored triplets and sparsity patterns.
        """


        self.dfx = self.build_sparse_matrix(self.dfx,
                                            [(row, col) for row, col in self.sparsity_fx],
                                            (self.nx, self.nx), 'dfx')

        self.dfy = self.build_sparse_matrix(self.dfy,
                                            [(row, col - self.nx) for row, col in self.sparsity_fy],
                                            (self.nx, self.ny), 'dfy')

        self.dgx = self.build_sparse_matrix(self.dgx,
                                            [(row - self.nx, col) for row, col in self.sparsity_gx],
                                            (self.ny, self.nx), 'dgx')

        self.dgy = self.build_sparse_matrix(self.dgy,
                                            [(row - self.nx, col - self.nx) for row, col in self.sparsity_gy],
                                            (self.ny, self.ny), 'dgy')


    def initilize_fg(self):
        self.f = None
        self.g = None

    def update_fg(self):
        self.f = None
        self.g = None
