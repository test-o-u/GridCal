import networkx as nx
from typing import Iterable, List
from GridCalEngine.Utils.Symbolic.block import Block, Port


class Engine:
    """
    Fixed-step scheduler for a directed block graph.

    Parameters
    ----------
    blocks : list[Block]
        All *top-level* blocks (leaf or Subsystem) that form the model.
        Connection wires are discovered automatically.

    Notes
    -----
    * Uses NetworkX to topo-sort once on construction.
    * If a loop is detected (algebraic cycle) a `ValueError` is raised.
    * Each major step does:
        1. block.step(dt, t)  in topo order
        2. immediately propagates the block’s outputs to connected inputs
    """

    # ------------------------------------------------------------------
    def __init__(self, blocks: List["Block"]):
        self.blocks = blocks
        self.order  = self._toposort(blocks)

    # ------------------------------------------------------------------
    def _toposort(self, blks: Iterable["Block"]) -> List["Block"]:
        g = nx.DiGraph()
        for b in blks:
            g.add_node(b)

        # add edges: src block  → dst block
        for b in blks:
            for p_out in b.outputs.values():
                for p_dst in p_out.connections:
                    g.add_edge(b, p_dst.owner)

        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("Algebraic loop detected in block diagram. "
                             "Insert delay/integrator blocks to break the loop.")

        return list(nx.topological_sort(g))

    # ------------------------------------------------------------------
    def step_once(self, dt: float, t: float) -> None:
        """
        Advance the entire model by a single major step *dt*.

        * Each block’s `step(dt, t)` is called in topological order.
        * After a block finishes, its outputs are pushed to all connected
          destination ports so downstream blocks see fresh values.
        """
        for blk in self.order:
            blk.step(dt, t)

            # propagate this block's outputs
            for p_out in blk.outputs.values():
                for p_dst in p_out.connections:
                    p_dst.value = p_out.value

    # ------------------------------------------------------------------
    def simulate(self,
                 t0: float,
                 tf: float,
                 dt: float):
        """
        Generator that advances the system from *t0* to *tf* (inclusive),
        yielding the new time after each successful step.

            >>> for t in engine.simulate(0.0, 1.0, 0.01):
            ...     print(t)
        """
        t = t0
        steps = int(round((tf - t0) / dt))
        for _ in range(steps):
            self.step_once(dt, t)
            t += dt
            yield t
