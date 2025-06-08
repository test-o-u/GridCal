# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from enum import Enum, auto
from typing import List, Dict, Optional
from collections import defaultdict
import sympy as sym

from PySide6.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsItem,
                               QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem, QMenu, QGraphicsPathItem,
                               QDialog, QVBoxLayout, QComboBox, QDialogButtonBox)
from PySide6.QtGui import QPen, QBrush, QPainterPath, QAction, QPainter
from PySide6.QtCore import Qt, QPointF
import sys


class BlockType(Enum):
    GAIN = auto()
    SUM = auto()
    INTEGRATOR = auto()
    DERIVATIVE = auto()
    PRODUCT = auto()
    DIVIDE = auto()
    SQRT = auto()
    SQUARE = auto()
    ABS = auto()
    MIN = auto()
    MAX = auto()
    STEP = auto()
    CONSTANT = auto()
    SATURATION = auto()
    RELATIONAL = auto()
    LOGICAL = auto()
    SOURCE = auto()
    DRAIN = auto()
    GENERIC = auto()


class BlockTypeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Block Type")
        self.layout = QVBoxLayout(self)

        self.combo = QComboBox(self)
        for bt in BlockType:
            self.combo.addItem(bt.name, bt)
        self.layout.addWidget(self.combo)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def selected_block_type(self) -> BlockType:
        return self.combo.currentData()


class SymbolicBlock:
    def __init__(self, block_id: int, name: str, block_type: BlockType,
                 inputs: List[object], outputs: List[object]) -> None:
        self.block_id: int = block_id
        self.name: str = name
        self.block_type: BlockType = block_type
        self.inputs: List[object] = inputs
        self.outputs: List[object] = outputs
        self.input_symbols: List[sym.Symbol] = []
        self.output_symbols: List[sym.Symbol] = []
        self.state_symbol: Optional[sym.Symbol] = None
        self.equations: List[sym.Expr] = []

    def generate_equations(self) -> None:
        inp = self.input_symbols
        if not self.output_symbols:
            return  # No outputs to define
        out = self.output_symbols[0]
        match self.block_type:
            case BlockType.GAIN:
                k = sym.Symbol(f"{self.name}_k")
                self.equations.append(sym.Eq(out, k * inp[0]))
            case BlockType.SUM:
                self.equations.append(sym.Eq(out, sum(inp)))
            case BlockType.INTEGRATOR:
                x = sym.Symbol(f"x_{self.block_id}")
                self.state_symbol = x
                self.equations.append(sym.Eq(sym.Derivative(x), inp[0]))
                self.equations.append(sym.Eq(out, x))
            case BlockType.DERIVATIVE:
                self.equations.append(sym.Eq(out, sym.Derivative(inp[0])))
            case BlockType.PRODUCT:
                prod = inp[0]
                for i in inp[1:]:
                    prod *= i
                self.equations.append(sym.Eq(out, prod))
            case BlockType.DIVIDE:
                self.equations.append(sym.Eq(out, inp[0] / inp[1]))
            case BlockType.SQRT:
                self.equations.append(sym.Eq(out, inp[0] ** 0.5))
            case BlockType.SQUARE:
                self.equations.append(sym.Eq(out, inp[0] ** 2))
            case BlockType.ABS:
                self.equations.append(sym.Eq(out, sym.Abs(inp[0])))
            case BlockType.MIN:
                self.equations.append(sym.Eq(out, sym.Min(*inp)))
            case BlockType.MAX:
                self.equations.append(sym.Eq(out, sym.Max(*inp)))
            case BlockType.STEP:
                threshold = sym.Symbol(f"{self.name}_threshold")
                level = sym.Symbol(f"{self.name}_level")
                self.equations.append(sym.Eq(out, sym.Piecewise((0, inp[0] < threshold), (level, True))))
            case BlockType.CONSTANT:
                const_val = sym.Symbol(f"{self.name}_val")
                self.equations.append(sym.Eq(out, const_val))
            case BlockType.SATURATION:
                u_min = sym.Symbol(f"{self.name}_min")
                u_max = sym.Symbol(f"{self.name}_max")
                self.equations.append(sym.Eq(out, sym.Min(sym.Max(inp[0], u_min), u_max)))
            case BlockType.RELATIONAL:
                self.equations.append(sym.Eq(out, sym.Piecewise((1, inp[0] > inp[1]), (0, True))))
            case BlockType.LOGICAL:
                self.equations.append(sym.Eq(out, sym.Piecewise((1, inp[0] & inp[1]), (0, True))))
            case BlockType.SOURCE:
                src_val = sym.Symbol(f"{self.name}_val")
                self.equations.append(sym.Eq(out, src_val))
            case BlockType.DRAIN:
                pass
            case _:
                self.equations.append(sym.Eq(out, sum(inp)))


class Port(QGraphicsEllipseItem):
    """
    Port of a block
    """

    def __init__(self, block: "Block", is_input: bool, index: int, total: int, radius=6):
        """

        :param block:
        :param is_input:
        :param index:
        :param total:
        :param radius:
        """
        super().__init__(-radius, -radius, 2 * radius, 2 * radius, block)
        self.setBrush(QBrush(Qt.GlobalColor.blue if is_input else Qt.GlobalColor.green))
        self.setPen(QPen(Qt.GlobalColor.black))
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        self.block = block
        self.is_input = is_input
        self.connection = None

        spacing = block.rect().height() / (total + 1)
        y = spacing * (index + 1)
        x = 0 if is_input else block.rect().width()
        self.setPos(x, y)

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def is_connected(self):
        return self.connection is not None


class Connection(QGraphicsPathItem):
    def __init__(self, source_port, target_port):
        super().__init__()
        self.setZValue(-1)
        self.source_port = source_port
        self.target_port = target_port
        self.source_port.connection = self
        self.target_port.connection = self
        self.setPen(QPen(Qt.GlobalColor.darkBlue, 2))
        self.setAcceptHoverEvents(True)

        self.update_path()

    def update_path(self):
        start = self.source_port.scenePos()
        end = self.target_port.scenePos()
        mid_x = (start.x() + end.x()) / 2
        c1 = QPointF(mid_x, start.y())
        c2 = QPointF(mid_x, end.y())
        path = QPainterPath(start)
        path.cubicTo(c1, c2, end)
        self.setPath(path)

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def contextMenuEvent(self, event):
        menu = QMenu()
        remove_action = QAction("Remove Connection", menu)
        menu.addAction(remove_action)
        if menu.exec(event.screenPos()) == remove_action:
            self.scene().removeItem(self)
            self.source_port.connection = None
            self.target_port.connection = None


class ResizeHandle(QGraphicsRectItem):
    def __init__(self, block, size=10):
        super().__init__(0, 0, size, size, block)
        self.setBrush(QBrush(Qt.GlobalColor.darkGray))
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setZValue(2)
        self.block = block
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setAcceptHoverEvents(True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if not self.block._resizing_from_handle:
                return super().itemChange(change, value)

            new_pos = value  # already QPointF
            min_width, min_height = 40, 30
            new_width = max(new_pos.x(), min_width)
            new_height = max(new_pos.y(), min_height)

            self.block.resize_block(new_width, new_height)

            return QPointF(new_width, new_height)
        return super().itemChange(change, value)


class Block(QGraphicsRectItem):
    def __init__(self, name, inputs=1, outputs=1, block_type=BlockType.GENERIC):
        """

        :param name:
        :param inputs:
        :param outputs:
        :param block_type:
        """
        super().__init__(0, 0, 100, 60)
        self.setBrush(Qt.GlobalColor.lightGray)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges
        )
        self.setAcceptHoverEvents(True)
        self.setAcceptHoverEvents(True)

        self.name_item = QGraphicsTextItem(name, self)

        self.name_item.setPos(10, 5)

        self.inputs = [Port(self, True, i, inputs) for i in range(inputs)]
        self.outputs = [Port(self, False, i, outputs) for i in range(outputs)]

        self.block_type: BlockType = block_type

        self.resize_handle = ResizeHandle(self)

        # ✅ Avoid triggering overridden setRect during init
        super().setRect(0, 0, 100, 60)
        self.update_ports()
        self.update_handle_position()

        self._resizing_from_handle = False

    def resize_block(self, width, height):
        # Update geometry safely
        self.prepareGeometryChange()
        QGraphicsRectItem.setRect(self, 0, 0, width, height)
        self.update_ports()
        self.update_handle_position()

    def update_handle_position(self):
        rect = self.rect()
        self._resizing_from_handle = False
        self.resize_handle.setPos(rect.width(), rect.height())
        self._resizing_from_handle = True

    def _set_rect_internal(self, w, h):
        QGraphicsRectItem.setRect(self, 0, 0, w, h)
        self.update_ports()
        self.update_handle_position()

    def setRect(self, x, y, w, h):
        if not getattr(self, '_suppress_resize', False):
            self._set_rect_internal(w, h)

    def update_ports(self):
        for i, port in enumerate(self.inputs):
            spacing = self.rect().height() / (len(self.inputs) + 1)
            port.setPos(0, spacing * (i + 1))
        for i, port in enumerate(self.outputs):
            spacing = self.rect().height() / (len(self.outputs) + 1)
            port.setPos(self.rect().width(), spacing * (i + 1))
        self.update_handle_position()
        # Also update connections
        for port in self.inputs + self.outputs:
            if port.connection:
                port.connection.update_path()

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            for port in self.inputs + self.outputs:
                if port.connection:
                    port.connection.update_path()
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        menu = QMenu()
        delete_action = QAction("Remove Block", menu)
        menu.addAction(delete_action)
        if menu.exec(event.screenPos()) == delete_action:
            # Remove connections
            for port in self.inputs + self.outputs:
                if port.connection:
                    self.scene().removeItem(port.connection)
                    if port.connection.source_port:
                        port.connection.source_port.connection = None
                    if port.connection.target_port:
                        port.connection.target_port.connection = None
            # Remove the block itself
            self.scene().removeItem(self)


class GraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self._panning = False
        self._pan_start = QPointF()

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = 1.15 if zoom_in else 1 / 1.15
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)


class DiagramScene(QGraphicsScene):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.temp_line = None
        self.source_port = None

    def contextMenuEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())

        # ✅ Let lines, block and ports handle their own context menus
        if item is not None:
            if not isinstance(item, DiagramScene):
                return super().contextMenuEvent(event)

        # menu = QMenu()
        # scene_pos = event.scenePos()
        # for label, ins, outs in [
        #     ("1 in / 1 out", 1, 1),
        #     ("2 in / 1 out", 2, 1),
        #     ("2 in / 2 out", 2, 2),
        #     ("Source", 0, 1),
        #     ("Drain", 1, 0),
        # ]:
        #     action = QAction(label, menu)
        #     action.triggered.connect(lambda _, x=scene_pos,
        #                                     i=ins,
        #                                     o=outs:
        #                              self.add_block(pos=x, ins=i, outs=o))
        #     menu.addAction(action)
        # menu.exec(event.screenPos())

        dialog = BlockTypeDialog()
        if dialog.exec():
            block_type = dialog.selected_block_type()
            ins = 2 if block_type in {BlockType.SUM, BlockType.PRODUCT, BlockType.MIN, BlockType.MAX} else 1
            outs = 1
            if block_type == BlockType.SOURCE:
                ins = 0
            elif block_type == BlockType.DRAIN:
                outs = 0
            self.add_block(event.scenePos(), ins, outs, block_type)

    def add_block(self, pos, ins, outs, block_type=BlockType.GENERIC):
        """

        :param pos:
        :param ins:
        :param outs:
        :param block_type:
        :return:
        """
        block = Block(name=f"{block_type.name}_{len(self.items())}",
                      inputs=ins,
                      outputs=outs,
                      block_type=block_type)
        self.addItem(block)
        block.setPos(pos)

    def mousePressEvent(self, event):
        for item in self.items(event.scenePos()):
            if isinstance(item, Port) and not item.is_input and not item.is_connected():
                self.source_port = item
                path = QPainterPath(item.scenePos())
                self.temp_line = self.addPath(path, QPen(Qt.PenStyle.DashLine))
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.temp_line:
            start = self.source_port.scenePos()
            end = event.scenePos()
            mid_x = (start.x() + end.x()) / 2
            c1 = QPointF(mid_x, start.y())
            c2 = QPointF(mid_x, end.y())
            path = QPainterPath(start)
            path.cubicTo(c1, c2, end)
            self.temp_line.setPath(path)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.temp_line:
            # FIX: scan items under mouse for a valid input Port
            for item in self.items(event.scenePos()):
                if isinstance(item, Port) and item.is_input and not item.is_connected():
                    connection = Connection(self.source_port, item)
                    self.addItem(connection)
                    break
            self.removeItem(self.temp_line)
            self.temp_line = None
            self.source_port = None
        else:
            super().mouseReleaseEvent(event)


def extract_dae_from_scene(scene: QGraphicsScene) -> List[sym.Expr]:
    """

    :param scene:
    :return:
    """  # Adjust if these classes are moved
    symbol_counter: Dict[str, int] = defaultdict(int)
    block_map: Dict[object, SymbolicBlock] = {}

    for i, item in enumerate(scene.items()):
        if isinstance(item, Block):
            block = SymbolicBlock(i, item.name_item.toPlainText(), item.block_type, item.inputs, item.outputs)
            block.input_symbols = [sym.Symbol(f"u_{i}_{j}") for j in range(len(block.inputs))]
            block.output_symbols = [sym.Symbol(f"y_{i}_{j}") for j in range(len(block.outputs))]
            block_map[item] = block

    for item in scene.items():
        if isinstance(item, Connection):
            src_block = item.source_port.block
            tgt_block = item.target_port.block
            src_idx = src_block.outputs.index(item.source_port)
            tgt_idx = tgt_block.inputs.index(item.target_port)
            output_sym = block_map[src_block].output_symbols[src_idx]
            block_map[tgt_block].input_symbols[tgt_idx] = output_sym

    equations: List[sym.Expr] = []
    for block in block_map.values():
        block.generate_equations()
        equations.extend(block.equations)

    return equations


class BlockEditor(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Block Editor with Ports")

        self.scene = DiagramScene(self)
        self.view = GraphicsView(self.scene)
        self.setCentralWidget(self.view)

        self.resize(800, 600)

    def extract_dae(self):
        eqs = extract_dae_from_scene(self.scene)
        return eqs


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BlockEditor()
    window.show()
    sys.exit(app.exec())
