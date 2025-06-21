# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import math
from GridCalEngine.Utils.Symbolic.symbolic import Var
from GridCalEngine.Utils.Symbolic.block import EquationBlock, connect
from GridCalEngine.Utils.Symbolic.leaf_blocks import Constant
from GridCalEngine.Utils.Symbolic.engine import Engine



# parameters & symbolic variables
Rs, Ld, Lq = 0.01, 0.8, 0.7
id_, iq = Var("i_d"), Var("i_q")           # states
vd, vq, w, phi = (Var(n) for n in ("v_d", "v_q", "omega", "phi_f"))

# RHS expressions
f_id = (vd - Rs*id_ + w*Lq*iq) / Ld
f_iq = (vq - Rs*iq - w*Ld*id_ - w*phi) / Lq

# Outputs simply expose the currents
outputs = {"i_d": id_, "i_q": iq}

RMSGen = EquationBlock(
    "Gen",
    inputs=[vd, vq, w, phi],
    states={id_: f_id, iq: f_iq},
    outputs=outputs
)


const_vd, const_vq = Constant(1.0, "vd"), Constant(0.0, "vq")
const_w  = Constant(2*math.pi*50, "w")
const_phi= Constant(1.1, "phi")

connect(const_vd.out_port("out"), RMSGen.inputs["v_d"])
connect(const_vq.out_port("out"), RMSGen.inputs["v_q"])
connect(const_w .out_port("out"), RMSGen.inputs["omega"])
connect(const_phi.out_port("out"), RMSGen.inputs["phi_f"])

eng = Engine([const_vd, const_vq, const_w, const_phi, RMSGen])
for t in eng.simulate(0, 0.05, 1e-4):
    print(t, RMSGen.outputs["i_d"].value, RMSGen.outputs["i_q"].value)
