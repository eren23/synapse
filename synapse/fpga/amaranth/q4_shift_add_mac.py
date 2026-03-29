"""
Q4 Shift-and-Add MAC Unit for FPGA.

A single Q4 block MAC computes the dot product of 32 input activations
with 32 hardwired 4-bit weights using only shifts and adds — no multipliers.

The weights are baked into the combinational logic at generation time.
Each weight value (-8 to 7) becomes a fixed shift-add tree.

Architecture:
    32 inputs x 16-bit fixed-point
        |
    32 parallel shift-add trees (hardwired per weight)
        |
    Adder tree (5 stages: 32 -> 16 -> 8 -> 4 -> 2 -> 1)
        |
    1 output x 32-bit accumulator
"""

from amaranth.hdl import *


def shift_add_tree(m: Module, x: Signal, w: int, prefix: str) -> Signal:
    """Generate combinational logic for x * w using shifts and adds.

    w is a 4-bit signed integer in [-8, 7].
    x is a signed 16-bit input.
    Returns a signed signal wide enough to hold the result.
    """
    # x * w needs at most 16 + 4 = 20 bits (signed)
    result = Signal(signed(20), name=f"{prefix}_result")

    if w == 0:
        m.d.comb += result.eq(0)
    elif w == 1:
        m.d.comb += result.eq(x)
    elif w == -1:
        m.d.comb += result.eq(-x)
    elif w == 2:
        m.d.comb += result.eq(x << 1)
    elif w == -2:
        m.d.comb += result.eq(-(x << 1))
    elif w == 3:
        m.d.comb += result.eq((x << 1) + x)
    elif w == -3:
        m.d.comb += result.eq(-((x << 1) + x))
    elif w == 4:
        m.d.comb += result.eq(x << 2)
    elif w == -4:
        m.d.comb += result.eq(-(x << 2))
    elif w == 5:
        m.d.comb += result.eq((x << 2) + x)
    elif w == -5:
        m.d.comb += result.eq(-((x << 2) + x))
    elif w == 6:
        m.d.comb += result.eq((x << 2) + (x << 1))
    elif w == -6:
        m.d.comb += result.eq(-((x << 2) + (x << 1)))
    elif w == 7:
        m.d.comb += result.eq((x << 3) - x)
    elif w == -7:
        m.d.comb += result.eq(-((x << 3) - x))
    elif w == -8:
        m.d.comb += result.eq(-(x << 3))
    else:
        raise ValueError(f"Weight {w} out of Q4 range [-8, 7]")

    return result


class Q4ShiftAddMAC(Elaboratable):
    """A single Q4 block MAC unit with hardwired weights.

    Computes dot product of 32 inputs with 32 fixed weights using shift-add.
    Weights are baked into the logic at construction time.

    Parameters
    ----------
    weights : list[int]
        32 signed 4-bit integers in [-8, 7], the hardwired weights for this block.
    input_width : int
        Bit width of input activations (default 16, signed fixed-point).

    Ports
    -----
    inputs : list of Signal(signed(input_width))
        32 input activation values.
    output : Signal(signed(32))
        The accumulated dot product (integer, before scale multiplication).
    """

    def __init__(self, weights: list, input_width: int = 16):
        assert len(weights) == 32, f"Q4 block must have 32 weights, got {len(weights)}"
        assert all(-8 <= w <= 7 for w in weights), f"Weights must be in [-8, 7]"
        self.weights = weights
        self.input_width = input_width

        # Ports
        self.inputs = [Signal(signed(input_width), name=f"in_{i}") for i in range(32)]
        self.output = Signal(signed(32), name="dot_product")

    def elaborate(self, platform):
        m = Module()

        # Stage 1: 32 parallel shift-add trees
        products = []
        for i, w in enumerate(self.weights):
            if w == 0:
                continue  # Skip zero weights entirely — no logic generated
            prod = shift_add_tree(m, self.inputs[i], w, f"w{i}")
            products.append(prod)

        if not products:
            # All weights are zero
            m.d.comb += self.output.eq(0)
            return m

        # Stage 2: Adder tree to sum all products
        # Binary reduction: pairs -> pairs -> ... -> single value
        level = products
        stage = 0
        while len(level) > 1:
            next_level = []
            for j in range(0, len(level), 2):
                if j + 1 < len(level):
                    # Width grows by 1 bit per addition stage
                    width = max(level[j].shape().width, level[j+1].shape().width) + 1
                    s = Signal(signed(width), name=f"sum_s{stage}_{j//2}")
                    m.d.comb += s.eq(level[j] + level[j+1])
                    next_level.append(s)
                else:
                    next_level.append(level[j])
            level = next_level
            stage += 1

        # Final output (extend to 32 bits)
        m.d.comb += self.output.eq(level[0])

        return m


class Q4LinearHardwired(Elaboratable):
    """A full Q4 linear layer with all weights hardwired.

    Time-multiplexed: processes one output row per clock cycle.
    All input activations must be presented simultaneously.

    Parameters
    ----------
    weight_blocks : list[list[int]]
        List of Q4 blocks (each block is 32 ints). Layout: [out_features * blocks_per_row].
    out_features : int
    in_features : int
    scales : list[float]
        Block scales as floats. Stored as fixed-point constants.
    input_width : int
        Bit width of input activations.

    Ports
    -----
    x : list of Signal(signed(input_width))
        Input activations [in_features].
    start : Signal(1)
        Pulse high to begin computation.
    output : Signal(signed(32))
        Current output value (valid when output_valid is high).
    output_idx : Signal
        Index of current output row.
    output_valid : Signal(1)
        High when output is valid.
    done : Signal(1)
        High when all outputs computed.
    """

    def __init__(self, weight_blocks, scales, out_features, in_features,
                 input_width=16, scale_width=16, scale_frac=10):
        self.weight_blocks = weight_blocks
        self.scales = scales
        self.out_features = out_features
        self.in_features = in_features
        self.input_width = input_width
        self.scale_width = scale_width
        self.scale_frac = scale_frac

        padded_k = ((in_features + 31) // 32) * 32
        self.blocks_per_row = padded_k // 32

        # Ports
        self.x = [Signal(signed(input_width), name=f"x_{i}") for i in range(in_features)]
        self.start = Signal(name="start")
        self.output = Signal(signed(32), name="output")
        self.output_idx = Signal(range(out_features), name="output_idx")
        self.output_valid = Signal(name="output_valid")
        self.done = Signal(name="done")

    def elaborate(self, platform):
        m = Module()

        bpr = self.blocks_per_row
        n = self.out_features
        k = self.in_features

        # Instantiate MAC units — one per block position (bpr units)
        macs = []
        for b in range(bpr):
            # Use weights from row 0 initially; we'll mux per row
            mac = Q4ShiftAddMAC([0] * 32, self.input_width)
            m.submodules[f"mac_{b}"] = mac
            macs.append(mac)

        # Wire inputs to MACs
        for b in range(bpr):
            for i in range(32):
                col = b * 32 + i
                if col < k:
                    m.d.comb += macs[b].inputs[i].eq(self.x[col])
                else:
                    m.d.comb += macs[b].inputs[i].eq(0)

        # Since weights are HARDWIRED, we can't mux them at runtime.
        # Instead, for a full hardwired layer, we generate ALL output rows
        # as combinational logic and select with a mux.
        #
        # For large layers, this is too much logic. The time-multiplexed
        # version instantiates bpr MAC units and reconfigures weight selection.
        # But since weights are constants, "reconfiguring" means a big MUX.
        #
        # For the proof-of-concept, we generate all rows combinationally
        # and use the row counter to select.

        # Generate all row dot products combinationally
        row_results = []
        for row in range(n):
            # Accumulate block results for this row
            block_results = []
            for b in range(bpr):
                block_idx = row * bpr + b
                weights = self.weight_blocks[block_idx]
                scale = self.scales[block_idx]

                # Create a MAC for this specific block
                mac_name = f"row{row}_blk{b}"
                blk_mac = Q4ShiftAddMAC(weights, self.input_width)
                m.submodules[mac_name] = blk_mac

                # Wire inputs
                for i in range(32):
                    col = b * 32 + i
                    if col < k:
                        m.d.comb += blk_mac.inputs[i].eq(self.x[col])
                    else:
                        m.d.comb += blk_mac.inputs[i].eq(0)

                # Scale the block result (fixed-point multiply)
                scale_int = int(round(scale * (1 << self.scale_frac)))
                scale_const = Const(scale_int, signed(self.scale_width))
                scaled = Signal(signed(48), name=f"{mac_name}_scaled")
                m.d.comb += scaled.eq(blk_mac.output * scale_const)
                block_results.append(scaled)

            # Sum all blocks for this row
            if block_results:
                row_sum = block_results[0]
                for br in block_results[1:]:
                    new_sum = Signal(signed(48), name=f"row{row}_sum")
                    m.d.comb += new_sum.eq(row_sum + br)
                    row_sum = new_sum
                row_results.append(row_sum)
            else:
                z = Signal(signed(48), name=f"row{row}_zero")
                m.d.comb += z.eq(0)
                row_results.append(z)

        # Row counter for sequential output
        row_counter = Signal(range(n + 1), name="row_counter")
        running = Signal(name="running")

        with m.If(self.start):
            m.d.sync += row_counter.eq(0)
            m.d.sync += running.eq(1)
            m.d.sync += self.done.eq(0)

        with m.Elif(running):
            m.d.sync += self.output_valid.eq(1)
            m.d.sync += self.output_idx.eq(row_counter)

            # MUX: select the pre-computed row result
            with m.Switch(row_counter):
                for row in range(n):
                    with m.Case(row):
                        # Right-shift to remove scale fraction bits
                        m.d.sync += self.output.eq(row_results[row] >> self.scale_frac)

            with m.If(row_counter == n - 1):
                m.d.sync += running.eq(0)
                m.d.sync += self.done.eq(1)
            with m.Else():
                m.d.sync += row_counter.eq(row_counter + 1)

        with m.Else():
            m.d.sync += self.output_valid.eq(0)

        return m
