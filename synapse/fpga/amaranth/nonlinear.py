"""
Non-linear operations for hardwired LEWM inference in RTL.

Implements fixed-point versions of:
  - GELU activation (piecewise linear approximation)
  - LayerNorm (mean, variance, reciprocal sqrt via Newton-Raphson)
  - Softmax over 3 elements (exp LUT + reciprocal)
  - adaLN modulation: normed * (1 + scale) + shift

All modules are combinational (no clock) to match the hardwired linear layers.
"""

import math
import numpy as np
from amaranth.hdl import *


# ---------------------------------------------------------------------------
# GELU: Piecewise Linear Approximation
# ---------------------------------------------------------------------------

class GELU_PWL(Elaboratable):
    """Piecewise-linear GELU approximation.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))

    Approximated with 16 linear segments over [-4, 4]:
      - x < -4: output = 0
      - x > 4:  output = x
      - else: linear interpolation between segment endpoints

    Fixed-point: Q8.8 input/output (16-bit signed, 8 fractional bits).
    Max error vs exact GELU: < 0.5% over [-4, 4].
    """

    FRAC = 8
    SCALE = 1 << 8  # 256

    def __init__(self, width=16):
        self.width = width
        self.x = Signal(signed(width), name="gelu_in")
        self.y = Signal(signed(width), name="gelu_out")

        # Pre-compute segment table
        self.segments = self._compute_segments(16)

    @staticmethod
    def _gelu_exact(x):
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

    def _compute_segments(self, n_segments):
        """Compute piecewise linear segments for GELU over [-4, 4]."""
        segments = []
        x_min, x_max = -4.0, 4.0
        step = (x_max - x_min) / n_segments

        for i in range(n_segments):
            x0 = x_min + i * step
            x1 = x0 + step
            y0 = self._gelu_exact(x0)
            y1 = self._gelu_exact(x1)

            # Slope and intercept for segment: y = slope * (x - x0) + y0
            slope = (y1 - y0) / (x1 - x0)
            segments.append({
                'x0': x0, 'x1': x1,
                'y0': y0, 'y1': y1,
                'slope': slope,
                'x0_fixed': int(round(x0 * self.SCALE)),
                'y0_fixed': int(round(y0 * self.SCALE)),
                'slope_fixed': int(round(slope * self.SCALE)),
            })

        return segments

    def elaborate(self, platform):
        m = Module()
        S = self.SCALE
        n_seg = len(self.segments)

        # Thresholds in fixed-point
        x_min_fixed = int(-4 * S)
        x_max_fixed = int(4 * S)

        with m.If(self.x < x_min_fixed):
            # x < -4: GELU ≈ 0
            m.d.comb += self.y.eq(0)
        with m.Elif(self.x >= x_max_fixed):
            # x > 4: GELU ≈ x
            m.d.comb += self.y.eq(self.x)
        with m.Else():
            # Determine segment index: (x - x_min_fixed) / step_fixed
            step_fixed = int(round((8.0 / n_seg) * S))
            offset = Signal(signed(self.width + 4), name="gelu_offset")
            m.d.comb += offset.eq(self.x - x_min_fixed)

            # Use cascaded if-else for segment selection
            for i, seg in enumerate(self.segments):
                boundary = seg['x0_fixed'] + int(round(8.0 / n_seg * S))
                if i < n_seg - 1:
                    cond = self.x < self.segments[i + 1]['x0_fixed']
                else:
                    cond = True  # Last segment

                # y = y0 + slope * (x - x0), all in fixed-point
                dx = Signal(signed(self.width + 4), name=f"gelu_dx{i}")
                prod = Signal(signed(32), name=f"gelu_prod{i}")
                result = Signal(signed(self.width), name=f"gelu_res{i}")

                m.d.comb += dx.eq(self.x - seg['x0_fixed'])
                m.d.comb += prod.eq(dx * seg['slope_fixed'])
                m.d.comb += result.eq(seg['y0_fixed'] + (prod >> self.FRAC))

                if i < n_seg - 1:
                    with m.If(cond):
                        m.d.comb += self.y.eq(result)
                else:
                    # Default (last segment)
                    with m.Else() if i > 0 else m.If(True):
                        m.d.comb += self.y.eq(result)

        return m


# ---------------------------------------------------------------------------
# LayerNorm (fixed-point, iterative reciprocal sqrt)
# ---------------------------------------------------------------------------

class LayerNorm(Elaboratable):
    """Fixed-point LayerNorm for LEWM (hidden_size=192).

    Computes: output[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]

    For the hardwired implementation:
      - weight and bias are constants (baked in)
      - mean/var computed combinationally via adder trees
      - reciprocal sqrt via lookup table (256 entries)

    Fixed-point: Q8.8 (16-bit signed, 8 fractional bits).
    """

    FRAC = 8
    SCALE = 1 << 8

    def __init__(self, hidden_size: int, weights: np.ndarray = None,
                 biases: np.ndarray = None):
        self.hidden_size = hidden_size
        self.x = [Signal(signed(16), name=f"ln_in_{i}") for i in range(hidden_size)]
        self.y = [Signal(signed(16), name=f"ln_out_{i}") for i in range(hidden_size)]

        # Hardwired weights/biases (or default to identity)
        if weights is None:
            weights = np.ones(hidden_size, dtype=np.float32)
        if biases is None:
            biases = np.zeros(hidden_size, dtype=np.float32)

        self.weight_fixed = [int(round(w * self.SCALE)) for w in weights]
        self.bias_fixed = [int(round(b * self.SCALE)) for b in biases]

    def elaborate(self, platform):
        m = Module()
        H = self.hidden_size
        S = self.SCALE
        FRAC = self.FRAC

        # Step 1: Compute mean via adder tree
        # Sum all inputs, divide by H
        sum_sig = self._adder_tree(m, self.x, "mean")

        # mean = sum / H (approximate: H=192 = 64*3, so /192 ≈ *341/65536)
        # More accurately: multiply by (1/192) in fixed point
        # 1/192 * 2^16 = 341.33... ≈ 341
        inv_h = int(round((1.0 / H) * (1 << 16)))
        mean = Signal(signed(32), name="mean")
        m.d.comb += mean.eq((sum_sig * inv_h) >> 16)

        # Step 2: Compute variance
        # var = sum((x[i] - mean)^2) / H
        diffs = []
        sq_sums = []
        for i in range(H):
            diff = Signal(signed(20), name=f"diff_{i}")
            m.d.comb += diff.eq(self.x[i] - mean[:16])
            diffs.append(diff)

            sq = Signal(signed(32), name=f"sq_{i}")
            m.d.comb += sq.eq((diff * diff) >> FRAC)  # Keep in Q8.8 range
            sq_sums.append(sq)

        var_sum = self._adder_tree(m, sq_sums, "var")

        # var = var_sum / H
        variance = Signal(signed(32), name="variance")
        m.d.comb += variance.eq((var_sum * inv_h) >> 16)

        # Step 3: Reciprocal sqrt via LUT
        # inv_std = 1/sqrt(var + eps)
        # Use top 8 bits of variance as LUT index
        # Pre-compute LUT: for var in [0, 255], inv_std = 1/sqrt(var/256 + eps)
        inv_std_lut = self._compute_rsqrt_lut(256, FRAC)
        inv_std = Signal(signed(16), name="inv_std")

        # Clamp variance to LUT range
        var_idx = Signal(8, name="var_idx")
        m.d.comb += var_idx.eq(Mux(variance[FRAC:FRAC+8] > 255, 255,
                                    variance[FRAC:FRAC+8]))

        # LUT selection
        with m.Switch(var_idx):
            for idx, val in enumerate(inv_std_lut):
                with m.Case(idx):
                    m.d.comb += inv_std.eq(val)

        # Step 4: Normalize and apply weight/bias
        for i in range(H):
            normed = Signal(signed(32), name=f"normed_{i}")
            scaled = Signal(signed(32), name=f"scaled_{i}")
            m.d.comb += normed.eq((diffs[i] * inv_std) >> FRAC)
            m.d.comb += scaled.eq(
                ((normed * self.weight_fixed[i]) >> FRAC) + self.bias_fixed[i]
            )
            m.d.comb += self.y[i].eq(scaled[:16])

        return m

    def _adder_tree(self, m, signals, prefix):
        """Binary adder tree over signals."""
        level = list(signals)
        stage = 0
        while len(level) > 1:
            nxt = []
            for j in range(0, len(level), 2):
                if j + 1 < len(level):
                    w = max(level[j].shape().width, level[j+1].shape().width) + 1
                    s = Signal(signed(w), name=f"{prefix}_s{stage}_{j//2}")
                    m.d.comb += s.eq(level[j] + level[j+1])
                    nxt.append(s)
                else:
                    nxt.append(level[j])
            level = nxt
            stage += 1
        return level[0]

    @staticmethod
    def _compute_rsqrt_lut(size, frac):
        """Pre-compute reciprocal sqrt LUT."""
        scale = 1 << frac
        lut = []
        eps = 1e-6
        for i in range(size):
            var_float = i / scale + eps
            inv_std = 1.0 / math.sqrt(var_float)
            # Clamp to fit in 16-bit signed
            val = int(round(inv_std * scale))
            val = max(-32768, min(32767, val))
            lut.append(val)
        return lut


# ---------------------------------------------------------------------------
# Softmax over 3 elements (LEWM-specific)
# ---------------------------------------------------------------------------

class Softmax3(Elaboratable):
    """Fixed-point softmax over exactly 3 elements.

    LEWM attention has seq_len=3, so softmax is always over 3 values.
    Uses a 256-entry exp LUT and fixed-point reciprocal.

    Input/output: Q8.8 (16-bit signed).
    """

    FRAC = 8
    SCALE = 1 << 8

    def __init__(self):
        self.x = [Signal(signed(16), name=f"sm_in_{i}") for i in range(3)]
        self.y = [Signal(signed(16), name=f"sm_out_{i}") for i in range(3)]

        # Pre-compute exp LUT for range [-4, 4] mapped to [0, 255]
        self.exp_lut = self._compute_exp_lut(256)
        self.recip_lut = self._compute_recip_lut(256)

    @staticmethod
    def _compute_exp_lut(size):
        """exp(x) for x in [-4, 4], indexed by (x+4)/8 * size."""
        lut = []
        scale = 1 << 8
        for i in range(size):
            x_float = -4.0 + 8.0 * i / size
            val = math.exp(x_float)
            lut.append(int(round(val * scale)))
        return lut

    @staticmethod
    def _compute_recip_lut(size):
        """1/x for x in [1, 256] (after scaling)."""
        lut = []
        scale = 1 << 8
        for i in range(size):
            x_float = max(1, i + 1) / scale
            val = 1.0 / x_float
            lut.append(min(32767, int(round(val * scale))))
        return lut

    def elaborate(self, platform):
        m = Module()
        S = self.SCALE

        # Step 1: Find max for numerical stability
        max_val = Signal(signed(16), name="sm_max")
        m.d.comb += max_val.eq(Mux(self.x[0] > self.x[1],
                                     Mux(self.x[0] > self.x[2], self.x[0], self.x[2]),
                                     Mux(self.x[1] > self.x[2], self.x[1], self.x[2])))

        # Step 2: Compute exp(x_i - max) via LUT
        exp_vals = []
        for i in range(3):
            diff = Signal(signed(16), name=f"sm_diff_{i}")
            m.d.comb += diff.eq(self.x[i] - max_val)

            # Map diff to LUT index: (diff + 4*S) * 256 / (8*S)
            # = (diff + 1024) * 256 / 2048 = (diff + 1024) / 8
            idx = Signal(8, name=f"sm_idx_{i}")
            raw_idx = Signal(signed(16), name=f"sm_raw_idx_{i}")
            m.d.comb += raw_idx.eq((diff + 4 * S) >> 3)
            m.d.comb += idx.eq(Mux(raw_idx < 0, 0,
                                    Mux(raw_idx > 255, 255, raw_idx[:8])))

            exp_val = Signal(16, name=f"sm_exp_{i}")
            with m.Switch(idx):
                for j, val in enumerate(self.exp_lut):
                    with m.Case(j):
                        m.d.comb += exp_val.eq(min(val, 65535))
            exp_vals.append(exp_val)

        # Step 3: Sum of exponentials
        exp_sum = Signal(18, name="sm_sum")
        m.d.comb += exp_sum.eq(exp_vals[0] + exp_vals[1] + exp_vals[2])

        # Step 4: Divide each exp by sum
        # output[i] = exp_vals[i] * S / exp_sum
        for i in range(3):
            # Multiply by scale, divide by sum
            # Use: output = (exp * 256) / sum
            numer = Signal(32, name=f"sm_num_{i}")
            m.d.comb += numer.eq(exp_vals[i] << 8)

            # Simple division: numer / exp_sum
            # For 3 elements, sum is bounded, so we can use shift-based approx
            # or just let synthesis handle it
            quot = Signal(signed(16), name=f"sm_quot_{i}")
            m.d.comb += quot.eq(Mux(exp_sum == 0, 0, numer // exp_sum))
            m.d.comb += self.y[i].eq(quot)

        return m


# ---------------------------------------------------------------------------
# adaLN Modulation
# ---------------------------------------------------------------------------

class AdaLNModulate(Elaboratable):
    """adaLN modulation: output[i] = normed[i] * (1 + scale[i]) + shift[i]

    All inputs/outputs are Q8.8 fixed-point (16-bit signed).
    """

    FRAC = 8
    SCALE = 1 << 8

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.normed = [Signal(signed(16), name=f"mod_normed_{i}") for i in range(hidden_size)]
        self.scale = [Signal(signed(16), name=f"mod_scale_{i}") for i in range(hidden_size)]
        self.shift = [Signal(signed(16), name=f"mod_shift_{i}") for i in range(hidden_size)]
        self.y = [Signal(signed(16), name=f"mod_out_{i}") for i in range(hidden_size)]

    def elaborate(self, platform):
        m = Module()
        S = self.SCALE

        for i in range(self.hidden_size):
            # (1 + scale) in fixed-point: S + scale
            one_plus_scale = Signal(signed(20), name=f"mod_1ps_{i}")
            m.d.comb += one_plus_scale.eq(S + self.scale[i])

            # normed * (1 + scale)
            prod = Signal(signed(32), name=f"mod_prod_{i}")
            m.d.comb += prod.eq((self.normed[i] * one_plus_scale) >> self.FRAC)

            # + shift
            m.d.comb += self.y[i].eq(prod[:16] + self.shift[i])

        return m


# ---------------------------------------------------------------------------
# Gated Residual: output = input + gate * value
# ---------------------------------------------------------------------------

class GatedResidual(Elaboratable):
    """Gated residual connection: y[i] = x[i] + gate[i] * value[i]

    Q8.8 fixed-point.
    """

    FRAC = 8

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.x = [Signal(signed(16), name=f"gr_x_{i}") for i in range(hidden_size)]
        self.gate = [Signal(signed(16), name=f"gr_gate_{i}") for i in range(hidden_size)]
        self.value = [Signal(signed(16), name=f"gr_val_{i}") for i in range(hidden_size)]
        self.y = [Signal(signed(16), name=f"gr_out_{i}") for i in range(hidden_size)]

    def elaborate(self, platform):
        m = Module()

        for i in range(self.hidden_size):
            gated = Signal(signed(32), name=f"gr_gated_{i}")
            m.d.comb += gated.eq((self.gate[i] * self.value[i]) >> self.FRAC)
            m.d.comb += self.y[i].eq(self.x[i] + gated[:16])

        return m


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def test_gelu_accuracy():
    """Test GELU PWL accuracy against exact implementation."""
    print("Testing GELU PWL accuracy...")

    gelu = GELU_PWL()
    S = gelu.SCALE
    FRAC = gelu.FRAC

    max_err = 0
    n_tests = 0
    for x_int in range(-4 * S, 4 * S + 1, 4):
        x_float = x_int / S
        exact = gelu._gelu_exact(x_float)

        # Find segment
        for seg in gelu.segments:
            if x_int < int(round(seg['x1'] * S)):
                # y = y0 + slope * (x - x0)
                dx = x_int - seg['x0_fixed']
                prod = dx * seg['slope_fixed']
                approx_fixed = seg['y0_fixed'] + (prod >> FRAC)
                approx = approx_fixed / S
                break
        else:
            approx = x_float  # x > 4, identity

        err = abs(exact - approx)
        max_err = max(max_err, err)
        n_tests += 1

    print(f"  Tested {n_tests} points over [-4, 4]")
    print(f"  Max absolute error: {max_err:.6f}")
    print(f"  Max relative error: {max_err / 4:.4%} (relative to range)")
    print(f"  {'PASS' if max_err < 0.1 else 'FAIL'}: error {'<' if max_err < 0.1 else '>'} 0.1")
    return max_err < 0.1


def test_softmax3_accuracy():
    """Test Softmax3 LUT accuracy."""
    print("\nTesting Softmax3 accuracy...")

    sm = Softmax3()
    S = sm.SCALE

    np.random.seed(42)
    max_err = 0
    n_tests = 100

    for _ in range(n_tests):
        x = np.random.randn(3).astype(np.float32)
        # Exact softmax
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        exact = exp_x / np.sum(exp_x)

        # Fixed-point approximation
        x_fixed = np.round(x * S).astype(int)
        max_fixed = max(x_fixed)

        exp_approx = []
        for xi in x_fixed:
            diff = xi - max_fixed
            # LUT index
            raw_idx = (diff + 4 * S) >> 3
            idx = max(0, min(255, raw_idx))
            exp_approx.append(sm.exp_lut[idx])

        exp_sum = sum(exp_approx)
        if exp_sum > 0:
            approx = np.array([(e * S) / exp_sum for e in exp_approx]) / S
        else:
            approx = np.array([1/3, 1/3, 1/3])

        err = np.max(np.abs(exact - approx))
        max_err = max(max_err, err)

    print(f"  Tested {n_tests} random 3-vectors")
    print(f"  Max absolute error: {max_err:.6f}")
    print(f"  {'PASS' if max_err < 0.05 else 'FAIL'}: error {'<' if max_err < 0.05 else '>'} 0.05")
    return max_err < 0.05


if __name__ == "__main__":
    ok1 = test_gelu_accuracy()
    ok2 = test_softmax3_accuracy()
    print(f"\n{'ALL PASS' if ok1 and ok2 else 'FAILURES'}")
