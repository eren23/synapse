/// Lightweight f32 tensor for autograd operations.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "data length {} != shape product {}",
            data.len(),
            expected
        );
        Tensor { data, shape }
    }

    pub fn scalar(val: f32) -> Self {
        Tensor {
            data: vec![val],
            shape: vec![1],
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![1.0; n],
            shape: shape.to_vec(),
        }
    }

    pub fn zeros_like(&self) -> Self {
        Self::zeros(&self.shape)
    }

    pub fn ones_like(&self) -> Self {
        Self::ones(&self.shape)
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch in add");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch in sub");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch in mul");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn neg(&self) -> Tensor {
        let data = self.data.iter().map(|a| -a).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    // ── Broadcasting ────────────────────────────────────────────────

    pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
        let ndim = a.len().max(b.len());
        let mut result = vec![0usize; ndim];
        for i in 0..ndim {
            let da = if i < ndim - a.len() {
                1
            } else {
                a[i - (ndim - a.len())]
            };
            let db = if i < ndim - b.len() {
                1
            } else {
                b[i - (ndim - b.len())]
            };
            assert!(
                da == db || da == 1 || db == 1,
                "cannot broadcast dims {} and {}",
                da,
                db
            );
            result[i] = da.max(db);
        }
        result
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Tensor {
        if self.shape == shape {
            return self.clone();
        }
        let ndim = shape.len();
        let self_ndim = self.shape.len();
        let mut padded = vec![1usize; ndim];
        for i in 0..self_ndim {
            padded[ndim - self_ndim + i] = self.shape[i];
        }
        for i in 0..ndim {
            assert!(padded[i] == shape[i] || padded[i] == 1);
        }
        let mut src_strides = vec![0usize; ndim];
        {
            let mut stride = 1usize;
            for i in (0..ndim).rev() {
                src_strides[i] = if padded[i] == 1 { 0 } else { stride };
                stride *= padded[i];
            }
        }
        let mut out_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * shape[i + 1];
        }
        let n: usize = shape.iter().product();
        let mut data = vec![0.0f32; n];
        for out_flat in 0..n {
            let mut src_flat = 0usize;
            let mut remaining = out_flat;
            for dim in 0..ndim {
                let coord = remaining / out_strides[dim];
                remaining %= out_strides[dim];
                src_flat += coord * src_strides[dim];
            }
            data[out_flat] = self.data[src_flat];
        }
        Tensor {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn reduce_sum_to(&self, target_shape: &[usize]) -> Tensor {
        if self.shape == target_shape {
            return self.clone();
        }
        let ndim = self.shape.len();
        let target_ndim = target_shape.len();
        let mut padded_target = vec![1usize; ndim];
        for i in 0..target_ndim {
            padded_target[ndim - target_ndim + i] = target_shape[i];
        }
        let target_n: usize = target_shape.iter().product();
        let mut result = vec![0.0f32; target_n];
        let mut self_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            self_strides[i] = self_strides[i + 1] * self.shape[i + 1];
        }
        let mut target_strides_padded = vec![0usize; ndim];
        {
            let mut stride = 1usize;
            for i in (0..ndim).rev() {
                target_strides_padded[i] = if padded_target[i] == 1 { 0 } else { stride };
                stride *= padded_target[i];
            }
        }
        for flat in 0..self.numel() {
            let mut target_flat = 0usize;
            let mut remaining = flat;
            for dim in 0..ndim {
                let coord = remaining / self_strides[dim];
                remaining %= self_strides[dim];
                target_flat += coord * target_strides_padded[dim];
            }
            result[target_flat] += self.data[flat];
        }
        Tensor {
            data: result,
            shape: target_shape.to_vec(),
        }
    }

    // ── Element-wise with broadcasting ──────────────────────────────

    pub fn add_broadcast(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            return self.add(other);
        }
        let out_shape = Self::broadcast_shapes(&self.shape, &other.shape);
        self.broadcast_to(&out_shape)
            .add(&other.broadcast_to(&out_shape))
    }

    pub fn sub_broadcast(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            return self.sub(other);
        }
        let out_shape = Self::broadcast_shapes(&self.shape, &other.shape);
        self.broadcast_to(&out_shape)
            .sub(&other.broadcast_to(&out_shape))
    }

    pub fn mul_broadcast(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            return self.mul(other);
        }
        let out_shape = Self::broadcast_shapes(&self.shape, &other.shape);
        self.broadcast_to(&out_shape)
            .mul(&other.broadcast_to(&out_shape))
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch in div");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a / b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn div_broadcast(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            return self.div(other);
        }
        let out_shape = Self::broadcast_shapes(&self.shape, &other.shape);
        self.broadcast_to(&out_shape)
            .div(&other.broadcast_to(&out_shape))
    }

    // ── Scalar ops ──────────────────────────────────────────────────

    pub fn scale(&self, s: f32) -> Tensor {
        let data = self.data.iter().map(|&x| x * s).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn add_scalar(&self, s: f32) -> Tensor {
        let data = self.data.iter().map(|&x| x + s).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    // ── Math ────────────────────────────────────────────────────────

    pub fn exp(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.exp()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn log(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.ln()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn sqrt(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.sqrt()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn pow_scalar(&self, p: f32) -> Tensor {
        let data = self.data.iter().map(|&x| x.powf(p)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    // ── Activations ─────────────────────────────────────────────────

    pub fn relu(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        let data = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn tanh_act(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.tanh()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn gelu(&self) -> Tensor {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        let data = self
            .data
            .iter()
            .map(|&x| {
                let inner = c * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    // ── Matrix operations ───────────────────────────────────────────

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensors");
        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(other.shape[0], k, "matmul inner dim mismatch");
        let n = other.shape[1];
        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                data[i * n + j] = sum;
            }
        }
        Tensor {
            data,
            shape: vec![m, n],
        }
    }

    pub fn transpose_2d(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                data[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor {
            data,
            shape: vec![n, m],
        }
    }

    pub fn transpose_dims(&self, dim0: usize, dim1: usize) -> Tensor {
        let ndim = self.shape.len();
        assert!(dim0 < ndim && dim1 < ndim);
        if dim0 == dim1 {
            return self.clone();
        }
        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);
        let n = self.numel();
        let mut data = vec![0.0f32; n];
        let mut src_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            src_strides[i] = src_strides[i + 1] * self.shape[i + 1];
        }
        let mut dst_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
        }
        for src_flat in 0..n {
            let mut coords = vec![0usize; ndim];
            let mut remaining = src_flat;
            for dim in 0..ndim {
                coords[dim] = remaining / src_strides[dim];
                remaining %= src_strides[dim];
            }
            coords.swap(dim0, dim1);
            let mut dst_flat = 0usize;
            for dim in 0..ndim {
                dst_flat += coords[dim] * dst_strides[dim];
            }
            data[dst_flat] = self.data[src_flat];
        }
        Tensor {
            data,
            shape: new_shape,
        }
    }

    // ── Reductions ──────────────────────────────────────────────────

    pub fn sum_all(&self) -> Tensor {
        Tensor::scalar(self.data.iter().sum())
    }

    pub fn sum_axis(&self, axis: usize, keepdim: bool) -> Tensor {
        let ndim = self.shape.len();
        assert!(axis < ndim);
        let mut out_shape_kd: Vec<usize> = self.shape.clone();
        out_shape_kd[axis] = 1;
        let out_n: usize = out_shape_kd.iter().product();
        let mut data = vec![0.0f32; out_n];
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        let mut out_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape_kd[i + 1];
        }
        for flat in 0..self.numel() {
            let mut out_flat = 0;
            let mut remaining = flat;
            for dim in 0..ndim {
                let coord = remaining / strides[dim];
                remaining %= strides[dim];
                if dim != axis {
                    out_flat += coord * out_strides[dim];
                }
            }
            data[out_flat] += self.data[flat];
        }
        if keepdim {
            Tensor {
                data,
                shape: out_shape_kd,
            }
        } else {
            let mut final_shape: Vec<usize> = self.shape.clone();
            final_shape.remove(axis);
            if final_shape.is_empty() {
                final_shape.push(1);
            }
            Tensor {
                data,
                shape: final_shape,
            }
        }
    }

    pub fn mean_all(&self) -> Tensor {
        Tensor::scalar(self.data.iter().sum::<f32>() / self.numel() as f32)
    }

    pub fn mean_axis(&self, axis: usize, keepdim: bool) -> Tensor {
        self.sum_axis(axis, keepdim)
            .scale(1.0 / self.shape[axis] as f32)
    }

    pub fn variance_axis(&self, axis: usize, keepdim: bool) -> Tensor {
        let mean = self.mean_axis(axis, true);
        let diff = self.sub_broadcast(&mean);
        diff.mul(&diff).mean_axis(axis, keepdim)
    }

    // ── Shape ops ───────────────────────────────────────────────────

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.numel(), "reshape: incompatible shapes");
        Tensor {
            data: self.data.clone(),
            shape: shape.to_vec(),
        }
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let mut shape = self.shape.clone();
        shape.insert(dim, 1);
        Tensor {
            data: self.data.clone(),
            shape,
        }
    }

    // ── Softmax ─────────────────────────────────────────────────────

    pub fn softmax_axis(&self, axis: usize) -> Tensor {
        let ndim = self.shape.len();
        assert!(axis < ndim);
        let axis_size = self.shape[axis];
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut data = vec![0.0f32; self.numel()];
        for o in 0..outer {
            for i in 0..inner {
                let mut max_val = f32::NEG_INFINITY;
                for a in 0..axis_size {
                    let idx = o * axis_size * inner + a * inner + i;
                    max_val = max_val.max(self.data[idx]);
                }
                let mut sum = 0.0f32;
                for a in 0..axis_size {
                    let idx = o * axis_size * inner + a * inner + i;
                    let v = (self.data[idx] - max_val).exp();
                    data[idx] = v;
                    sum += v;
                }
                for a in 0..axis_size {
                    let idx = o * axis_size * inner + a * inner + i;
                    data[idx] /= sum;
                }
            }
        }
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn log_softmax_axis(&self, axis: usize) -> Tensor {
        let ndim = self.shape.len();
        assert!(axis < ndim);
        let axis_size = self.shape[axis];
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut data = vec![0.0f32; self.numel()];
        for o in 0..outer {
            for i in 0..inner {
                let mut max_val = f32::NEG_INFINITY;
                for a in 0..axis_size {
                    let idx = o * axis_size * inner + a * inner + i;
                    max_val = max_val.max(self.data[idx]);
                }
                let mut lse = 0.0f32;
                for a in 0..axis_size {
                    let idx = o * axis_size * inner + a * inner + i;
                    lse += (self.data[idx] - max_val).exp();
                }
                lse = lse.ln();
                for a in 0..axis_size {
                    let idx = o * axis_size * inner + a * inner + i;
                    data[idx] = self.data[idx] - max_val - lse;
                }
            }
        }
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?}, shape={:?})", self.data, self.shape)
    }
}
