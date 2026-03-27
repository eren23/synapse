pub(crate) fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub(crate) fn add_vecs_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += *y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_vecs_correctness() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let out = add_vecs(&a, &b);
        let expected = [5.0f32, 7.0, 9.0];
        assert_eq!(out.len(), expected.len());
        for (i, (&e, &g)) in expected.iter().zip(out.iter()).enumerate() {
            assert!((g - e).abs() < 1e-7, "add_vecs mismatch at {i}: {g} != {e}");
        }
    }

    #[test]
    fn add_vecs_inplace_modifies_first() {
        let mut a = vec![10.0f32, 20.0, 30.0];
        let b = vec![1.0f32, 2.0, 3.0];
        add_vecs_inplace(&mut a, &b);
        let expected = [11.0f32, 22.0, 33.0];
        for (i, (&e, &g)) in expected.iter().zip(a.iter()).enumerate() {
            assert!((g - e).abs() < 1e-7, "add_vecs_inplace mismatch at {i}: {g} != {e}");
        }
    }

    #[test]
    fn add_vecs_zero_vector() {
        let a = vec![1.0f32, -2.0, 3.14];
        let zeros = vec![0.0f32; 3];
        let out = add_vecs(&a, &zeros);
        for (i, (&orig, &got)) in a.iter().zip(out.iter()).enumerate() {
            assert!((got - orig).abs() < 1e-7, "add with zeros should be identity at {i}");
        }
    }
}
