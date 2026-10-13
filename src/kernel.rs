//! The kernels. No dependencies, no BLAS — these are the loops themselves.
//!
//! Everything here works on flat `&[f64]` buffers with a separate shape, the
//! same layout `numsol`'s `kernel.sau` uses, so a result can be compared
//! element for element against the pure-Saule reference. That comparison is
//! the whole correctness story: the Saule implementation is the oracle, and a
//! kernel here is wrong unless it agrees with it.
//!
//! `f64` throughout, deliberately. Saule's `float` is 64-bit, so an `f64`
//! kernel can be checked for *exact* agreement with the reference on the
//! operations that are exactly representable, and to a tight tolerance on the
//! transcendental ones. `f32` doubles arithmetic throughput and halves memory
//! traffic, and it is what a GPU will want — but introducing it at the same
//! time as the kernels would mean never knowing whether a disagreement was a
//! bug or the precision. It goes in behind an explicit switch, after this
//! layer is known good.

/// Row-major strides for `dims`, innermost last.
pub fn strides(dims: &[usize]) -> Vec<usize> {
    let mut out = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        out[i] = out[i + 1] * dims[i + 1];
    }
    out
}

/// Element count for `dims`. An empty shape is one scalar, as in numpy.
pub fn size_of(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// numpy's broadcasting rule: axes line up from the right, and each pair must
/// be equal or one of them must be 1.
pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        // Walk from the right: index `i` from the end of each shape.
        let av = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let bv = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        let dim = if av == bv {
            av
        } else if av == 1 {
            bv
        } else if bv == 1 {
            av
        } else {
            return Err(format!(
                "shapes {a:?} and {b:?} do not broadcast: axis {i} from the right is {av} vs {bv}"
            ));
        };
        out[rank - 1 - i] = dim;
    }
    Ok(out)
}

/// The flat offset into a buffer of shape `dims` for the point `idx` in a
/// broadcast output — an axis of extent 1 is read repeatedly.
#[inline]
fn broadcast_offset(idx: &[usize], dims: &[usize], strides: &[usize]) -> usize {
    let lead = idx.len() - dims.len();
    let mut off = 0;
    for ax in 0..dims.len() {
        // An axis of extent 1 contributes nothing: that is the stretch.
        if dims[ax] != 1 {
            off += idx[lead + ax] * strides[ax];
        }
    }
    off
}

/// Elementwise `f(a, b)` with broadcasting, into a fresh buffer.
///
/// The fast path — equal shapes — is a flat zip with no index arithmetic at
/// all, which is the case that actually runs in a training loop. The general
/// path walks an odometer over the output shape.
pub fn zip_with(
    a: &[f64],
    a_dims: &[usize],
    b: &[f64],
    b_dims: &[usize],
    out_dims: &[usize],
    f: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    if a_dims == b_dims && a_dims == out_dims {
        return a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
    }

    // A lone scalar on either side is the next most common shape by far
    // (`a * 2.0`), and it needs no odometer either.
    if b.len() == 1 {
        let y = b[0];
        return a.iter().map(|&x| f(x, y)).collect();
    }
    if a.len() == 1 {
        let x = a[0];
        return b.iter().map(|&y| f(x, y)).collect();
    }

    let total = size_of(out_dims);
    let a_str = strides(a_dims);
    let b_str = strides(b_dims);
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_dims.len()];

    for _ in 0..total {
        let ao = broadcast_offset(&idx, a_dims, &a_str);
        let bo = broadcast_offset(&idx, b_dims, &b_str);
        out.push(f(a[ao], b[bo]));

        // Odometer increment, innermost axis first.
        for ax in (0..out_dims.len()).rev() {
            idx[ax] += 1;
            if idx[ax] < out_dims[ax] {
                break;
            }
            idx[ax] = 0;
        }
    }
    out
}

/// Stretch `data` of shape `dims` up to `out_dims`.
pub fn stretch(data: &[f64], dims: &[usize], out_dims: &[usize]) -> Vec<f64> {
    if dims == out_dims {
        return data.to_vec();
    }
    let total = size_of(out_dims);
    let str_in = strides(dims);
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_dims.len()];
    for _ in 0..total {
        out.push(data[broadcast_offset(&idx, dims, &str_in)]);
        for ax in (0..out_dims.len()).rev() {
            idx[ax] += 1;
            if idx[ax] < out_dims[ax] {
                break;
            }
            idx[ax] = 0;
        }
    }
    out
}

/// Block size for the `matmul` tiling, in elements per side.
///
/// 64×64 `f64` tiles are 32 KiB each, so the three live tiles sit inside a
/// 128 KiB L1 on the machines this targets. It is a starting point chosen by
/// arithmetic rather than measurement; `benches` is where it earns a change.
const TILE: usize = 64;

/// Rank-2 matrix product: `(height × shared) @ (shared × width)`.
///
/// Three things make this quicker than the obvious triple loop, and all three
/// are about memory rather than arithmetic:
///
/// 1. **Tiling.** Blocking all three axes keeps the working set in L1, so the
///    `shared` dimension is walked once per tile rather than once per output
///    element.
/// 2. **`i-k-j` order inside a tile.** With row-major storage, the innermost
///    loop then strides contiguously through both `b` and `out`, which is what
///    lets the compiler keep them in registers and vectorise.
/// 3. **Hoisting `a[i,k]`.** It is invariant in the inner loop, so it is read
///    once per `(i, k)` instead of once per `(i, k, j)`.
///
/// No `unsafe`. The slice indexing inside the inner loop is bounds-checked,
/// and removing those checks is a measurable-win question, not an assumption.
pub fn matmul(a: &[f64], height: usize, shared: usize, b: &[f64], width: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; height * width];

    for i0 in (0..height).step_by(TILE) {
        let i_end = (i0 + TILE).min(height);
        for k0 in (0..shared).step_by(TILE) {
            let k_end = (k0 + TILE).min(shared);
            for j0 in (0..width).step_by(TILE) {
                let j_end = (j0 + TILE).min(width);

                for i in i0..i_end {
                    let a_row = i * shared;
                    let out_row = i * width;
                    for k in k0..k_end {
                        let aik = a[a_row + k];
                        // Skipping a zero is not a fast path worth taking in
                        // general, but it costs one compare and the weight
                        // matrices this will see are not sparse, so it is left
                        // out rather than guessed at.
                        let b_row = k * width;
                        let b_slice = &b[b_row + j0..b_row + j_end];
                        let out_slice = &mut out[out_row + j0..out_row + j_end];
                        for (o, &bv) in out_slice.iter_mut().zip(b_slice.iter()) {
                            *o += aik * bv;
                        }
                    }
                }
            }
        }
    }
    out
}

/// The matrix product over a leading batch axis.
///
/// Three shape combinations, which are the ones a transformer needs:
///
/// | left | right | result | used for |
/// |---|---|---|---|
/// | `(M, K)` | `(K, N)` | `(M, N)` | an ordinary projection |
/// | `(B, M, K)` | `(B, K, N)` | `(B, M, N)` | per-head attention scores |
/// | `(B, M, K)` | `(K, N)` | `(B, M, N)` | one weight shared across a batch |
///
/// The last is worth having on its own: a transformer's projections apply the
/// same matrix to every sequence in the batch, and without this the caller has
/// to broadcast the weight into `B` copies first — allocating `B` times the
/// weight and reading it `B` times from memory instead of once.
///
/// Each batch delegates to [`matmul`], so the tiling and the ordering are the
/// kernel that is already tested rather than a second implementation of it.
pub fn matmul_batched(
    a: &[f64],
    a_dims: &[usize],
    b: &[f64],
    b_dims: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), String> {
    // (batch, rows, shared) on the left; (batch_or_1, shared, cols) on the
    // right. Normalising to this shape first is what keeps the loop below
    // free of rank checks.
    let (batch, m, k) = match a_dims {
        [m, k] => (1usize, *m, *k),
        [b0, m, k] => (*b0, *m, *k),
        other => {
            return Err(format!(
                "matmul: the left operand must be rank 2 or 3, got rank {}",
                other.len()
            ));
        }
    };
    let (b_batch, k2, n) = match b_dims {
        [k2, n] => (1usize, *k2, *n),
        [b1, k2, n] => (*b1, *k2, *n),
        other => {
            return Err(format!(
                "matmul: the right operand must be rank 2 or 3, got rank {}",
                other.len()
            ));
        }
    };

    if k != k2 {
        return Err(format!(
            "matmul: inner dimensions do not line up — {k} vs {k2}"
        ));
    }
    // Checked before the batch comparison below, not after: a rank-2 left
    // operand reads as batch 1, so the generic message would say "batch sizes
    // do not line up — 1 vs B" and send the caller looking for a batch
    // mismatch instead of telling them the shape they actually want.
    if a_dims.len() == 2 && b_dims.len() == 3 {
        return Err(
            "matmul: a rank-2 left operand with a rank-3 right one is not supported — \
             reshape the left to (1, M, K) to say which axis is the batch"
                .to_string(),
        );
    }
    // A batched right-hand side must either match the left's batch or be the
    // single shared matrix. Anything else would need real broadcasting over
    // the batch axis, which nothing has asked for yet and which would be
    // silently wrong if guessed at.
    if b_batch != 1 && b_batch != batch {
        return Err(format!(
            "matmul: batch sizes do not line up — {batch} vs {b_batch}"
        ));
    }

    let mut out = Vec::with_capacity(batch * m * n);
    for bi in 0..batch {
        let a_slice = &a[bi * m * k..(bi + 1) * m * k];
        // The shared matrix is read again for each batch rather than copied.
        let b_slice = if b_batch == 1 {
            &b[..k * n]
        } else {
            &b[bi * k * n..(bi + 1) * k * n]
        };
        out.extend_from_slice(&matmul(a_slice, m, k, b_slice, n));
    }

    let dims = if a_dims.len() == 2 {
        vec![m, n]
    } else {
        vec![batch, m, n]
    };
    Ok((out, dims))
}

/// Swap the last two axes, leaving any leading batch axes alone.
///
/// Distinct from [`transpose`], which reverses *every* axis — that is what
/// numsol does and what rank-2 needs, but on a batch it would turn
/// `(B, L, d)` into `(d, L, B)` and move the batch into the middle. Attention
/// needs `(B, L, d) -> (B, d, L)`, which is this. numpy spells it `.mT`.
///
/// Each batch is transposed with the same blocked path rank 2 uses.
pub fn transpose_last2(data: &[f64], dims: &[usize]) -> Result<(Vec<f64>, Vec<usize>), String> {
    if dims.len() < 2 {
        return Err(format!(
            "transposeLast: needs at least rank 2, got rank {}",
            dims.len()
        ));
    }
    let rank = dims.len();
    let (h, w) = (dims[rank - 2], dims[rank - 1]);
    let batch: usize = dims[..rank - 2].iter().product();

    let mut out = Vec::with_capacity(batch * h * w);
    for bi in 0..batch {
        let slice = &data[bi * h * w..(bi + 1) * h * w];
        out.extend_from_slice(&transpose(slice, &[h, w]));
    }

    let mut out_dims = dims.to_vec();
    out_dims[rank - 2] = w;
    out_dims[rank - 1] = h;
    Ok((out, out_dims))
}

/// Transpose, for any rank. Rank 2 gets a blocked path; higher ranks reverse
/// every axis, which is what numsol's reference does.
pub fn transpose(data: &[f64], dims: &[usize]) -> Vec<f64> {
    match dims.len() {
        0 | 1 => data.to_vec(),
        2 => {
            let (h, w) = (dims[0], dims[1]);
            let mut out = vec![0.0f64; h * w];
            // Blocked, because the naive version strides the full row length
            // on one side of the assignment and thrashes a cache line per
            // element on anything large.
            for i0 in (0..h).step_by(TILE) {
                let i_end = (i0 + TILE).min(h);
                for j0 in (0..w).step_by(TILE) {
                    let j_end = (j0 + TILE).min(w);
                    for i in i0..i_end {
                        for j in j0..j_end {
                            out[j * h + i] = data[i * w + j];
                        }
                    }
                }
            }
            out
        }
        _ => {
            let rev: Vec<usize> = dims.iter().rev().copied().collect();
            let total = size_of(dims);
            let in_str = strides(dims);
            let mut out = vec![0.0f64; total];
            let mut idx = vec![0usize; dims.len()];
            for _ in 0..total {
                let src: usize = idx.iter().zip(in_str.iter()).map(|(i, s)| i * s).sum();
                let mut dst = 0usize;
                let out_str = strides(&rev);
                for ax in 0..dims.len() {
                    dst += idx[dims.len() - 1 - ax] * out_str[ax];
                }
                out[dst] = data[src];
                for ax in (0..dims.len()).rev() {
                    idx[ax] += 1;
                    if idx[ax] < dims[ax] {
                        break;
                    }
                    idx[ax] = 0;
                }
            }
            out
        }
    }
}

/// How an axis is collapsed by [`reduce_axis`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reduce {
    Sum,
    Max,
    ArgMax,
}

/// Collapse `axis` (0-based here; the Saule face is 1-based) of `dims`.
///
/// The buffer is viewed as `outer × extent × inner`, which turns every
/// reduction into the same two loops regardless of rank and keeps the inner
/// one contiguous.
pub fn reduce_axis(data: &[f64], dims: &[usize], axis: usize, how: Reduce) -> Vec<f64> {
    let extent = dims[axis];
    let inner: usize = dims[axis + 1..].iter().product();
    let outer: usize = dims[..axis].iter().product();
    let mut out = vec![0.0f64; outer * inner];

    for o in 0..outer {
        let base = o * extent * inner;
        for i in 0..inner {
            match how {
                Reduce::Sum => {
                    let mut total = 0.0f64;
                    for e in 0..extent {
                        total += data[base + e * inner + i];
                    }
                    out[o * inner + i] = total;
                }
                Reduce::Max => {
                    let mut best = data[base + i];
                    for e in 1..extent {
                        let v = data[base + e * inner + i];
                        if v > best {
                            best = v;
                        }
                    }
                    out[o * inner + i] = best;
                }
                Reduce::ArgMax => {
                    let mut best = data[base + i];
                    let mut at = 0usize;
                    for e in 1..extent {
                        let v = data[base + e * inner + i];
                        if v > best {
                            best = v;
                            at = e;
                        }
                    }
                    // 1-based, to match numsol's indexing convention.
                    out[o * inner + i] = (at + 1) as f64;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides(&[5]), vec![1]);
    }

    #[test]
    fn broadcasting_follows_numpy() {
        assert_eq!(broadcast(&[2, 3], &[1, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast(&[2, 3], &[3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast(&[2, 3], &[2, 1]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast(&[2, 1, 4], &[3, 1]).unwrap(), vec![2, 3, 4]);
        assert!(broadcast(&[2, 3], &[4, 3]).is_err());
    }

    /// The tiled product must equal the textbook triple loop exactly — same
    /// operations in the same order per output element, so `f64` addition
    /// associativity is not in question.
    #[test]
    fn matmul_matches_the_naive_product() {
        for (h, s, w) in [(1, 1, 1), (3, 4, 5), (64, 64, 64), (70, 33, 129)] {
            let a: Vec<f64> = (0..h * s).map(|i| (i % 17) as f64 - 8.0).collect();
            let b: Vec<f64> = (0..s * w).map(|i| (i % 13) as f64 - 6.0).collect();

            let mut want = vec![0.0f64; h * w];
            for i in 0..h {
                for j in 0..w {
                    let mut acc = 0.0;
                    for k in 0..s {
                        acc += a[i * s + k] * b[k * w + j];
                    }
                    want[i * w + j] = acc;
                }
            }

            assert_eq!(matmul(&a, h, s, &b, w), want, "shape {h}x{s}x{w}");
        }
    }

    #[test]
    fn transpose_is_its_own_inverse() {
        let dims = vec![3usize, 5];
        let data: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let once = transpose(&data, &dims);
        let twice = transpose(&once, &[5, 3]);
        assert_eq!(twice, data);
    }

    #[test]
    fn zip_broadcasts_a_row_across_a_matrix() {
        // (2,3) + (1,3): the row is reused.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0];
        let out = zip_with(&a, &[2, 3], &b, &[1, 3], &[2, 3], |x, y| x + y);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn zip_broadcasts_a_column_across_a_matrix() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0];
        let out = zip_with(&a, &[2, 3], &b, &[2, 1], &[2, 3], |x, y| x + y);
        assert_eq!(out, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
    }

    #[test]
    fn reductions_collapse_the_named_axis() {
        // (2,3) row-major: [[1,2,3],[4,5,6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            reduce_axis(&data, &[2, 3], 0, Reduce::Sum),
            vec![5.0, 7.0, 9.0]
        );
        assert_eq!(reduce_axis(&data, &[2, 3], 1, Reduce::Sum), vec![6.0, 15.0]);
        assert_eq!(reduce_axis(&data, &[2, 3], 1, Reduce::Max), vec![3.0, 6.0]);
        // 1-based: the largest in each row is at index 3.
        assert_eq!(
            reduce_axis(&data, &[2, 3], 1, Reduce::ArgMax),
            vec![3.0, 3.0]
        );
    }

    #[test]
    fn stretch_repeats_a_degenerate_axis() {
        let data = vec![7.0, 8.0];
        assert_eq!(
            stretch(&data, &[2, 1], &[2, 3]),
            vec![7.0, 7.0, 7.0, 8.0, 8.0, 8.0]
        );
    }
}

#[cfg(test)]
mod batched_tests {
    use super::*;

    /// A batched product must equal the rank-2 product of each slice, which
    /// is the definition and is also how a caller would check it by hand.
    #[test]
    fn batched_matmul_equals_per_slice_products() {
        let (b, m, k, n) = (3usize, 4usize, 5usize, 2usize);
        let a: Vec<f64> = (0..b * m * k).map(|i| (i % 11) as f64 - 5.0).collect();
        let rhs: Vec<f64> = (0..b * k * n).map(|i| (i % 7) as f64 - 3.0).collect();

        let (got, dims) = matmul_batched(&a, &[b, m, k], &rhs, &[b, k, n]).expect("shapes line up");
        assert_eq!(dims, vec![b, m, n]);

        for bi in 0..b {
            let want = matmul(
                &a[bi * m * k..(bi + 1) * m * k],
                m,
                k,
                &rhs[bi * k * n..(bi + 1) * k * n],
                n,
            );
            assert_eq!(&got[bi * m * n..(bi + 1) * m * n], &want[..], "batch {bi}");
        }
    }

    /// A shared rank-2 right operand applies the same matrix to every batch.
    #[test]
    fn a_shared_weight_applies_to_every_batch() {
        let (b, m, k, n) = (3usize, 2usize, 3usize, 4usize);
        let a: Vec<f64> = (0..b * m * k).map(|i| i as f64).collect();
        let w: Vec<f64> = (0..k * n).map(|i| (i % 5) as f64 - 2.0).collect();

        let (got, dims) = matmul_batched(&a, &[b, m, k], &w, &[k, n]).expect("shapes line up");
        assert_eq!(dims, vec![b, m, n]);

        for bi in 0..b {
            let want = matmul(&a[bi * m * k..(bi + 1) * m * k], m, k, &w, n);
            assert_eq!(&got[bi * m * n..(bi + 1) * m * n], &want[..], "batch {bi}");
        }
    }

    /// Rank 2 against rank 2 still goes through and stays rank 2, so the
    /// batched entry point is a superset of the old one.
    #[test]
    fn rank_two_is_unchanged_and_stays_rank_two() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (got, dims) = matmul_batched(&a, &[2, 3], &b, &[3, 2]).expect("lines up");
        assert_eq!(dims, vec![2, 2]);
        assert_eq!(got, matmul(&a, 2, 3, &b, 2));
    }

    #[test]
    fn mismatched_shapes_are_named_not_guessed() {
        let a = vec![0.0; 2 * 3 * 4];
        // inner dimension disagrees
        assert!(matmul_batched(&a, &[2, 3, 4], &[0.0; 2 * 5 * 6], &[2, 5, 6]).is_err());
        // batch disagrees
        assert!(matmul_batched(&a, &[2, 3, 4], &[0.0; 3 * 4 * 6], &[3, 4, 6]).is_err());
        // rank 4 is not supported
        assert!(matmul_batched(&a, &[1, 2, 3, 4], &[0.0; 4], &[4, 1]).is_err());
        // rank-2 left with rank-3 right would need the left broadcast
        let err = matmul_batched(&[0.0; 6], &[2, 3], &[0.0; 2 * 3 * 2], &[2, 3, 2])
            .expect_err("not supported");
        assert!(err.contains("reshape the left"), "{err}");
    }

    /// `(B, L, d) -> (B, d, L)`: the batch must stay in front, which is the
    /// whole reason this is not `transpose`.
    #[test]
    fn transpose_last2_keeps_the_batch_in_front() {
        let dims = vec![2usize, 2, 3];
        // batch 0: [[1,2,3],[4,5,6]]   batch 1: [[7,8,9],[10,11,12]]
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();

        let (got, out_dims) = transpose_last2(&data, &dims).expect("rank 3");
        assert_eq!(out_dims, vec![2, 3, 2]);
        // batch 0 transposed: [[1,4],[2,5],[3,6]]
        assert_eq!(&got[0..6], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // batch 1 transposed: [[7,10],[8,11],[9,12]]
        assert_eq!(&got[6..12], &[7.0, 10.0, 8.0, 11.0, 9.0, 12.0]);
    }

    #[test]
    fn transpose_last2_on_rank_two_is_the_ordinary_transpose() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (got, dims) = transpose_last2(&data, &[2, 3]).expect("rank 2");
        assert_eq!(dims, vec![3, 2]);
        assert_eq!(got, transpose(&data, &[2, 3]));
    }

    /// Doing it twice is the identity, batch axis included.
    #[test]
    fn transpose_last2_round_trips() {
        let dims = vec![3usize, 4, 5];
        let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let (once, d1) = transpose_last2(&data, &dims).expect("rank 3");
        let (twice, d2) = transpose_last2(&once, &d1).expect("rank 3");
        assert_eq!(d2, dims);
        assert_eq!(twice, data);
    }

    #[test]
    fn transpose_last2_needs_rank_two() {
        assert!(transpose_last2(&[1.0], &[1]).is_err());
    }
}
