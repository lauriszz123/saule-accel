//! numpy's shape, in Saule's types — accelerated.
//!
//! This is the native kernel layer `numsol`'s README always anticipated:
//! *"`kernel.sau` is the single file a native backend would replace."* The
//! layering there is one-way — `Kernel` and `Shape` know nothing about
//! `NDArray` — which is what makes a native backend a replacement rather
//! than a rewrite.
//!
//! ## Why the buffer lives here
//!
//! The SDK copies a `Vec<T>` across the boundary. So a design where Saule
//! holds the data in a `table<float>` and hands it to a native kernel per
//! operation would marshal the entire buffer in and out on every call — 8 MB
//! each way on a million-element array — and the copying would cost more than
//! the arithmetic it was meant to accelerate.
//!
//! So an [`NDArray`] *is* the buffer. It crosses the boundary as an opaque handle,
//! `a.matmul(b)` passes two handles and returns a third, and the floats never
//! move. `toTable` exists for the edges — reading a result, seeding weights —
//! and is the only thing here that copies.
//!
//! That is also what makes a GPU backend possible later without changing this
//! API: a device buffer is just as opaque as a host one.
//!
//! ## Correctness
//!
//! `numsol` is the oracle. Every kernel here has to agree with the pure-Saule
//! implementation element for element, and `examples/conformance` is the
//! program that checks it. A kernel that is quick and disagrees is not a
//! faster backend, it is a broken one — which is the whole reason the Saule
//! implementation stays in the tree rather than being deleted once this
//! works.
//!
//! ## Building and installing
//!
//! ```text
//! cargo build --release
//! cp target/release/libsaule_accel.dylib ~/.saule/native_packages/
//! ```

pub mod kernel;

use std::sync::atomic::{AtomicBool, Ordering};

use saule_sdk::prelude::*;

saule_sdk::saule_package! {
    name = "numsol",
    version = "0.1.0",
    doc = "numpy's shape, in Saule's types — accelerated. A drop-in for the pure-Saule numsol.",
    classes {
        Num = "Where arrays are made, and where the compute device is chosen."
    }
}

// ─── the device switch ─────────────────────────────────────────────────────

/// Whether the caller has asked for a GPU. Separate from whether one is
/// *being used*, because the honest answer to `autoUseGPU(true)` on a machine
/// with no usable device is "cpu", and a flag that recorded only the request
/// would make `device()` lie.
static WANT_GPU: AtomicBool = AtomicBool::new(false);

/// Whether a GPU backend is compiled in at all.
///
/// False today: the kernels in [`kernel`] are host-side. The constant exists
/// so `autoUseGPU` and `device` are already truthful, and so the day a device
/// backend lands nothing above this line has to change.
const HAS_GPU_BACKEND: bool = false;

fn active_device() -> &'static str {
    if WANT_GPU.load(Ordering::Relaxed) && HAS_GPU_BACKEND {
        "gpu"
    } else {
        "cpu"
    }
}

/// Ask for GPU execution where it is available.
///
/// Returns the device actually selected, so the answer is never a promise:
/// `Num.autoUseGPU(true)` returns `"cpu"` on a machine with no usable
/// device, and the program runs correctly anyway. Nothing above this call
/// changes — the point of the switch is that model code never mentions a
/// device.
// Spelled out rather than left to the macro's camelCasing, which would derive
// `autoUseGpu` from the Rust name. `GPU` is an initialism and reads as one.
#[saule_export(class = "Num", name = "autoUseGPU")]
fn auto_use_gpu(on: bool) -> String {
    WANT_GPU.store(on, Ordering::Relaxed);
    active_device().to_string()
}

/// The device operations are running on: `"cpu"` or `"gpu"`.
#[saule_export(class = "Num")]
fn device() -> String {
    active_device().to_string()
}

/// Whether this build has a GPU backend at all, regardless of the switch.
#[saule_export(class = "Num")]
fn gpu_available() -> bool {
    HAS_GPU_BACKEND
}

// ─── shapes ────────────────────────────────────────────────────────────────

/// Saule speaks `table<integer>` for a shape; the kernels want `[usize]`.
fn dims_from(raw: &[i64], what: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for (i, &d) in raw.iter().enumerate() {
        if d < 0 {
            return Err(format!(
                "accel: {what} got a negative extent {d} on axis {}",
                i + 1
            ));
        }
        out.push(d as usize);
    }
    Ok(out)
}

fn dims_to(dims: &[usize]) -> Vec<i64> {
    dims.iter().map(|&d| d as i64).collect()
}

fn shape_text(dims: &[usize]) -> String {
    let inner: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!("({})", inner.join(", "))
}

// ─── the array ─────────────────────────────────────────────────────────────

/// A dense, row-major, contiguous array of `float`.
///
/// Row-major and contiguous with no views, matching `numsol`: `reshape`,
/// `transpose` and `slice` all return fresh arrays. That costs a copy and
/// buys the guarantee that writing through one name cannot surprise another.
/// The operator contracts this class answers.
///
/// Each one is satisfied by a method of the conventional name — `OpAdd` by
/// `add`, `OpNeg` by `neg`, `OpLen` by `len` — exactly as a Saule class
/// satisfies them. The typechecker reads the list off the package
/// description, so `a + b` on an `NDArray` type-checks before anything loads, and
/// a contract declared here without its method is a compile error in the
/// program that uses it rather than a surprise at run time.
#[saule_class(implements(
    OpAdd<any, NDArray>,
    OpSub<any, NDArray>,
    OpMul<any, NDArray>,
    OpDiv<any, NDArray>,
    OpMod<any, NDArray>,
    OpPow<any, NDArray>,
    OpNeg<NDArray>,
    OpLen,
    OpEq<NDArray>,
    OpToString,
    OpIndex<integer, float>,
    OpNewIndex<integer, float>,
    Assignable<table<float>>
))]
pub struct NDArray {
    data: Vec<f64>,
    dims: Vec<usize>,
}

/// Shape and size rather than the buffer.
///
/// Hand-written because a derived `Debug` prints every element, and the one
/// place this shows up is an assertion message about an array that may have a
/// million of them.
impl std::fmt::Debug for NDArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NDArray{} [{} element(s)]",
            shape_text(&self.dims),
            self.data.len()
        )
    }
}

/// What an `any` operand turned out to be.
///
/// The right-hand side of an operator may be another array, a lone `float`,
/// or a `table<float>` read as a vector — the same three `numsol` accepts,
/// because this layer is meant to stand in for it.
enum Operand {
    /// Borrowed, so an array operand never copies.
    Array(SObject<NDArray>),
    /// A lone `float`, applied to every element without building an array.
    Scalar(f64),
    /// A `table<float>`, read as a rank-1 array. This one does copy: the
    /// table's memory belongs to the host.
    Vector(Vec<f64>),
}

/// Narrow an `any` into an [`Operand`], or say why it cannot be one.
///
/// `any` is how one method takes both its own class and a scalar —
/// `OpAdd<any, NDArray>` is what makes `a + b` and `a + 2.0` the same method.
/// Narrowing it is a checked cast on the Saule side and a match here; either
/// way the failure is a named error rather than a guess.
fn operand(value: &SValue, what: &str) -> Result<Operand, String> {
    match value {
        SValue::Float(f) => Ok(Operand::Scalar(*f)),
        // Saule never promotes `integer` to `float`, and this layer does not
        // get a private exemption — `a + 2` names the fix rather than
        // silently converting, exactly as numsol's README promises.
        SValue::Int(n) => Err(format!(
            "accel: {what}: elements are `float`, so `{n}` has to be written \
             `{n}.0` or `{n}f`"
        )),
        SValue::Object(obj) => obj
            .downcast::<NDArray>()
            .map(Operand::Array)
            .ok_or_else(|| {
                format!(
                    "accel: {what}: expected an `NDArray`, a `float` or a `table<float>`, \
                     got a `{}`",
                    obj.class_name()
                )
            }),
        SValue::Table(table) => {
            // 1-based array part, which is what `table<float>` is.
            let values = table
                .to_vec()
                .map_err(|e| format!("accel: {what}: reading the table: {e}"))?;
            let mut out = Vec::with_capacity(values.len());
            for (i, v) in values.iter().enumerate() {
                match v {
                    SValue::Float(f) => out.push(*f),
                    SValue::Int(n) => {
                        return Err(format!(
                            "accel: {what}: element {} of the table is the integer \
                             `{n}`; elements are `float`, so write `{n}.0`",
                            i + 1
                        ));
                    }
                    other => {
                        return Err(format!(
                            "accel: {what}: element {} of the table is not a `float` \
                             ({other:?})",
                            i + 1
                        ));
                    }
                }
            }
            Ok(Operand::Vector(out))
        }
        other => Err(format!(
            "accel: {what}: expected an `NDArray`, a `float` or a `table<float>`, \
             got {other:?}"
        )),
    }
}

impl NDArray {
    fn new(data: Vec<f64>, dims: Vec<usize>) -> NDArray {
        debug_assert_eq!(data.len(), kernel::size_of(&dims));
        NDArray { data, dims }
    }

    /// Combine with an `any` operand, elementwise, broadcasting.
    ///
    /// The scalar case does not build an array for the scalar: `a * 2.0` is
    /// the commonest shape in real code and allocating a one-element array
    /// for it, then broadcasting that, is pure waste.
    fn combine(
        &self,
        other: &SValue,
        what: &str,
        f: impl Fn(f64, f64) -> f64,
    ) -> Result<NDArray, String> {
        match operand(other, what)? {
            Operand::Scalar(v) => Ok(self.unary(|x| f(x, v))),
            Operand::Array(rhs) => {
                let rhs = rhs.borrow().map_err(|e| format!("accel: {what}: {e}"))?;
                self.binary(&rhs, what, f)
            }
            Operand::Vector(values) => {
                let n = values.len();
                let rhs = NDArray::new(values, vec![n]);
                self.binary(&rhs, what, f)
            }
        }
    }

    /// The buffer and shape, for the kernels and for tests.
    pub fn parts(&self) -> (&[f64], &[usize]) {
        (&self.data, &self.dims)
    }

    fn binary(
        &self,
        other: &NDArray,
        what: &str,
        f: impl Fn(f64, f64) -> f64,
    ) -> Result<NDArray, String> {
        let out_dims = kernel::broadcast(&self.dims, &other.dims)
            .map_err(|e| format!("accel: {what}: {e}"))?;
        let data = kernel::zip_with(
            &self.data,
            &self.dims,
            &other.data,
            &other.dims,
            &out_dims,
            f,
        );
        Ok(NDArray::new(data, out_dims))
    }

    fn unary(&self, f: impl Fn(f64) -> f64) -> NDArray {
        NDArray::new(self.data.iter().map(|&x| f(x)).collect(), self.dims.clone())
    }

    /// The matrix product against a known `NDArray`.
    ///
    /// The typed core of [`NDArray::matmul`]: the public method narrows an `any`
    /// and lands here, and the unit tests call it directly rather than
    /// fabricating an `SValue`, which needs a live host.
    /// A rank-1 right operand follows numpy's `@` rule: it is promoted to a
    /// column, multiplied, and the axis it added is dropped again — so
    /// `(M, K) @ (K,)` is `(M,)` and not `(M, 1)`. numsol documents the same
    /// behaviour, and getting it wrong in either direction (promoting to a
    /// row, or leaving the axis in place) is a shape bug that type-checks.
    fn matmul_arr(&self, other: &NDArray) -> Result<NDArray, String> {
        let promoted = other.dims.len() == 1;
        let rhs_dims: Vec<usize> = if promoted {
            vec![other.dims[0], 1]
        } else {
            other.dims.clone()
        };

        let (data, mut dims) =
            kernel::matmul_batched(&self.data, &self.dims, &other.data, &rhs_dims)
                .map_err(|e| format!("accel: {e}"))?;

        if promoted {
            // Drop the trailing 1 the promotion introduced.
            dims.pop();
        }
        Ok(NDArray::new(data, dims))
    }

    /// A multi-axis, 1-based index as a flat offset.
    fn offset_of(&self, idx: &[i64], what: &str) -> Result<usize, String> {
        if idx.len() != self.dims.len() {
            return Err(format!(
                "accel: {what}: {} needs {} index/indices, got {}",
                shape_text(&self.dims),
                self.dims.len(),
                idx.len()
            ));
        }
        let strides = kernel::strides(&self.dims);
        let mut off = 0usize;
        for (ax, (&i, &extent)) in idx.iter().zip(self.dims.iter()).enumerate() {
            if i < 1 || i as usize > extent {
                return Err(format!(
                    "accel: {what}: index {i} on axis {} is out of range 1..={extent}",
                    ax + 1
                ));
            }
            off += (i as usize - 1) * strides[ax];
        }
        Ok(off)
    }

    /// The in-place add, once the right-hand side is a flat buffer.
    ///
    /// Broadcasting is allowed only *into* this array's shape: an in-place
    /// operation cannot change the shape it writes into, so a right-hand side
    /// that would widen it is an error rather than a silent reallocation.
    fn add_slice_in_place(&mut self, rhs: &[f64], rhs_dims: &[usize]) -> Result<(), String> {
        let out = kernel::broadcast(&self.dims, rhs_dims)
            .map_err(|e| format!("accel: addInPlace: {e}"))?;
        if out != self.dims {
            return Err(format!(
                "accel: addInPlace: {} would have to grow to {} — an in-place add \
                 cannot change the shape it writes into",
                shape_text(&self.dims),
                shape_text(&out)
            ));
        }
        let stretched = kernel::stretch(rhs, rhs_dims, &self.dims);
        for (x, y) in self.data.iter_mut().zip(stretched.iter()) {
            *x += y;
        }
        Ok(())
    }

    /// `axis` as Saule writes it (1-based) checked and converted to 0-based.
    fn axis_of(&self, axis: i64, what: &str) -> Result<usize, String> {
        if axis < 1 || axis as usize > self.dims.len() {
            return Err(format!(
                "accel: {what}: axis {axis} is out of range for {}",
                shape_text(&self.dims)
            ));
        }
        Ok(axis as usize - 1)
    }
}

#[saule_methods]
impl NDArray {
    // ---------------------------------------------------------- inspection

    /// The shape, as a 1-based `table<integer>`.
    pub fn get_shape(&self) -> Vec<i64> {
        dims_to(&self.dims)
    }

    /// Number of axes.
    pub fn rank(&self) -> i64 {
        self.dims.len() as i64
    }

    /// Total element count.
    pub fn size(&self) -> i64 {
        self.data.len() as i64
    }

    /// `#a` — the element count, satisfying `OpLen`.
    ///
    /// The same number [`NDArray::size`] reports. Both exist because `#a` is the
    /// idiom and `a.size()` is what reads well in a chain.
    pub fn len(&self) -> i64 {
        self.data.len() as i64
    }

    /// Whether the array holds no elements. Present because a `len` without
    /// an `is_empty` beside it is a clippy lint, and the question is a fair
    /// one to ask.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `NDArray.of(x)` — the `Assignable` contract.
    ///
    /// What makes `local v: NDArray = {1.0, 2.0}` work: wherever a declared
    /// `NDArray` meets a bare value, the language calls this.
    ///
    /// Returns an `SObject`, not a fresh `NDArray`, and that is the whole
    /// subtlety. The conversion is applied wherever the declared type is
    /// `NDArray`, including `local b: NDArray = a` where `a` already is one —
    /// and an array is a reference value, so that assignment has to **alias**
    /// rather than copy. Handing back the same counted reference preserves
    /// identity; building a new array here would silently turn every such
    /// assignment into a deep copy, and `b` would stop tracking `a`.
    ///
    /// Wider than numsol's, which takes `table<float>`: this takes `any` so it
    /// can recognise an array and pass it through. More permissive, never
    /// less, so nothing that worked against the reference stops working.
    pub fn of(value: SValue) -> Result<SObject<NDArray>, String> {
        match operand(&value, "of")? {
            // Already an array: the same object back, identity intact.
            Operand::Array(obj) => Ok(obj),
            Operand::Vector(values) => {
                let n = values.len();
                Ok(SObject::new(NDArray::new(values, vec![n])))
            }
            // A lone float fills a slot as a one-element array, matching what
            // `coerce` does with one.
            Operand::Scalar(v) => Ok(SObject::new(NDArray::new(vec![v], vec![1]))),
        }
    }

    /// `NDArray.coerce(x)` — read an operand as an array.
    ///
    /// Static, and the reason `a + 2.0` and `a + {1.0, 2.0}` mean what they
    /// look like: an array passes through, a `float` becomes a one-element
    /// array that broadcasts, a `table<float>` becomes a vector. zenith calls
    /// this directly to build a `Tensor` from anything.
    pub fn coerce(other: SValue) -> Result<NDArray, String> {
        match operand(&other, "coerce")? {
            Operand::Array(obj) => {
                let got = obj.borrow().map_err(|e| format!("accel: coerce: {e}"))?;
                Ok(NDArray::new(got.data.clone(), got.dims.clone()))
            }
            // Rank 0, so it broadcasts against anything.
            Operand::Scalar(v) => Ok(NDArray::new(vec![v], vec![])),
            Operand::Vector(values) => {
                let n = values.len();
                Ok(NDArray::new(values, vec![n]))
            }
        }
    }

    /// Rows `first..=last` of a rank-2 array, 1-based and inclusive.
    ///
    /// A fresh array, like every other reshaping operation here: there are no
    /// views, so a slice cannot alias the array it came from.
    pub fn slice(&self, first: i64, last: i64) -> Result<NDArray, String> {
        if self.dims.len() != 2 {
            return Err(format!(
                "accel: slice() needs a rank-2 array, got {}",
                shape_text(&self.dims)
            ));
        }
        let (h, w) = (self.dims[0], self.dims[1]);
        if first < 1 || last < first || last as usize > h {
            return Err(format!(
                "accel: slice: rows {first}..{last} are out of range 1..={h}"
            ));
        }
        let (f, l) = (first as usize - 1, last as usize);
        Ok(NDArray::new(
            self.data[f * w..l * w].to_vec(),
            vec![l - f, w],
        ))
    }

    /// Row `i` of a rank-2 array, 1-based, as a **rank-1** array.
    ///
    /// Rank 1 rather than `(1, cols)`, matching numsol: a row read out to be
    /// worked on is a vector, and leaving the degenerate axis on would make
    /// it broadcast differently than the reference does.
    pub fn row(&self, i: i64) -> Result<NDArray, String> {
        let flat = self.slice(i, i)?;
        let n = flat.data.len();
        Ok(NDArray::new(flat.data, vec![n]))
    }

    /// Column `j` of a rank-2 array, 1-based, as a **rank-1** array.
    pub fn col(&self, j: i64) -> Result<NDArray, String> {
        if self.dims.len() != 2 {
            return Err(format!(
                "accel: col() needs a rank-2 array, got {}",
                shape_text(&self.dims)
            ));
        }
        let (h, w) = (self.dims[0], self.dims[1]);
        if j < 1 || j as usize > w {
            return Err(format!("accel: col: {j} is out of range 1..={w}"));
        }
        let jj = j as usize - 1;
        // Rank 1, as `row` is and as numsol's is.
        Ok(NDArray::new(
            (0..h).map(|r| self.data[r * w + jj]).collect(),
            vec![h],
        ))
    }

    /// Whether the shape is exactly `other`. Cheaper than comparing the
    /// tables the caller would otherwise build.
    pub fn has_shape(&self, other: Vec<i64>) -> bool {
        self.dims.len() == other.len()
            && self
                .dims
                .iter()
                .zip(other.iter())
                .all(|(&d, &o)| o >= 0 && d == o as usize)
    }

    /// One element by multi-axis index, 1-based on every axis.
    pub fn get(&self, idx: Vec<i64>) -> Result<f64, String> {
        Ok(self.data[self.offset_of(&idx, "get")?])
    }

    /// Write one element by multi-axis index, 1-based on every axis.
    pub fn set(&mut self, idx: Vec<i64>, value: f64) -> Result<(), String> {
        let at = self.offset_of(&idx, "set")?;
        self.data[at] = value;
        Ok(())
    }

    /// `a[i]` — the flat, 1-based element, satisfying `OpIndex`.
    pub fn index(&self, i: i64) -> Result<f64, String> {
        self.at(i)
    }

    /// `a[i] = v`, satisfying `OpNewIndex`.
    pub fn new_index(&mut self, i: i64, value: f64) -> Result<(), String> {
        self.set_at(i, value)
    }

    /// `a == b` — same shape, same contents, satisfying `OpEq`.
    ///
    /// Exact rather than tolerant: `allClose` is the tolerant one, and an
    /// operator that quietly rounded would make `==` mean something different
    /// here than everywhere else in the language.
    pub fn equals(&self, other: &NDArray) -> bool {
        self.dims == other.dims && self.data == other.data
    }

    /// `a % b` — elementwise remainder, satisfying `OpMod`.
    pub fn modulo(&self, other: SValue) -> Result<NDArray, String> {
        // Rust's `%` on floats is the truncated remainder, which is what
        // Lua's and Saule's `%` give for floats.
        self.combine(&other, "mod", |x, y| x % y)
    }

    /// Add `other` into this array **in place**, returning self so it chains.
    ///
    /// The in-place family exists for accumulators and optimiser inner loops,
    /// where allocating a fresh array per step is the dominant cost. It is the
    /// deliberate exception to "arrays are never aliased".
    #[saule(chain)]
    pub fn add_in_place(&mut self, other: SValue) -> Result<(), String> {
        match operand(&other, "addInPlace")? {
            Operand::Scalar(v) => {
                for x in &mut self.data {
                    *x += v;
                }
                Ok(())
            }
            Operand::Array(rhs) => {
                let rhs = rhs
                    .borrow()
                    .map_err(|e| format!("accel: addInPlace: {e}"))?;
                self.add_slice_in_place(&rhs.data, &rhs.dims)
            }
            Operand::Vector(values) => {
                let n = values.len();
                self.add_slice_in_place(&values, &[n])
            }
        }
    }

    /// Multiply every element by `k` in place.
    #[saule(chain)]
    pub fn scale_in_place(&mut self, k: f64) {
        for x in &mut self.data {
            *x *= k;
        }
    }

    /// Set every element to `value`, in place.
    #[saule(chain)]
    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// The Heaviside step: 1 where positive, 0 elsewhere.
    pub fn step(&self) -> NDArray {
        self.unary(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    /// Elementwise minimum against a scalar, as `Num.minimum` does.
    pub fn minimum(&self, k: f64) -> NDArray {
        self.unary(|x| x.min(k))
    }

    /// The Euclidean norm of every element.
    pub fn norm(&self) -> f64 {
        // Summing squares then taking the root, rather than `hypot` folded
        // across the buffer: one pass, and the inputs here are not at the
        // extremes where the overflow-safe form earns its cost.
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Mean along one axis, 1-based.
    pub fn mean_axis(&self, axis: i64, keep: Option<bool>) -> Result<NDArray, String> {
        let ax = self.axis_of(axis, "meanAxis")?;
        let extent = self.dims[ax] as f64;
        if extent == 0.0 {
            return Err("accel: meanAxis: the axis has no elements".to_string());
        }
        let summed = self.reduce(axis, keep.unwrap_or(false), kernel::Reduce::Sum, "meanAxis")?;
        Ok(summed.unary(|x| x / extent))
    }

    /// `tostring(a)` and `print(a)`, satisfying `OpToString`.
    ///
    /// Shape and a short prefix of the buffer rather than every element: an
    /// array printed in a loop is usually being inspected, not serialised,
    /// and a million floats in a terminal is not an inspection.
    ///
    /// Named `render` in Rust because an inherent `to_string` shadows the one
    /// `Display` provides, which clippy rejects; the contract only cares
    /// about the Saule-visible name.
    #[saule(name = "toString")]
    pub fn render(&self) -> String {
        const SHOWN: usize = 8;
        let head: Vec<String> = self
            .data
            .iter()
            .take(SHOWN)
            .map(|v| format!("{v:.4}"))
            .collect();
        let tail = if self.data.len() > SHOWN { ", …" } else { "" };
        format!(
            "NDArray{} [{}{}]",
            shape_text(&self.dims),
            head.join(", "),
            tail
        )
    }

    /// Extent of one axis, 1-based.
    pub fn dim(&self, axis: i64) -> Result<i64, String> {
        let ax = self.axis_of(axis, "dim")?;
        Ok(self.dims[ax] as i64)
    }

    /// One element by flat index, 1-based — as `String.sub`, `Table.insert`
    /// and `for i, v in t` all count.
    pub fn at(&self, index: i64) -> Result<f64, String> {
        if index < 1 || index as usize > self.data.len() {
            return Err(format!(
                "accel: at: index {index} is out of range for {} element(s)",
                self.data.len()
            ));
        }
        Ok(self.data[index as usize - 1])
    }

    /// Write one element by flat index, 1-based.
    pub fn set_at(&mut self, index: i64, value: f64) -> Result<(), String> {
        if index < 1 || index as usize > self.data.len() {
            return Err(format!(
                "accel: setAt: index {index} is out of range for {} element(s)",
                self.data.len()
            ));
        }
        self.data[index as usize - 1] = value;
        Ok(())
    }

    /// The sole element of a one-element array.
    pub fn item(&self) -> Result<f64, String> {
        if self.data.len() != 1 {
            return Err(format!(
                "accel: item() needs a one-element array, got {}",
                shape_text(&self.dims)
            ));
        }
        Ok(self.data[0])
    }

    /// The buffer as a flat `table<float>`.
    ///
    /// **This copies** — it is the boundary, and the only method here that
    /// moves the data. Use it to read a result or compare against the
    /// reference, not inside a loop.
    pub fn to_table(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// `(2, 3)` — the shape as text, for diagnostics.
    pub fn shape_string(&self) -> String {
        shape_text(&self.dims)
    }

    // ------------------------------------------------------------- reshape

    /// The same elements under a new shape. Element count must match.
    pub fn reshape(&self, dims: Vec<i64>) -> Result<NDArray, String> {
        let want = dims_from(&dims, "reshape")?;
        if kernel::size_of(&want) != self.data.len() {
            return Err(format!(
                "accel: reshape: {} has {} element(s), {} wants {}",
                shape_text(&self.dims),
                self.data.len(),
                shape_text(&want),
                kernel::size_of(&want)
            ));
        }
        Ok(NDArray::new(self.data.clone(), want))
    }

    /// Axes reversed. Rank 2 is the ordinary matrix transpose.
    pub fn transpose(&self) -> NDArray {
        let out_dims: Vec<usize> = self.dims.iter().rev().copied().collect();
        NDArray::new(kernel::transpose(&self.data, &self.dims), out_dims)
    }

    /// An independent copy.
    pub fn copy(&self) -> NDArray {
        NDArray::new(self.data.clone(), self.dims.clone())
    }

    /// Stretched up to `dims` by numpy's broadcasting rule.
    pub fn broadcast_to(&self, dims: Vec<i64>) -> Result<NDArray, String> {
        let want = dims_from(&dims, "broadcastTo")?;
        let check =
            kernel::broadcast(&self.dims, &want).map_err(|e| format!("accel: broadcastTo: {e}"))?;
        if check != want {
            return Err(format!(
                "accel: broadcastTo: {} does not stretch to {}",
                shape_text(&self.dims),
                shape_text(&want)
            ));
        }
        Ok(NDArray::new(
            kernel::stretch(&self.data, &self.dims, &want),
            want,
        ))
    }

    // ---------------------------------------------------------- arithmetic

    /// `a + b` — elementwise, broadcasting.
    ///
    /// Takes `any`, so one method serves `a + b`, `a + 2.0` and
    /// `a + {1.0, 2.0}`. That is what `OpAdd<any, NDArray>` means, and it is how
    /// `numsol` declares the same operator.
    pub fn add(&self, other: SValue) -> Result<NDArray, String> {
        self.combine(&other, "add", |x, y| x + y)
    }

    /// `a - b`
    pub fn sub(&self, other: SValue) -> Result<NDArray, String> {
        self.combine(&other, "sub", |x, y| x - y)
    }

    /// `a * b` — **elementwise**. The matrix product is [`NDArray::matmul`].
    pub fn mul(&self, other: SValue) -> Result<NDArray, String> {
        self.combine(&other, "mul", |x, y| x * y)
    }

    /// `a / b`
    pub fn div(&self, other: SValue) -> Result<NDArray, String> {
        self.combine(&other, "div", |x, y| x / y)
    }

    /// `a ^ b`
    pub fn pow(&self, other: SValue) -> Result<NDArray, String> {
        self.combine(&other, "pow", |x, y| x.powf(y))
    }

    /// Elementwise maximum.
    pub fn maximum(&self, other: SValue) -> Result<NDArray, String> {
        // `f64::max` rather than a compare, so a NaN on either side
        // propagates the way the reference's `Math.max` does.
        self.combine(&other, "maximum", f64::max)
    }

    /// A lone `float` added to every element, without building an array for
    /// it. The scalar family exists because `a + 2.0` is the commonest shape
    /// in real code and allocating a one-element `NDArray` for it is waste.
    pub fn add_scalar(&self, v: f64) -> NDArray {
        self.unary(|x| x + v)
    }

    pub fn mul_scalar(&self, v: f64) -> NDArray {
        self.unary(|x| x * v)
    }

    pub fn pow_scalar(&self, v: f64) -> NDArray {
        self.unary(|x| x.powf(v))
    }

    pub fn neg(&self) -> NDArray {
        self.unary(|x| -x)
    }

    // ------------------------------------------------------------ unary fns

    pub fn exp(&self) -> NDArray {
        self.unary(f64::exp)
    }

    pub fn log(&self) -> NDArray {
        self.unary(f64::ln)
    }

    pub fn sqrt(&self) -> NDArray {
        self.unary(f64::sqrt)
    }

    pub fn abs(&self) -> NDArray {
        self.unary(f64::abs)
    }

    pub fn tanh(&self) -> NDArray {
        self.unary(f64::tanh)
    }

    /// The logistic sigmoid.
    ///
    /// Split on the sign of `x` so neither branch can overflow `exp`: the
    /// naive `1/(1+exp(-x))` loses to `inf` for very negative `x`, which is
    /// exactly where a saturating activation must stay finite.
    pub fn sigmoid(&self) -> NDArray {
        self.unary(|x| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        })
    }

    pub fn relu(&self) -> NDArray {
        self.unary(|x| if x > 0.0 { x } else { 0.0 })
    }

    /// Clamped into `[lo, hi]`.
    pub fn clamp(&self, lo: f64, hi: f64) -> Result<NDArray, String> {
        // Spelled out rather than `!(lo <= hi)`: a NaN bound compares false
        // against everything, so the negated form was right but read as a
        // typo. The NaN case has to be rejected, because `f64::clamp` panics
        // on a NaN bound.
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return Err(format!("accel: clamp: bad range lo {lo}, hi {hi}"));
        }
        Ok(self.unary(|x| x.clamp(lo, hi)))
    }

    // ------------------------------------------------------- linear algebra

    /// The matrix product, rank 2 **or batched rank 3**.
    ///
    /// | left | right | result |
    /// |---|---|---|
    /// | `(M, K)` | `(K, N)` | `(M, N)` |
    /// | `(B, M, K)` | `(B, K, N)` | `(B, M, N)` |
    /// | `(B, M, K)` | `(K, N)` | `(B, M, N)` |
    ///
    /// The third form is what a transformer's projections want: one weight
    /// applied to every sequence in the batch, read once rather than
    /// broadcast into `B` copies first.
    ///
    /// Takes `any` for consistency with the operators, but only an `NDArray` makes
    /// sense here — a scalar product is `mulScalar` and says so.
    pub fn matmul(&self, other: SValue) -> Result<NDArray, String> {
        let rhs = match operand(&other, "matmul")? {
            Operand::Array(obj) => obj,
            Operand::Scalar(_) => {
                return Err(
                    "accel: matmul: the matrix product needs an array; for a scalar \
                     use `mulScalar` or `a * 2.0`"
                        .to_string(),
                );
            }
            Operand::Vector(values) => {
                // A vector is promoted to a single row, as numpy's `@` does.
                // Rank 1, so `matmul_arr` promotes it to a column and
                // demotes the result, per numpy's rule.
                let n = values.len();
                return self.matmul_arr(&NDArray::new(values, vec![n]));
            }
        };
        let rhs = rhs.borrow().map_err(|e| format!("accel: matmul: {e}"))?;
        self.matmul_arr(&rhs)
    }

    /// Swap the last two axes, leaving a leading batch axis alone.
    ///
    /// `(B, L, d) -> (B, d, L)`, which is what batched attention needs.
    /// [`NDArray::transpose`] reverses *every* axis — right for rank 2, and on a
    /// batch it would move the batch into the middle. numpy spells this
    /// `.mT`.
    pub fn transpose_last(&self) -> Result<NDArray, String> {
        let (data, dims) =
            kernel::transpose_last2(&self.data, &self.dims).map_err(|e| format!("accel: {e}"))?;
        Ok(NDArray::new(data, dims))
    }

    // ------------------------------------------------------------ reductions

    /// Sum of every element.
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Mean of every element.
    pub fn mean(&self) -> Result<f64, String> {
        if self.data.is_empty() {
            return Err("accel: mean() of an empty array".to_string());
        }
        Ok(self.data.iter().sum::<f64>() / self.data.len() as f64)
    }

    /// Largest element.
    pub fn max(&self) -> Result<f64, String> {
        self.data
            .iter()
            .copied()
            .reduce(f64::max)
            .ok_or_else(|| "accel: max() of an empty array".to_string())
    }

    /// Sum along one axis, 1-based. `keep` leaves the axis in place with
    /// extent 1, which is what makes the result broadcast back.
    pub fn sum_axis(&self, axis: i64, keep: Option<bool>) -> Result<NDArray, String> {
        self.reduce(axis, keep.unwrap_or(false), kernel::Reduce::Sum, "sumAxis")
    }

    pub fn max_axis(&self, axis: i64, keep: Option<bool>) -> Result<NDArray, String> {
        self.reduce(axis, keep.unwrap_or(false), kernel::Reduce::Max, "maxAxis")
    }

    /// Index of the largest element along one axis, 1-based in and out.
    /// Index of the largest element along one axis, 1-based in and out.
    ///
    /// No `keepDims`, matching `numsol`: the axis is always dropped, because
    /// an index array is read positionally rather than broadcast back.
    pub fn argmax_axis(&self, axis: i64) -> Result<NDArray, String> {
        self.reduce(axis, false, kernel::Reduce::ArgMax, "argmaxAxis")
    }

    /// Whether every element agrees with `other` to within `tol`.
    ///
    /// The conformance check leans on this: a transcendental kernel is not
    /// expected to be bit-identical to the reference, only indistinguishable
    /// at the tolerance the caller names.
    pub fn all_close(&self, other: &NDArray, tol: Option<f64>) -> bool {
        // numsol's default. Spelled out because Saule has no `1e-9` literal
        // and the two have to agree on the tolerance, not just the comparison.
        let tol = tol.unwrap_or(0.000_000_001);
        self.dims == other.dims
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(&x, &y)| (x - y).abs() <= tol)
    }
}

impl NDArray {
    fn reduce(
        &self,
        axis: i64,
        keep: bool,
        how: kernel::Reduce,
        what: &str,
    ) -> Result<NDArray, String> {
        let ax = self.axis_of(axis, what)?;
        let data = kernel::reduce_axis(&self.data, &self.dims, ax, how);
        let mut out_dims = self.dims.clone();
        if keep {
            out_dims[ax] = 1;
        } else {
            out_dims.remove(ax);
        }
        Ok(NDArray::new(data, out_dims))
    }
}

// ─── factories ─────────────────────────────────────────────────────────────

/// `Num.zeros({2, 3})`
#[saule_export(class = "Num")]
fn zeros(dims: Vec<i64>) -> Result<NDArray, String> {
    let d = dims_from(&dims, "zeros")?;
    Ok(NDArray::new(vec![0.0; kernel::size_of(&d)], d))
}

/// `Num.ones({2, 3})`
#[saule_export(class = "Num")]
fn ones(dims: Vec<i64>) -> Result<NDArray, String> {
    let d = dims_from(&dims, "ones")?;
    Ok(NDArray::new(vec![1.0; kernel::size_of(&d)], d))
}

/// `Num.full({2, 3}, 7.0)`
#[saule_export(class = "Num")]
fn full(dims: Vec<i64>, value: f64) -> Result<NDArray, String> {
    let d = dims_from(&dims, "full")?;
    Ok(NDArray::new(vec![value; kernel::size_of(&d)], d))
}

/// `Num.arange(0.0, 6.0)` — the half-open range `[start, stop)`.
///
/// `step` is optional and defaults to 1, matching `Num.arange`. An `Option`
/// parameter is how the SDK spells an optional argument: a missing or `nil`
/// argument arrives as `None`.
#[saule_export(class = "Num")]
fn arange(start: f64, stop: f64, step: Option<f64>) -> Result<NDArray, String> {
    let step = step.unwrap_or(1.0);
    if step == 0.0 {
        return Err("accel: arange: step must not be zero".to_string());
    }
    // Counted rather than accumulated: `start + step * i` cannot drift the
    // way repeated `v += step` does on a fractional step.
    let span = (stop - start) / step;
    let count = if span <= 0.0 { 0 } else { span.ceil() as usize };
    let data: Vec<f64> = (0..count).map(|i| start + step * i as f64).collect();
    let n = data.len();
    Ok(NDArray::new(data, vec![n]))
}

/// `Num.array({1.0, 2.0, 3.0})` — a rank-1 array from a table.
#[saule_export(class = "Num")]
fn array(values: Vec<f64>) -> NDArray {
    let n = values.len();
    NDArray::new(values, vec![n])
}

/// `Num.shaped({1.0, 2.0, ...}, {2, 3})` — a flat buffer under a shape.
///
/// Data first, then the shape, matching `Num.shaped` exactly. The argument
/// order is not a preference: this layer is meant to become a drop-in for
/// `numsol`, and a factory whose arguments are transposed relative to the
/// thing it replaces would be a silent, type-correct bug at every call site
/// where both extents happen to be equal.
#[saule_export(class = "Num")]
fn shaped(data: Vec<f64>, dims: Vec<i64>) -> Result<NDArray, String> {
    let d = dims_from(&dims, "shaped")?;
    if kernel::size_of(&d) != data.len() {
        return Err(format!(
            "accel: shaped: {} wants {} element(s), got {}",
            shape_text(&d),
            kernel::size_of(&d),
            data.len()
        ));
    }
    Ok(NDArray::new(data, d))
}

/// `Num.scalar(3.0)` — a one-element array.
///
/// Shape `(1)`, not rank 0: that is the shape `Num.scalar` gives, and the
/// difference is observable through `shapeString`, `rank`, and anything that
/// broadcasts against it.
#[saule_export(class = "Num")]
fn scalar(value: f64) -> NDArray {
    NDArray::new(vec![value], vec![1])
}

/// `Num.zerosLike(a)` — zeros in `a`'s shape.
#[saule_export(class = "Num")]
fn zeros_like(other: &NDArray) -> NDArray {
    NDArray::new(vec![0.0; other.data.len()], other.dims.clone())
}

/// `Num.onesLike(a)`
#[saule_export(class = "Num")]
fn ones_like(other: &NDArray) -> NDArray {
    NDArray::new(vec![1.0; other.data.len()], other.dims.clone())
}

/// `Num.eye(3)` — the identity matrix.
#[saule_export(class = "Num")]
fn eye(n: i64) -> Result<NDArray, String> {
    if n < 0 {
        return Err(format!("accel: eye: negative size {n}"));
    }
    let n = n as usize;
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Ok(NDArray::new(data, vec![n, n]))
}

/// `Num.seeded({2, 3}, 1337)` — reproducible pseudo-random values in
/// `[0, 1)`.
///
/// A seeded SplitMix64 rather than a system source, because a benchmark and a
/// conformance run both want the same numbers twice. It is not for
/// cryptography and makes no claim to be.
#[saule_export(class = "Num")]
fn seeded(dims: Vec<i64>, seed: i64) -> Result<NDArray, String> {
    let d = dims_from(&dims, "seeded")?;
    let total = kernel::size_of(&d);
    let mut state = seed as u64;
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // The top 53 bits are the mantissa's worth, so this is uniform.
        data.push((z >> 11) as f64 / (1u64 << 53) as f64);
    }
    Ok(NDArray::new(data, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(dims: &[usize], data: &[f64]) -> NDArray {
        NDArray::new(data.to_vec(), dims.to_vec())
    }

    #[test]
    fn the_device_switch_never_promises_more_than_it_has() {
        // With no GPU backend compiled in, asking for one must still report
        // the truth rather than the request.
        assert_eq!(auto_use_gpu(true), "cpu");
        assert_eq!(device(), "cpu");
        assert!(!gpu_available());
        assert_eq!(auto_use_gpu(false), "cpu");
    }

    #[test]
    fn arithmetic_broadcasts() {
        let a = arr(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let row = arr(&[1, 3], &[10.0, 20.0, 30.0]);
        let got = a.binary(&row, "add", |x, y| x + y).expect("broadcasts");
        assert_eq!(got.get_shape(), vec![2, 3]);
        assert_eq!(got.to_table(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn mismatched_shapes_are_an_error_not_a_panic() {
        let a = arr(&[2, 3], &[0.0; 6]);
        let b = arr(&[4, 3], &[0.0; 12]);
        let err = a
            .binary(&b, "add", |x, y| x + y)
            .expect_err("must not broadcast");
        assert!(err.contains("do not broadcast"), "{err}");
    }

    /// numpy's rule for a 1-D right operand: promoted to a column, then the
    /// added axis is dropped. `(2,3) @ (3,) -> (2,)`.
    #[test]
    fn a_rank_one_right_operand_is_promoted_and_demoted() {
        let a = arr(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = arr(&[3], &[1.0, 1.0, 1.0]);
        let got = a.matmul_arr(&v).expect("promoted");
        assert_eq!(got.get_shape(), vec![2], "the added axis must be dropped");
        assert_eq!(got.to_table(), vec![6.0, 15.0]);
    }

    #[test]
    fn matmul_rejects_an_unsupported_rank_by_name() {
        let a = arr(&[2, 3], &[0.0; 6]);
        // Rank 4 is not a product this kernel defines.
        let four = NDArray::new(vec![0.0; 24], vec![1, 2, 3, 4]);
        let err = a.matmul_arr(&four).expect_err("rank 4 is not supported");
        assert!(err.contains("rank 2 or 3"), "{err}");
    }

    /// The batched forms, through the public shape rules.
    #[test]
    fn matmul_batches_and_shares_a_weight() {
        // (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
        let a = NDArray::new((0..12).map(|i| i as f64).collect(), vec![2, 2, 3]);
        let b = NDArray::new((0..12).map(|i| (i % 5) as f64).collect(), vec![2, 3, 2]);
        assert_eq!(a.matmul_arr(&b).unwrap().get_shape(), vec![2, 2, 2]);

        // (2, 2, 3) @ (3, 4) -> (2, 2, 4): one weight across the batch.
        let w = NDArray::new((0..12).map(|i| i as f64).collect(), vec![3, 4]);
        let out = a.matmul_arr(&w).expect("shared weight");
        assert_eq!(out.get_shape(), vec![2, 2, 4]);
        // Each batch must equal the rank-2 product of that slice.
        let slice0 = NDArray::new(a.to_table()[0..6].to_vec(), vec![2, 3]);
        let want0 = slice0.matmul_arr(&w).unwrap();
        assert_eq!(&out.to_table()[0..8], &want0.to_table()[..]);
    }

    /// `transposeLast` swaps the last two axes and leaves the batch alone,
    /// where `transpose` would move the batch into the middle.
    #[test]
    fn transpose_last_differs_from_transpose_on_a_batch() {
        let a = NDArray::new((0..12).map(|i| i as f64).collect(), vec![2, 2, 3]);
        assert_eq!(a.transpose_last().unwrap().get_shape(), vec![2, 3, 2]);
        assert_eq!(a.transpose().get_shape(), vec![3, 2, 2]);
    }

    #[test]
    fn matmul_lines_the_shared_axis_up() {
        let a = arr(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = arr(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let got = a.matmul_arr(&b).expect("lines up");
        assert_eq!(got.get_shape(), vec![2, 2]);
        // [[22,28],[49,64]]
        assert_eq!(got.to_table(), vec![22.0, 28.0, 49.0, 64.0]);

        let bad = arr(&[2, 2], &[0.0; 4]);
        assert!(a.matmul_arr(&bad).is_err());
    }

    #[test]
    fn identity_leaves_a_matrix_alone() {
        let a = arr(&[3, 3], &(1..=9).map(|i| i as f64).collect::<Vec<_>>());
        let i = eye(3).expect("eye");
        assert!(a.matmul_arr(&i).unwrap().all_close(&a, Some(0.0)));
        assert!(i.matmul_arr(&a).unwrap().all_close(&a, Some(0.0)));
    }

    #[test]
    fn indices_are_one_based_both_ways() {
        let a = arr(&[3], &[7.0, 8.0, 9.0]);
        assert_eq!(a.at(1).unwrap(), 7.0);
        assert_eq!(a.at(3).unwrap(), 9.0);
        assert!(a.at(0).is_err(), "0 is out of range in a 1-based API");
        assert!(a.at(4).is_err());

        // argmax reports a 1-based index too.
        let m = arr(&[2, 3], &[1.0, 9.0, 2.0, 5.0, 4.0, 3.0]);
        assert_eq!(m.argmax_axis(2).unwrap().to_table(), vec![2.0, 1.0]);
    }

    #[test]
    fn sigmoid_stays_finite_at_the_extremes() {
        let a = arr(&[4], &[-1000.0, -40.0, 40.0, 1000.0]);
        let s = a.sigmoid();
        for v in s.to_table() {
            assert!(v.is_finite(), "sigmoid overflowed: {v}");
            assert!((0.0..=1.0).contains(&v), "out of range: {v}");
        }
        // The naive 1/(1+exp(-x)) form gives inf for -1000 and so NaN here.
        assert_eq!(s.to_table()[0], 0.0);
        assert_eq!(s.to_table()[3], 1.0);
    }

    #[test]
    fn reductions_can_keep_the_axis_for_broadcasting_back() {
        let a = arr(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let dropped = a.sum_axis(2, None).unwrap();
        assert_eq!(dropped.get_shape(), vec![2]);
        assert_eq!(dropped.to_table(), vec![6.0, 15.0]);

        let kept = a.sum_axis(2, Some(true)).unwrap();
        assert_eq!(kept.get_shape(), vec![2, 1]);
        // Kept, so it broadcasts back against the original.
        assert!(a.binary(&kept, "div", |x, y| x / y).is_ok());
    }

    #[test]
    fn an_out_of_range_axis_is_named_not_panicked() {
        let a = arr(&[2, 3], &[0.0; 6]);
        let err = a.sum_axis(3, None).expect_err("rank is 2");
        assert!(err.contains("out of range"), "{err}");
        assert!(a.sum_axis(0, None).is_err(), "axes are 1-based");
    }

    #[test]
    fn reshape_must_preserve_the_element_count() {
        let a = arr(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(a.reshape(vec![3, 2]).unwrap().get_shape(), vec![3, 2]);
        assert_eq!(a.reshape(vec![6]).unwrap().to_table(), a.to_table());
        let err = a.reshape(vec![4, 2]).expect_err("6 != 8");
        assert!(err.contains("element(s)"), "{err}");
    }

    #[test]
    fn a_seeded_array_is_reproducible_and_in_range() {
        let a = seeded(vec![4, 4], 1337).unwrap();
        let b = seeded(vec![4, 4], 1337).unwrap();
        assert!(
            a.all_close(&b, Some(0.0)),
            "same seed must give the same array"
        );

        let c = seeded(vec![4, 4], 1338).unwrap();
        assert!(!a.all_close(&c, Some(0.0)), "a different seed must differ");

        assert!(a.to_table().iter().all(|v| (0.0..1.0).contains(v)));
    }

    #[test]
    fn transpose_round_trips() {
        let a = arr(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.get_shape(), vec![3, 2]);
        assert!(t.transpose().all_close(&a, Some(0.0)));
    }

    #[test]
    fn broadcast_to_refuses_to_shrink() {
        let a = arr(&[2, 3], &[0.0; 6]);
        assert!(a.broadcast_to(vec![2, 3]).is_ok());
        assert!(
            a.broadcast_to(vec![2, 1]).is_err(),
            "stretching is not squeezing"
        );
    }

    #[test]
    fn scalar_forms_match_the_array_forms() {
        let a = arr(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let two = arr(&[1], &[2.0]);
        assert!(
            a.add_scalar(2.0)
                .all_close(&a.binary(&two, "add", |x, y| x + y).unwrap(), Some(0.0))
        );
        assert!(
            a.mul_scalar(2.0)
                .all_close(&a.binary(&two, "mul", |x, y| x * y).unwrap(), Some(0.0))
        );
        assert!(a.pow_scalar(2.0).all_close(
            &a.binary(&two, "pow", |x, y| x.powf(y)).unwrap(),
            Some(1e-12)
        ));
    }
}

// ─── random factories ──────────────────────────────────────────────────────

/// The PRNG, matching Saule's `Math.random` exactly.
///
/// Same algorithm (xorshift64*), same multiplier, same default seed, same
/// top-53-bits float conversion — so `Num.random`, `uniform` and `randn`
/// produce the **identical stream** the pure-Saule numsol does. Two reasons
/// that is worth copying rather than improving on:
///
///   * **Reproducibility.** Saule seeds `Math.random` from a fixed constant,
///     so a numsol program gives the same numbers every run. A native package
///     that seeded itself from the clock would make a training run
///     irreproducible, which is the opposite of what a backend swap should
///     change. It cost exactly that once: the toy GPT passed, then failed on
///     the next run, because its initialisation moved under it.
///   * **It keeps the random factories inside the conformance check.** A
///     different stream would make `Num.randn` untestable against the
///     reference, and those are precisely the functions a network's
///     initialisation depends on.
///
/// `Math.randomseed` reseeds Saule's generator and not this one, so the two
/// streams diverge after it is called. That is a known limit, written down
/// rather than papered over; nothing here can see the interpreter's state.
static RNG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x6a09_e667_f3bc_c909);

/// One xorshift64* step, bit-for-bit as `stdlib::math::next_u64` does it.
fn next_u64() -> u64 {
    use std::sync::atomic::Ordering;
    // `fetch_update` rather than load/store, so two threads cannot take the
    // same step; the interpreter is single-threaded, but a harness that runs
    // several programs in one process would otherwise correlate their streams.
    let mut out = 0u64;
    let _ = RNG.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |state| {
        let mut x = if state == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            state
        };
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        out = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
        Some(x)
    });
    out
}

/// Uniform in `[0, 1)` from the top 53 bits — an IEEE 754 fraction, exactly
/// as `random_unit` builds it.
fn next_f64() -> f64 {
    let bits = next_u64() >> 11;
    (bits as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// `Num.random({2, 3})` — uniform in `[0, 1)`.
#[saule_export(class = "Num")]
fn random(dims: Vec<i64>) -> Result<NDArray, String> {
    let d = dims_from(&dims, "random")?;
    let total = kernel::size_of(&d);
    Ok(NDArray::new((0..total).map(|_| next_f64()).collect(), d))
}

/// `Num.uniform({2, 3}, -1.0, 1.0)` — uniform in `[lo, hi)`.
#[saule_export(class = "Num")]
fn uniform(dims: Vec<i64>, lo: f64, hi: f64) -> Result<NDArray, String> {
    let d = dims_from(&dims, "uniform")?;
    let total = kernel::size_of(&d);
    let span = hi - lo;
    Ok(NDArray::new(
        (0..total).map(|_| lo + next_f64() * span).collect(),
        d,
    ))
}

/// `Num.randn({2, 3})` — standard normal, mean 0, variance 1.
///
/// Box–Muller in its **trigonometric** form, and in that form specifically
/// because it is the one `numsol` uses: each pair of uniforms becomes a radius
/// and an angle whose cosine and sine are two independent normals. The polar
/// rejection form is the usual choice in Rust and is arguably better — it
/// trades the two transcendentals for a loop — but it consumes a different
/// number of uniforms per output, so it would draw a different stream and the
/// two implementations could never be compared.
///
/// `u1` is nudged off zero for the same reason numsol nudges it: `random()`
/// can return exactly 0 and `log(0)` is `-inf`.
#[saule_export(class = "Num")]
fn randn(dims: Vec<i64>) -> Result<NDArray, String> {
    let d = dims_from(&dims, "randn")?;
    let total = kernel::size_of(&d);
    let mut data = vec![0.0f64; total];

    let mut i = 0usize;
    while i < total {
        let u1 = next_f64().max(0.000_000_000_001);
        let u2 = next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;

        data[i] = radius * angle.cos();
        if i + 1 < total {
            data[i + 1] = radius * angle.sin();
        }
        i += 2;
    }

    Ok(NDArray::new(data, d))
}

/// `Num.matrix({{1.0, 2.0}, {3.0, 4.0}})` — a rank-2 array from rows.
///
/// Every row must be the same length; a ragged literal is a shape error
/// rather than a silently padded array.
#[saule_export(class = "Num")]
fn matrix(rows: Vec<Vec<f64>>) -> Result<NDArray, String> {
    let height = rows.len();
    if height == 0 {
        return Err("numsol: matrix: needs at least one row".to_string());
    }
    let width = rows[0].len();
    if width == 0 {
        return Err("numsol: matrix: rows cannot be empty".to_string());
    }
    let mut data = Vec::with_capacity(height * width);
    for (i, row) in rows.iter().enumerate() {
        if row.len() != width {
            return Err(format!(
                "numsol: matrix: row {} has {} element(s), row 1 has {width}",
                i + 1,
                row.len()
            ));
        }
        data.extend_from_slice(row);
    }
    Ok(NDArray::new(data, vec![height, width]))
}
