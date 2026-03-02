use std::f32;
use std::ptr;

pub const MAX_VOCAB_SIZE: usize = 100;

pub type DataT = f32;
pub type GradT = f32;

pub struct Matrix {
    pub data_start: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { data_start: 0, rows, cols }
    }

    pub fn at(&self, i: usize, j: usize) -> usize {
        self.data_start + i * self.cols + j
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(u8)]
pub enum Op {
    Const,
    SubConst,
    Relu,
    InvLog,
    InvSqrt,
    Exp,
    Add,
    Mul,
    Div,
}

pub struct Tape {
    pub data: Vec<DataT>,
    pub grad: Vec<GradT>,
    pub i_child0: Vec<usize>,
    pub i_child1: Vec<usize>,
    pub op: Vec<Op>,
    pub size: usize,
    pub cap: usize,
}

impl Tape {
pub fn new(n: usize) -> Self {
        Self {
            data: vec![0.0; n],
            grad: vec![0.0; n],
            i_child0: vec![0; n],
            i_child1: vec![0; n],
            op: vec![Op::Const; n],
            size: 0,
            cap: n,
        }
    }

    #[inline(always)]
    fn ensure(&mut self) {
        if self.size >= self.cap {
            let new_cap = self.cap * 2;
            self.data.resize(new_cap, 0.0);
            self.grad.resize(new_cap, 0.0);
            self.i_child0.resize(new_cap, 0);
            self.i_child1.resize(new_cap, 0);
            self.op.resize(new_cap, Op::Const);
            self.cap = new_cap;
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize { self.size }

    #[inline(always)]
    pub fn truncate(&mut self, n: usize) { self.size = n; }

#[inline(always)]
    pub fn push_const(&mut self, d: DataT) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = d;
        self.op[i] = Op::Const;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn relu(&mut self, a: usize) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = self.data[a].max(0.0);
        self.i_child0[i] = a;
        self.op[i] = Op::Relu;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn inv_log(&mut self, a: usize) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = -self.data[a].ln();
        self.i_child0[i] = a;
        self.op[i] = Op::InvLog;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn inv_sqrt(&mut self, a: usize) -> usize {
        self.ensure();
        let i = self.size;
        let val = (self.data[a] + 1e-5).powf(-0.5);
        self.data[i] = val;
        self.i_child0[i] = a;
        self.op[i] = Op::InvSqrt;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn exp(&mut self, a: usize) -> usize {
        self.ensure();
        let i = self.size;
        let val = self.data[a].exp();
        self.data[i] = val;
        self.i_child0[i] = a;
        self.op[i] = Op::Exp;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn sub_const(&mut self, a: usize, c: DataT) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = self.data[a] - c;
        self.i_child0[i] = a;
        self.op[i] = Op::SubConst;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = self.data[a] + self.data[b];
        self.i_child0[i] = a;
        self.i_child1[i] = b;
        self.op[i] = Op::Add;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = self.data[a] * self.data[b];
        self.i_child0[i] = a;
        self.i_child1[i] = b;
        self.op[i] = Op::Mul;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn mul_const(&mut self, a: usize, c: DataT) -> usize {
        let i_c = self.push_const(c);
        self.mul(a, i_c)
    }

    #[inline(always)]
    pub fn div(&mut self, a: usize, b: usize) -> usize {
        self.ensure();
        let i = self.size;
        self.data[i] = self.data[a] / self.data[b];
        self.i_child0[i] = a;
        self.i_child1[i] = b;
        self.op[i] = Op::Div;
        self.size += 1;
        i
    }

    #[inline(always)]
    pub fn div_const(&mut self, a: usize, c: DataT) -> usize {
        let i_c = self.push_const(c);
        self.div(a, i_c)
    }

    #[inline(always)]
    pub fn mul_add(&mut self, a: usize, b: usize, c: usize) -> usize {
        let i1 = self.size;
        let i2 = i1 + 1;
        self.size += 2;
        self.ensure(); // Ensure space for 2 nodes

        // Using raw pointers for maximum speed in the hot loop
        unsafe {
            let base_data = self.data.as_mut_ptr();
            let base_op = self.op.as_mut_ptr();
            let base_c0 = self.i_child0.as_mut_ptr();
            let base_c1 = self.i_child1.as_mut_ptr();

            // Fetch values from data pointer
            let va = *base_data.add(a);
            let vb = *base_data.add(b);
            let product = va * vb;

            // i1: The Multiplication Node
            *base_data.add(i1) = product;
            *base_op.add(i1) = Op::Mul;
            *base_c0.add(i1) = a;
            *base_c1.add(i1) = b;

            // i2: The Addition Node
            *base_data.add(i2) = product + *base_data.add(c);
            *base_op.add(i2) = Op::Add;
            *base_c0.add(i2) = c;
            *base_c1.add(i2) = i1;
        }
        i2
    }

    pub fn softmax(&mut self, out: &mut [usize], logits: &[usize]) {
        let mut max_val = f32::MIN;
        for &idx in logits {
            if self.data[idx] > max_val { max_val = self.data[idx]; }
        }

        let mut exps = [0usize; MAX_VOCAB_SIZE];
        for i in 0..logits.len() {
            let sub = self.sub_const(logits[i], max_val);
            exps[i] = self.exp(sub);
        }

        let mut sum_exponents = exps[0];
        for i in 1..logits.len() {
            sum_exponents = self.add(sum_exponents, exps[i]);
        }

        for i in 0..logits.len() {
            out[i] = self.div(exps[i], sum_exponents);
        }
    }

    pub fn rmsnorm(&mut self, out: &mut [usize], x: &[usize]) {
        let mut sum_squares = self.mul(x[0], x[0]);
        for i in 1..x.len() {
            sum_squares = self.mul_add(x[i], x[i], sum_squares);
        }
        let mean_square = self.div_const(sum_squares, x.len() as DataT);
        let scale = self.inv_sqrt(mean_square);
        for i in 0..x.len() {
            out[i] = self.mul(x[i], scale);
        }
    }

    pub fn linear(&mut self, out: &mut [usize], x: &[usize], w: &Matrix) {
        for i in 0..w.rows {
            let w_at_row_i = w.at(i, 0);
            let mut sum = self.mul(w_at_row_i, x[0]);
            for j in 1..w.cols {
                sum = self.mul_add(w_at_row_i + j, x[j], sum);
            }
            out[i] = sum;
        }
    }

pub fn backward(&mut self, loss_idx: usize) {
        // Equivalent to std::memset(grad, 0, n * sizeof(grad_T))
        unsafe {
            ptr::write_bytes(self.grad.as_mut_ptr(), 0, loss_idx + 1);
        }
        self.grad[loss_idx] = 1.0;

        // Using raw pointers for maximum speed in the hot loop
        let p_data = self.data.as_ptr();
        let p_grad = self.grad.as_mut_ptr();
        let p_c0 = self.i_child0.as_ptr();
        let p_c1 = self.i_child1.as_ptr();
        let p_op = self.op.as_ptr();

        for i in (0..=loss_idx).rev() {
            unsafe {
                let op = *p_op.add(i);
                if op == Op::Const { continue; }

                let g = *p_grad.add(i);
                if g == 0.0 { continue; }

                let c0 = *p_c0.add(i);

                match op {
                    Op::SubConst | Op::Add => {
                        *p_grad.add(c0) += g;
                        if op == Op::Add {
                            *p_grad.add(*p_c1.add(i)) += g;
                        }
                    }
                    Op::Relu => {
                        if *p_data.add(c0) > 0.0 {
                            *p_grad.add(c0) += g;
                        }
                    }
                    Op::InvLog => {
                        *p_grad.add(c0) -= g / *p_data.add(c0);
                    }
                    Op::InvSqrt => {
                        *p_grad.add(c0) -= 0.5 * g * (*p_data.add(i)) / (*p_data.add(c0) + 1e-5);
                    }
                    Op::Exp => {
                        *p_grad.add(c0) += g * (*p_data.add(i));
                    }
                    Op::Mul => {
                        let c1 = *p_c1.add(i);
                        *p_grad.add(c0) += g * (*p_data.add(c1));
                        *p_grad.add(c1) += g * (*p_data.add(c0));
                    }
                    Op::Div => {
                        let c1 = *p_c1.add(i);
                        let d_c1 = *p_data.add(c1);
                        *p_grad.add(c0) += g / d_c1;
                        *p_grad.add(c1) -= g * (*p_data.add(c0)) / (d_c1 * d_c1);
                    }
                    _ => {}
                }
            }
        }
    }
}
