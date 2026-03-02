use std::f64::consts::PI;

const STATE_SIZE: usize = 624;

pub struct PythonRandom {
    state: [u32; STATE_SIZE],
    index: usize,
    gauss_next: Option<f64>,
}

impl PythonRandom {
    pub fn new(seed: u32) -> Self {
        let mut pr = Self {
            state: [0; STATE_SIZE],
            index: STATE_SIZE,
            gauss_next: None,
        };
        pr.seed(seed);
        pr
    }

    /// Replicates Python's init_by_array/init_genrand seeding logic
    fn seed(&mut self, seed: u32) {
        // init_genrand
        self.state[0] = 19650218_u32;
        for i in 1..STATE_SIZE {
            let prev = self.state[i - 1];
            self.state[i] = 1812433253_u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }

        // init_by_array logic
        let mut i = 1;
        for _ in (1..=STATE_SIZE).rev() {
            let prev = self.state[i - 1];
            self.state[i] = (self.state[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1664525_u32)))
                .wrapping_add(seed);
            i += 1;
            if i >= STATE_SIZE {
                self.state[0] = self.state[STATE_SIZE - 1];
                i = 1;
            }
        }

        for _ in (1..STATE_SIZE).rev() {
            let prev = self.state[i - 1];
            self.state[i] = (self.state[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1566083941_u32)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= STATE_SIZE {
                self.state[0] = self.state[STATE_SIZE - 1];
                i = 1;
            }
        }

        self.state[0] = 0x80000000_u32;
        self.index = STATE_SIZE;
    }

    /// Core MT19937 generator
    fn next_u32(&mut self) -> u32 {
        if self.index >= STATE_SIZE {
            self.twist();
        }

        let mut y = self.state[self.index];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;

        self.index += 1;
        y
    }

    fn twist(&mut self) {
        for i in 0..STATE_SIZE {
            let y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % STATE_SIZE] & 0x7fffffff);
            let mut next = self.state[(i + 397) % STATE_SIZE] ^ (y >> 1);
            if y % 2 != 0 {
                next ^= 0x9908b0df;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    /// Replicates Python's random(): 53-bit float from two 32-bit ints
    pub fn random(&mut self) -> f64 {
        let a = self.next_u32() >> 5;
        let b = self.next_u32() >> 6;
        let value = ((a as u64) << 26) | (b as u64);
        (value as f64) / (1u64 << 53) as f64
    }

    fn bit_width(&self, x: u32) -> u32 {
        if x == 0 { return 0; }
        32 - x.leading_zeros()
    }

    fn getrandbits(&mut self, k: u32) -> u32 {
        if k == 0 { return 0; }
        self.next_u32() >> (32 - k)
    }

    fn randbelow(&mut self, n: u32) -> u32 {
        if n == 0 { return 0; }
        let k = self.bit_width(n);
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        if let Some(z) = self.gauss_next.take() {
            return mu + z * sigma;
        }

        let x2pi = self.random() * 2.0 * PI;
        let g2rad = (-2.0 * (1.0 - self.random()).ln()).sqrt();
        
        let z = x2pi.cos() * g2rad;
        self.gauss_next = Some(x2pi.sin() * g2rad);
        
        mu + z * sigma
    }

    pub fn shuffle<T>(&mut self, vec: &mut [T]) {
        for i in (1..vec.len()).rev() {
            let j = self.randbelow((i + 1) as u32) as usize;
            vec.swap(i, j);
        }
    }

    pub fn choices(&mut self, weights: &[f32], k: usize) -> Vec<usize> {
        if weights.is_empty() { return vec![]; }

        let mut cum_weights = Vec::with_capacity(weights.len());
        let mut acc = 0.0_f64;
        for &w in weights {
            acc += w as f64;
            cum_weights.push(acc);
        }

        let total = *cum_weights.last().unwrap();
        let mut res = Vec::with_capacity(k);

        for _ in 0..k {
            let choice = self.random() * total;
            // binary search for the first element >= choice
            let index = match cum_weights.binary_search_by(|w| {
                if *w < choice { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater }
            }) {
                Ok(i) => i,
                Err(i) => i,
            };
            res.push(index.min(weights.len() - 1));
        }
        res
    }
}
