// Model building — placeholder module
// C FFI bindings to system_architecture_definitions.h (planned)

pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: vec![0.0f32; in_features * out_features],
            bias: vec![0.0f32; out_features],
            in_features,
            out_features,
        }
    }
}
