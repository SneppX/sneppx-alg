// Tensor operations — placeholder module
// C FFI bindings to multidimensional_tensor_engine.h (planned)

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0f32; size],
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![1.0f32; size],
        }
    }
}
