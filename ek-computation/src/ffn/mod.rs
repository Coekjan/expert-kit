use ek_base::error::EKResult;
use expert_ort::OnnxFFN;
use expert_torch::TorchFFN;
use meta::{Expert, ExpertWeight};
use safetensors::tensor::TensorView;
use tracing::instrument;

use crate::{
    backend::{Device, EkTensor, torch::TchTensor},
    ffn::expert_torch::stream,
    x::{self},
};

#[allow(dead_code)]
pub mod expert_ort;
pub mod expert_torch;
pub mod meta;

pub enum ExpertBackend {
    Torch(TorchFFN),
    OnnxF32(OnnxFFN<f32>),
}

impl ExpertBackend {
    pub async fn build<'a>(
        instance: x::EKInstance,
        tensor: &'a safetensors::SafeTensors<'a>,
    ) -> EKResult<ExpertBackend> {
        let backend = match instance.backend {
            x::ExpertBackendType::Torch => {
                let weight = ExpertWeight::<TchTensor>::from_safetensor(tensor, instance.device)?;
                ExpertBackend::Torch(TorchFFN::construct(instance, weight)?)
            }
            x::ExpertBackendType::Onnx => todo!(),
        };
        Ok(backend)
    }
}

impl ExpertBackend {
    #[instrument(skip(self, view))]
    pub fn forward(&self, view: &TensorView) -> EKResult<TchTensor> {
        match self {
            ExpertBackend::Torch(exp) if matches!(exp.device(), Device::CUDA(_)) => {
                let stream = stream::TorchStream::new(exp.device().into());
                let _guard = stream.guard();
                let inp = TchTensor::from_tensor_view(view);
                let shape = inp.inner().size();
                let inp = inp.to_device(exp.device());
                log::debug!("input shape {shape:?}");
                assert!(shape.len() == 2);
                Ok(exp.forward(&inp))
            }
            ExpertBackend::Torch(exp) => {
                let inp = TchTensor::from_tensor_view(view);
                let shape = inp.inner().size();
                let inp = inp.to_device(exp.device());
                log::debug!("input shape {shape:?}");
                assert!(shape.len() == 2);
                Ok(exp.forward(&inp))
            }
            ExpertBackend::OnnxF32(_exp) => {
                todo!()
            }
        }
    }
}
