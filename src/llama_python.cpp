#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <llama.hpp>
#include <memory>
namespace py = pybind11;

PYBIND11_MODULE(libllama_engine, m) {
  py::enum_<llama::DeviceType>(m, "DeviceType")
      .value("CPU", llama::DeviceType::CPU)
      .value("CUDA", llama::DeviceType::CUDA);

  py::class_<llama::LlamaConfig>(m, "LlamaConfig")
      .def(py::init<>())
      .def_readwrite("checkpoint_path", &llama::LlamaConfig::checkpoint_path)
      .def_readwrite("tokenizer_path", &llama::LlamaConfig::tokenizer_path)
      .def_readwrite("device_type", &llama::LlamaConfig::device_type);

  // For llama::RunConfig
  py::class_<llama::RunConfig>(m, "RunConfig")
      .def(py::init<>())
      .def_readwrite("temperature", &llama::RunConfig::temperature)
      .def_readwrite("topp", &llama::RunConfig::topp)
      .def_readwrite("rng_seed", &llama::RunConfig::rng_seed);

  py::class_<llama::Llama2<float>>(m, "Llama2FP32")
      .def(py::init<const llama::LlamaConfig&>())
      .def("Generate", &llama::Llama2<float>::Generate);

  py::class_<llama::Llama3<float>>(m, "Llama3FP32")
      .def(py::init<const llama::LlamaConfig&>())
      .def("Generate", &llama::Llama3<float>::Generate);
}