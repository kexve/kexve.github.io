---
layout: post
title: TorchScript探索
categories: 编译原理
---

## 简介  

TorchScript是PyTorch模型（nn.Module的子类）的**中间表示**，可以在高性能环境（例如C ++）中运行。  

> TorchScript是PyTorch模型的一种表示方法，可以被TorchScript编译器理解、编译和序列化。从根本上说，TorchScript本身就是一种编程语言。它是使用PyTorch API的Python的一个子集。用于TorchScript的C++接口包括三个主要功能：  
1 用于加载和执行Python中定义的序列化TorchScript模型的机制。  
2 用于定义扩展TorchScript标准操作库的自定义操作符的API。  
3 从C++中对TorchScript程序进行及时的编译。  
如果你想尽可能地用Python定义你的模型，但随后将它们输出到C++中用于生产环境和无Python推理，那么第一个机制可能会引起你的极大兴趣。你可以通过[这个链接](https://pytorch.org/tutorials/advanced/cpp_export.html)了解更多信息。第二个API关注的是你想用自定义操作符扩展TorchScript的情况，这些操作符同样可以被序列化并在推理过程中从C++调用。最后，torch::jit::compile函数可用于直接从C++访问TorchScript编译器。  

TorchScript软件栈包括两部分：TorchScript（python）和LibTorch（C++）。TorchScript负责将Python代码转成一个中间表示，LibTorch负责解析运行这个中间表示。  

## 保存模型，生成中间表示

对应编译器的前端（语法分析、类型检查、中间代码生成）。

TorchScript保存模型有两种模式：trace模式和script模式。  

### Tracing

跟踪模型的执行，然后将其路径记录下来。在使用trace模式时，需要构造一个符合要求的输入，然后使用TorchScript tracer运行一遍。每执行一个算子，就会往当前的graph中加入一个node。PyTorch导出ONNX也是使用了这部分代码，所以理论上能够导出ONNX的模型也能够使用trace模式导出torch模型。

trace 模式有比较大的限制：

1. 不能有 if-else 等控制流
2. 只支持 Tensor 操作

通过上述对实现方式的解释，很容易理解为什么有这种限制：1. 跟踪出的 graph 是静态的，如果有控制流，那么记录下来的只是当时生成模型时走的那条路；2. 追踪代码是跟 Tensor 算子绑定在一起的，如果是非 Tensor 的操作，是无法被记录的。

通过 trace 模式的特点可以看出，trace 模式通常用于深度模型的导出（深度模型通常没有 if-else 控制流且没有非 Tensor 操作）。

### Scripting

TorchScript 实现了一个完整的编译器以支持 script 模式。保存模型阶段对应编译器的前端（语法分析、类型检查、中间代码生成）。在保存模型时，TorchScript 编译器解析 Python 代码，并构建代码的 AST（抽象语法树）。

script 模式在的限制比较小，不仅支持 if-else 等控制流，还支持非 Tensor 操作，如 List、Tuple、Map 等容器操作。

## 运行模型

对应编译器后端（代码优化、目标代码生成、目标代码优化）。

LibTorch 还实现了一个可以运行该编译器所生成代码的解释器。在运行代码时，在 LibTorch 中，AST 被加载，在进行一系列代码优化后生成目标代码（并非机器码），然后由解释器运行。

## 使用

### Trace模式

对于只有Tensor操作的模型，比较适合使用trace模式：

``` python
class Module_0(torch.nn.Module):
    def __init__(self, N, M):
        super(Module_0, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))
        self.linear = torch.nn.Linear(N, M)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.weight.mm(input)
        output = self.linear(output)
        return output

scripted_module = torch.jit.trace(Module_0(2, 3).eval(), (torch.zeros(3, 2)))
scripted_module.save("Module_0.pt")

print(scripted_module.graph)
# graph(%self.1 : __torch__.Module_0,
#       %input.1 : Float(3, 2, strides=[2, 1], requires_grad=0, device=cpu)):
#   %linear : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear"](%self.1)
#   %weight.1 : Tensor = prim::GetAttr[name="weight"](%self.1)
#   %input : Float(2, 2, strides=[2, 1], requires_grad=1, device=cpu) = aten::mm(%weight.1, %input.1) # /mnt/petrelfs/yukexue/ykx_model_test/trace_test.py:10:0
#   %18 : Tensor = prim::CallMethod[name="forward"](%linear, %input)
#   return (%18)

print(scripted_module.code)
# def forward(self,
#     input: Tensor) -> Tensor:
#   linear = self.linear
#   weight = self.weight
#   input0 = torch.mm(weight, input)
#   return (linear).forward(input0, )
```

### Script模式

对于下面这种存在控制流和非 Tensor 操作的模型，比较适合使用 script 模式：

``` python
class Module_1(torch.nn.Module):
    def __init__(self, N, M):
        super(Module_1, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))
        self.linear = torch.nn.Linear(N, M)

    def forward(self, input: torch.Tensor, do_linear: bool) -> torch.Tensor:
        output = self.weight.mm(input)
        if do_linear:
            output = self.linear(output)
        return output

scripted_module = torch.jit.script(Module_1(3, 3).eval())
scripted_module.save("Module_1.pt")

print(scripted_module.graph)
# graph(%self : __torch__.Module_1,
#       %input.1 : Tensor,
#       %do_linear.1 : bool):
#   %weight : Tensor = prim::GetAttr[name="weight"](%self)
#   %output.1 : Tensor = aten::mm(%weight, %input.1) # /mnt/petrelfs/yukexue/ykx_model_test/trace_test.py:22:17
#   %output : Tensor = prim::If(%do_linear.1) # /mnt/petrelfs/yukexue/ykx_model_test/trace_test.py:23:8
#     block0():
#       %linear : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear"](%self)
#       %output.5 : Tensor = prim::CallMethod[name="forward"](%linear, %output.1) # /mnt/petrelfs/yukexue/ykx_model_test/trace_test.py:24:21
#       -> (%output.5)
#     block1():
#       -> (%output.1)
#   return (%output)

print(scripted_module.code)
# def forward(self,
#     input: Tensor,
#     do_linear: bool) -> Tensor:
#   weight = self.weight
#   output = torch.mm(weight, input)
#   if do_linear:
#     linear = self.linear
#     output0 = (linear).forward(output, )
#   else:
#     output0 = output
#   return output0
```

### 混合模式

trace 模式和 script 模式各有千秋也各有局限，在使用时将两种模式结合在一起使用可以最大化发挥 TorchScript 的优势。例如，一个 module 包含控制流，同时也包含一个只有 Tensor 操作的子模型。这种情况下当然可以直接使用 script 模式，但是 script 模式需要对部分变量进行类型标注，比较繁琐。这种情况下就可以仅对上述子模型进行 trace，整体再进行 script：

``` python
class Module_2(torch.nn.Module):
    def __init__(self, N, M):
        super(Module_2, self).__init__()
        self.linear = torch.nn.Linear(N, M)
        self.sub_module = torch.jit.trace(Module_0(2, 3).eval(), (torch.zeros(3, 2)))

    def forward(self, input: torch.Tensor, do_linear: bool) -> torch.Tensor:
        output = self.sub_module(input)
        if do_linear:
            output = self.linear(output)
        return output


scripted_module = torch.jit.script(Module_2(2, 3).eval())
```

### C++运行模型

针对上面模型导出的例子，C++ 中加载使用的方式如下：

``` python
#include <torch/script.h>

int main() {
  // load module
  torch::jit::script::Module torch_module;
  try {
    torch_module = torch::jit::load("my_module.pt");
  } catch (const c10::Error& e) {
    std::cerr << "error loading the module" << std::endl;
    return -1;
  }

  // make inputs
  std::vector<float> vec(9);
  std::vector<torch::jit::IValue> torch_inputs;
  torch::Tensor torch_tensor =
      torch::from_blob(vec.data(), {3, 3}, torch::kFloat32);
  torch_inputs.emplace_back(torch_tensor);
  torch_inputs.emplace_back(false);

  // run module
  torch::jit::IValue torch_outputs;
  try {
    torch_outputs = torch_module.forward(torch_inputs);
  } catch (const c10::Error& e) {
    std::cerr << "error running the module" << std::endl;
    return -1;
  }

  auto outputs_tensor = torch_outputs.toTensor();
}
```

## TorchScript的语法限制

1. 支持的类型有限（包括Tensor、Tuple[T0,T1,...,TN]、bool、int、float、str、List[T]、Optional[T]、Dict[K,V]），指在运行（而非初始化）过程中使用的对象或者函数参数  
    - 这其中不包括 set 数据类型，这意味着需要使用 set 的地方就要通过其他的方式绕过，比如先用 list 然后去重

    - 使用 tuple 时需要声明其中的类型，例如 Tuple[int, int, int]，这也就意味着 tuple 在运行时长度不能变化，所以要使用 list 代替

    - 创建字典时，只有 int、float、comple、string、torch.Tensor 可以作为 key

2. 不支持 lambda 函数，但是可以通过自定义排序类的方式实现，略微麻烦，但是可以解决

3. 因为 TorchScript 是静态类型语言，运行时不能变换变量类型

4. 因为编码问题，所以对中文字符串进行遍历时会抛异常，所以尽量不要处理中文，如果需要处理中文，则需要将中文切分成字符粒度后再送入模型中进行处理
