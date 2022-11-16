---
layout: post
title: TorchScript 探索
categories: [编译原理, pytorch]
---

## 简介

TorchScript 是 PyTorch 模型（nn.Module 的子类）的**中间表示**，可以在高性能环境（例如 C ++）中运行。

> TorchScript 是 PyTorch 模型的一种表示方法，可以被 TorchScript 编译器理解、编译和序列化。从根本上说，TorchScript 本身就是一种编程语言。它是使用 PyTorch API 的 Python 的一个子集。用于 TorchScript 的 C++接口包括三个主要功能：  
> 1 用于加载和执行 Python 中定义的序列化 TorchScript 模型的机制。  
> 2 用于定义扩展 TorchScript 标准操作库的自定义操作符的 API。  
> 3 从 C++中对 TorchScript 程序进行及时的编译。  
> 如果你想尽可能地用 Python 定义你的模型，但随后将它们输出到 C++中用于生产环境和无 Python 推理，那么第一个机制可能会引起你的极大兴趣。你可以通过[这个链接](https://pytorch.org/tutorials/advanced/cpp_export.html)了解更多信息。第二个 API 关注的是你想用自定义操作符扩展 TorchScript 的情况，这些操作符同样可以被序列化并在推理过程中从 C++调用。最后，torch::jit::compile 函数可用于直接从 C++访问 TorchScript 编译器。

TorchScript 软件栈包括两部分：TorchScript（python）和 LibTorch（C++）。TorchScript 负责将 Python 代码转成一个中间表示，LibTorch 负责解析运行这个中间表示。

## 保存模型，生成中间表示

对应编译器的前端（语法分析、类型检查、中间代码生成）。

TorchScript 保存模型有两种模式：trace 模式和 script 模式。

### Tracing

跟踪模型的执行，然后将其路径记录下来。在使用 trace 模式时，需要构造一个符合要求的输入，然后使用 TorchScript tracer 运行一遍。每执行一个算子，就会往当前的 graph 中加入一个 node。PyTorch 导出 ONNX 也是使用了这部分代码，所以理论上能够导出 ONNX 的模型也能够使用 trace 模式导出 torch 模型。

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

### Trace 模式

对于只有 Tensor 操作的模型，比较适合使用 trace 模式：

```python
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

### Script 模式

对于下面这种存在控制流和非 Tensor 操作的模型，比较适合使用 script 模式：

```python
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

```python
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

```python
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

## TorchScript 的语法限制

1. 支持的类型有限（包括 Tensor、Tuple[T0,T1,...,TN]、bool、int、float、str、List[T]、Optional[T]、Dict[K,V]），指在运行（而非初始化）过程中使用的对象或者函数参数

   - 这其中不包括 set 数据类型，这意味着需要使用 set 的地方就要通过其他的方式绕过，比如先用 list 然后去重

   - 使用 tuple 时需要声明其中的类型，例如 Tuple[int, int, int]，这也就意味着 tuple 在运行时长度不能变化，所以要使用 list 代替

   - 创建字典时，只有 int、float、comple、string、torch.Tensor 可以作为 key

2. 不支持 lambda 函数，但是可以通过自定义排序类的方式实现，略微麻烦，但是可以解决

3. 因为 TorchScript 是静态类型语言，运行时不能变换变量类型

4. 因为编码问题，所以对中文字符串进行遍历时会抛异常，所以尽量不要处理中文，如果需要处理中文，则需要将中文切分成字符粒度后再送入模型中进行处理

## Pytorch 源码

### IR（TorchScript）的基本表示

Pytorch 中的设计（parameter，计算节点等）在 torchScript 中的对应：

| 名称           | source code       | 简介                                                                                            |
| -------------- | ----------------- | ----------------------------------------------------------------------------------------------- |
| Modules        | module.h          | 对标 nn.Module                                                                                  |
| Parameters     | module.h          | 对标 PyTorch 的 parameter                                                                       |
| Method         | Method.h          | 包括 FunctionSchema 方法描述，Graph 实际计算图，GraphExecutor do the optimization and execution |
| FunctionSchema | function_schema.h | 描述参数与返回类型                                                                              |
| Graph          | ir.h              | 定义 function 的具体实现，包括 Nodes，Blocks，Values                                            |
| Nodes          | ir.h              | 一个指令，如一次卷积运算，一次矩阵运算                                                          |
| Block          | ir.h              | 控制语句 if，loop + list of nodes                                                               |

还有 with,Value,Type 等

### trace 实现

```python
# pytorch/torch/jit/_trace.py

def trace(
    func,
    example_inputs=None,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
    example_kwarg_inputs=None
):
    # 省略很多代码
    # 。。。

    # 发现是nn.Module instacene forward, 追踪forward
    if isinstance(func, torch.nn.Module):
        if example_inputs is None:
            if isinstance(example_kwarg_inputs, dict):
                example_inputs = example_kwarg_inputs
            else:
                raise RuntimeError("example_kwarg_inputs should be a dict")
        return trace_module(
            func,
            {"forward": example_inputs},
            None,
            check_trace,
            wrap_check_inputs(check_inputs),
            check_tolerance,
            strict,
            _force_outplace,
            _module_class,
            example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
        )

    # 传进来的是某个module instance的 forward
    if (
        hasattr(func, "__self__")
        and isinstance(func.__self__, torch.nn.Module)
        and func.__name__ == "forward"
    ):
        if example_inputs is None:
            if isinstance(example_kwarg_inputs, dict):
                example_inputs = example_kwarg_inputs
            else:
                raise RuntimeError("example_kwarg_inputs should be a dict")
        return trace_module(
            func.__self__,
            {"forward": example_inputs},
            None,
            check_trace,
            wrap_check_inputs(check_inputs),
            check_tolerance,
            strict,
            _force_outplace,
            _module_class,
            example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
        )

    # 。。。

    # 一个查找变量名的接口
    var_lookup_fn = _create_interpreter_name_lookup_fn(0)

    # 。。。

    name = _qualified_name(func)
    if isinstance(example_kwarg_inputs, dict):
        example_inputs = example_kwarg_inputs
        # C++ 入口
        traced = torch._C._create_function_from_trace_with_dict(
            name,
            func,
            example_kwarg_inputs,
            var_lookup_fn,
            strict,
            _force_outplace,
            get_callable_argument_names(func)
        )
    else:
        # C++ 入口
        traced = torch._C._create_function_from_trace(
            name,
            func,
            example_inputs,
            var_lookup_fn,
            strict,
            _force_outplace,
            get_callable_argument_names(func)
        )

    # 检查traced 与 原func是否有差异
    if check_trace:
        if check_inputs is not None:
            _check_trace(
                check_inputs,
                func,
                traced,
                check_tolerance,
                strict,
                _force_outplace,
                False,
                _module_class,
                example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
            )
        else:
            _check_trace(
                [example_inputs],
                func,
                traced,
                check_tolerance,
                strict,
                _force_outplace,
                False,
                _module_class,
                example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
            )

    return traced
```

去 C++ 中看下发生了什么

```c++
// pytorch/torch/csrc/jit/frontend/tracer.cpp

std::pair<std::shared_ptr<TracingState>, Stack> trace(
    Stack inputs,
    const std::function<Stack(Stack)>& traced_fn,
    std::function<std::string(const Variable&)> var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self,
    const std::vector<std::string>& argument_names) {
  try {

    // 。。。

    auto state = std::make_shared<TracingState>();

    // setTracingState 将state 这个实例set下来，在之后计算节点get出来insert计算过程
    setTracingState(state);

    // state这个数据结构会在forward过程中存储trace到的计算过程
    if (self) {
      Value* self_value = state->graph->insertInput(0, "self")->setType(
          self->_ivalue()->type());
      gatherParametersAndBuffers(state, self_value, *self, {"__module"});
    }

    // 。。。
    } else {
      for (IValue& input : inputs) {
        input = addInput(state, input, input.type(), state->graph->addInput());
      }
    }

    auto graph = state->graph;

    // 将python中的变量名解析函数绑定下来
    getTracingState()->lookup_var_name_fn = std::move(var_name_lookup_fn);
    getTracingState()->strict = strict;
    getTracingState()->force_outplace = force_outplace;

    // 开始forward，在计算发生时，会把计算记录到state中
    auto out_stack = traced_fn(inputs);

    // Exit a trace, treating 'out_stack' as the outputs of the trace.  These
    // are the variables whose values will be computed upon subsequent
    // invocations of the trace.
    size_t i = 0;
    for (auto& output : out_stack) {
      // NB: The stack is in "reverse" order, so when we pass the diagnostic
      // number we need to flip it based on size.
      state->graph->registerOutput(
          state->getOutput(output, out_stack.size() - i));
      i++;
    }
    setTracingState(nullptr);

    if (getInlineEverythingMode()) {
      Inline(*graph);
    }
    FixupTraceScopeBlocks(graph, self);
    NormalizeOps(graph);
    return {state, out_stack};
  } catch (...) {
    tracer::abandon();
    throw;
  }
}
```

那么具体记录 operation 的过程发生在哪里呢？

```c++
    // 这个好像在新版本中改变了
    // pytorch/torch/csrc/jit/runtime/register_c10_ops.cpp
    Operator createOperatorFromC10_withTracingHandledHere(
        const c10::OperatorHandle& op) {
        return Operator(op, [op](Stack& stack) {
        const auto input_size = op.schema().arguments().size();
        const auto output_size = op.schema().returns().size();

        Node* node = nullptr;
        std::shared_ptr<jit::tracer::TracingState> tracer_state;

        // trace the input before unwrapping, otherwise we may lose
        // the input information
        if (jit::tracer::isTracing()) {
            // 获取 tracer_state
            tracer_state = jit::tracer::getTracingState();
            auto symbol = Symbol::fromQualString(op.schema().name());
            const auto& graph = tracer::getTracingState()->graph;
            node = graph->create(symbol, 0);
            tracer::recordSourceLocation(node);
            const auto& args = op.schema().arguments();
            int i = 0;
            // # 记录args
            for (auto iter = stack.end() - input_size; iter != stack.end();
                ++iter, ++i) {
            // TODO we need to refactor graph APIs (e.g., addInputs)
            // appropriately; after that, we can get rid of the giant if-else
            // block we will clean this tech debt together in the following PRs
            auto type = args[i].type();
            if (type->kind() == TypeKind::OptionalType) {
              if (iter->isNone()) {
                Value* none = graph->insertNode(graph->createNone())->output();
                node->addInput(none);
                continue;
              } else {
                type = type->expect<OptionalType>()->getElementType();
              }
            }
            if (type->isSubtypeOf(TensorType::get())) {
              AT_ASSERT(iter->isTensor());
              tracer::addInputs(node, args[i].name().c_str(), iter->toTensor());
            } else if (type->kind() == TypeKind::FloatType) {
              AT_ASSERT(iter->isDouble());
              tracer::addInputs(node, args[i].name().c_str(), iter->toDouble());
            } else if (type->kind() == TypeKind::IntType) {
              AT_ASSERT(iter->isInt());
              tracer::addInputs(node, args[i].name().c_str(), iter->toInt());
            } else if (type->kind() == TypeKind::BoolType) {
              AT_ASSERT(iter->isBool());
              tracer::addInputs(node, args[i].name().c_str(), iter->toBool());
            } else if (type->kind() == TypeKind::StringType) {
              AT_ASSERT(iter->isString());
              tracer::addInputs(node, args[i].name().c_str(), iter->toStringRef());
            } else if (type->kind() == TypeKind::NumberType) {
              tracer::addInputs(node, args[i].name().c_str(), iter->toScalar());
            } else if (type->kind() == TypeKind::ListType) {
              const auto& elem_type = type->expect<ListType>()->getElementType();
              if (elem_type->isSubtypeOf(TensorType::get())) {
                AT_ASSERT(iter->isTensorList());
                auto list = iter->toTensorVector();
                tracer::addInputs(node, args[i].name().c_str(), list);
              } else if (elem_type->kind() == TypeKind::FloatType) {
                AT_ASSERT(iter->isDoubleList());
                // NB: now, tracer doesn't support tracing double list. We add
                // special handling here, since in our case, we assume that all the
                // doubles in the list are constants
                auto value = iter->toDoubleVector();
                std::vector<Value*> info(value.size());
                for (size_t value_index = 0; value_index < value.size();
                     ++value_index) {
                  info[value_index] = graph->insertConstant(value[value_index]);
                  tracer::recordSourceLocation(info[value_index]->node());
                }
                node->addInput(
                    graph
                        ->insertNode(graph->createList(jit::FloatType::get(), info))
                        ->output());
              } else if (elem_type->kind() == TypeKind::IntType) {
                AT_ASSERT(iter->isIntList());
                tracer::addInputs(
                    node, args[i].name().c_str(), iter->toIntVector());
              } else if (elem_type->kind() == TypeKind::BoolType) {
                AT_ASSERT(iter->isBoolList());
                tracer::addInputs(
                    node, args[i].name().c_str(), iter->toBoolList().vec());
              } else {
                throw std::runtime_error(
                    "unsupported input list type: " + elem_type->str());
              }
            } else if (iter->isObject()) {
              tracer::addInputs(node, args[i].name().c_str(), iter->toObject());
            } else {
              throw std::runtime_error("unsupported input type: " + type->str());
            }
          }
            // node嵌入graph
            graph->insertNode(node);

          jit::tracer::setTracingState(nullptr);
        }
```

`这个好像在新版本中改变了`。可以看到，在具体运算发生时，会使用 getTracingState() 得到 forward 开始去创建的 state，然后看到根据 op.schema().name() 得到计算类型（比如相加），根据计算类型通过 createNone 方法创建一个计算节点，然后创建计算输入，最后把计算 node insert 到 graph 中，完成一次对计算的记录。

### Script 实现

因为 script 得到 IR 的方式是解析源码，因此对于不同的代码形式会略有不同(函数，class，nn.Module 的 instance)：

```python
# pytorch/torch/jit/_script.py

def script(obj, optimize=None, _frames_up=0, _rcb=None,
           example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None):

        # 。。。

        # 检查重载
        if hasattr(obj, "__script_if_tracing_wrapper"):
            obj = obj.__original_fn  # type: ignore[union-attr]
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
        # 。。。

        # 检查重载
        _check_directly_compile_overloaded(obj)

        # 是否之前被script过了
        maybe_already_compiled_fn = _try_get_jit_cached_function(obj)
        if maybe_already_compiled_fn:
            return maybe_already_compiled_fn
        # 得到ast语法树
        ast = get_jit_def(obj, obj.__name__)
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
        #c++ 入口,根据ast得到ir
        fn = torch._C._jit_script_compile(
            qualified_name, ast, _rcb, get_default_args(obj)
        )
        # Forward docstrings
        fn.__doc__ = obj.__doc__
        # cache起来
        _set_jit_function_cache(obj, fn)
        return fn
    # 。。。

```

我们看下 get_jit_def 是如何得到 jit 规定的 ast 语法树的

```python
# pytorch/torch/jit/frontend.py
def get_jit_def(fn, def_name, self_name=None):

    # 得到源代码的一些信息
    sourcelines, file_lineno, filename = get_source_lines_and_file(fn, torch._C.ErrorReport.call_stack())
    sourcelines = normalize_source_lines(sourcelines)
    source =  dedent_src ''.join(sourcelines)
    # dedent_src 为包含了要script函数的字符串
    dedent_src = dedent(source)
    # 调用python ast包将字符串解析为Python的ast
    py_ast = ast.parse(dedent_src)

    # 得到python类型注释
    type_line = torch.jit.annotations.get_type_line(source)
    #ctx中包含了函数所有原信息
    ctx = SourceContext(source, filename, file_lineno, leading_whitespace_len, True)
    fn_def = py_ast.body[0]

    # build_def将python 的ast 转化为torchjit 使用的ast格式
    return build_def(ctx, fn_def, type_line, def_name, self_name=self_name)
```

解释下 py_ast.body[0] 是什么

```python
import ast
... func_def= \
... """def test(a):
...     a = a + 2
...     return a + 1"""
... results = ast.parse(func_def)
```

![20221104172640](https://cdn.jsdelivr.net/gh/kexve/img@main/image_blog20221104172640.png)

可见，ast.body 是一个 list，其长度等于解析的 string 中包含的函数的个数，我们看第一个元素，其中 value 是一个 Binop 具体为一个 Add，left 是 Name 类型，id 为 a，right 是 Num，也就是 2，这个 Binop 即解析的 a = a + 2。

因为我们 get_source_lines_and_file 返回的一定是一个 single top-level function， 因此我们直接取用第 0 个元素，即 py_ast.body[0] 就可以了。

接下来看 build_def 是如何将 Python 的 ast 转化为自己需要的 ast 的。

```python
# pytorch/torch/jit/frontend.py

def build_def(ctx, py_def, type_line, def_name, self_name=None, pdt_arg_types=None):
    # 。。。

    return Def(Ident(r, def_name),
               decl,
               build_stmts(ctx, body))
```

因为 ctx 包含 source code 所有信息, body 是 Python ast 解析结果,那么 build_stmts 中应该包含我们想要的答案。

```python
# pytorch/torch/jit/frontend.py

    from torch._C._jit_tree_views import (
        ClassDef, Ident, Stmt, Decl, Def, Var,
        EmptyTypeAnnotation, Param, ExprStmt, Assign,
        Delete, Return, Raise, Assert, AugAssign, While,
        For, If, Pass, Break, Continue, Apply, Dots, Select,
        TrueLiteral, FalseLiteral, NoneLiteral, Starred,
        ListLiteral, TupleLiteral, DictLiteral, Const,
        StringLiteral, ListComp, Attribute, BinOp, UnaryOp,
        SliceExpr, Subscript, TernaryIf, With, WithItem, Property,
        DictComp,
    )
    # jit中定义的ast基本结构

    def build_stmts(ctx, stmts):
        #发现其调用了`build_stmt`
        stmts = [build_stmt(ctx, s) for s in stmts]
        return list(filter(None, stmts))

    #`build_stmt` 是一个StmtBuilder()的instance
    build_stmt = StmtBuilder()
    build_expr = ExprBuilder()

    class Builder(object):
        def __call__(self, ctx, node):
            # 可见会根据解析出的ast的类型返回相应的build方法，从截图可以看到`a+2`是一个`Assign`类型
            # 因此会调用build_Assign
            method = getattr(self, 'build_' + node.__class__.__name__, None)
            if method is None:
                raise UnsupportedNodeError(ctx, node)
            return method(ctx, node)

    class StmtBuilder(Builder):
        @staticmethod
        def build_Assign(ctx, stmt):
            # 截图可以看到stmt.value是一个Binop
            # build_expr是ExprBuilder的INSTANCE，其会调用`build_BinOp`
            rhs = build_expr(ctx, stmt.value)
            lhs = [build_expr(ctx, x) for x in stmt.targets]
            return Assign(lhs, rhs)

        @staticmethod
        def build_Expr(ctx, stmt):
            # Binop
            value = stmt.value
            if value.__class__.__name__ == 'Str':
                # If a statement is a string literal expression,
                # then it is a docstring. Just ignore it.
                return None
            else:
                return ExprStmt(build_expr(ctx, value))

     class ExprBuilder(Builder):
            binop_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Pow: '**',
            ast.Mod: '%',
            ast.FloorDiv: '//',
            ast.BitAnd: '&',
            ast.BitXor: '^',
            ast.BitOr: '|',
            ast.LShift: '<<',
            ast.RShift: '>>',
        }
            @staticmethod
        def build_BinOp(ctx, expr):
            #expr.left是个`Name`调用build_Name
            lhs = build_expr(ctx, expr.left)
            rhs = build_expr(ctx, expr.right)
            op = type(expr.op)
            # 转化为约定的代表运算类型的string 符号
            op_token = ExprBuilder.binop_map.get(op)
            return BinOp(op_token, lhs, rhs)
```

最终转化为的格式，类似于 S-expression.

```
    (def
      (ident test)
      (decl
        (list
          (param
            (ident a)
            (option)
            (option)
            (False)))
        (option))
      (list
        (assign
          (list (variable (ident a)))
          (option
            (+
              (variable (ident a))
              (const 2)))
          (option))
        (return
          (+
            (variable (ident a))
            (const 1)))))
```

好的，我们已经得到得到 jit 约定的 AST 树了，接下来我们要进入 torch.\_C.\_jit_script_compile 查看如何将这样的 ast 树转化为 IR.

```c++
// pytorch/torch/csrc/jit/python/script_init.cpp

    static StrongFunctionPtr script_compile_function(
        const c10::QualifiedName& name,
        const Def& def,
        const FunctionDefaults& defaults,
        const ResolutionCallback& rcb) {
       #  def 中包含ast，跟着它就能找到答案
      auto cu = get_python_cu();
      #看来是get_python_cu这个类中的define函数完成的
      auto defined_functions = cu->define(
          QualifiedName(name.prefix()),
          /*properties=*/{},
          /*propResolvers=*/{},
          {def},
          {pythonResolver(rcb)},
          nullptr,
          true);
      TORCH_INTERNAL_ASSERT(defined_functions.size() == 1);
      auto& defined = defined_functions[0];
      defined->setSchema(getSchemaWithNameAndDefaults(
          def.range(), defined->getSchema(), def.name().name(), defaults));
      StrongFunctionPtr ret(std::move(cu), defined);
      didFinishEmitFunction(ret);
      return ret;
    }
    # 发现只是wapper了下CompilationUnit
    inline std::shared_ptr<CompilationUnit> get_python_cu() {
      return py::module::import("torch.jit._state")
          .attr("_python_cu")
          .cast<std::shared_ptr<CompilationUnit>>();
    }

    #关于compilation_unit
    #/torch/csrc/jit/api/compilation_unit.h
     // for historic reasons, these are defined in ir_emitter.cpp
     // Returns the list of Functions just defined.
      std::vector<Function*> define(
          const c10::optional<c10::QualifiedName>& prefix,
          const std::vector<Property>& properties,
          const std::vector<ResolverPtr>& propResolvers,
          const std::vector<Def>& definitions,
          const std::vector<ResolverPtr>&
              defResolvers, /* determines how we handle free
                         variables in each definition*/
          // if non-null, the first argument to each def, is bound to this value
          const Self* self,
          // see [name mangling]
          bool shouldMangle = false);
    #实现在torch/csrc/jit/frontend/ir_emitter.cpp
    std::unique_ptr<Function> CompilationUnit::define(
        const c10::optional<QualifiedName>& prefix,
        const Def& def,
        const ResolverPtr& resolver,
        const Self* self,
        const std::unordered_map<std::string, Function*>& function_table,
        bool shouldMangle) const {

      auto _resolver = resolver;
      .....
      auto creator = [def, _resolver, self](Function& method) {
        ....
        ##核心代码to_ir
        to_ir(def, _resolver, self, method);
      };

      auto fn = torch::make_unique<GraphFunction>(
          std::move(name), std::make_shared<Graph>(), creator);
      return fn;
    }
```

我们跟随 def，找到了一个转化为 IR 的关键的 **struct** to_ir，其输入中有 def，也就是 ast，\_resolver 是 Python 中传过来的解析名字的函数，我们可以在内部找到关键部分

```c++
// pytorch/torch/csrc/jit/frontend/ir_emitter.cpp

    to_ir(
          const Def& def,
          ResolverPtr resolver_,
          const Self* self,
          Function& method) // method being constructed
          : method(method),
            graph(method.graph()),
            resolver(std::move(resolver_)),
            typeParser_(resolver),
            environment_stack(nullptr) {
        AT_ASSERT(resolver);
        pushFrame(graph->block(), /*starts_def=*/true);

        #emitDef 中会调用emitStatements
        method.setSchema(emitDef(def, self, graph->block()));
        ConvertToSSA(graph);
        CanonicalizeModifiedLoops(graph);
        NormalizeOps(graph);
        runCleanupPasses(graph);
      }
    private:
    //  #在to_ir 的private中我们可以看到Graph Function这些我们之前介绍的IR的组成部分
      Function& method;
      std::shared_ptr<Graph> graph;
      ResolverPtr resolver;
      std::unordered_map<int64_t, Value*> integral_constants;

    //  #emitDef 中会调用emitStatements
     FunctionSchema emitDef(const Def& def, const Self* self, Block* block) {
        ......
        // body
        auto stmts_list = def.statements();
        emitStatements(stmts_list.begin(), stmts_list.end());
         ........
      }
     void emitStatements(
          List<Stmt>::const_iterator begin,
          List<Stmt>::const_iterator end) {
        for (; begin != end; ++begin) {
          auto stmt = *begin;
          ErrorReport::CallStack::update_pending_range(stmt.range());
          switch (stmt.kind()) {
            case TK_IF:
              emitIf(If(stmt));
              break;
            case TK_WHILE:
              emitWhile(While(stmt));
              break;
            case TK_FOR:
              emitFor(For(stmt));
              break;
            case TK_ASSIGN:
              emitAssignment(Assign(stmt));
           .................
              break;
            default:
              throw ErrorReport(stmt)
                  << "Unrecognized statement kind " << kindToString(stmt.kind());
          }
          // Found an exit statement in this block. The remaining statements aren't
          // reachable so we don't emit them.
          if (exit_blocks.count(environment_stack->block()))
            return;
        }
      }


// 我们可以看到根据stmt.kind(),会进入而各种emit里面，其中一定可以找到
// graph->insertNode(graph->create(.....));
// 类似的操作，对应我们建立IR graph
```

以上是我们 script 一个 function 为例子。**怎么 script 一个 module？** 因为有一些变量的指代，是需要初始化后才知道的，同时，我们希望 script 完的 module 对外还能保持一样的接口，即可以正常访问原有 module 的属性，那么应该怎么做呢？在 module 原有的 init 结束后随即开始完整的 script forward 函数，替换涉及到的所有函数为 script 后的函数如何正常访问原有的属性。

如何在一个类的 init 函数后面绑定行为呢，我们想到 metaclass，torch.jit 实现了 ScriptMeta 这个 metaclass（元类）。

### 关于计算图优化

IR 的 Method 中内置 GraphExecutor object，创建于第一次执行的时候，负责优化。

```c++
// pytorch/torch/csrc/jit/api/method.h

  GraphExecutor& get_executor() {
    return toGraphFunction(*function_).get_executor();
  }
```

GraphExecutor 的定义在/torch/csrc/jit/runtime/graph_executor.cpp，可见其由 graph 产生，定义了 run 方法执行

```c++
    GraphExecutor::GraphExecutor(
        const std::shared_ptr<Graph>& graph,
        std::string function_name)
        : pImpl(
              IsNewExecutorEnabled()
                  ? dynamic_cast<GraphExecutorImplBase*>(
                        new ProfilingGraphExecutorImpl(
                            graph,
                            std::move(function_name)))
                  : dynamic_cast<GraphExecutorImplBase*>(
                        new GraphExecutorImpl(graph, std::move(function_name)))) {}
    std::shared_ptr<Graph> GraphExecutor::graph() const {
      return pImpl->graph;
    }
    const ExecutionPlan& GraphExecutor::getPlanFor(
        Stack& inputs,
        size_t remaining_bailout_depth) {
      return pImpl->getPlanFor(inputs, remaining_bailout_depth);
    }

     std::shared_ptr<GraphExecutorImplBase> pImpl;
    .....

// 关于 GraphExecutorImplBase,/torch/csrc/jit/runtime/graph_executor.cpp


    const ExecutionPlan& getOrCompile(const Stack& stack) {
          .....
          auto plan = compileSpec(spec);

        }
      }
    // # compileSpec 会返回一个plan
    ExecutionPlan compileSpec(const ArgumentSpec& spec) {
        auto opt_graph = graph->copy();
        GRAPH_DUMP("Optimizing the following function:", opt_graph);
        arg_spec_creator_.specializeTypes(*opt_graph, spec);

        // Phase 0. Inline functions, then clean up any artifacts that the inliner
        //          left in that may inhibit optimization
         .....
        runRequiredPasses(opt_graph);
        GRAPH_DEBUG(
            "After runRequiredPasses, before ConstantPropagation\n", *opt_graph);

        // Phase 2. Propagate detailed information about the spec through the
        //          graph (enabled more specializations in later passes).
        //          Shape propagation sometimes depends on certain arguments being
        //          constants, and constant propagation doesn't need shape
        //          information anyway, so it's better to run it first.
        ConstantPropagation(opt_graph);
        GRAPH_DEBUG(
            "After ConstantPropagation, before PropagateInputShapes\n", *opt_graph);
        PropagateInputShapes(opt_graph);
        GRAPH_DEBUG(
            "After PropagateInputShapes, before PropagateRequiresGrad\n",
            *opt_graph);
        PropagateRequiresGrad(opt_graph);
        GRAPH_DEBUG(
            "After PropagateRequiresGrad, before runOptimization\n", *opt_graph);

        // Phase 3. Run differentiable optimizations (i.e. simple graph rewrites
        //          that we can still execute using autograd).
        runOptimization(opt_graph);
        // .....各种优化
        return ExecutionPlan(opt_graph, function_name_);
      }
```

这些优化在 torch/csrc/jit/passes/ 文件夹

torch/csrc/jit/passes/dead_code_elimination.cpp

/torch/csrc/jit/passes/fuse_linear.cpp

torch/csrc/jit/passes/remove_dropout.cpp

torch/csrc/jit/passes/fold_conv_bn.cpp
