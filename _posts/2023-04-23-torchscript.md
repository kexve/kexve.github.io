---
layout: post
title: torchscript
categories: [Pytorch]
---

## 与 ONNX 的关系

ONNX 是业界广泛使用的一种神经网络中间表示，PyTorch 自然也对 ONNX 提供了支持。torch.onnx.export 函数可以帮助我们把 PyTorch 模型转换成 ONNX 模型，这个函数会使用 trace 的方式记录 PyTorch 的推理过程。聪明的同学可能已经想到了，没错，**ONNX 的导出，使用的正是 TorchScript 的 trace 工具**。具体步骤如下：

1. 使用 trace 的方式先生成一个 TorchScipt 模型，如果你转换的本身就是 TorchScript 模型，则可以跳过这一步。
2. 使用**许多 pass 对 1 中生成的模型进行变换**，其中对 ONNX 导出最重要的一个 pass 就是 **ToONNX**，这个 pass 会进行一个映射，将 **TorchScript 中 prim、aten 空间下的算子映射到 onnx 空间下的算子**。
3. 使用 ONNX 的 **proto** 格式对模型进行**序列化**，完成 ONNX 的导出。
4. 关于 ONNX 导出的实现以及算子映射的方式将会在未来的分享中详细展开。

## 与 torch.fx 的关系？

PyTorch1.9 开始添加了 torch.fx 工具，根据官方的介绍，它由**符号追踪器（symbolic tracer）**，**中间表示（IR）**， **Python 代码生成（Python code generation）**等组件组成，实现了 python->python 的翻译。是不是和 TorchScript 看起来有点像？

其实他们之间联系不大，可以算是互相垂直的两个工具，为解决两个不同的任务而诞生。

1. TorchScript 的主要用途是进行**模型部署**，**需要记录生成一个便于推理优化的 IR**，对计算图的编辑通常都是面向性能提升等等，**不会给模型本身添加新的功能**。
2. FX 的主要用途是进行 python->python 的翻译，**它的 IR 中节点类型更简单，比如函数调用、属性提取**等等，这样的 IR 学习成本更低更容易编辑。使用 **FX 来编辑图通常是为了实现某种特定功能，比如给模型插入量化节点等，避免手动编辑网络造成的重复劳动**。

这两个工具可以同时使用，比如使用 FX 工具编辑模型来让训练更便利、功能更强大；然后用 TorchScript 将模型加速部署到特定平台。

## Pass

严格地说这不是 Graph 的一部分，pass 是一个来源于编译原理的概念，它会接收一种中间表示（IR），遍历它并且进行一些变换，生成满足某种条件的新 IR。

TorchScript 中定义了许多 pass 来优化 Graph。比如对于常规编译器很常见的 DeadCodeElimination（DCE），CommonSubgraphElimination(CSE)等等；也有一些针对深度学习的融合优化，比如 FuseConvBN 等；还有针对特殊任务的 pass，ONNX 的导出就是其中一类 pass。

![20230423154052](https://cdn.jsdelivr.net/gh/kexve/img@main/image_blog20230423154052.png)

![20230423154107](https://cdn.jsdelivr.net/gh/kexve/img@main/image_blog20230423154107.png)

## JIT Trace

Jit trace 在 python 侧的接口为 torch.jit.trace，输入的参数会经过层层传递，最终会进入 torch/jit/frontend/trace.cpp 中的 trace 函数中。这个函数是 Jit trace 的核心，大致执行了下面几个步骤：

1. 创建新的 TracingState 对象，该对象会维护 trace 的 Graph 以及一些必要的环境参数。
2. 根据 trace 时的模型输入参数，生成 Graph 的输入节点。
3. 进行模型推理，同时生成 Graph 中的各个元素。
4. 生成 Graph 的输出节点。
5. 进行一些简单的优化。

下面介绍细节:

1. 创建 TracingState 对象

   TracingState 对象包含了 Graph 的指针、函数名映射、栈帧信息等，trace 的过程就是不断更新 TracingState 的过程。

   ```c++
   struct TORCH_API TracingState
       : public std::enable_shared_from_this<TracingState> {
   // 部分接口，可以帮助Graph的构建
   std::shared_ptr<Graph> graph;

   void enterFrame();
   void leaveFrame();

   void setValue(const IValue& v, Value* value);
   void delValue(const IValue& var);
   Value* getValue(const IValue& var);
   Value* getOutput(const IValue& var, size_t i);
   bool hasValue(const IValue& var) const;

   Node* createNode(c10::Symbol op_name, size_t num_outputs);
   void insertNode(Node* node);
   };
   ```

2. 生成 Graph 输入

   这个步骤会根据输入的 IValue 的类型，在 graph 中插入新的输入 Value。还记得在基本概念章节中我们提到的 IValue 与 Value 的区别吗？

   [TorchScript IR 中的类型 C++ 接口](http://zh0ngtian.tech/posts/dbbbf040.html)

   torch::jit::IValue 和 torch::jit::Value 的一个区别是：前者主要出现在程序运行的过程中，后者仅在 IR 中。torch::jit::IValue 是运行过程中出现的所有数据类型的容器。

   ```c++
   for (IValue& input : inputs) {
       // addInput这个函数会unpack一些容器类型的IValue，创建对应的Node
       input = addInput(state, input, input.type(), state->graph->addInput());
   }
   ```

3. 进行 Tracing

   Tracing 的过程就是使用样本数据进行一次推理的过程，但是实际在 github 的源码中，并不能找到关于推理时如何更新 TracingState 的代码。

   那么 PyTorch 到底是如何做到在**推理时更新 TracingState** 的呢？我们首先介绍关于 PyTorch 源码编译的一些小细节。

   PyTorch 要适配各种硬件以及环境，为所有这些情况定制代码工作量大得可怕，也不方便后续的维护更新。因此 **PyTorch 中许多代码是根据 build 时的参数生成出来**，更新 TracingState 的代码就是其中之一。生成 Tracing 代码的脚本如下：

   ```python
   python -m tools.autograd.gen_autograd \
       aten/src/ATen/native/native_functions.yaml \
       ${OUTPUT_DIR} \
       tools/autograd

   # derivatives.yaml和native_functions.yaml中包含
   # 许多FunctionSchema以及生成代码需要的信息
   ```

   大家可以跑一下看看都生成了些什么。生成的代码中 TraceTypeEverything.cpp 包含了许多关于更新 TracingState 的内容，我们还是以 add 算子举例如下：

   ```yaml
   # yaml
   - func: scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
     structured_delegate: scatter_add.out
     variants: function, method

   - func: scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
     structured_delegate: scatter_add.out
     variants: method

   - func: scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
     structured: True
     variants: function
     dispatch:
       CPU, CUDA: scatter_add

     # func的内容是一个FunctionSchema，定义了函数的输入输出、别名信息等。
   ```

   ```cpp
   at::Tensor scatter_add(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
     torch::jit::Node* node = nullptr;
     std::shared_ptr<jit::tracer::TracingState> tracer_state;
     if (jit::tracer::isTracing()) {
     // 步骤1： 如果tracing时，使用TracingState创建ops对应的Node并插入Graph
       tracer_state = jit::tracer::getTracingState();
       at::Symbol op_name;
       op_name = c10::Symbol::fromQualString("aten::scatter_add");
       node = tracer_state->createNode(op_name, /*num_outputs=*/0);
       jit::tracer::recordSourceLocation(node);
       jit::tracer::addInputs(node, "self", self);
       jit::tracer::addInputs(node, "dim", dim);
       jit::tracer::addInputs(node, "index", index);
       jit::tracer::addInputs(node, "src", src);
       tracer_state->insertNode(node);

       jit::tracer::setTracingState(nullptr);
     }
     // 步骤2： ops计算，不管是否进行Tracing都会执行
     auto result =at::_ops::scatter_add::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src);
     if (tracer_state) {
     // 步骤3： 在TracingState中设置ops输出
       jit::tracer::setTracingState(std::move(tracer_state));
       jit::tracer::addOutput(node, result);
     }
     return result;
   }
   ```

   上方是 FunctionSchema，下方为生成的代码。代码会根据是否 isTracing 来选择是否记录 Graph 的结构信息。

   实际在 Tracing 时，每经过一个 ops，都会调用一个类似上面生成的函数，执行如下步骤：

   1. 在推理前根据解析的 FunctionSchema 生成 Node 以及各个输入 Value；
   2. 然后进行 ops 的正常计算；
   3. 最后根据 ops 的输出生成 Node 的输出 Value。

4. 注册 Graph 输出

   这部分没有太多值得说的，就是挨个把推理的输出注册成 Graph 的输出 Value。由于输出在一个栈中，因此输出的编号要逆序。

   ```c++
   size_t i = 0;
   for (auto& output : out_stack) {
   // NB: The stack is in "reverse" order, so when we pass the diagnostic
   // number we need to flip it based on size.
   state->graph->registerOutput(
   state->getOutput(output, out_stack.size() - i));
   i++;
   }
   ```

5. Graph 优化

   完成 Tracing 后，会对 Graph 进行一些简单的优化，包括如下数个 passes：

   Inline(Optional)：网络定义经常会包含很多嵌套结构，比如 Resnet 会由很多 BottleNeck 组成。这就会涉及到对 sub module 的调用，这种调用会生成 prim::CallMethod 等 Node。Inline 优化会将 sub module 的 Graph 内联到当前的 Graph 中，消除 CallMethod、CallFunction 等节点。

   FixupTraceScopeBlock：处理一些与 scope 相关的 node，比如将诸如 prim::TracedAttr[scope="\_\_module.f.param"]()这样的 Node 拆成数个 prim::GetAttr 的组合。
   NormalizeOps：有些不同名 Node 可能有相同的功能，比如 aten::absolute 和 aten::abs，N ormalizeOps 会把这些 Node 的类型名字统一（通常为较短的那个）。

经过上述步骤，就可以得到经过 trace 的结果。

## ONNX Export

Onnx 模型的导出同样要用到 jit trace 的过程，大致的步骤如下：

1. 加载 ops 的 symbolic 函数，主要是 torch 中预定义的 symbolic。
2. 设置环境，包括 opset_version，是否折叠常量等等。
3. 使用 jit trace 生成 Graph。
4. 将 Graph 中的 Node 映射成 ONNX 的 Node，并进行必要的优化。
5. 将模型导出成 ONNX 的序列化格式。

接下来，我们将按照顺序介绍以上几个步骤：

1. 加载 Symbolic

   严格地说这一步在 export 之前就已经完成。在 symbolic_registry.py 中，会维护一个\_symbolic_versions 对象，在导入这个模块时会使用 importlib 将预先定义的 symbolic（torch.onnx.symbolic_opset<xx>)加载到其中。

   ```python
   _symbolic_versions: Dict[Union[int, str], Any] = {}
   from torch.onnx.symbolic_helper import _onnx_stable_opsets, _onnx_main_opset
   for opset_version in _onnx_stable_opsets + [_onnx_main_opset]:
       module = importlib.import_module("torch.onnx.symbolic_opset{}".format(opset_version))
       _symbolic_versions[opset_version] = module
   ```

   \_symbolic_versions 中 key 为 opset_version，value 为对应的 symbolic 集合。symbolic 是一种映射函数，用来把对应的 aten/prim Node 映射成 onnx 的 Node。可以阅读 torch/onnx/symbolic_opset<xx>.py 了解更多细节。

2. 设置环境

   根据 export 的输入参数调整环境信息，比如 opset 的版本、是否将 init 导出成 Input、是否进行常量折叠等等。后续的优化会根据这些环境运行特定的 passes。

3. Graph Tracing

   这一步实际执行的就是上面介绍过的 Jit Tracing 过程，如果遗忘的话可以再复习一下哦。

4. ToONNX

   Graph 在实际使用之前会经过很多的 pass，每个 pass 都会对 Graph 进行一些变换，可以在 torch/csrc/jit/passes 中查看实现细节。这些 pass 很多功能与常见的编译器中的类似，篇幅关系就不在这里展开介绍了。对于 torchscript->ONNX 而言，**最重要的 pass 当属 ToONNX**。

   ToONNX 的 python 接口为 torch.\_C.\_jit_pass_onnx，对应的实现为 onnx.cpp。它会遍历 Graph 中所有的 Node，生成对应的 ONNX Node，插入新的 Graph 中：

   ```c++
   auto k = old_node->kind();    // 取得Node的ops类型
   if (k.is_caffe2()) {
     // ToONNX之前的会有一些对caffe2算子的pass
     // 因此这里只要直接clone到新的graph中即可
     cloneNode(old_node);
   } else if (k == prim::PythonOp) {
     // 如果是Python自定义的函数，比如继承自torch.autograd.Function的函数
     // 就会查找并调用对应的symbolic函数进行转换
     callPySymbolicMethod(static_cast<ConcretePythonOp*>(old_node));
   } else {
     // 如果是其他情况（通常是aten的算子）调用步骤1加载的symbolic进行转换
     callPySymbolicFunction(old_node);
   }
   ```

   cloneNode 的功能就和名字一样，就是简单的拷贝 old_node，然后塞进新的 Graph 中。

   callPySymbolicFunction

   当 Node 的类型为 PyTorch 的内置类型时，会调用这个函数来处理。

   该函数会调用 python 侧的 torch.onnx.utils.\_run_symbolic_function 函数，将 Node 进行转换，并插入新的 Graph，我们可以尝试如下 python 代码：

   ```python
   graph = torch._C.Graph()  # 创建Graph
   [graph.addInput() for _ in range(2)]  # 插入两个输入
   node = graph.create('aten::add', list(graph.inputs()))  # 创建节点
   node = graph.insertNode(node)  # 插入节点
   graph.registerOutput(node.output())  # 注册输出
   print(f'old graph:\n {graph}')

   new_graph = torch._C.Graph()  # 创建新的Graph用于ONNX
   [new_graph.addInput() for _ in range(2)]  # 插入两个输入
   _run_symbolic_function(
       new_graph, node, inputs=list(new_graph.inputs()),
       env={})  # 将aten Node转换为onnx Node， 插入新的Graph
   # 如果是torch>1.8，那么可能还要传入block
   print(f'new graph:\n {new_graph}')
   ```

   然后看一下可视化的结果：

   Old graph

   ```
   graph(%0 : Tensor,
         %1 : Tensor):
     %2 : Tensor = aten::add(%0, %1)
     return (%2)
   ```

   New graph

   ```
   graph(%0 : Tensor,
         %1 : Tensor):
     %2 : Tensor = onnx::Add(%0, %1)
     return ()
   ```

   可以看见，原来的 aten::add 节点已经被替换为了 onnx::Add。那么这个映射是如何完成的呢？还记得第一步记录的\_symbolic_versions 吗？\_run_symbolic_function 会调用 torch.onnx.symbolic_registry 中的\_find_symbolic_in_registry 函数，查找\_symbolic_versions 中是否存在满足条件的映射，如果存在，就会进行如上图中的转换。

   注意：**转换的新 Graph 中没有输出 Value，这是因为这部分是在 ToONNX 的 c++ 代码中实现，\_run_symbolic_function 仅负责 Node 的映射**。

   callPySymbolicMethod

   一些非 pytorch 原生的计算会被标记为 PythonOp。碰到这种 Node 时，会有三种可能的处理方式：

   1. 如果这个 PythonOp 带有名为 symbolic 的属性，那么就会尝试使用这个 symbolic 当作映射函数，生成 ONNX 节点
   2. 如果没有 symbolic 属性，但是在步骤 1 的时候注册了 prim::PythonOp 的 symbolic 函数，那么就会使用这个函数生成节点。
   3. 如果都没有，则直接 clone PythonOp 节点到新的 Graph。
   4. symbolic 函数的写法很简单，基本上就是调用 python bind 的 Graph 接口创建新节点，比如：

   ```py
   class CustomAdd(torch.autograd.Function):

   @staticmethod
   def forward(ctx, x, val):
       return x + val

   @staticmethod
   def symbolic(g, x, val):
       # g.op 可以创建新的Node
       # Node的名字 为 <domain>::<node_name>，如果domain为onnx，可以只写node_name
       # Node可以有很多属性，这些属性名必须有_<type>后缀，比如val如果为float类型，则必须有_f后缀
       return g.op("custom_domain::add", x, val_f=val)
   ```

   实际在使用上面的函数时，就会生成 custom_domain::add 这个 Node。当然，能否被用于推理这就要看推理引擎的支持情况了。

   通过 callPySymbolicFunction 和 callPySymbolicMethod，就可以生成一个由 ONNX（或自定义的 domain 下的 Node）组成的新 Graph。这之后还会执行一些优化 ONNX Graph 的 pass，这里不详细展开了。

5. 序列化
   到这里为止建图算是完成了，但是要给其他后端使用的话，需要将这个 Graph 序列化并导出。序列化的过程比较简单，基本上只是**调用 ONNX 的 proto 接口**，**将 Graph 中的各个元素映射到 ONNX 的 GraphProto 上**。没有太多值得展开的内容，可以阅读 export.cpp 中的 EncodeGraph，EncodeBlock，EncodeNode 函数了解更多细节。

之后只要根据具体的 export_type，将序列化后的 proto 写入文件即可。

至此，ONNX export 完成，可以开始享受各种推理引擎带来的速度提升了。
