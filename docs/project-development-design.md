# lw.PPOCR.C 项目详细开发设计

> 文档状态：开发基线设计
>
> 适用基线：`v1.0.0` 稳定发布线，以及当前工作树中的 Runtime 内存/性能实验。
>
> 目标读者：维护者、性能优化开发者、平台移植开发者、CI/发布维护者，以及需要在新会话中继续开发的协作者。

## 1. 设计结论

`lw.PPOCR.C` 已经不是单一的 OCR Demo，而是一套分层的纯 C OCR Runtime 和多入口产品：

```text
PP-OCRv6 ONNX
      |
Python converter / analyzer
      |
LWM v0.1 model pack
      |
pure-C loader + session planner + executor + SIMD kernels
      |
stable high-level C ABI
   /       |        |        \
 C     HTTP/Web   WASM     Java/.NET/Android
```

后续开发必须遵守以下总原则：

1. **先保持语义和可诊断性，再追求速度。** 任何优化都必须保留 Scalar 参考路径、可关闭开关和可复现 A/B 数据。
2. **C ABI 与 Runtime 分离。** 业务导出、HTTP、Java、Android、WASM 的逻辑不能反向污染 C 核心；模型、坐标、score 和 UTF-8 语义由核心统一提供。
3. **以完整 OCR 为最终目标。** 单算子 benchmark 只能用于定位热点，不能单独作为发布收益。
4. **平台声明分层。** “能编译”“CI 通过”“目标机器验证”“性能承诺”是四种不同结论，不得混写。
5. **内存增加必须有收益证据。** 当前硬件允许适度增加 workspace 和常驻缓存，但必须同时记录峰值工作集、模型大小和回退行为。
6. **LWM v0.1 暂不冻结。** 高层 C ABI v1 的兼容范围保持稳定；低层 session/planner 与 LWM 格式继续标为实验性。

## 2. 当前产品边界

### 2.1 模型与运行时

| 变体 | 默认产品 | 运行时 | 字典 | 当前定位 |
|---|---|---|---|---|
| PP-OCRv6 Tiny | C/HTTP/Android/Java/Node/WASM 默认包 | DET + 共享 CLS + REC | Tiny 字典 | 稳定默认模型 |
| PP-OCRv6 Small | 独立 Runtime model pack / 浏览器变体 | Small DET + 共享 CLS + Small/Medium 共用 REC 字典 | `PP-OCRv6_small_rec_dict.txt` | 预览质量档 |
| PP-OCRv6 Medium | 独立 Runtime model pack / 桌面优先浏览器变体 | Medium DET + 共享 CLS + Small/Medium 共用 REC 字典 | `PP-OCRv6_small_rec_dict.txt` | 高资源预览档 |

Small 和 Medium 复用 Tiny CLS，不重复发布 CLS 模型。模型包必须包含 `manifest.json`、`SHA256SUMS`、三个 `.lwm` 文件、匹配字典和许可证/构建信息。

### 2.2 入口

- **C API**：识别、方向分类、检测、完整 OCR；输入为调用者提供的 BGR8 像素，不解码 JPEG/PNG。
- **HTTP Demo**：JSON/Base64 与二进制图像请求，输出稳定 JSON、request id、错误码和 OCR 结果。
- **浏览器/WASM**：离线单 HTML、`lw-ppocr.js` SDK、Node WASM 包；现代构建默认 WASM SIMD128，Legacy 构建提供 Chrome 70 兼容路径。
- **Android**：AAR、ARM64 Demo；模型缓存按 manifest `asset_set_id` 失效，发布不要求本机安装 Android 工具链，由 CI 验证。
- **Desktop Java/JNI**：Windows/Linux/macOS 目标包，JNI 只负责边界转换，不复制 Runtime 逻辑。
- **C# WinForms**：保持既有 P/Invoke/结果导出边界，使用稳定高层 C API。

## 3. 分层架构与职责

### 3.1 Converter/Model 层

目录与职责：

| 路径 | 职责 |
|---|---|
| `converter/`、`tools/` | ONNX 检查、shape 推导、算子清单、LWM 转换、模型包准备 |
| `models/` | ONNX 输入、字典、样例和发布模型元数据 |
| `ci/` | Golden 文本、模型哈希、变体校验和测试契约 |

Converter 可以依赖 Python、ONNX、NumPy、protobuf；部署 Runtime 不依赖这些组件。

模型转换必须记录：输入 ONNX SHA-256、转换器版本、LWM 格式版本、节点/张量数量、输出 SHA-256、字典 SHA-256 和 `asset_set_id`。

### 3.2 Runtime 层

```text
src/runtime/model.c       LWM 边界检查和只读模型
src/runtime/shape.c       动态 shape 推导
src/runtime/session.c     resolved tensor table、workspace、常量准备
src/runtime/memory.c      生命周期规划和对齐分配
src/runtime/executor.c    节点执行、融合、CTC/布局专用路径
src/runtime/prepared_execution.c  实验性 prepared dispatch
src/kernels/              Scalar 参考算子
src/simd/                 SSE2/AVX2/NEON/LSX/LASX/WASM 后端
src/ppocr/                前处理、检测、裁剪、CLS、REC、阅读顺序、CTC
```

Runtime 层不得知道 HTTP、DOM、Java/Kotlin、文件选择器或业务导出格式。所有平台入口都通过高层 C API 调用。

### 3.3 应用层

应用层负责：

- 图像解码、PDF 页面渲染和 EXIF 方向；
- 请求验证、鉴权、request id、日志和超时；
- 文本/TXT/JSON/剪贴板/文件保存；
- UI 状态、标注显隐、PDF 翻页和模型选择；
- 对不同平台的线程调度和生命周期。

应用层不能假设内部 workspace、LWM 节点编号或 SIMD 后端一定存在。

## 4. 稳定契约与兼容策略

### 4.1 C ABI v1

当前稳定范围是 `include/lw_infer.h` 中的高层 REC、CLS、DET、full-OCR API。必须保持：

- `struct_size` 前缀兼容规则；
- caller-owned BGR 输入和 UTF-8 输出；
- `lw_*_get_info` 的容量查询语义；
- 失败时不写入不满足容量的输出缓冲区；
- `score`、box 坐标、阅读顺序、CLS 0/180 度和 CTC blank/repeat 语义；
- 单 handle 不支持并发，多个独立 handle 可以并行；
- create/free 成对，free 接受 null。

低层 `lw_model_*`、`lw_session_*`、workspace planner 和 prepared execution 仍是实验性接口，后续可以在 ABI v2 或内部接口中调整。

### 4.2 LWM

LWM v0.1 继续采用 bounds-checked loader 和 deterministic converter，但暂不承诺格式冻结。任何字段、算子 ID 或版本改变必须：

1. 更新 loader/validator；
2. 更新 converter 和模型 manifest；
3. 增加旧包/损坏包/未知字段测试；
4. 更新模型包 SHA-256 和 release notes；
5. 在文档中明确是否需要新模型包。

### 4.3 HTTP/Web/SDK

HTTP 和 JS SDK 的 JSON schema 采用显式 `schema_version`；未知配置字段必须拒绝。Web 导出 schema v1 的字段名、类型和 `elapsed_ms` 语义不得悄悄改变。Legacy 构建的语法门禁必须以 Chrome 70 为准，不能仅以 Node 解析通过为准。

## 5. 性能与内存目标

### 5.1 指标

每次性能实验必须同时保存：

- 完整 OCR 总延迟：warm-up 后中位数和 P95；
- DET、crop、CLS、REC、CTC、layout 转换和调度阶段耗时；
- worker 数：1、4，必要时 8；
- REC 实际宽度分布和 padding ratio；
- 峰值 RSS/working set、workspace、模型文件和 packed weight bytes；
- 全文 checksum、行数、exact-line rate、CER；
- CPU/ISA、编译器、模型包和数据集 manifest SHA-256。

不得把不同 CPU、不同 REC 宽度策略或不同模型的绝对耗时直接混合。比率只能在同一 runner、同一 replica、同一 workload 内计算。

### 5.2 当前重点结论

- Tiny/Small/Medium 的 REC 默认宽度保持 **960**；192/320/480/640/960 自适应仍然保留。
- Medium profile 显示主要 Conv1x1 时间集中在长几何 `H×W` map，square-map packed 扩展不是当前第一优先级。
- Conv1x1 FMA 目前是独立 opt-in 实验：Tiny 1 worker 约有低个位数收益，Small/Medium 的现有样本收益低于 2% 推广门槛，默认仍关闭。
- broad NHWC 改造和没有完整 OCR 收益证据的 SIMD kernel 不进入默认路径。
- 适度增加内存可以接受，但任何增长都必须通过 workspace report、重复运行和跨模型对比说明原因。

### 5.3 外部方案借鉴边界（SimdPaddleOCR）

最新的 SimdPaddleOCR 公开实现提供了几条值得验证、但不能直接照搬的路线：图级 NHWC、在 resize/crop 阶段直接写目标布局、按图内卷积线程与行级 worker 分开配置，以及把模型作为可流式读取的资源提供者。它的输入边界同样是不负责图片解码，而是接收 BGR/RGB/BGRA/RGBA 交错像素；这与本项目“解码在应用层、Runtime 只接收像素”的边界一致。

本项目只借鉴机制，不直接移植 C# 类型、线程模型或绝对性能数字：

| 借鉴点 | 本项目的验证方式 | 暂不做的事情 |
|---|---|---|
| 图级 NHWC / 预处理直接落目标布局 | 在 `rec_preprocess_layout_driver` 中比较 NCHW/NHWC 写入、峰值 workspace、完整 OCR checksum；只保留有收益的形状 | 不把 NHWC 作为默认 ABI 或直接改 LWM 布局 |
| DET/REC 图内线程与行级 worker 分离 | 分别记录 `det_threads`、`line_workers`，对 1/4 worker 做 paired profile | 不把线程数硬编码成某个 CPU 的最佳值 |
| 模型资源按 manifest/stream 提供 | C 侧继续使用 bounds-checked LWM；应用层/Android/Java 复用 `asset_set_id` 和 SHA-256 | 不在 Runtime 内加入压缩包、文件系统或下载逻辑 |
| 预处理避免中间 BGR 拷贝 | 对外仍接受 BGR8；在 Web/Android/Java 入口测量 decode→resize→OCR 的临时 buffer 数量 | 不为了减少一次拷贝而破坏现有 C ABI 的 caller-owned 输入语义 |
| 多 ISA 分层 fallback | 保持 Scalar/SSE2/AVX2/NEON/LSX/LASX/WASM 后端逐层回退，并在同一 corpus 做 parity | 不把某个 x64/ARM runner 的特化结果写成全平台承诺 |

对比报告必须同时给出：提交 SHA、模型/数据集 manifest SHA-256、CPU/ISA、线程配置、REC 宽度策略、完整 OCR 文本 checksum、exact-line/CER、峰值工作集和报告生成命令。SimdPaddleOCR 的仓库说明显示其近期重点是图级 NHWC 和降低工作集；这些是研究输入，不是本项目的兼容性或性能结论。

## 6. 分阶段开发计划

### Phase 0：基线与可观测性（必须先完成）

目标：让每个优化都能回答“快在哪里、占了什么内存、结果是否一致”。

实施：

1. 固定 Tiny/Small/Medium 的 100 图 corpus、manifest 和 golden 文本；图片生成物不进入 Git，manifest、生成器和 README 进入 Git。
2. 统一 `tools/generate_ocr_dataset.py`、`tools/prepare_ocr_dataset_ppm.py`、`examples/ocr_dataset_benchmark.c` 和 `tools/compare_ocr_dataset_runtime.py` 的输出 schema。
3. 扩展 `full_ocr_profile_driver` 的 REC 节点 shape、weights、kernel、stride、dilation、pads 遥测。
4. 使用 `workspace-report-driver` 记录 model bytes、workspace bytes、semantic lower bound、packed weights 和 prepared constants。
5. 每次实验输出 JSON 和 Markdown，不手工抄写单个 benchmark 数字。

验收：同一提交、同一 runner、同一 manifest 可重复得到同一 checksum；报告可区分基线和候选构建。

### Phase 1：Runtime 内存收口

目标：降低峰值工作集，尤其是 Medium 和多 worker 场景；不改变公共 ABI。

按以下顺序实施：

1. **workspace planner v2**：为融合链和不再读取的中间张量建立真实 last-use/lifetime，给出 semantic lower bound 与实际规划差值。
2. **prepared constant cache**：按模型、shape、ISA 和兼容性复用 packed Conv/MatMul 常量；失败时回退 canonical layout。
3. **CTC terminal-output elision**：保留 score 兼容性，只省掉不再需要的最终输出存储；不能使用只取最大 logit 的不兼容解码。
4. **crop/line buffer reuse**：整页 crop、worker-local text/line buffer 和 BGR scratch 只增不减，避免重复 malloc/free。
5. **融合内存回收**：GELU 等融合只在常量、shape、private lifetime 全部满足时启用；不能只跳过节点而继续占用完整规划空间。

文件范围：`src/runtime/memory.c`、`src/runtime/session.c`、`src/runtime/executor.c`、`src/runtime/prepared_execution.c`、`src/ppocr/recognizer.c`、`src/ppocr/ocr.c`。

门槛：Tiny/Small/Medium 的 1/4 worker Golden、determinism、workspace limit、sanitizer 全部通过；峰值工作集有明确变化说明；失败路径仍释放所有资源。

### Phase 1A：布局与缓冲区实验（与 Phase 1 并行、默认不启用）

这是把外部方案转成可控实验的最小闭环：

1. 为 REC 预处理记录输入格式、目标布局、stride、分配次数和总字节数；同一输入分别跑现有路径与候选布局。
2. 先只覆盖真实模型 shape（Tiny/Small/Medium、REC 宽度 960）；合成 shape 仅用于 kernel parity，不用于宣布收益。
3. 候选路径必须保留 scalar reference，并以编译开关隔离；默认构建、WASM、Android、Java 不自动带入实验开关。
4. 若减少一次中间拷贝，必须将收益拆成 preprocess、REC、full OCR 三层；若 full OCR 中位数没有至少 3% 稳定收益，回退并记录负结果。
5. 峰值工作集若增加，只有在延迟收益超过门槛且 1/4 worker 都没有失控时才继续；报告必须同时给出每 worker 增量。

第一轮验收命令：

```powershell
cmake --build build-dev --config Release --parallel --target rec-preprocess-layout-driver
build-dev\Release\rec-preprocess-layout-driver.exe --width 960 --workers 1 --workers 4
ctest --test-dir build-dev -C Release -R "rec_preprocess|full_ocr|golden" --output-on-failure
```

该阶段的产物是 JSON/Markdown A/B 报告，不是公共 ABI 变化；没有完整 OCR 证据时不得合入默认 dispatch。

### Phase 2：x64 算子优化

目标：只优化 profile 中仍占大头的形状，并保留 Scalar/SSE2 回退。

建议顺序：

1. **Conv1x1 长几何路径**：针对 Medium 主要 `C_in/C_out/H/W` 组合做局部 A/B；FMA dispatch 使用独立开关，不与 MatMul 混合。
2. **regular 3×3**：只对 profile 中真实出现的 stride/pad/output-channel 组合做四输出平面 blocking；边界和非整除通道保留旧路径。
3. **ConvTranspose**：先记录输入/权重/输出形状和 scatter 写入模式，再决定是否做 output-tile 或 channel blocking；没有完整 OCR 收益时不改默认 dispatch。
4. **MatMul**：继续使用 session-packed wide weights 和 4-row/16-column microkernel；每个候选都要独立 benchmark 和 full OCR A/B。
5. **FMA**：只在完整 Tiny/Small/Medium corpus 达到预设门槛后推广；否则保持 opt-in。

每个 kernel 必须提供：Scalar reference、随机/边界测试、ISA 检测、非支持 CPU 回退、单算子 benchmark、完整 OCR paired report。

### Phase 3：ARM64、LoongArch64、WASM parity

目标：保证正确性和合理回退，暂不承诺与 x64 相同的性能收益。

- ARM64：NEON 只在编译和运行时都确认可用时 dispatch；保持 scalar reference；优先 Conv1x1、regular 3×3、ConvTranspose 的真实模型形状。
- LoongArch64：LSX/LASX 作为独立后端，不能用 x86 intrinsics；QEMU 只证明构建/通路/无非法指令，不代表客户真机性能。
- WASM：现代 SIMD128 与 scalar build 并列验证；legacy JS 通过 Chrome 70 语法检查，不能引入 optional chaining、numeric separator 或新版 runtime helper。

文件与 CI：`src/simd/*`、`tests/*reference*`、`tests/simd_backend_driver`、`arm64-performance.yml`、`customer-linux-architectures.yml`、`wasm-html.yml`。

### Phase 4：入口与发布产品化

目标：保证同一 Runtime 能被所有入口稳定使用。

1. HTTP：继续验证二进制和 JSON/Base64 请求、错误码、request id、超时、并发和 staged package。
2. Web：保持单 HTML 离线能力；导出 TXT/JSON、PDF fallback、检测标注显隐、Legacy 移动浏览器兼容均由 UI 层实现。
3. Android：模型 manifest 校验、缓存 asset set 失效、大图 bounds/downsample/EXIF；不引入 Camera/CameraX。
4. Java/JNI：Windows/Linux/macOS 包只暴露稳定高层结果；避免把 C++/OpenCV 类型穿过 JNI。
5. C#：保持 .NET Framework 兼容；Result API 与 Web/HTTP 字段语义一致。
6. 发布：每个模型包和平台包包含 BUILD-INFO、许可证、第三方声明、SBOM、manifest、SHA256SUMS；release workflow 只上传经过 staged/package smoke 的产物。

### Phase 5：下一个版本的收口

建议版本策略：

- `v1.0.x`：只修发布、CI、兼容性和不改变语义的 bug。
- `v1.1.0`：在完整基准和跨平台门禁通过后，收录默认启用的内存/算子优化；不自动冻结 LWM。
- ABI 或 LWM 发生破坏性变化时，使用新版本/新契约，不覆盖旧 tag。

## 7. 具体开发操作

### 7.1 Windows x64 基础验证

```powershell
cmake -S . -B build-dev -G "Visual Studio 17 2022" -A x64 `
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-dev --config Release --parallel
ctest --test-dir build-dev -C Release --output-on-failure
```

### 7.2 运行 workspace/profile

```powershell
build-dev\Release\workspace-report-driver.exe `
  build-dev\models\rec.lwm 960 1

build-dev\Release\full-ocr-profile-driver.exe `
  build-dev\models\det.lwm build-dev\models\cls.lwm `
  build-dev\models\rec.lwm models\ppocrv6-tiny\ppocr_keys.txt `
  build-local-data\lw-generated-ocr-ppm\images.txt 1 `
  > build-local-data\tiny-profile-w1.json
```

### 7.3 生成数据集与比较运行时

```powershell
python tools\generate_ocr_dataset.py `
  --output build-local-data\lw-generated-ocr --count 100 --seed 20260907 --force
python tools\prepare_ocr_dataset_ppm.py `
  --dataset build-local-data\lw-generated-ocr\metadata.json `
  --output build-local-data\lw-generated-ocr-ppm --force

build-dev\Release\full-ocr-dataset-benchmark.exe `
  build-dev\models\det.lwm build-dev\models\cls.lwm `
  build-dev\models\rec.lwm models\ppocrv6-tiny\ppocr_keys.txt `
  build-local-data\lw-generated-ocr-ppm\images.txt 1 3 1 960
```

### 7.4 FMA/布局实验

实验构建必须显式写出开关，避免把候选路径误当默认路径：

```powershell
cmake -S . -B build-fma-conv-only -G "Visual Studio 17 2022" -A x64 `
  -DBUILD_TESTING=ON `
  -DLW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH=ON `
  -DLW_EXPERIMENTAL_AVX2_FMA_MATMUL_DISPATCH=OFF
```

NHWC 只作为 parity/A-B 工具，不作为默认布局；任何改变布局的实验必须同时跑 shape、Golden、score、内存和完整 OCR。

### 7.5 CI 验证顺序

1. 本地 Windows x64：编译、CTest、真实模型 OCR。
2. Pull Request：Ubuntu x64、Windows x64、macOS ARM64、WASM、Java、Android、sanitizer。
3. 手动架构工作流：amd64、ARM64、LoongArch64；确认 ELF machine、动态依赖、SIMD backend 和 installed package。
4. 发布前：staged package、HTTP live demo、模型 manifest、SHA256、SBOM、release asset allowlist。

本机没有 WASM/Android/LoongArch 编译环境时，不安装临时工具链；修改必须依靠 CI 工作流验证，并在文档中明确“CI 验证”与“真机验证”的区别。

## 8. 测试与验收矩阵

| 层级 | 必须通过的内容 |
|---|---|
| 编译 | Windows x64/x86、Linux x64、macOS ARM64、WASM、Android CI |
| 算子 | Scalar reference、SSE2/AVX2/NEON/LSX/LASX/WASM parity、边界/随机输入 |
| Runtime | LWM bounds、shape、workspace limit、prepared fallback、重复 create/free |
| OCR | Tiny/Small/Medium 16 行、全文 checksum、score finite、determinism、CER/exact-line |
| 内存 | warm-up 后 heap/RSS 收敛、1/4 worker、长文本和多图序列、无 sanitizer 报告 |
| API | ABI export allowlist、struct_size、容量失败不写输出、recognize-only 与 full OCR |
| HTTP | JSON/Base64、二进制请求、错误码、超时、并发、staged package live request |
| Web | modern/legacy syntax、单文件离线、PDF fallback、导出、移动端启动 |
| 发布 | manifest/SHA256、SBOM、许可证、动态依赖、精确 archive smoke |

性能和准确率在报告中是信息性指标；契约、checksum、结果结构、包完整性和动态依赖是发布门禁。

## 9. 风险与禁止事项

- 不要为了通过 benchmark 把默认 REC 宽度改成 320；默认保持 960。
- 不要把 SimdPaddleOCR 的绝对数字直接当成当前项目的性能结论；只能借鉴机制并在同一 workload 下复现。
- 不要在纯 C Runtime 中加入 PDF.js、OpenCV、剪贴板、文件下载或 UI 导出逻辑。
- 不要把 FMA、NHWC 或 prepared execution 的实验开关默认打开，除非完整模型和多平台门禁通过。
- 不要只看单算子 benchmark；必须跑完整 OCR、全文 checksum、score 和峰值内存。
- 不要为了消除编译警告而删除安全检查；针对 `-Werror=type-limits`、unused function 等问题应调整条件或编译门控。
- 不要覆盖已发布 tag，不要把未验证的 LoongArch/Windows 7/实体手机性能写成正式支持承诺。
- 不要提交生成的测试图片；提交生成器、manifest schema 和复现命令即可。

## 10. 下一轮工作清单

按优先级建议如下：

### P0：把当前工作树收口

- [x] 为 `full_ocr_profile_driver` 新 shape telemetry 增加 Python schema/assertion 测试。
- [x] 跑完整 CTest，Windows x64 baseline 56/56 通过；候选关键测试也通过。
- [x] 更新 `docs/runtime-memory-performance-design.md`，记录 telemetry contract 和 Medium 长几何 Conv1x1 证据。
- [x] 审核所有实验开关默认值和 CMake target 定义，确保 WASM/Java/Android 不误带 x64 FMA。

### P1：继续做有数据支撑的优化

- [ ] Medium 主要长几何 Conv1x1 做单形状 A/B，先看 packed weight/cache、线程分工和内存带宽。
- [x] 建立 DET regular 3×3/ConvTranspose 真实节点 shape profile；下一步才进入 output-channel blocking A/B。
- [ ] 完成 Small/Medium 4-worker、完整 corpus 和至少一个非 x64 CI 的 paired report。
- [ ] 做 workspace planner 与 prepared constants 的组件级峰值报告。
- [ ] 完成布局/预处理 A/B：验证是否能在不改变 C ABI 的前提下减少一次中间 buffer；负结果也要进入报告。

### P2：产品质量与发布维护

- [ ] 增加 recognition-only 的 HTTP/Java/Android 使用示例和容量语义测试。
- [ ] 完善模型升级、manifest asset set、回滚和缓存失效说明。
- [ ] 预留 ABI v2/LWM v0.2 评审，但在真实用户反馈前不冻结新格式。
- [ ] 视客户需求增加签名 tag、artifact attestation、SBOM/provenance；未做签名不影响 v1.0.0，但文档需如实说明。

## 11. 完成定义

本设计对应的阶段只有在以下条件全部满足时才算完成：

1. 代码、文档、CI 和发布包对模型/平台/ABI 的说法一致；
2. 每个默认优化都有 Scalar fallback 和可复现 A/B 数据；
3. Tiny/Small/Medium 在同一数据集上的全文 checksum、score、行数和 determinism 不回归；
4. 1/4 worker 的 workspace/RSS 变化有报告，增长有明确理由；
5. staged package 能在目标环境启动并完成真实 OCR，而不仅是 build tree 中的单元测试；
6. 新会话只阅读本文件和链接文档，就能按命令重现基线、开发、测试和 CI 验收。

## 12. 关联文档

- [C API 与 ABI v1](c-api.md)
- [总体架构](architecture.md)
- [平台矩阵](platform-matrix.md)
- [模型支持矩阵](supported-models.md)
- [Runtime 内存与性能设计](runtime-memory-performance-design.md)
- [Prepared Execution](prepared-execution.md)
- [x64 SIMD Phase 2](x64-simd-phase2.md)
- [完整 OCR profile](full-ocr-profile.md)
- [模型包说明](model-packs.md)
- [发布就绪检查](release-readiness-v1.0.md)
