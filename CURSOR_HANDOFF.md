# Cursor Handoff: Py-ART Compatible Rust Kernel Rewrite

本文档给 Cursor 使用，目标是把 `nriet-rust-art` 继续推进到完整 Rust 内核改造完成。不要把本文当成设计讨论稿；它是执行清单。每一项实现前都必须先冻结 Python/Py-ART oracle 行为，实现后必须跑对应门禁。

## 0. 当前结论

- 当前真实工作目录：`F:\nriet-rust-art\repo`
- 当前外部 oracle 源码包：`F:\nriet-rust-art\pyart-main.zip`
- 当前外部 operational parity 数据：`F:\nriet-rust-art\闽侯对比用数据`
- 旧文档里出现的 `D:\nriet-rust-art\...` 全部视为历史路径。后续执行只以 `F:\nriet-rust-art` 为准。
- 公开 Python API 必须继续是 `import pyart`
- Python distribution identity 必须继续是 `arm_pyart`
- Rust 扩展只能作为私有 native module：`pyart._rust`
- Python 保留公开对象模型、mutable object model、Radar/Grid 等上层结构
- Rust 只提供私有热点内核，不向用户暴露新的公开 API
- 默认 parity 是 exact parity，不是近似 parity

## 1. 不可破坏的兼容合同

任何代码变更都必须满足以下规则：

- `import pyart` 继续可用
- `pyart._rust` 只能是私有实现细节
- `pyproject.toml` 中 `project.name` 继续是 `arm_pyart`
- `pyproject.toml` 中 maturin `module-name` 继续是 `pyart._rust`
- 不改变 public module / class / function 名称
- 不改变 public signature
- 不改变 `__all__`
- 不改变 warning 类型、触发条件和大致消息
- 不改变 exception 类型、触发条件和大致消息
- 不改变 masked array 的 `mask`
- 不改变 masked array 的 `fill_value`
- 不改变 payload dtype
- 不改变 payload shape
- 不改变 NaN / inf 位置
- 不改变 metadata dict/list/tuple 内容
- 不改变 Radar/Grid 对象布局和可变语义
- 不为 Rust 方便而修正 Py-ART 历史怪行为

只有一种情况允许非 exact numeric parity：先写明原因，再把 tolerance 记录到 `README.md` 的 `Floating Tolerance Exceptions` 段落，并加测试锁住 tolerance 范围。没有记录的 tolerance 一律视为 bug。

## 2. Cursor 开始前必须做的检查

在 Cursor 里打开 repo 后，先执行：

```powershell
cd F:\nriet-rust-art\repo
git status --short
cargo --version
python --version
python -m pip show maturin
```

当前仓库预计是 dirty 状态，包含大量已完成但未提交的迁移工作。不要使用 `git reset --hard`，不要删除未跟踪目录。

当前已知 dirty 状态大致包括：

- modified: `Cargo.toml`
- modified: `README.md`
- modified: `pyproject.toml`
- modified: `src/lib.rs`
- modified: `tests/test_smoke.py`
- deleted: `python/nriet_rust_art/__init__.py`
- untracked: `Cargo.lock`
- untracked: `LICENSE-PYART.txt`
- untracked: `python/pyart/`
- untracked: `src/*.rs`
- untracked: `tests/api/`
- untracked: `tests/io/`
- untracked: `tests/parity/`
- untracked: `tests/rstm/`
- untracked: `tools/`

建议 Cursor 第一件事是建立分支并提交当前基线，防止后续多轮修改混在一起：

```powershell
git branch --show-current
git switch -c rust-kernel-full-rewrite
git add Cargo.toml Cargo.lock pyproject.toml README.md LICENSE-PYART.txt src python tests tools CURSOR_HANDOFF.md
git status --short
git commit -m "bootstrap pyart-compatible rust kernel rewrite"
```

如果当前已经在合适分支，不要强行新建；先确认分支名，再提交基线。

## 3. 不允许提交的外部输入

这些文件和目录只能作为 oracle / data 使用，不能加入 git：

- `F:\nriet-rust-art\pyart-main.zip`
- `F:\nriet-rust-art\pyart-main\`
- `F:\nriet-rust-art\_oracle\`
- `F:\nriet-rust-art\闽侯对比用数据\`
- 大型 benchmark 输出
- 大型 manifest 输出
- wheel 构建产物 `dist\*.whl`
- `target\`

若 `.gitignore` 还没有保护这些路径，先补 `.gitignore`，再提交。

## 4. 当前项目布局

关键文件：

- `F:\nriet-rust-art\repo\pyproject.toml`
  - Python distribution: `arm_pyart`
  - Python source root: `python`
  - native module: `pyart._rust`

- `F:\nriet-rust-art\repo\Cargo.toml`
  - Rust crate metadata
  - crate version 必须和 `src/lib.rs` 暴露的版本一致

- `F:\nriet-rust-art\repo\src\lib.rs`
  - PyO3 module 入口
  - 注册所有 native modules
  - 当前模块包括：
    - `advection`
    - `cappi`
    - `cfad`
    - `correct`
    - `despeckle`
    - `echo_class`
    - `filters`
    - `io`
    - `kdp`
    - `map`
    - `qpe`
    - `qvp`
    - `rstm`
    - `sigmath`
    - `simple_moment`
    - `spectra`
    - `srv`
    - `transforms`
    - `util`
    - `vad`

- `F:\nriet-rust-art\repo\python\pyart\`
  - vendored Py-ART compatible public shell
  - 公开 API 在这里，不要把用户 API 挪到 Rust

- `F:\nriet-rust-art\repo\tests\conftest.py`
  - source tree 测试和 installed wheel 测试的导入隔离
  - `PYART_TEST_INSTALLED=1` 时强制检查 installed `pyart` 和 installed `pyart._rust` 同源
  - 检查 installed `arm_pyart` 版本、`pyart.__version__`、Rust crate version

- `F:\nriet-rust-art\repo\tests\parity\`
  - 每个 native slice 的 exact parity 测试

- `F:\nriet-rust-art\repo\tests\rstm\`
  - RSTM synthetic 和 MinHou operational parity 测试

- `F:\nriet-rust-art\repo\tools\api_manifest.py`
  - 生成 public API manifest

- `F:\nriet-rust-art\repo\tools\rstm_reference.py`
  - RSTM Python reference freezing 工具

- `F:\nriet-rust-art\repo\tools\minhou_manifest.py`
  - MinHou operational 数据 manifest 工具

## 5. 已完成工作，Cursor 不要推翻

当前 `README.md` 的 `Compatibility Contract` 下面列出的 native slices 是已迁移/已冻结的兼容行为来源。Cursor 后续新增 slice 时必须同步更新该段。不要把 README 已记录的 native slice 删除或改成泛泛描述。

最近已完成且经过 Spark 审查的切片：

### 5.1 `correct.attenuation.get_mask_fzl`

- Rust helper：`_attenuation_end_gate_from_excluded_mask`
- Python wrapper：`_end_gate_arr_from_excluded_mask`
- parity test：`tests/parity/test_attenuation_end_gate_mask.py`
- 行为：
  - 输入是 2D bool excluded mask
  - 输出是 `int32` end gate array
  - zero-gate ray 返回 `-1`
  - first excluded gate clamp 规则保持 oracle
  - fallback 保留 Python 原异常表面

### 5.2 `map.GateMapper.mapped_radar`

- Rust helper：`_gate_mapper_apply_field_f64`
- Python dispatch：`python/pyart/map/gate_mapper.py`
- parity test：`tests/parity/test_gate_mapper_apply_field.py`
- 行为：
  - row-major assignment 顺序保持 oracle
  - duplicate mapping 使用后写覆盖
  - ray 0 skip 行为保持
  - masked source 只置 mask，不改变 payload
  - direct helper 对超大 float index 拒绝
  - sub-integer `1.9` 截断行为保持 oracle

### 5.3 `retrieve._echo_class_wt.label_classes`

- Rust helper：`_echo_class_wt_label_classes_f64`
- Python 最终 `.astype(np.int32)` 必须留在 Python
- parity test：`tests/parity/test_echo_class_wt.py`
- 行为：
  - 保留 NumPy `astype` 的 warning / `np.seterr` 行为
  - oracle 中第一轮 `wt_sum >= 3` 会被后续 `np.where(..., -3, 0)` 覆盖
  - `wt_sum=3, dbz=50` 结果是 `2`，不是 `3`
  - 不允许把这个逻辑“修正”为更合理的分类

## 6. 最近一次已知通过门禁

以下是上次已知通过的结果。Cursor 接手后必须重新跑，因为这些结果可能因后续改动变旧：

```text
cargo fmt --check
cargo test -q                         # 105 Rust tests passed
source pytest                         # 1613 passed, 973 skipped, 13 warnings
installed wheel + RSTM                # 2586 passed
installed echo_class_wt                # 27 passed
Spark review                          # no high/medium after fixes
```

重新验证命令见后面的测试门禁。

## 7. 构建和测试门禁

### 7.1 Rust 基础门禁

每次 Rust 改动后跑：

```powershell
cd F:\nriet-rust-art\repo
cargo fmt --check
cargo test -q
```

如果 `cargo fmt --check` 失败：

```powershell
cargo fmt
cargo fmt --check
```

### 7.2 source-tree Python 门禁

source-tree 测试使用 `python\pyart`，不是 installed site-packages：

```powershell
cd F:\nriet-rust-art\repo
Remove-Item Env:PYART_TEST_INSTALLED -ErrorAction SilentlyContinue
python -m pytest tests -q
```

如果只是快速 source smoke，且没有 native extension，可临时：

```powershell
$env:PYART_ALLOW_MISSING_RUST='1'
python -m pytest tests\test_smoke.py tests\api -q
Remove-Item Env:PYART_ALLOW_MISSING_RUST -ErrorAction SilentlyContinue
```

不能用这个变量绕过最终验收。

### 7.3 wheel 构建和安装门禁

Windows 上 `maturin develop` 可能因为没有 venv 失败。优先使用 wheel：

```powershell
cd F:\nriet-rust-art\repo
python -m maturin build --release --out dist
python -m pip install --force-reinstall --no-deps dist\arm_pyart-0.1.0-cp312-cp312-win_amd64.whl
```

如果 Python 版本不是 cp312，wheel 文件名会变化。用 `Get-ChildItem dist\*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1` 找最新 wheel。

### 7.4 installed package 门禁

installed mode 必须检查 `pyart` 和 `pyart._rust` 都来自 installed wheel：

```powershell
cd F:\nriet-rust-art\repo
$env:PYART_TEST_INSTALLED='1'
python -m pytest tests -q
Remove-Item Env:PYART_TEST_INSTALLED -ErrorAction SilentlyContinue
```

### 7.5 installed + RSTM operational 门禁

有 MinHou 数据时跑：

```powershell
cd F:\nriet-rust-art\repo
$env:PYART_TEST_INSTALLED='1'
$env:RSTM_DATA_ROOT='F:\nriet-rust-art\闽侯对比用数据'
python -m pytest tests -q
Remove-Item Env:PYART_TEST_INSTALLED -ErrorAction SilentlyContinue
Remove-Item Env:RSTM_DATA_ROOT -ErrorAction SilentlyContinue
```

### 7.6 RSTM 单独门禁

```powershell
cd F:\nriet-rust-art\repo
$env:RSTM_DATA_ROOT='F:\nriet-rust-art\闽侯对比用数据'
python -m pytest tests\rstm -q
Remove-Item Env:RSTM_DATA_ROOT -ErrorAction SilentlyContinue
```

### 7.7 API manifest 门禁

基础测试：

```powershell
cd F:\nriet-rust-art\repo
python -m pytest tests\api -q
```

手动生成当前 package manifest：

```powershell
cd F:\nriet-rust-art\repo
python tools\api_manifest.py --package pyart --path python --output .tmp\api-current.json --fail-on-import-error
```

oracle manifest 的路径要以实际解压 oracle 为准。如果 oracle 解压在 `F:\nriet-rust-art\_oracle\pyart-main\pyart-main`，命令形状如下：

```powershell
python tools\api_manifest.py --package pyart --path F:\nriet-rust-art\_oracle\pyart-main\pyart-main --output .tmp\api-oracle.json --fail-on-import-error
```

如果 oracle 解压路径不同，先用：

```powershell
Get-ChildItem F:\nriet-rust-art -Recurse -Directory -Filter pyart | Select-Object -First 20 FullName
```

确认实际 `pyart` package parent。

## 8. 单个 Rust native slice 的标准执行流程

每个函数都必须按这个模板执行。

### 8.1 选择候选函数

优先选择满足以下条件的函数：

- 热点明显
- 计算主体是 ndarray / scalar loop
- public API 不需要改变
- 输入输出边界明确
- oracle 行为可以用小测试冻结
- 不依赖复杂 Python object mutation
- 可以安全 fallback 到原 Python 路径

暂缓选择以下函数：

- 公开对象结构高度耦合
- 需要读写复杂 metadata
- warning/exception 表面不容易冻结
- 依赖 SciPy 复杂数值算法且边界行为未确认
- 需要多文件 I/O 状态机
- 需要 cross-record streaming 状态
- 需要改变 Radar/Grid object model

### 8.2 冻结 oracle 行为

先读当前 vendored Python 文件，再对照 frozen oracle。不要只看当前被改过的代码。

常用搜索：

```powershell
cd F:\nriet-rust-art\repo
rg "function_name" python\pyart tests src
rg "function_name" F:\nriet-rust-art\_oracle
```

如果 oracle 解压目录不存在，先从 `F:\nriet-rust-art\pyart-main.zip` 解压到外部目录，不要解压进 git tracked repo。

### 8.3 先写 parity 测试

测试必须先覆盖 Python oracle 行为，再写 Rust。每个测试文件至少包含：

- dense normal case
- empty / one-element / minimal shape
- dtype 边界
- masked array case
- NaN case
- inf case
- non-contiguous view
- read-only array 如适用
- wrong rank
- wrong shape
- negative index / oversized index 如适用
- warning parity
- exception parity
- installed-mode direct helper test

如果函数涉及 masked array，必须比较：

- `np.ma.getdata`
- `np.ma.getmaskarray`
- `fill_value`
- dtype
- shape
- masked payload 是否保留原值

如果函数涉及 dict/list metadata，必须比较：

- key 集合
- value 类型
- value 顺序
- numpy scalar vs Python scalar
- in-place mutation 位置

### 8.4 写 Rust helper

Rust helper 命名建议：

```text
_<module>_<function>_<dtype_or_variant>
```

示例：

- `_gate_mapper_apply_field_f64`
- `_echo_class_wt_label_classes_f64`
- `_attenuation_end_gate_from_excluded_mask`

Rust helper 只能是私有 native 函数。不能变成公开 `pyart` API。

### 8.5 Rust 输入校验

Rust 入口必须校验：

- ndim
- shape
- dtype
- C-contiguous
- writable output
- input/output shape 一致性
- finite 值要求
- index 范围
- allocation size 上限
- integer overflow
- float to int 转换

禁止：

- 对 Python 输入路径使用 `unwrap()`
- 对 Python 输入路径使用 `expect()`
- unchecked output indexing
- 假设 shape 一定正确
- 假设 dtype 一定正确
- 假设 contiguous
- 假设 `as_slice()` 一定成功
- float 直接 `as usize`
- 让 Rust panic 作为 Python 错误处理

### 8.6 Python dispatch 规则

Python 层必须保留原函数作为 fallback。典型结构：

```python
try:
    from pyart import _rust
except Exception:
    _rust = None

def original_public_function(...):
    if _rust is not None and _can_use_rust_fast_path(...):
        try:
            result = _rust._private_helper(...)
            _validate_rust_result(result)
            return _finish_python_surface(result)
        except (TypeError, ValueError, FloatingPointError):
            pass

    return _original_python_path(...)
```

注意：

- 不支持的输入必须 fallback，不是报新错
- 如果 oracle 本来会报错，fallback 后报原来的错
- 如果 warning 依赖 NumPy，最后一步留在 Python
- 如果 masked array fill/payload 依赖 NumPy，构造 surface 留在 Python
- 不要吞掉原本应该冒出的异常
- direct helper 测试可以要求 Rust 抛 `ValueError`，但 public function 必须保持 oracle 行为

### 8.7 验证 Rust 输出

Python 接收到 Rust 输出后，至少验证：

- `shape`
- `dtype`
- contiguous 需求
- mask shape
- payload shape
- 输出是否 alias 了不该 alias 的输入

如果验证失败，走 fallback 或抛和 oracle 一致的异常。

### 8.8 更新文档

每新增一个 native slice，更新 `README.md`：

- 写明 public function
- 写明 private helper
- 写明 fast path 支持的 dtype/shape
- 写明 fallback 条件
- 写明 exact parity 点
- 写明特殊 oracle 行为
- 如果有 tolerance，写入 `Floating Tolerance Exceptions`

## 9. 下一阶段优先级

### P0. 稳定基线

必须先做：

- [ ] 创建/确认工作分支
- [ ] 提交当前 bootstrap 基线
- [ ] 跑 `cargo fmt --check`
- [ ] 跑 `cargo test -q`
- [ ] 跑 source `python -m pytest tests -q`
- [ ] 构建 release wheel
- [ ] 强制重装 wheel
- [ ] 跑 installed `python -m pytest tests -q`
- [ ] 有数据时跑 installed + RSTM
- [ ] 把实际通过结果写入 `README.md` 或后续交接记录

### P1. 下一个推荐切片：`correct.phase_proc.smooth_and_trim_scan`

这是当前最适合继续推进的候选。

目标：

- 把 2D scan smoothing 的纯数组部分迁移到 Rust
- public API 不变
- unsupported cases fallback Python

已知 oracle 线索：

- Python 使用 `scipy.ndimage.convolve1d(x, w / w.sum(), axis=1)`
- SciPy 默认 `mode="reflect"`
- SciPy 默认 `origin=0`
- scan 方向是 axis 1
- `sg_smooth` 分支先只支持完全确认的 window

建议 Rust helper：

```text
_phase_proc_smooth_and_trim_scan_f64
```

建议 fast path 条件：

- `x` 是 dense ndarray
- `x.dtype == np.float64`
- `x.ndim == 2`
- C-contiguous
- finite
- not masked
- `weights.dtype == np.float64`
- weights 1D C-contiguous
- weights sum finite and nonzero
- `window_len >= 3`
- `x.shape[1] >= window_len`
- `sg_smooth` 只在已冻结 exact 行为时启用，否则 fallback

必须测试：

- 1 行多列
- 多行多列
- window 3
- window 5
- even/odd 相关 oracle 行为
- 边界 reflect 行为
- one-hot weights 反推出边界 index
- non-contiguous `x[:, ::2]`
- masked array fallback
- NaN fallback
- inf fallback
- float32 fallback
- width 小于 window fallback
- weights sum zero fallback
- direct helper invalid input rejection
- installed wheel direct helper exists

实现前必须用小测试冻结 SciPy reflect 行为。不要手写自认为正确的 reflect。

### P2. `correct.phase_proc` 其他切片

候选：

- unwrap masked 局部 helper
- phase edge helper
- fzl related helper
- smoothing scalar/vector helper

规则：

- 只迁移可 exact freeze 的内部 helper
- phase unwrap 涉及 mask 时必须比较 payload 和 mask
- 如果 NumPy/SciPy warning 很复杂，保留 Python

### P3. `retrieve._kdp_proc`

原计划第一批真实 native 目标：

- `_kdp_proc.lowpass_maesaka_term`
- `_kdp_proc.lowpass_maesaka_jac`

执行要求：

- 先读 `tests/parity/test_kdp_proc.py`
- 先把当前 Python oracle 行为扩充到边界测试
- Rust 只做 dense numeric core
- 任何 sparse/linear algebra 行为不确定时保留 Python
- 对浮点结果默认 exact；若确实因运算顺序出现差异，必须记录 tolerance exception

### P4. `filters` / `util`

候选：

- `_unwrap_1d`
- `_fast_edge_finder`
- circular statistics helper
- gatefilter merge helper

要求：

- index/boolean mask 行为必须 exact
- bool 输出 dtype 必须 exact
- mask merge 顺序必须 exact
- NaN 比较和 sentinel 行为必须 exact

### P5. `map` / KDTree / gridding

已完成 `GateMapper.mapped_radar` 的一部分。后续顺序：

1. 扩展 f32 field 路径，前提是 dtype/fill/mask parity 可完全锁住
2. nearest-neighbor load
3. KDTree 查询后的纯数组 assignment
4. gate-to-grid 支持函数
5. 完整 gridding 最后做

不要一开始迁移完整 gridding。gridding 涉及对象、metadata、mask、坐标、半径、权重和边界行为，风险最高。

### P6. `retrieve` 其他算法

优先纯数组函数：

- QPE coefficient / blend helpers
- QVP find index / projection helpers
- simple moment calculations
- echo classification helpers

规则：

- 分类逻辑不要重排阈值顺序
- `np.where` 链必须按 oracle 覆盖顺序实现
- `astype` 如果触发 warning，保留在 Python
- masked array surface 优先由 Python 构造

### P7. binary I/O

已有大量 `io` native slices。后续继续时要非常保守：

- SIGMET 跨 record 状态留在 Python
- malformed header 的 exception surface 留在 Python
- byte order / byte-swapped array 默认 fallback
- object dtype fallback
- noncontiguous fallback
- oversized allocation 拒绝
- checksum/hash 可在 Rust 做，但 parser 行为必须先 Python freeze

### P8. RSTM operational parser

这是单独阶段，不能直接从 Rust 开始。

顺序：

1. 用 `tools\rstm_reference.py` 冻结 Python byte-level reference
2. 用 `tools\minhou_manifest.py` 生成 operational manifest
3. 确认 gzip 识别只用 magic bytes `1F 8B`，不用扩展名
4. 确认 `.bz2` 和 `.Z` 文件实际是 gzip payload 的事实
5. 冻结 header fields
6. 冻结 payload record layout
7. 冻结 malformed 文件行为
8. 写 Rust parser
9. Rust parser 与 Python reference 逐字段比对

operational 验收事实：

- file count: `107`
- compressed bytes: `6,021,086,889`
- gzip `.bz2` CCJ files: `30`
- gzip `.Z` SPAR files: `77`
- logical decompressed payload header starts with `RSTM`

RSTM manifest 命令：

```powershell
cd F:\nriet-rust-art\repo
$env:RSTM_DATA_ROOT='F:\nriet-rust-art\闽侯对比用数据'
python tools\minhou_manifest.py --data-root $env:RSTM_DATA_ROOT --output .tmp\minhou-rstm-manifest.json
Remove-Item Env:RSTM_DATA_ROOT -ErrorAction SilentlyContinue
```

如果需要 SHA256，先确认磁盘和时间成本，再跑：

```powershell
python tools\minhou_manifest.py --data-root F:\nriet-rust-art\闽侯对比用数据 --compressed-sha256 --decompressed-sha256 --output .tmp\minhou-rstm-manifest-sha256.json
```

大 manifest 不要提交。

### P9. API parity

目标：

- public import graph 与 oracle 对齐
- public signatures 与 oracle 对齐
- public import errors 与 oracle 对齐
- package identity 与 `arm_pyart` 对齐

执行：

- 增强 `tools\api_manifest.py` 或新增 compare 工具
- 生成 oracle manifest
- 生成 rewritten manifest
- 对比 imported modules
- 对比 public names
- 对比 signatures
- 对比 import errors

不能因为 Rust 私有 helper 增加而污染 public manifest。

### P10. benchmark

benchmark 只能在 parity 全通过后做。不要为了 benchmark 改 API。

建议 benchmark 结构：

- 每个 native slice 一个 microbenchmark
- source Python path vs Rust fast path
- 小数组、中数组、 operational-like 数组
- 记录 median / p95
- 记录输入 shape/dtype
- benchmark 输出到 untracked 目录

推荐路径：

```text
benchmarks/
.tmp/benchmarks/
```

大输出不要提交。

## 10. 模块所有权拆分

如果 Cursor 用多个窗口/分支并行，按这个所有权拆，避免冲突。

### Worker A: API/package identity

只碰：

- `pyproject.toml`
- `python\pyart\__init__.py`
- `tests\api\`
- `tools\api_manifest.py`
- README 的 API/package 相关段落

职责：

- `import pyart`
- `arm_pyart`
- `pyart._rust`
- public manifest parity
- installed package import checks

### Worker B: Rust bridge + core transforms

只碰：

- `src\lib.rs`
- `src\transforms.rs`
- `python\pyart\core\`
- `tests\parity\test_transforms.py`
- `tests\parity\test_transform_edges.py`

职责：

- PyO3 registration
- ndarray bridge pattern
- coordinate transform kernels

### Worker C: correct / filters / util

只碰：

- `src\correct.rs`
- `src\filters.rs`
- `src\util.rs`
- `python\pyart\correct\`
- `python\pyart\filters\`
- `python\pyart\util\`
- 对应 `tests\parity\test_*`

职责：

- phase processing
- attenuation
- gate filters
- unwrap / edge helpers

### Worker D: map / KDTree / gridding

只碰：

- `src\map.rs`
- `python\pyart\map\`
- `tests\parity\test_gate_mapper_apply_field.py`
- `tests\parity\test_gate_to_grid_map.py`
- `tests\parity\test_map_kernels.py`

职责：

- GateMapper
- nearest neighbor assignment
- gridding 支持函数
- 完整 gridding 放最后

### Worker E: retrieve

只碰：

- `src\kdp.rs`
- `src\qpe.rs`
- `src\qvp.rs`
- `src\simple_moment.rs`
- `src\echo_class.rs`
- `python\pyart\retrieve\`
- 对应 parity tests

职责：

- `_kdp_proc`
- QPE
- QVP
- simple moment
- echo classification

### Worker F: binary I/O and RSTM

只碰：

- `src\io.rs`
- `src\rstm.rs`
- `python\pyart\io\`
- `python\pyart\aux_io\`
- `tools\rstm_reference.py`
- `tools\minhou_manifest.py`
- `tests\rstm\`
- `tests\io\`
- I/O parity tests

职责：

- SIGMET/MDV/NEXRAD/UF 等 binary helper
- RSTM freeze
- RSTM Rust parser

### Worker G: parity/benchmark harness

只碰：

- `tests\parity\`
- `tests\api\`
- `tools\`
- benchmark scripts
- README 测试结果段落

职责：

- exact compare helper
- installed-mode tests
- benchmark harness
- result recording

合并规则：每个 worker 合并前必须跑局部测试。主线合并后必须跑全量 source + installed。

## 11. Spark review 规则

每个高风险 slice 完成后，用 codex-spark/code-review 做审查。把下面模板直接给 Spark：

```text
请审查本次 Rust native slice。重点不是代码风格，而是兼容和安全。

必须检查：
1. 是否有 public API drift：import pyart、函数签名、__all__、metadata、package identity arm_pyart。
2. 是否有 Rust panic 路径：unwrap/expect、unchecked indexing、slice 越界、shape 假设。
3. 是否有 dtype/mask/fill_value drift，尤其是 masked array payload 和 mask 写入顺序。
4. 是否有 float->int/index 不安全转换，NaN/inf/超大值/负数/小数行为是否和 oracle 一致。
5. warning/exception 是否和 Python oracle 一致。
6. 非 C-contiguous、masked、unsupported dtype 是否正确 fallback。
7. installed wheel 下 pyart._rust 是否和当前 pyart 同源加载。
8. 是否需要新增 parity 测试。

请只列 high/medium 风险和必须修复项。无 high/medium 时明确说无阻塞项。
```

处理规则：

- high 必须修
- medium 原则上必须修；如不修，必须写入 README 的 residual risk
- low 可以排期，但不能影响 exact parity

## 12. 代码实现细节规范

### 12.1 Rust

Rust 代码要遵循：

- 小 helper
- 明确错误返回
- 明确 shape guard
- 明确 dtype guard
- 不共享 mutable alias
- 不暴露 unchecked output mutation
- 不引入全局状态
- 不缓存 Python object 指针
- 大 allocation 前检查元素数和字节数
- index 计算使用 checked add/mul
- offset 计算使用 checked arithmetic
- bool mask 使用清晰语义，不能反转

建议错误类型：

- shape/rank 错误：`PyValueError`
- dtype/type 错误：`PyTypeError` 或 `PyValueError`，以现有模块模式为准
- overflow/oversize：`PyValueError`
- unsupported direct helper input：`PyValueError`

public Python 函数不一定暴露这些 Rust 错误，因为很多情况应该 fallback 到 oracle Python path。

### 12.2 Python

Python fast path guard 要独立成小函数，便于测试：

```python
def _can_use_rust_foo(...):
    ...
```

不要让 guard 本身产生和 oracle 不一致的新异常。guard 失败时直接返回 False。

Python fallback 要保留原代码结构。推荐做法：

1. 把原实现抽成 `_foo_python(...)`
2. public `foo(...)` 先尝试 Rust fast path
3. fast path 不满足时调用 `_foo_python(...)`

如果原函数很短，也可以只在原函数中插入 fast path，但不要让原逻辑变得不可读。

### 12.3 Masked array

masked array 是最高风险点。必须明确：

- Rust 是否看见 mask
- Rust 是否写 mask
- Rust 是否写 payload
- masked source 的 payload 是否保留
- masked output 的未写区域 payload 是什么
- fill value 从哪里来

优先策略：

- Python 分配 masked array surface
- Rust 只填 dense payload 和 bool mask
- Python 设置 fill value
- Python 做最终 dict/object mutation

### 12.4 Warning and exception

如果原 oracle 的 warning 来自 NumPy/SciPy，不要在 Rust 里模拟，优先保留触发 warning 的 Python 最后一步。

典型例子：

- `astype(np.int32)` 的 invalid warning
- `np.sqrt` 对负数的 RuntimeWarning
- divide by zero warning
- broadcast exception
- reshape exception

Rust 可以负责 warning 前的 deterministic precursor。

## 13. 常见失败模式

遇到以下情况，不要硬写 Rust，先 fallback：

- object dtype
- byte-swapped dtype
- non-native endian
- non-contiguous view
- Fortran-order array
- negative stride
- masked scalar
- scalar array vs Python scalar 行为不清楚
- NaN sentinel 行为不清楚
- `np.asarray` 会复制并改变 alias 语义
- 原函数依赖 uninitialized masked payload
- 原函数依赖 NumPy integer wraparound
- 原函数依赖 Python list negative indexing
- 原函数依赖 `np.where` 覆盖顺序
- 原函数依赖 stable sort ties
- 文件 parser 跨 record 状态
- I/O malformed payload exception 表面

## 14. Cursor 单切片工作提示模板

每个新 slice 可以用下面提示开一个 Cursor 子任务：

```text
在 F:\nriet-rust-art\repo 中实现一个 Py-ART exact parity Rust native slice。

目标函数：
<填 public/private Python 函数>

要求：
- public API 不变
- Rust 只暴露 pyart._rust 私有 helper
- Python 保留 fallback path
- 先补 parity tests，再写 Rust
- exact 比较 shape/dtype/mask/fill_value/warnings/exceptions/NaN/numeric values
- unsupported dtype/masked/noncontiguous/nonfinite 输入 fallback Python
- Rust 不允许 unwrap/expect/unchecked indexing/unchecked float->index cast
- 完成后运行：
  cargo fmt --check
  cargo test -q
  python -m pytest <相关测试> -q
  python -m pytest tests -q
  python -m maturin build --release --out dist
  python -m pip install --force-reinstall --no-deps <最新 wheel>
  $env:PYART_TEST_INSTALLED='1'; python -m pytest tests -q
- 更新 README Compatibility Contract
```

## 15. 当前最推荐的 Cursor 第一轮任务

不要一上来做全库重构。第一轮只做以下事项：

1. 提交当前基线
2. 重跑所有门禁，确认当前状态
3. 实现 `correct.phase_proc.smooth_and_trim_scan` fast path
4. 扩充 `tests/parity/test_phase_proc_smooth.py`
5. 更新 README
6. 跑 source 全量
7. 构建 wheel
8. 跑 installed 全量
9. 跑 Spark review
10. 修复 Spark high/medium
11. 提交该 slice

建议 commit：

```text
add rust fast path for phase_proc smooth_and_trim_scan
```

## 16. 完整项目 Definition of Done

只有全部满足，才算“完整 Rust 改造完成”：

- [ ] `import pyart` 可用
- [ ] installed package metadata 是 `arm_pyart`
- [ ] installed `pyart._rust` 位于 installed `pyart` 同目录
- [ ] `pyart.__version__` 与 `pyproject.toml` 一致
- [ ] `pyart._rust.version()` 与 `Cargo.toml` 一致
- [ ] public API manifest 与 oracle 无未解释 drift
- [ ] 所有 native slices 都有 parity tests
- [ ] 所有 native slices 都在 README 记录 fast path 和 fallback 条件
- [ ] 没有未记录的 floating tolerance
- [ ] source-tree `python -m pytest tests -q` 通过
- [ ] installed wheel `PYART_TEST_INSTALLED=1 python -m pytest tests -q` 通过
- [ ] `cargo fmt --check` 通过
- [ ] `cargo test -q` 通过
- [ ] RSTM Python reference 已冻结
- [ ] RSTM Rust parser 与 Python reference exact parity
- [ ] MinHou operational RSTM acceptance 通过
- [ ] benchmark harness 可运行
- [ ] benchmark 结果记录关键热点加速
- [ ] Spark 最终审查无 high/medium
- [ ] 外部 oracle/data 没有被提交
- [ ] repo 只剩预期 tracked changes
- [ ] 最终交接文档更新到最新状态

## 17. 最终交付建议

最终交付时给出：

- branch name
- commit hash
- 测试命令和结果
- wheel 文件名
- API manifest 对比结论
- RSTM manifest/acceptance 结论
- benchmark 摘要
- Spark review 结论
- 已知 residual risk
- 下一批可优化候选

不要声称“完全兼容”除非 API、numeric parity、installed wheel、RSTM operational、Spark review 全部通过。

