# Drone-Kalman-Filter

用于 `DetectionTarget` 消息流的 Python 轨迹平滑项目。

当前目标只面向前端 2D 轨迹显示：
- 只平滑 `spatial.position.latitude`
- 只平滑 `spatial.position.longitude`
- 不改变输入消息的字段结构
- 不做跨 `deviceId` 的轨迹融合

项目分三部分：
- `drone_kalman_filter`：实时平滑主包
- `drone_kalman_validation`：离线验证与技术诊断
- `drone_kalman_visualization`：最直观的原始轨迹 / 平滑轨迹对比图

详细算法与公式说明见：
- 技术实现说明.md

代码目录上：
- `src/` 里只放实时主链路
- `tooling/` 里放验证、诊断和可视化工具

## 安装

```bash
python -m pip install -e .
```

## 这套代码解决什么问题

输入的原始无人机轨迹会出现：
- 短时跳变
- 来回抽动
- 局部锯齿

这套代码的目标是把前端画出来的 2D 轨迹线变得更 smooth，同时又尽量不把轨迹拉离真实位置。

## 输入是什么

输入是一行一个 JSON 的 `JSONL` 文件，或者实时逐条送入的 `DetectionTarget` 字典对象。

单条消息至少需要这些字段，主链路才能对它做平滑：

```json
{
  "identity": {
    "targetId": "目标唯一标识",
    "traceId": "轨迹会话标识"
  },
  "source": {
    "deviceId": "设备标识"
  },
  "spatial": {
    "position": {
      "latitude": "23.048578",
      "longitude": "113.09021"
    }
  },
  "eventTime": "2026-03-09T03:28:29.859Z"
}
```

说明：
- `targetId + deviceId` 用来隔离实时状态
- `traceId` 用来判断是否需要重新起一段新轨迹
- `eventTime` 用来计算运动学时间间隔
- `latitude / longitude` 必须能转成合法数值

如果这些关键字段缺失或非法，消息不会被平滑，会按原样透传。

## 输出是什么

输出仍然是同结构的 `DetectionTarget` JSON。

只会修改：
- `spatial.position.latitude`
- `spatial.position.longitude`

不会修改：
- `identity`
- `source`
- `traceId`
- `eventTime`
- `processTime`
- `altitude`
- `heightAgl`
- `velocity`
- `orientation`
- `extension`
- `originalData`

也就是说：
- 输入格式是什么，输出格式就还是什么
- 只是经纬度数值被替换成平滑后的结果

## 怎么喂输入，怎么拿输出

### 1. 作为 Python 插件接实时流

```python
from drone_kalman_filter import DroneKalmanFilterPlugin, PluginConfig

plugin = DroneKalmanFilterPlugin(
    PluginConfig(
        smoother_mode="robust_prefilter_kalman",
        window_size=5,
        lag_points=2,
        max_segment_gap_seconds=10.0,
        idle_flush_seconds=10.0,
    )
)

# 每来一条消息就喂一条 dict
outputs = plugin.process(message_dict)
for item in outputs:
    consume(item)

# 流结束、切流或程序退出前，记得 flush
for item in plugin.flush():
    consume(item)
```

接口说明：
- `process(message: dict) -> list[dict]`
- `process_json_line(line: str) -> list[str]`
- `flush() -> list[dict]`
- `flush_json() -> list[str]`

为什么 `process()` 返回的是 `list`：
- 这条链路是固定滞后窗口，不一定每输入一条就立刻产出一条
- 但会保证整体顺序正确
- 最终 `flush()` 后，输入条数和输出条数一致

### 2. 直接处理 JSONL 文件

生成 `kalman` 模式输出：

```bash
python -m drone_kalman_filter.cli smooth ^
  --smoother-mode kalman ^
  --input data\core-target-raw.jsonl ^
  --output out\core-target-smoothed-kalman.jsonl
```

生成 `robust_prefilter_kalman` 模式输出：

```bash
python -m drone_kalman_filter.cli smooth ^
  --smoother-mode robust_prefilter_kalman ^
  --input data\core-target-raw.jsonl ^
  --output out\core-target-smoothed-robust.jsonl
```

## 两种实时模式的区别

### `kalman`
- 直接做固定滞后窗口的 Kalman / RTS 风格平滑
- 逻辑更简单
- 对普通抖动有效

### `robust_prefilter_kalman`
- 先做一层鲁棒预处理，再进入 Kalman 主链路
- 主要用于压制短时跳变、交替尖刺和局部来回抽动
- 当前更接近前端展示目标

当前默认模式仍然是 `kalman`，是为了保持保守兼容。
如果你现在的目标是“前端轨迹线更稳”，通常优先试 `robust_prefilter_kalman`。

## 运行后会产出什么

### 1. 平滑后的 JSONL

在 `out/` 下，例如：
- `out/core-target-smoothed-kalman.jsonl`
- `out/core-target-smoothed-robust.jsonl`

这些文件就是可以继续喂给下游或前端的平滑结果。

### 2. 基础运动统计

```bash
python -m drone_kalman_filter.cli report --input data\core-target-raw.jsonl
python -m drone_kalman_filter.cli report --input out\core-target-smoothed-kalman.jsonl
python -m drone_kalman_filter.cli report --input out\core-target-smoothed-robust.jsonl
```

这会输出：
- `step_distance_m`
- `implied_speed_mps`

主要用于粗看跳变有没有下降，不是最终展示图。

### 3. 验收报告 `acceptance_summary.json`

```bash
python -m drone_kalman_filter.cli acceptance ^
  --raw data\core-target-raw.jsonl ^
  --smoothed out\core-target-smoothed-robust.jsonl ^
  --output acceptance-robust\acceptance_summary.json
```

这个文件是当前最贴近“前端能不能用”的摘要，不替代 `diagnostics/summary.json`。

顶层结构：
- `global`
- `by_device`
- `violations`

`by_device` 的重点字段：
- `point_count`
- `dt_median_s`
- `dt_p95_s`
- `normal_point_offset_p95_m`
- `global_offset_median_m`
- `global_offset_p75_m`
- `global_offset_p95_m`
- `global_offset_p99_m`
- `recovery_points_p95`
- `raw_direction_flip_count`
- `direction_flip_count`
- `release_latency_median_s`
- `release_latency_p95_s`
- `warning`

字段解释：
- `normal_point_offset_p95_m`
  - 用来衡量正常点有没有被平滑拉偏太多
  - 会排除高疑似跳变点及其后续少量点
- `global_offset_*`
  - 统计所有点的平滑偏移分布
- `recovery_points_p95`
  - 发生明显跳变后，轨迹恢复正常大致需要多少个点
- `direction_flip_count`
  - 用来衡量轨迹是否还在高频折返、来回抽动
- `warning`
  - 某个设备平滑后反而比原始数据更容易折返时，会打警告

## 最直观的轨迹对比图

如果你只想直接看：
- 原始轨迹长什么样
- 平滑后轨迹长什么样

请用独立包 `drone_kalman_visualization`：

```bash
python -m drone_kalman_visualization.compare ^
  --raw data\core-target-raw.jsonl ^
  --smoothed out\core-target-smoothed-robust.jsonl ^
  --out-dir visualization-robust-single-segment
```

输出：
- `visualization-robust-single-segment/summary.json`
- `visualization-robust-single-segment/devices/*.png`

这套图的特点：
- 每个 `(targetId, deviceId)` 一张图
- 默认只选该设备“点数最多的连续片段”
- 左图是 `raw`
- 右图是 `smoothed`
- 直接用 `longitude` 做横轴、`latitude` 做纵轴

这套图最适合肉眼判断：
- 原始线是不是有急促转折
- 平滑后是不是更顺、更圆润

## 验证和 diagnose 有什么区别

### `validate`

```bash
python -m drone_kalman_validation.validate ^
  --raw data\core-target-raw.jsonl ^
  --smoothed out\core-target-smoothed-robust.jsonl ^
  --out-dir validation-robust
```

这是展示型验证，主要看 raw 和 smoothed 的整体对比。

### `diagnose`

```bash
python -m drone_kalman_validation.diagnose ^
  --raw data\core-target-raw.jsonl ^
  --smoothed out\core-target-smoothed-robust.jsonl ^
  --baseline-output out\core-target-baseline.jsonl ^
  --out-dir diagnostics-robust
```

这是技术诊断，不是前端效果图。

它会同时看三条轨迹：
- `raw`
- `current`：当前实时算法输出
- `baseline RTS`：离线整段平滑结果

用途是判断：
- 当前实时算法是不是明显跑飞了
- 还是 baseline 自己就不可信

所以：
- 想看前端效果，用 `drone_kalman_visualization.compare`
- 想查算法责任归属，用 `diagnose`

## baseline 是什么

```bash
python -m drone_kalman_validation.baseline ^
  --input data\core-target-raw.jsonl ^
  --output out\core-target-baseline.jsonl
```

baseline 是离线整段 RTS 平滑结果，只用于辅助诊断。

它和实时插件的区别：
- 实时插件：固定滞后小窗口，面向实时前端
- baseline：固定区间整段平滑，面向离线对照

当前 baseline 仍然不是前端验收裁判，只是辅助参考。

## 目录说明

- `src/drone_kalman_filter`
  - 实时主链路代码
- `tooling/drone_kalman_validation`
  - 离线验证与 diagnose
- `tooling/drone_kalman_visualization`
  - 原始轨迹 / 平滑轨迹对比图
- `data`
  - 原始输入样本
- `out`
  - 平滑输出与 baseline 输出
- `acceptance-*`
  - 验收摘要
- `validation-*`
  - 展示型验证结果
- `diagnostics-*`
  - 技术诊断结果
- `visualization-*`
  - 最直观的前端视角轨迹图

## 当前默认约束

- 只做 2D `lat/lon` 平滑
- 前端按 `deviceId` 分别画线
- 不做跨设备融合
- 当前真实样本是 `1 个 targetId × 5 个 deviceId`
- 输出 JSON 结构必须与输入保持一致

## 测试

```bash
python -m pytest
```

当前测试覆盖：
- 实时平滑输入输出结构保持
- `kalman` 与 `robust_prefilter_kalman`
- 单点跳变、连续跳变、burst 污染回归
- 多目标交错与同目标多设备隔离
- 验收指标计算
- 真实样本回放
- baseline 与 diagnose
