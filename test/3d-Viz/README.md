# 3D Viz Demo

这个目录是一个独立的 Three.js 视觉原型，用来测试：

- 三维空间中的问卷纸布局
- Trip Diary 与 SP Survey 的双页展示
- 文本框、checkbox、radio 的自动填写动画
- 旋转、缩放、平移相机观察

## 文件

- `index.html`: 演示页面入口
- `demo.js`: Three.js 场景与动画逻辑

## 运行方式

建议通过本地静态服务打开，而不是直接双击 `file://`：

```bash
cd /Users/convel/PycharmProjects/spSurvey/test/3d-Viz
python3 -m http.server 8080
```

然后打开：

- <http://127.0.0.1:8080>

## 当前实现

- 使用 `Three.js + CSS3DRenderer + OrbitControls`
- 左页：Trip Diary
- 右页：SP Survey
- 顶部按钮可触发单页或整套动画
- 支持自动旋转开关

## 后续可扩展方向

- 把问卷纸之间的连线改成数据流粒子
- 在场景里加入训练节点、权重节点与 dashboard 节点
- 用颜色/发光表现 reward、loss、权重更新状态
