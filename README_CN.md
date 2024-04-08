# hala-renderer

[English](README.md) | [中文](README_CN.md) | [日本語](README_JP.md) | [한국어](README_KO.md)

## 简介
`hala-renderer`是一个基于`hala-gfx`开发的渲染器库。它包含了各种类型的渲染器，但目前仅有一款基于硬件光追的渲染器。

## 功能特点
- **硬件光追**

## 安装
要在你的Rust项目中使用`hala-renderer`，同级目录下必须现有`hala-gfx`，`hala-renderer`编译时会去"../hala-gfx"搜索。
你可以通过在`Cargo.toml`文件中添加以下依赖来使用`hala-renderer`：

```toml
[dependencies]
hala-renderer = { path = "./hala-renderer" }
```

确保你的系统已经安装了Rust编程环境和cargo包管理器。

## 依赖关系
`hala-renderer`依赖于[hala-gfx](https://github.com/zhing2006/hala-gfx)。

请确保`hala-gfx`依赖项在使用`hala-renderer`之前已正确放到同级目录。

## 贡献
欢迎任何形式的贡献，无论是bug报告或是代码贡献。

## 许可证
`hala-renderer`根据《[GNU General Public License v3.0许可证](LICENSE)》开源。

## 联系方式
如果你有任何问题或建议，请通过创建一个issue来联系。