# hala-renderer
[![License](https://img.shields.io/badge/License-GPL3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![MSRV](https://img.shields.io/badge/rustc-1.70.0+-ab6000.svg)](https://blog.rust-lang.org/2023/06/01/Rust-1.70.0.html)

[English](README.md) | [中文](README_CN.md) | [日本語](README_JP.md) | [한국어](README_KO.md)

## Introduction
`hala-renderer` is a rendering library developed on top of `hala-gfx`. It includes various types of renderers, but currently, it only features a renderer based on hardware ray tracing.

## Features
- **Hardware Ray Tracing**

## Installation
To use `hala-renderer` in your Rust project, `hala-gfx` must already be present in the sibling directory, as `hala-renderer` will search for it at "../hala-gfx".
You can use `hala-renderer` by adding the following dependency to your `Cargo.toml` file:

```toml
[dependencies]
hala-renderer = { path = "./hala-renderer" }
```

Make sure that you have the Rust programming environment and the cargo package manager installed on your system.

## Dependencies
`hala-renderer` depends on [hala-gfx](https://github.com/zhing2006/hala-gfx).

Please ensure that the `hala-gfx` dependency is correctly placed in the sibling directory before using `hala-renderer`.

## Contribution
Contributions of any kind are welcome, whether it's bug reporting or code contributions.

## License
`hala-renderer` is open-sourced under the [GNU General Public License v3.0](LICENSE).

## Contact
If you have any questions or suggestions, please contact us by creating an issue.