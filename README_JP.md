# hala-renderer

[English](README.md) | [中文](README_CN.md) | [日本語](README_JP.md) | [한국어](README_KO.md)

## 紹介
`hala-renderer`は`hala-gfx`を基に開発されたレンダリングライブラリです。様々なタイプのレンダラーを含んでいますが、現在はハードウェアレイトレーシングに基づいたレンダラーのみを特徴としています。

## 機能特徴
- **ハードウェアレイトレーシング**

## インストール
Rustプロジェクトで`hala-renderer`を使用するには、同じディレクトリに`hala-gfx`が存在している必要があります。`hala-renderer`は"../hala-gfx"で検索します。
`Cargo.toml`ファイルに以下の依存関係を追加することで`hala-renderer`を使用できます：

```toml
[dependencies]
hala-renderer = { path = "./hala-renderer" }
```

システムにRustプログラミング環境とcargoパッケージマネージャがインストールされていることを確認してください。

## 依存関係
`hala-renderer`は[hala-gfx](https://github.com/zhing2006/hala-gfx)に依存しています。

`hala-renderer`を使用する前に、`hala-gfx`依存関係が同じディレクトリに正しく配置されていることを確認してください。

## 貢献
バグ報告やコードの貢献など、あらゆる種類の貢献を歓迎します。

## ライセンス
`hala-renderer`は[GNU General Public License v3.0](LICENSE)でオープンソース化されています。

## 連絡先
ご質問や提案がある場合は、issueを作成してご連絡ください。